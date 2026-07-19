from types import SimpleNamespace

import pytest
import torch

import sglang.srt.speculative.dvr_worker as dvr_worker_module
from sglang.srt.speculative.dvr_worker import (
    DecodeVerifyRollbackWorkerV2,
    _dvr_proposal_probs,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _sampling_info(top_ks, top_ps, min_ps):
    return SimpleNamespace(
        top_ks=torch.tensor(top_ks),
        top_ps=torch.tensor(top_ps),
        min_ps=torch.tensor(min_ps),
        need_top_k_sampling=any(value != 0 for value in top_ks),
        need_top_p_sampling=any(value != 1.0 for value in top_ps),
        need_min_p_sampling=any(value != 0.0 for value in min_ps),
    )


def test_dvr_algorithm_contracts():
    self_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
    eagle_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE

    assert self_draft.is_dvr_self_draft() and not self_draft.is_dvr_eagle()
    assert eagle_draft.is_dvr_eagle() and not eagle_draft.is_eagle()
    assert not self_draft.has_draft_kv()
    assert eagle_draft.has_draft_kv()


def test_proposal_filter_matches_target_distribution():
    proposal = _dvr_proposal_probs(
        torch.tensor([[0.6, 0.3, 0.1]]),
        _sampling_info([3], [1.0], [0.5]),
        "pytorch",
    )
    torch.testing.assert_close(proposal, torch.tensor([[2 / 3, 1 / 3, 0.0]]))

    joint = _dvr_proposal_probs(
        torch.tensor([[0.6, 0.25, 0.15]]),
        _sampling_info([2], [0.7], [0.0]),
        "pytorch",
    )
    torch.testing.assert_close(joint, torch.tensor([[12 / 17, 5 / 17, 0.0]]))


def test_proposal_filter_handles_mixed_and_repeated_sampling_rows():
    probs = torch.tensor([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])
    filtered = _dvr_proposal_probs(
        probs,
        _sampling_info([1, 3], [1.0, 1.0], [0.0, 0.0]),
        "pytorch",
    )
    torch.testing.assert_close(filtered[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(filtered[1], probs[1])

    repeated = _dvr_proposal_probs(
        torch.tensor([[0.6, 0.4]] * 4),
        _sampling_info([1, 2], [1.0, 1.0], [0.0, 0.0]),
        "pytorch",
        repeat=2,
    )
    torch.testing.assert_close(repeated[:2], torch.tensor([[1.0, 0.0]] * 2))
    torch.testing.assert_close(repeated[2:], torch.tensor([[0.6, 0.4]] * 2))


@pytest.mark.parametrize("is_dvr_eagle", [False, True])
def test_short_prefix_uses_one_root_verify_sentinel(is_dvr_eagle):
    worker = object.__new__(DecodeVerifyRollbackWorkerV2)
    worker.is_dvr_eagle = is_dvr_eagle
    worker.device = "cpu"
    worker.num_draft_tokens = 4
    worker._chain_retrieve_index = torch.arange(8).view(2, 4)
    worker._chain_retrieve_sibling = torch.full((2, 4), -1)
    worker._chain_position_offsets = torch.arange(4)
    batch = SimpleNamespace(
        spec_info=dvr_worker_module.EagleDraftInput(bonus_tokens=torch.tensor([6, 7])),
        seq_lens=torch.tensor([1, 65]),
        seq_lens_cpu=torch.tensor([1, 65]),
        seq_lens_sum=66,
    )

    verify_input = worker._build_one_root_verify_input(batch)

    assert verify_input.draft_token.tolist() == [6] * 4 + [7] * 4
    assert verify_input.spec_steps == 0
    assert verify_input.retrieve_next_token.eq(-1).all()
    assert verify_input.positions.tolist() == [1, 2, 3, 4, 65, 66, 67, 68]
