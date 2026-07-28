import pytest
import torch

from sglang.srt.speculative.dvr_reject_sampling import (
    dvr_chain_rejection_sample,
    dvr_sample_from_probs,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def run_chain_sample(target_probs, draft_probs, candidate, seed=2026):
    device = target_probs.device
    candidates = torch.tensor([[0, candidate]], dtype=torch.int64, device=device)
    retrieve_index = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    predicts = torch.full((2,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((1, 2), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.empty((1,), dtype=torch.int32, device=device)

    dvr_chain_rejection_sample(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrieve_index=retrieve_index,
        target_probs=target_probs,
        draft_probs=draft_probs,
        sampling_seed=torch.tensor([seed], dtype=torch.int64, device=device),
        positions=torch.tensor([[64, 65]], dtype=torch.int64, device=device),
    )
    return predicts, accept_index, accept_token_num


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("residual_kind", ["zero", "infinite"])
def test_degenerate_residual_samples_target_without_padded_token(residual_kind):
    tokenizer_vocab_size = 248077
    padded_vocab_size = 248320
    target_row = torch.zeros(padded_vocab_size, dtype=torch.float32, device="cuda")
    target_row[:3] = torch.tensor([0.2, 0.3, 0.5], device="cuda")
    target_probs = torch.stack((target_row, target_row)).unsqueeze(0)
    draft_probs = target_row.reshape(1, 1, -1).clone()
    if residual_kind == "infinite":
        draft_probs[0, 0, 0] = -float("inf")

    predicts, accept_index, accept_token_num = run_chain_sample(
        target_probs,
        draft_probs,
        candidate=padded_vocab_size - 1,
    )

    assert accept_token_num.item() == 0
    assert accept_index[0, 0].item() == 0
    assert 0 <= predicts[0].item() < tokenizer_vocab_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_proposal_never_selects_zero_probability_padded_vocab():
    tokenizer_vocab_size = 248077
    padded_vocab_size = 248320
    probs = torch.zeros((8, padded_vocab_size), device="cuda")
    probs[:, :3] = torch.tensor([0.2, 0.3, 0.5], device="cuda")

    tokens = dvr_sample_from_probs(
        probs,
        torch.arange(8, dtype=torch.int64, device="cuda") + 2026,
        torch.arange(8, dtype=torch.int64, device="cuda") + 64,
    )

    assert torch.all(tokens < tokenizer_vocab_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_all_accepted_path_samples_final_target_row():
    target_probs = torch.tensor(
        [[[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]],
        dtype=torch.float32,
        device="cuda",
    )
    draft_probs = torch.tensor([[[0.1, 0.2, 0.7]]], dtype=torch.float32, device="cuda")

    predicts, accept_index, accept_token_num = run_chain_sample(
        target_probs,
        draft_probs,
        candidate=2,
    )

    assert accept_token_num.item() == 1
    assert accept_index.tolist() == [[0, 1]]
    assert predicts[0].item() == 2
    assert 0 <= predicts[1].item() < target_probs.shape[-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_equal_distributions_accept_full_dvr16_chain():
    batch_size = 3
    num_draft_tokens = 15
    num_slots = num_draft_tokens + 1
    vocab_size = 32
    device = "cuda"

    generator = torch.Generator().manual_seed(2026)
    draft_probs = torch.rand(
        batch_size, num_draft_tokens, vocab_size, generator=generator
    )
    draft_probs /= draft_probs.sum(dim=-1, keepdim=True)
    draft_probs = draft_probs.to(device)
    draft_tokens = draft_probs.argmax(dim=-1)

    bonus_tokens = torch.tensor([3, 11, 29], device=device)
    bonus_probs = torch.zeros(
        batch_size, vocab_size, dtype=torch.float32, device=device
    )
    bonus_probs.scatter_(1, bonus_tokens.unsqueeze(1), 1.0)
    target_probs = torch.cat((draft_probs, bonus_probs.unsqueeze(1)), dim=1)
    candidates = torch.cat(
        (
            torch.zeros(batch_size, 1, dtype=torch.int64, device=device),
            draft_tokens,
        ),
        dim=1,
    )
    retrieve_index = torch.arange(
        batch_size * num_slots, dtype=torch.int32, device=device
    ).view(batch_size, num_slots)
    predicts = torch.full(
        (batch_size * num_slots,), -1, dtype=torch.int32, device=device
    )
    accept_index = torch.full(
        (batch_size, num_slots), -1, dtype=torch.int32, device=device
    )
    accepted_drafts = torch.empty(batch_size, dtype=torch.int32, device=device)

    dvr_chain_rejection_sample(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accepted_drafts,
        candidates=candidates,
        retrieve_index=retrieve_index,
        target_probs=target_probs,
        draft_probs=draft_probs,
        sampling_seed=torch.tensor([2026, 2030, 99], dtype=torch.int64, device=device),
        positions=torch.arange(
            64,
            64 + batch_size * num_slots,
            dtype=torch.int64,
            device=device,
        ).view(batch_size, num_slots),
    )

    assert accepted_drafts.tolist() == [num_draft_tokens] * batch_size
    assert torch.equal(accept_index, retrieve_index)
    expected = torch.cat((draft_tokens, bonus_tokens.unsqueeze(1)), dim=1)
    assert torch.equal(predicts.view(batch_size, num_slots), expected.to(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_proposal_is_repeatable_and_batch_order_independent():
    probs = torch.tensor(
        [
            [0.05, 0.15, 0.30, 0.50],
            [0.60, 0.25, 0.10, 0.05],
            [0.20, 0.10, 0.40, 0.30],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    seeds = torch.tensor([2026, 2030, 99], dtype=torch.int64, device="cuda")
    positions = torch.tensor([64, 127, 511], dtype=torch.int64, device="cuda")

    expected = dvr_sample_from_probs(probs, seeds, positions)
    repeated = dvr_sample_from_probs(probs, seeds, positions)
    permutation = torch.tensor([2, 0, 1], device="cuda")
    permuted = dvr_sample_from_probs(
        probs[permutation], seeds[permutation], positions[permutation]
    )

    assert torch.equal(expected, repeated)
    assert torch.equal(permuted, expected[permutation])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_rejection_is_repeatable_and_batch_order_independent():
    batch_size = 3
    num_slots = 4
    device = "cuda"
    draft_probs = torch.tensor(
        [
            [[0.60, 0.20, 0.10, 0.05, 0.05]] * (num_slots - 1),
            [[0.05, 0.10, 0.15, 0.30, 0.40]] * (num_slots - 1),
            [[0.20, 0.20, 0.20, 0.20, 0.20]] * (num_slots - 1),
        ],
        dtype=torch.float32,
        device=device,
    )
    target_probs = torch.tensor(
        [
            [[0.10, 0.20, 0.30, 0.25, 0.15]] * num_slots,
            [[0.30, 0.25, 0.20, 0.15, 0.10]] * num_slots,
            [[0.05, 0.15, 0.50, 0.20, 0.10]] * num_slots,
        ],
        dtype=torch.float32,
        device=device,
    )
    candidates = torch.cat(
        (
            torch.zeros(batch_size, 1, dtype=torch.int64, device=device),
            draft_probs.argmax(dim=-1),
        ),
        dim=1,
    )
    seeds = torch.tensor([2026, 2030, 99], dtype=torch.int64, device=device)
    positions = torch.tensor(
        [[64, 65, 66, 67], [127, 128, 129, 130], [511, 512, 513, 514]],
        dtype=torch.int64,
        device=device,
    )

    def run(order):
        retrieve_index = torch.arange(
            batch_size * num_slots, dtype=torch.int32, device=device
        ).view(batch_size, num_slots)
        predicts = torch.full(
            (batch_size * num_slots,), -1, dtype=torch.int32, device=device
        )
        accept_index = torch.full(
            (batch_size, num_slots), -1, dtype=torch.int32, device=device
        )
        accept_lens = torch.empty(batch_size, dtype=torch.int32, device=device)
        dvr_chain_rejection_sample(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_lens,
            candidates=candidates[order],
            retrieve_index=retrieve_index,
            target_probs=target_probs[order],
            draft_probs=draft_probs[order],
            sampling_seed=seeds[order],
            positions=positions[order],
        )
        return predicts.view(batch_size, num_slots), accept_lens

    identity = torch.arange(batch_size, device=device)
    permutation = torch.tensor([2, 0, 1], device=device)
    expected_predicts, expected_lens = run(identity)
    repeated_predicts, repeated_lens = run(identity)
    permuted_predicts, permuted_lens = run(permutation)

    assert torch.equal(expected_predicts, repeated_predicts)
    assert torch.equal(expected_lens, repeated_lens)
    assert torch.equal(permuted_predicts, expected_predicts[permutation])
    assert torch.equal(permuted_lens, expected_lens[permutation])
