import pytest
import torch

from sglang.srt.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def run_chain_sample(target_probs, draft_probs, candidate, final_coin):
    device = target_probs.device
    candidates = torch.tensor([[0, candidate]], dtype=torch.int64, device=device)
    retrieve_index = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    predicts = torch.full((2,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((1, 2), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.empty((1,), dtype=torch.int32, device=device)

    chain_speculative_sampling_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=torch.full((1, 2), 0.5, device=device),
        uniform_samples_for_final_sampling=torch.tensor(
            [final_coin], dtype=torch.float32, device=device
        ),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    return predicts, accept_index, accept_token_num


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("residual_kind", ["zero", "infinite"])
def test_degenerate_residual_samples_target_without_padded_token(residual_kind):
    padded_vocab_size = 248320
    tokenizer_vocab_size = 248077
    target_row = torch.zeros(
        padded_vocab_size, dtype=torch.float32, device="cuda"
    )
    target_row[:3] = torch.tensor([0.2, 0.3, 0.5], device="cuda")
    target_probs = torch.stack((target_row, target_row)).unsqueeze(0)
    draft_probs = target_row.reshape(1, 1, -1).clone()
    if residual_kind == "infinite":
        draft_probs[0, 0, 0] = -float("inf")

    predicts, accept_index, accept_token_num = run_chain_sample(
        target_probs,
        draft_probs,
        candidate=padded_vocab_size - 1,
        final_coin=0.25,
    )

    assert accept_token_num.item() == 0
    assert accept_index[0, 0].item() == 0
    assert predicts[0].item() == 1
    assert predicts[0].item() < tokenizer_vocab_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_all_accepted_path_still_samples_final_target_row():
    target_probs = torch.tensor(
        [[[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]],
        dtype=torch.float32,
        device="cuda",
    )
    draft_probs = torch.tensor(
        [[[0.1, 0.2, 0.7]]], dtype=torch.float32, device="cuda"
    )

    predicts, accept_index, accept_token_num = run_chain_sample(
        target_probs,
        draft_probs,
        candidate=2,
        final_coin=0.6,
    )

    assert accept_token_num.item() == 1
    assert accept_index.tolist() == [[0, 1]]
    assert predicts.tolist() == [2, 2]
