"""Manual unit check for DVR chain speculative sampling.

DVR self draft runs topk=1, so EAGLE's tree sampling reduces to a single
chain. The production path intentionally keeps only the Triton CUDA kernel; this
test owns a small Python reference so serving code does not carry a slow fallback.

The reference mirrors the EAGLE verify contract used by DVR:

- candidates[:, 0] is the already verified anchor;
- candidates[:, 1:] are draft tokens checked against target probabilities;
- every accepted draft token is written to the previous retrieve slot;
- the last live retrieve slot receives either a target sample or a residual
  target-minus-draft sample.

Example:

PYTHONPATH=python conda run -n dvr_dev python \
  test/manual/dvr/test_dvr_chain_speculative_sampling.py \
  --device cuda --batch-sizes 1,2,8 --num-slots 16
"""

from __future__ import annotations

import argparse
from typing import List

import torch

from sglang.srt.speculative.dvr_utils import chain_speculative_sampling


def reference_chain_speculative_sampling(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> None:
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    for bid in range(batch_size):
        root_index = int(retrive_index[bid, 0].item())
        if root_index < 0 or root_index >= predicts.numel():
            accept_token_num[bid] = 0
            continue

        accept_index[bid, 0] = root_index
        last_index = root_index
        prob_row = 0
        num_accepted = 0
        all_accepted = True

        for step in range(1, num_slots):
            draft_token = int(candidates[bid, step].item())
            next_index = int(retrive_index[bid, step].item())
            can_accept = (
                0 <= draft_token < vocab_size
                and prob_row < target_probs.shape[1]
                and 0 <= next_index < predicts.numel()
                and num_accepted + 1 < accept_index.shape[1]
                and (
                    uniform_samples[bid, step - 1]
                    * draft_probs[bid, prob_row, draft_token]
                    < target_probs[bid, prob_row, draft_token]
                ).item()
            )
            if not can_accept:
                all_accepted = False
                break

            predicts[last_index] = draft_token
            num_accepted += 1
            prob_row = step
            last_index = next_index
            accept_index[bid, num_accepted] = last_index

        accept_token_num[bid] = num_accepted
        final_probs = target_probs[bid, prob_row]
        if not all_accepted:
            final_probs = torch.clamp(
                final_probs - draft_probs[bid, prob_row], min=0
            )

        norm = final_probs.sum()
        if norm <= 0:
            final_token = torch.argmax(target_probs[bid, prob_row])
        else:
            cdf = torch.cumsum(final_probs, dim=0)
            final_token = torch.searchsorted(
                cdf,
                uniform_samples_for_final_sampling[bid] * norm,
                right=True,
            ).clamp(max=vocab_size - 1)

        predicts[last_index] = final_token.to(predicts.dtype)


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def make_probs(
    *,
    batch: int,
    num_slots: int,
    vocab_size: int,
    device: str,
    seed: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    target_probs = torch.softmax(
        torch.randn(
            batch,
            num_slots,
            vocab_size,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        dim=-1,
    )
    draft_probs = torch.softmax(
        torch.randn(
            batch,
            num_slots,
            vocab_size,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        dim=-1,
    )
    candidates = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch, num_slots),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    coins = torch.rand(
        batch,
        num_slots,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    if mode == "all_accept":
        draft_probs = target_probs.clone()
        coins.zero_()
    elif mode == "early_reject":
        # Force rejection at the first draft token while keeping a non-trivial
        # residual distribution for final sampling.
        draft_probs = target_probs.clone()
        token = candidates[:, 1]
        row = torch.arange(batch, device=device)
        draft_probs[row, 0, token] = 0.75
        target_probs[row, 0, token] = 0.05
        draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
        coins.fill_(0.99)
    elif mode == "zero_residual":
        draft_probs = target_probs.clone()
        candidates[:, 1] = vocab_size
        coins.fill_(0.99)

    return candidates, target_probs.contiguous(), draft_probs.contiguous(), coins


def run_case(
    *,
    device: str,
    batch: int,
    num_slots: int,
    vocab_size: int,
    seed: int,
    mode: str,
) -> None:
    candidates, target_probs, draft_probs, coins = make_probs(
        batch=batch,
        num_slots=num_slots,
        vocab_size=vocab_size,
        device=device,
        seed=seed,
        mode=mode,
    )
    retrieve_index = (
        torch.arange(batch * num_slots, dtype=torch.long, device=device)
        .reshape(batch, num_slots)
        .contiguous()
    )
    final_coins = torch.rand(
        batch,
        dtype=torch.float32,
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed + 1009),
    )

    actual_predicts = torch.full(
        (batch * num_slots + 1,), -1, dtype=torch.int32, device=device
    )
    actual_accept_index = torch.full(
        (batch, num_slots), -1, dtype=torch.int32, device=device
    )
    actual_accept_num = torch.empty((batch,), dtype=torch.int32, device=device)

    chain_speculative_sampling(
        predicts=actual_predicts,
        accept_index=actual_accept_index,
        accept_token_num=actual_accept_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=torch.empty_like(retrieve_index),
        retrive_next_sibling=torch.empty_like(retrieve_index),
        uniform_samples=coins,
        uniform_samples_for_final_sampling=final_coins,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    torch.cuda.synchronize()

    expected_predicts = torch.full(
        actual_predicts.shape, -1, dtype=actual_predicts.dtype
    )
    expected_accept_index = torch.full(
        actual_accept_index.shape, -1, dtype=actual_accept_index.dtype
    )
    expected_accept_num = torch.empty(actual_accept_num.shape, dtype=torch.int32)
    reference_chain_speculative_sampling(
        predicts=expected_predicts,
        accept_index=expected_accept_index,
        accept_token_num=expected_accept_num,
        candidates=candidates.cpu(),
        retrive_index=retrieve_index.cpu(),
        uniform_samples=coins.cpu(),
        uniform_samples_for_final_sampling=final_coins.cpu(),
        target_probs=target_probs.cpu(),
        draft_probs=draft_probs.cpu(),
    )

    expected_predicts = expected_predicts.to(device)
    expected_accept_index = expected_accept_index.to(device)
    expected_accept_num = expected_accept_num.to(device)
    if not torch.equal(actual_predicts, expected_predicts):
        raise AssertionError(
            f"predict mismatch: mode={mode} batch={batch} "
            f"actual={actual_predicts.cpu().tolist()} "
            f"expected={expected_predicts.cpu().tolist()}"
        )
    if not torch.equal(actual_accept_index, expected_accept_index):
        raise AssertionError(
            f"accept_index mismatch: mode={mode} batch={batch} "
            f"actual={actual_accept_index.cpu().tolist()} "
            f"expected={expected_accept_index.cpu().tolist()}"
        )
    if not torch.equal(actual_accept_num, expected_accept_num):
        raise AssertionError(
            f"accept_num mismatch: mode={mode} batch={batch} "
            f"actual={actual_accept_num.cpu().tolist()} "
            f"expected={expected_accept_num.cpu().tolist()}"
        )

    print(
        "PASS mode={} batch={} num_slots={} vocab={} accept_num={}".format(
            mode,
            batch,
            num_slots,
            vocab_size,
            actual_accept_num.cpu().tolist(),
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=257)
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the DVR chain sampling test.")

    modes = ("random", "all_accept", "early_reject", "zero_residual")
    for batch_i, batch in enumerate(parse_int_list(args.batch_sizes)):
        for mode_i, mode in enumerate(modes):
            run_case(
                device=args.device,
                batch=batch,
                num_slots=args.num_slots,
                vocab_size=args.vocab_size,
                seed=args.seed + batch_i * 17 + mode_i,
                mode=mode,
            )
    print("ALL_OK True", flush=True)


if __name__ == "__main__":
    main()
