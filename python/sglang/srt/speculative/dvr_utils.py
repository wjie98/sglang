from __future__ import annotations

from contextlib import contextmanager

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - non-CUDA builds use the torch fallback.
    triton = None
    tl = None


@contextmanager
def dvr_causal_verify_cuda_graph_metadata(
    model_runner, attn_backend, forward_mode, spec_info, fallback_custom_mask=None
):
    """Keep DVR cuda graph verify on causal attention without backend edits.

    Some attention backends still read `spec_info.custom_mask.shape` while
    building cuda-graph metadata. For DVR topk=1 the real mask is causal, so
    temporarily provide the graph buffer only to satisfy metadata shape code,
    then clear the captured metadata before graph capture/replay uses it.
    """

    old_custom_mask = getattr(spec_info, "custom_mask", None)
    should_clear = (
        model_runner.spec_algorithm.is_decode_verify_rollback()
        and forward_mode.is_target_verify()
        and spec_info is not None
    )
    if should_clear and old_custom_mask is None and fallback_custom_mask is not None:
        # DVR target verify is a topk=1 chain, so the real attention mask is
        # causal and `custom_mask` should stay None. Some cuda-graph metadata
        # builders still read `spec_info.custom_mask.shape` before producing
        # their metadata, so provide the fixed graph buffer only for that shape
        # bookkeeping. The metadata is cleared below before graph capture/replay.
        spec_info.custom_mask = fallback_custom_mask
    try:
        yield
    finally:
        if should_clear:
            spec_info.custom_mask = old_custom_mask
            backends = [attn_backend]
            full_attn_backend = getattr(attn_backend, "full_attn_backend", None)
            if full_attn_backend is not None:
                backends.append(full_attn_backend)
            for backend in backends:
                metadata = getattr(backend, "forward_metadata", None)
                if metadata is None:
                    continue
                # Restore DVR's causal semantics after metadata construction:
                # do not let the temporary tree-mask buffer select a custom-mask
                # attention path in the captured graph or replay metadata. For
                # hybrid GDN models the real attention metadata lives inside the
                # nested full-attention backend, so clear both levels.
                if hasattr(metadata, "custom_mask"):
                    metadata.custom_mask = None
                if hasattr(metadata, "mask_indptr"):
                    metadata.mask_indptr = None


if triton is not None:

    @triton.jit
    def _chain_speculative_sampling_kernel(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        stride_cand_b: tl.constexpr,
        stride_cand_s: tl.constexpr,
        stride_idx_b: tl.constexpr,
        stride_idx_s: tl.constexpr,
        stride_uni_b: tl.constexpr,
        stride_uni_s: tl.constexpr,
        stride_tp_b: tl.constexpr,
        stride_tp_s: tl.constexpr,
        stride_tp_v: tl.constexpr,
        stride_dp_b: tl.constexpr,
        stride_dp_s: tl.constexpr,
        stride_dp_v: tl.constexpr,
        NUM_SLOTS: tl.constexpr,
        VOCAB_SIZE: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        bid = tl.program_id(0)
        cand_base = candidates + bid * stride_cand_b
        index_base = retrive_index + bid * stride_idx_b
        uniform_base = uniform_samples + bid * stride_uni_b

        prob_row = 0
        root_index = tl.load(index_base)
        tl.store(accept_index + bid * stride_idx_b, root_index)
        last_index = root_index
        num_accepted = 0
        step = 1
        continue_verifying = 1
        while (step < NUM_SLOTS) and (continue_verifying == 1):
            draft_token = tl.load(cand_base + step * stride_cand_s)
            target_prob = tl.load(
                target_probs
                + bid * stride_tp_b
                + prob_row * stride_tp_s
                + draft_token * stride_tp_v
            )
            draft_prob = tl.load(
                draft_probs
                + bid * stride_dp_b
                + prob_row * stride_dp_s
                + draft_token * stride_dp_v
            )
            coin = tl.load(uniform_base + (step - 1) * stride_uni_s)
            if coin * draft_prob < target_prob:
                tl.store(predicts + last_index, draft_token)
                num_accepted += 1
                prob_row = step
                last_index = tl.load(index_base + step * stride_idx_s)
                tl.store(
                    accept_index + bid * stride_idx_b + num_accepted * stride_idx_s,
                    last_index,
                )
                step += 1
            else:
                continue_verifying = 0

        tl.store(accept_token_num + bid, num_accepted)
        all_accepted = continue_verifying

        target_base = target_probs + bid * stride_tp_b + prob_row * stride_tp_s
        draft_base = draft_probs + bid * stride_dp_b + prob_row * stride_dp_s

        norm_sum = 0.0
        for v_start in range(0, VOCAB_SIZE, BLOCK_V):
            offsets = v_start + tl.arange(0, BLOCK_V)
            mask = offsets < VOCAB_SIZE
            target_val = tl.load(
                target_base + offsets * stride_tp_v, mask=mask, other=0.0
            )
            if all_accepted:
                residual_val = target_val
            else:
                draft_val = tl.load(
                    draft_base + offsets * stride_dp_v, mask=mask, other=0.0
                )
                residual_val = tl.maximum(target_val - draft_val, 0.0)
            norm_sum += tl.sum(residual_val)

        final_token = VOCAB_SIZE - 1
        if norm_sum <= 0.0:
            best_val = -float("inf")
            for v_start in range(0, VOCAB_SIZE, BLOCK_V):
                offsets = v_start + tl.arange(0, BLOCK_V)
                mask = offsets < VOCAB_SIZE
                target_val = tl.load(
                    target_base + offsets * stride_tp_v, mask=mask, other=-float("inf")
                )
                block_best = tl.max(target_val, axis=0)
                if block_best > best_val:
                    best_val = block_best
                    final_token = v_start + tl.argmax(target_val, axis=0)
        else:
            target_u = tl.load(uniform_samples_for_final_sampling + bid) * norm_sum
            cdf = 0.0
            found = 0
            for v_start in range(0, VOCAB_SIZE, BLOCK_V):
                if found == 0:
                    offsets = v_start + tl.arange(0, BLOCK_V)
                    mask = offsets < VOCAB_SIZE
                    target_val = tl.load(
                        target_base + offsets * stride_tp_v, mask=mask, other=0.0
                    )
                    if all_accepted:
                        residual_val = target_val
                    else:
                        draft_val = tl.load(
                            draft_base + offsets * stride_dp_v, mask=mask, other=0.0
                        )
                        residual_val = tl.maximum(target_val - draft_val, 0.0)
                    block_cdf = cdf + tl.cumsum(residual_val, axis=0)
                    matched = block_cdf > target_u
                    if tl.max(matched, axis=0):
                        final_token = v_start + tl.argmax(matched.to(tl.int32), axis=0)
                        found = 1
                    cdf += tl.sum(residual_val)

        tl.store(predicts + last_index, final_token)


def _chain_speculative_sampling_torch(
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
    for bid in range(batch_size):
        root_index = retrive_index[bid, 0]
        accept_index[bid, 0] = root_index
        last_index = root_index
        prob_row = 0
        num_accepted = 0
        all_accepted = True

        for step in range(1, num_slots):
            draft_token = candidates[bid, step]
            p = target_probs[bid, prob_row, draft_token]
            q = draft_probs[bid, prob_row, draft_token]
            if uniform_samples[bid, step - 1] * q < p:
                predicts[last_index] = draft_token
                num_accepted += 1
                prob_row = step
                last_index = retrive_index[bid, step]
                accept_index[bid, num_accepted] = last_index
            else:
                all_accepted = False
                break

        accept_token_num[bid] = num_accepted
        final_probs = target_probs[bid, prob_row]
        if not all_accepted:
            final_probs = torch.clamp(final_probs - draft_probs[bid, prob_row], min=0)

        norm = final_probs.sum()
        if norm <= 0:
            final_token = torch.argmax(target_probs[bid, prob_row])
        else:
            cdf = torch.cumsum(final_probs, dim=0)
            final_token = torch.searchsorted(
                cdf, uniform_samples_for_final_sampling[bid] * norm
            )
            final_token = torch.clamp(final_token, max=final_probs.numel() - 1)

        predicts[last_index] = final_token.to(predicts.dtype)


def chain_speculative_sampling(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    deterministic: bool,
) -> None:
    """Classic chain speculative sampling for DVR self draft.

    DVR uses topk=1, so tree topology is a single chain. The first candidate is
    the anchor token that has already been sampled by the target model; each
    following candidate is accepted with min(1, p / q). The final accepted slot
    is filled with target/residual sampling, matching EAGLE's verify contract.
    """

    del retrive_next_token, retrive_next_sibling
    del threshold_single, threshold_acc, deterministic

    if triton is not None and candidates.is_cuda and target_probs.is_cuda:
        batch_size, num_slots = candidates.shape
        _chain_speculative_sampling_kernel[(batch_size,)](
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index,
            uniform_samples,
            uniform_samples_for_final_sampling,
            target_probs,
            draft_probs,
            candidates.stride(0),
            candidates.stride(1),
            retrive_index.stride(0),
            retrive_index.stride(1),
            uniform_samples.stride(0),
            uniform_samples.stride(1),
            target_probs.stride(0),
            target_probs.stride(1),
            target_probs.stride(2),
            draft_probs.stride(0),
            draft_probs.stride(1),
            draft_probs.stride(2),
            NUM_SLOTS=num_slots,
            VOCAB_SIZE=target_probs.shape[-1],
            BLOCK_V=4096,
        )
        return

    _chain_speculative_sampling_torch(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
    )
