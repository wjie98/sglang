import torch
import triton
import triton.language as tl

from sglang.srt.layers.utils.hash import fmix32, murmur3_mix


DVR_PROPOSAL_RNG_DOMAIN = 0xA511E9B3
DVR_ACCEPT_RNG_DOMAIN = 0x63D83595
DVR_FINAL_RNG_DOMAIN = 0xB8F1A2C7


@triton.jit
def stateless_uniform(seed, position, domain: tl.constexpr):
    """Map a request seed, absolute position, and RNG domain to (0, 1)."""

    seed = seed.to(tl.uint64)
    h: tl.uint32 = 0
    h = murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
    h = murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    h = murmur3_mix(h, position.to(tl.uint32))
    h = murmur3_mix(h, domain)
    h ^= 16
    h = fmix32(h)

    # Use the high 24 hash bits so float32 conversion cannot round the largest
    # uint32 value to 1.0. The half-bin offset also keeps the result above zero.
    return ((h >> 8).to(tl.float32) + 0.5) * (1.0 / 16777216.0)


@triton.jit
def sample_from_probs_with_seed_kernel(
    Probs,
    Seeds,
    Positions,
    Output,
    stride_probs_b,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    RNG_DOMAIN: tl.constexpr,
):
    pid = tl.program_id(0)
    probs_base = Probs + pid * stride_probs_b

    norm_sum = 0.0
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        offsets = v_start + tl.arange(0, BLOCK_V)
        values = tl.load(
            probs_base + offsets, mask=offsets < VOCAB_SIZE, other=0.0
        )
        norm_sum += tl.sum(values)

    seed = tl.load(Seeds + pid)
    position = tl.load(Positions + pid)
    target = stateless_uniform(seed, position, RNG_DOMAIN) * norm_sum
    cumulative = 0.0
    sampled_token = 0
    fallback_token = 0
    found = 0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        offsets = v_start + tl.arange(0, BLOCK_V)
        mask = offsets < VOCAB_SIZE
        values = tl.load(probs_base + offsets, mask=mask, other=0.0)
        positive = mask & (values > 0.0)
        if tl.max(positive, axis=0):
            fallback_token = v_start + tl.argmax(
                tl.where(positive, offsets, 0), axis=0
            )
        if found == 0:
            block_cumulative = tl.cumsum(values, axis=0)
            matches = mask & (cumulative + block_cumulative > target)
            if tl.max(matches, axis=0):
                sampled_token = v_start + tl.argmax(
                    matches.to(tl.int32), axis=0
                )
                found = 1
            cumulative += tl.sum(values)

    sampled_token = tl.where(found != 0, sampled_token, fallback_token)
    tl.store(Output + pid, sampled_token)


def sample_from_probs_with_seed(probs, seeds, positions):
    """Deterministically sample each probability row without global RNG state."""

    if not probs.is_cuda:
        raise RuntimeError("Seeded DVR sampling requires CUDA tensors.")
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape {tuple(probs.shape)}.")
    if seeds is None:
        raise ValueError("Seeded DVR sampling requires one seed per request.")
    seeds = seeds.reshape(-1)
    positions = positions.reshape(-1)
    if seeds.shape != positions.shape or seeds.shape[0] != probs.shape[0]:
        raise ValueError(
            "DVR sampling seed/position rows must match probabilities: "
            f"probs={tuple(probs.shape)}, seeds={tuple(seeds.shape)}, "
            f"positions={tuple(positions.shape)}."
        )

    output = torch.empty(probs.shape[0], dtype=torch.int64, device=probs.device)
    sample_from_probs_with_seed_kernel[(probs.shape[0],)](
        probs,
        seeds,
        positions,
        output,
        probs.stride(0),
        VOCAB_SIZE=probs.shape[1],
        BLOCK_V=4096,
        RNG_DOMAIN=DVR_PROPOSAL_RNG_DOMAIN,
    )
    return output


@triton.jit
def speculative_sampling_classic_kernel(
    # Pointers
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    SamplingSeeds,
    Positions,
    TargetProbs,
    DraftProbs,
    # Strides
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_pos_b,
    stride_pos_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    # Constants
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    USE_SEEDED_RNG: tl.constexpr,
    ACCEPT_RNG_DOMAIN: tl.constexpr,
    FINAL_RNG_DOMAIN: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_prob_row = 0

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0

    # Verification Loop
    step = 1
    continue_verifying = 1

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)

        offset_prob = (
            (pid * stride_tp_b)
            + (cur_prob_row * stride_tp_s)
            + (draft_token * stride_tp_v)
        )
        offset_draft = (
            (pid * stride_dp_b)
            + (cur_prob_row * stride_dp_s)
            + (draft_token * stride_dp_v)
        )

        p = tl.load(TargetProbs + offset_prob)
        q = tl.load(DraftProbs + offset_draft)

        if USE_SEEDED_RNG:
            seed = tl.load(SamplingSeeds + pid)
            position = tl.load(
                Positions + pid * stride_pos_b + cur_prob_row * stride_pos_s
            )
            coin = stateless_uniform(seed, position, ACCEPT_RNG_DOMAIN)
        else:
            coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

        if coin * q < p:
            num_accept += 1
            cur_prob_row = step
            tl.store(Predicts + last_accepted_global_idx, draft_token)

            curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                curr_global_idx,
            )
            last_accepted_global_idx = curr_global_idx

            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # Final Sampling
    all_drafts_accepted = continue_verifying
    if USE_SEEDED_RNG:
        seed = tl.load(SamplingSeeds + pid)
        position = tl.load(
            Positions + pid * stride_pos_b + cur_prob_row * stride_pos_s
        )
        coin_final = stateless_uniform(seed, position, FINAL_RNG_DOMAIN)
    else:
        coin_final = tl.load(UniformSamplesFinal + pid)
    norm_sum = 0.0

    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    # DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When
    # all drafts are accepted cur_prob_row == num_steps is out of bounds for
    # DraftProbs, but the all-accepted branch samples pure target p and never
    # dereferences this pointer; on rejection cur_prob_row <= num_steps - 1.
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)

    # Pass 1: Sum
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE

        p_ptr = tp_base_ptr + v_offsets * stride_tp_v
        p_val = tl.load(p_ptr, mask=mask, other=0.0)

        if all_drafts_accepted:
            val = p_val
        else:
            q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
            q_val = tl.load(q_ptr, mask=mask, other=0.0)
            diff = p_val - q_val
            val = tl.where(diff > 0.0, diff, 0.0)

        norm_sum += tl.sum(val)

    # A rejected zero-probability proposal can leave p - q with no mass. Sample
    # from the valid target distribution instead of emitting a vocabulary sentinel.
    residual_is_valid = (norm_sum > 0.0) & (norm_sum < float("inf"))
    fallback_to_target = (all_drafts_accepted == 0) & ~residual_is_valid
    sample_target = (all_drafts_accepted == 1) | fallback_to_target
    sample_norm = norm_sum
    if fallback_to_target:
        sample_norm = 0.0
        for v_start in range(0, VOCAB_SIZE, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE
            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            sample_norm += tl.sum(tl.load(p_ptr, mask=mask, other=0.0))

    target_u = coin_final * sample_norm
    cum_sum = 0.0
    final_token = 0
    found = 0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if sample_target:
                val = p_val
            else:
                q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                q_val = tl.load(q_ptr, mask=mask, other=0.0)
                diff = p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            block_cumsum = tl.cumsum(val, axis=0)
            total_cumsum = cum_sum + block_cumsum

            candidates_mask = total_cumsum > target_u
            has_match = tl.max(candidates_mask, axis=0)

            if has_match:
                match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                final_token = v_start + match_idx
                found = 1

            cum_sum += tl.sum(val)

    tl.store(Predicts + last_accepted_global_idx, final_token)


def chain_speculative_sampling_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,  # not used in chain verification
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic,  # not used
    sampling_seed=None,
    positions=None,
):
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    use_seeded_rng = sampling_seed is not None
    if use_seeded_rng:
        sampling_seed = sampling_seed.reshape(-1)
        positions = positions.reshape(batch_size, num_slots)
        if sampling_seed.shape[0] != batch_size:
            raise ValueError(
                "Rejection sampling requires one seed per request: "
                f"batch_size={batch_size}, seeds={tuple(sampling_seed.shape)}."
            )
        uniform_samples = target_probs
        uniform_samples_for_final_sampling = target_probs
    elif uniform_samples is None or uniform_samples_for_final_sampling is None:
        raise ValueError("Unseeded rejection sampling requires uniform samples.")

    seed_arg = sampling_seed if use_seeded_rng else target_probs
    position_arg = positions if use_seeded_rng else target_probs

    grid = (batch_size,)
    speculative_sampling_classic_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        seed_arg,
        position_arg,
        target_probs,
        draft_probs,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        position_arg.stride(0),
        position_arg.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
        USE_SEEDED_RNG=use_seeded_rng,
        ACCEPT_RNG_DOMAIN=DVR_ACCEPT_RNG_DOMAIN,
        FINAL_RNG_DOMAIN=DVR_FINAL_RNG_DOMAIN,
    )
