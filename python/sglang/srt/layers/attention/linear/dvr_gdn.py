"""GDN-specific DVR state allocation, replay kernels, and lifecycle adapter."""

import math
from typing import Any, Optional

import torch
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.fla.op import exp
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_conv_window_scatter_with_mask,
    fused_mamba_state_scatter_with_mask,
)
from sglang.srt.utils import is_cpu
from sglang.srt.utils.nvtx_utils import profile_range

__all__ = [
    "DVRGDNStateAdapter",
    "dvr_gdn_intermediate_bytes_per_request",
]

if not is_cpu():
    import triton
    import triton.language as tl


if not is_cpu():

    @triton.jit
    def _dvr_compact_state_window_kernel(
        cache,
        indices,
        crosses_boundary,
        accepted_tail_lens,
        layer_stride,
        slot_stride,
        token_stride,
        E: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        i_req = tl.program_id(0)
        i_layer = tl.program_id(1).to(tl.int64)
        i_block = tl.program_id(2).to(tl.int64)
        if not tl.load(crosses_boundary + i_req):
            return

        slot = tl.load(indices + i_req).to(tl.int64)
        count = tl.load(accepted_tail_lens + i_req).to(tl.int64) * E
        offsets = i_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < count
        token = offsets // E
        element = offsets % E
        base = i_layer * layer_stride + slot * slot_stride
        values = tl.load(
            cache + base + (CHUNK_SIZE + token) * token_stride + element,
            mask=mask,
        )
        tl.store(cache + base + token * token_stride + element, values, mask=mask)

    @triton.jit
    def _dvr_pack_verify_window_kernel(
        cache0,
        candidate0,
        output0,
        cache1,
        candidate1,
        output1,
        request_rows,
        accepted_tail_lens,
        valid_mask,
        cache0_slot_stride,
        cache0_token_stride,
        candidate0_batch_stride,
        candidate0_token_stride,
        output0_batch_stride,
        output0_token_stride,
        cache1_slot_stride,
        cache1_token_stride,
        candidate1_batch_stride,
        candidate1_token_stride,
        output1_batch_stride,
        output1_token_stride,
        DRAFT_TOKENS: tl.constexpr,
        E0: tl.constexpr,
        E1: tl.constexpr,
        HAS_SECOND: tl.constexpr,
        READ_FIRST_CACHE: tl.constexpr,
        WRITE_FIRST_CACHE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        i_req = tl.program_id(0).to(tl.int64)
        i_token = tl.program_id(1).to(tl.int64)
        offsets = tl.program_id(2).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        slot = tl.load(request_rows + i_req).to(tl.int64)
        tail = tl.load(accepted_tail_lens + i_req).to(tl.int64)
        valid = tl.load(valid_mask + i_req)
        candidate_token = i_token - tail
        is_candidate = valid & (candidate_token >= 0) & (candidate_token < DRAFT_TOKENS)

        cache0_ptr = (
            cache0 + slot * cache0_slot_stride + i_token * cache0_token_stride + offsets
        )
        candidate0_ptr = (
            candidate0
            + i_req * candidate0_batch_stride
            + candidate_token * candidate0_token_stride
            + offsets
        )
        if READ_FIRST_CACHE:
            cache0_value = tl.load(cache0_ptr, mask=offsets < E0, other=0)
        else:
            cache0_value = 0.0
        candidate0_value = tl.load(
            candidate0_ptr, mask=(offsets < E0) & is_candidate, other=0
        )
        value0 = tl.where(is_candidate, candidate0_value, cache0_value)
        tl.store(
            output0
            + i_req * output0_batch_stride
            + i_token * output0_token_stride
            + offsets,
            value0,
            mask=offsets < E0,
        )
        if WRITE_FIRST_CACHE:
            tl.store(
                cache0_ptr,
                candidate0_value,
                mask=(offsets < E0) & is_candidate,
            )

        if HAS_SECOND:
            cache1_ptr = (
                cache1
                + slot * cache1_slot_stride
                + i_token * cache1_token_stride
                + offsets
            )
            candidate1_ptr = (
                candidate1
                + i_req * candidate1_batch_stride
                + candidate_token * candidate1_token_stride
                + offsets
            )
            cache1_value = tl.load(cache1_ptr, mask=offsets < E1, other=0)
            candidate1_value = tl.load(
                candidate1_ptr, mask=(offsets < E1) & is_candidate, other=0
            )
            value1 = tl.where(is_candidate, candidate1_value, cache1_value)
            tl.store(
                output1
                + i_req * output1_batch_stride
                + i_token * output1_token_stride
                + offsets,
                value1,
                mask=offsets < E1,
            )
            tl.store(
                cache1_ptr,
                candidate1_value,
                mask=(offsets < E1) & is_candidate,
            )

    @triton.jit
    def _dvr_gather_verify_output_kernel(
        source,
        output,
        accepted_tail_lens,
        source_batch_stride,
        source_token_stride,
        output_batch_stride,
        output_token_stride,
        E: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        i_req = tl.program_id(0).to(tl.int64)
        i_token = tl.program_id(1).to(tl.int64)
        offsets = tl.program_id(2).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        source_token = tl.load(accepted_tail_lens + i_req).to(tl.int64) + i_token
        values = tl.load(
            source
            + i_req * source_batch_stride
            + source_token * source_token_stride
            + offsets,
            mask=offsets < E,
        )
        tl.store(
            output
            + i_req * output_batch_stride
            + i_token * output_token_stride
            + offsets,
            values,
            mask=offsets < E,
        )

    @triton.jit
    def _dvr_gdn_rebuild_draft_state_kernel(
        k,
        v,
        g,
        beta,
        state_src,
        state_dst,
        request_rows,
        boundary_slots,
        destination_indices,
        token_count,
        N: tl.constexpr,
        S: tl.constexpr,
        CS: tl.constexpr,
        CD: tl.constexpr,
        WINDOW: tl.constexpr,
        MAX_STEPS: tl.constexpr,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        i_v, i_nh = tl.program_id(0), tl.program_id(1)
        i_ln, i_hv = i_nh // HV, i_nh % HV
        i_l, i_n = i_ln // N, i_ln % N
        i_h = i_hv // (HV // H)

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        state_input_idx = tl.load(request_rows + i_n).to(tl.int64)
        boundary_idx = tl.load(boundary_slots + i_n).to(tl.int64)
        destination_idx = tl.load(destination_indices + i_n).to(tl.int64)
        state_offset = (i_l * CS + boundary_idx) * HV * V * K + i_hv * V * K
        p_h0 = state_src + state_offset + o_v[:, None] * K + o_k[None, :]
        recurrent_state = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        steps = tl.load(token_count + i_n).to(tl.int64)
        p_k = k + (((i_l * S + state_input_idx) * WINDOW * H + i_h) * K + o_k)
        p_v = v + (((i_l * S + state_input_idx) * WINDOW * HV + i_hv) * V + o_v)
        p_g = g + ((i_l * S + state_input_idx) * WINDOW * HV + i_hv)
        p_beta = beta + ((i_l * S + state_input_idx) * WINDOW * HV + i_hv)

        for step in range(0, MAX_STEPS):
            if step < steps:
                key = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
                value = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
                key /= tl.sqrt(tl.sum(key * key) + 1e-6)
                recurrent_state *= exp(tl.load(p_g).to(tl.float32))
                value -= tl.sum(recurrent_state * key[None, :], 1)
                value *= tl.load(p_beta).to(tl.float32)
                recurrent_state += value[:, None] * key[None, :]

            p_k += H * K
            p_v += HV * V
            p_g += HV
            p_beta += HV

        state_offset = (i_l * CD + destination_idx) * HV * V * K + i_hv * V * K
        p_ht = state_dst + state_offset + o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, recurrent_state.to(p_ht.dtype.element_ty), mask=mask_h)


def _pack_verify_window_pair(
    cache0: torch.Tensor,
    candidate0: torch.Tensor,
    *,
    request_rows: torch.Tensor,
    accepted_tail_lens: torch.Tensor,
    valid_mask: torch.Tensor,
    cache1: Optional[torch.Tensor] = None,
    candidate1: Optional[torch.Tensor] = None,
    read_cache0: bool = True,
    persist_cache0: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Materialize one or two verify windows while persisting candidate rows."""

    has_second = cache1 is not None
    assert has_second == (candidate1 is not None)

    batch_size, draft_tokens = candidate0.shape[:2]
    output0 = cache0.new_empty((batch_size, *cache0.shape[1:]))
    output1 = cache1.new_empty((batch_size, *cache1.shape[1:])) if has_second else None
    e0 = math.prod(cache0.shape[2:])
    e1 = math.prod(cache1.shape[2:]) if has_second else 1
    if math.prod(candidate0.shape[2:]) != e0:
        raise ValueError("DVR candidate and cache element shapes do not match.")
    if has_second and math.prod(candidate1.shape[2:]) != e1:
        raise ValueError("DVR candidate and cache element shapes do not match.")

    block_size = 256
    grid = (
        batch_size,
        cache0.shape[1],
        triton.cdiv(max(e0, e1), block_size),
    )
    second_cache = cache1 if has_second else cache0
    second_candidate = candidate1 if has_second else candidate0
    second_output = output1 if has_second else output0
    _dvr_pack_verify_window_kernel[grid](
        cache0,
        candidate0,
        output0,
        second_cache,
        second_candidate,
        second_output,
        request_rows,
        accepted_tail_lens,
        valid_mask,
        cache0.stride(0),
        cache0.stride(1),
        candidate0.stride(0),
        candidate0.stride(1),
        output0.stride(0),
        output0.stride(1),
        second_cache.stride(0),
        second_cache.stride(1),
        second_candidate.stride(0),
        second_candidate.stride(1),
        second_output.stride(0),
        second_output.stride(1),
        DRAFT_TOKENS=draft_tokens,
        E0=e0,
        E1=e1,
        HAS_SECOND=has_second,
        READ_FIRST_CACHE=read_cache0,
        WRITE_FIRST_CACHE=persist_cache0,
        BLOCK_SIZE=block_size,
    )
    return output0, output1


def _gather_verify_output(
    source: torch.Tensor,
    *,
    accepted_tail_lens: torch.Tensor,
    draft_tokens: int,
) -> torch.Tensor:
    """Gather logical candidate rows directly into the target output layout."""

    batch_size = accepted_tail_lens.shape[0]
    source = source.view(batch_size, -1, *source.shape[-2:])
    output = source.new_empty((batch_size, draft_tokens, *source.shape[2:]))
    elements = math.prod(source.shape[2:])
    block_size = 256
    _dvr_gather_verify_output_kernel[
        (batch_size, draft_tokens, triton.cdiv(elements, block_size))
    ](
        source,
        output,
        accepted_tail_lens,
        source.stride(0),
        source.stride(1),
        output.stride(0),
        output.stride(1),
        E=elements,
        BLOCK_SIZE=block_size,
    )
    return output.reshape(1, batch_size * draft_tokens, *source.shape[2:])


def _local_gdn_dimensions(state_shape):
    local_value_heads, value_dim, key_dim = state_shape.temporal
    local_key_heads = state_shape.num_k_heads_per_tp
    assert local_key_heads > 0 and local_value_heads > 0
    return local_key_heads, local_value_heads, key_dim, value_dim


def dvr_gdn_intermediate_bytes_per_request(
    cache_params,
    num_draft_tokens: int,
    *,
    dedup_conv_window: bool,
    draft_reuses_target_state: bool,
) -> int:
    """Size DVR-only verify, rollback, and state-input buffers per request."""

    intermediate_conv_numel = sum(
        conv_dim
        * (
            num_draft_tokens + window_size - 1
            if dedup_conv_window
            else num_draft_tokens * window_size
        )
        for conv_dim, window_size in cache_params.shape.conv
    )
    num_layers = len(cache_params.layers)
    total = num_layers * (
        intermediate_conv_numel * cache_params.dtype.conv.itemsize
        + math.prod(cache_params.shape.temporal) * cache_params.dtype.temporal.itemsize
    )
    if draft_reuses_target_state:
        total += (
            num_layers
            * sum(math.prod(shape) for shape in cache_params.shape.conv)
            * cache_params.dtype.conv.itemsize
        )
    local_key_heads, local_value_heads, key_dim, value_dim = _local_gdn_dimensions(
        cache_params.shape
    )
    values_per_token = (
        local_key_heads * key_dim * cache_params.dtype.conv.itemsize
        + local_value_heads * value_dim * cache_params.dtype.conv.itemsize
        + 2 * local_value_heads * torch.float32.itemsize
    )
    # Persistent k/v/g/beta transition windows plus the int64 boundary slot.
    return (
        total + num_layers * (FLA_CHUNK_SIZE + num_draft_tokens) * values_per_token + 8
    )


def _compact_gdn_transition_windows(
    transition_windows: tuple[torch.Tensor, ...],
    *,
    indices: torch.Tensor,
    crosses_chunk_boundary: torch.Tensor,
    accepted_tail_lens: torch.Tensor,
    chunk_size: int,
) -> None:
    """Move the accepted post-boundary tail to the start of each GDN window."""

    tail_capacity = transition_windows[0].shape[2] - chunk_size
    if tail_capacity <= 0 or indices.numel() == 0:
        return

    crosses_chunk_boundary = crosses_chunk_boundary.to(torch.bool)
    accepted_tail_lens = accepted_tail_lens.to(
        device=crosses_chunk_boundary.device, dtype=torch.long
    ).contiguous()
    torch._assert_async(
        (
            ~crosses_chunk_boundary
            | ((accepted_tail_lens >= 0) & (accepted_tail_lens <= tail_capacity))
        ).all(),
        "DVR compact received an invalid linear-state tail length.",
    )
    indices = indices.to(
        device=crosses_chunk_boundary.device, dtype=torch.long
    ).contiguous()
    crosses_chunk_boundary = crosses_chunk_boundary.contiguous()

    for cache in transition_windows:
        elements_per_token = cache[0, 0, 0].numel()
        _dvr_compact_state_window_kernel[
            (
                indices.numel(),
                cache.shape[0],
                triton.cdiv(tail_capacity * elements_per_token, 256),
            )
        ](
            cache,
            indices,
            crosses_chunk_boundary,
            accepted_tail_lens,
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            E=elements_per_token,
            CHUNK_SIZE=chunk_size,
            BLOCK_SIZE=256,
        )


def _rebuild_gdn_self_draft_state(
    transition_windows: tuple[torch.Tensor, ...],
    *,
    state_cache,
    request_rows: torch.Tensor,
    boundary_slots: torch.Tensor,
    token_count: torch.Tensor,
) -> None:
    """Rebuild request-owned self-draft state from an exact boundary and tail."""

    k, v, g, beta = transition_windows
    num_layers, num_slots, num_tokens, num_key_heads, key_dim = k.shape
    _, _, _, num_value_heads, value_dim = v.shape
    num_reqs = request_rows.numel()
    if num_reqs == 0:
        return
    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 8)
    _dvr_gdn_rebuild_draft_state_kernel[
        (
            triton.cdiv(value_dim, block_v),
            num_layers * num_reqs * num_value_heads,
        )
    ](
        k=k,
        v=v,
        g=g,
        beta=beta,
        state_src=state_cache.temporal,
        state_dst=state_cache.intermediate_ssm[:, :, 0],
        request_rows=request_rows,
        boundary_slots=boundary_slots,
        destination_indices=request_rows,
        token_count=token_count.contiguous(),
        N=num_reqs,
        S=num_slots,
        CS=state_cache.temporal.shape[1],
        CD=state_cache.intermediate_ssm.shape[1],
        WINDOW=num_tokens,
        MAX_STEPS=FLA_CHUNK_SIZE,
        H=num_key_heads,
        HV=num_value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        num_warps=1,
        num_stages=3,
    )


class DVRGDNStateAdapter:
    """Adapter for DVR state replay in gated linear-state layers.

    The adapter owns the rolling-window and commit mechanics so model backends
    do not need to know the verify/post-verify lifecycle details.
    """

    def __init__(
        self,
        kernel_dispatcher: Any,
        *,
        state_cache: Any,
        transition_windows: tuple[torch.Tensor, ...],
        verify_boundary_slots: torch.Tensor,
        draft_reuses_target_state: bool,
        chunk_size: int = FLA_CHUNK_SIZE,
    ):
        self.kernel_dispatcher = kernel_dispatcher
        self.state_cache = state_cache
        self.transition_windows = transition_windows
        self.verify_boundary_slots = verify_boundary_slots
        self.chunk_size = chunk_size
        self._verify_plan: Optional[tuple[torch.Tensor, ...]] = None
        self.self_draft_conv_state = (
            tuple(
                tensor.new_zeros(
                    (
                        tensor.shape[0],
                        state_cache.intermediate_ssm.shape[1],
                        *tensor.shape[2:],
                    )
                )
                for tensor in state_cache.conv
            )
            if draft_reuses_target_state
            else None
        )

    @classmethod
    def for_gdn(
        cls,
        kernel_dispatcher,
        *,
        model_runner: Any,
    ) -> "DVRGDNStateAdapter":
        mamba_cache_params = model_runner.mambaish_config.mamba2_cache_params
        state_cache = (
            model_runner.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )
        state_tensors = (state_cache.temporal, *getattr(state_cache, "conv", ()))
        if any(not tensor.is_contiguous() for tensor in state_tensors):
            raise RuntimeError(
                "DVR GDN verify requires contiguous recurrent-state storage."
            )
        if any(
            getattr(state_cache, name, None) is not None
            for name in ("replayssm_d", "replayssm_k", "replayssm_g")
        ):
            raise RuntimeError("DVR GDN rollback does not support ReplaySSM state.")
        if (
            state_cache.temporal.dtype != torch.float32
            or state_cache.intermediate_ssm.dtype != torch.float32
        ):
            raise RuntimeError(
                "DVR GDN verify requires fp32 recurrent and intermediate states."
            )
        if state_cache.intermediate_ssm.shape[2] != 1:
            raise RuntimeError(
                "DVR self-draft requires one reusable recurrent workspace per request."
            )
        num_layers = state_cache.intermediate_ssm.shape[0]
        num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        if num_draft_tokens is None:
            raise RuntimeError("DVR requires speculative_num_draft_tokens.")

        state_shape = mamba_cache_params.shape
        local_key_heads, local_value_heads, key_dim, value_dim = _local_gdn_dimensions(
            state_shape
        )

        # ReqToTokenPool already reserves row 0 for padded CUDA graph requests.
        # Mirror that indexing directly instead of maintaining a second offset.
        num_slots = state_cache.intermediate_ssm.shape[1]
        window_len = FLA_CHUNK_SIZE + num_draft_tokens
        k = torch.zeros(
            num_layers,
            num_slots,
            window_len,
            local_key_heads,
            key_dim,
            dtype=mamba_cache_params.dtype.conv,
            device=model_runner.device,
        )
        v = torch.zeros(
            num_layers,
            num_slots,
            window_len,
            local_value_heads,
            value_dim,
            dtype=mamba_cache_params.dtype.conv,
            device=model_runner.device,
        )
        gate = torch.zeros(
            num_layers,
            num_slots,
            window_len,
            local_value_heads,
            dtype=torch.float32,
            device=model_runner.device,
        )

        return cls(
            kernel_dispatcher,
            state_cache=state_cache,
            transition_windows=(k, v, gate, torch.zeros_like(gate)),
            verify_boundary_slots=torch.zeros(
                num_slots, dtype=torch.long, device=model_runner.device
            ),
            draft_reuses_target_state=model_runner.spec_algorithm.is_dvr_self_draft(),
        )

    def resolve_request_slots(self, *, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve request-owned rows and accepted-endpoint state rows."""

        pool = batch.req_to_token_pool
        endpoint_slots = pool.get_mamba_indices(batch.req_pool_indices).to(torch.long)
        request_rows = batch.req_pool_indices.to(
            device=endpoint_slots.device, dtype=torch.long
        )
        return request_rows, endpoint_slots

    def zero_boundary_state(self, *, indices: torch.Tensor):
        indices = indices.to(device=self.state_cache.temporal.device, dtype=torch.long)
        for conv in self.state_cache.conv:
            conv[:, indices] = 0
        self.state_cache.temporal[:, indices] = 0

    def initialize_self_draft_state(
        self,
        *,
        request_rows: torch.Tensor,
        endpoint_slots: torch.Tensor,
    ) -> None:
        """Seed private self-draft state after target EXTEND.

        This full-state copy runs only when EXTEND establishes a new accepted
        endpoint. Steady-state decode reconstructs the same workspace directly
        from the authoritative boundary plus accepted transition rows.
        """

        if self.self_draft_conv_state is None:
            return
        endpoint_slots = endpoint_slots.to(
            device=self.state_cache.temporal.device, dtype=torch.long
        )
        request_rows = request_rows.to(
            device=self.state_cache.temporal.device, dtype=torch.long
        )
        with profile_range("draft_state_copy"):
            self.state_cache.intermediate_ssm[:, request_rows, 0] = (
                self.state_cache.temporal[:, endpoint_slots]
            )
            for target_conv, draft_conv in zip(
                self.state_cache.conv, self.self_draft_conv_state, strict=True
            ):
                draft_conv[:, request_rows] = target_conv[:, endpoint_slots]

    def decode_state(self, *, layer_cache, forward_batch, layer_idx: int):
        """Return request-owned state for provisional DVR-self decode."""

        if (
            self.self_draft_conv_state is None
            or not forward_batch.forward_mode.is_decode()
        ):
            return None
        request_rows = forward_batch.req_pool_indices.to(
            device=layer_cache.temporal.device, dtype=torch.long
        )
        return (
            self.self_draft_conv_state[0][layer_idx],
            layer_cache.intermediate_ssm[:, 0],
            request_rows,
        )

    def set_verify_boundaries(
        self, *, request_rows: torch.Tensor, boundary_slots: torch.Tensor
    ) -> None:
        """Bind logical request rows to the physical exact boundary for verify."""

        request_rows = request_rows.to(
            device=self.verify_boundary_slots.device, dtype=torch.long
        )
        self.verify_boundary_slots[request_rows] = boundary_slots.to(
            device=self.verify_boundary_slots.device, dtype=torch.long
        )

    def cache_prefill_transitions(
        self,
        *,
        forward_batch,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        layer_idx: int,
    ):
        if (
            forward_batch.extend_prefix_lens_cpu is None
            or forward_batch.extend_seq_lens_cpu is None
        ):
            raise RuntimeError(
                "DVR GDN target EXTEND requires CPU prefix and extend lengths."
            )

        # q affects only the output at its own token; it does not participate in
        # the recurrent transition. Target verify receives candidate q directly.
        transition_windows = (
            k.reshape(k.shape[1], k.shape[2], k.shape[3]),
            v.reshape(v.shape[1], v.shape[2], v.shape[3]),
            g.reshape(-1, g.shape[-1]),
            beta.reshape(-1, beta.shape[-1]),
        )
        request_rows = forward_batch.req_pool_indices.to(
            device=k.device, dtype=torch.long
        )
        layer_transition_windows = tuple(
            cache[layer_idx] for cache in self.transition_windows
        )
        src_base = 0
        for req_i, (prefix_len, extend_len) in enumerate(
            zip(
                forward_batch.extend_prefix_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
                strict=True,
            )
        ):
            seq_len = prefix_len + extend_len
            boundary = seq_len // self.chunk_size * self.chunk_size
            write_start = max(prefix_len, boundary)
            if write_start < seq_len:
                src_start = src_base + write_start - prefix_len
                src_end = src_start + seq_len - write_start
                cols = torch.arange(
                    write_start - boundary,
                    seq_len - boundary,
                    dtype=torch.long,
                    device=request_rows.device,
                )
                slot = request_rows[req_i]
                for cache, value in zip(
                    layer_transition_windows, transition_windows, strict=True
                ):
                    cache[slot, cols] = value[src_start:src_end]
            src_base += extend_len

    def prepare_forward(self, *, forward_batch, forward_metadata):
        """Prepare request metadata once before all GDN layers execute."""

        self._verify_plan = None
        if forward_batch.forward_mode.is_target_verify():
            self.prepare_target_verify(
                forward_batch=forward_batch,
                cache_indices=forward_metadata.mamba_cache_indices,
            )
            return None
        if not (
            forward_batch.forward_mode.is_extend()
            and forward_metadata.has_mamba_track_mask
        ):
            return None

        lengths_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        boundary_steps = torch.where(
            forward_batch.mamba_track_mask
            & (lengths_to_track.remainder(self.chunk_size) != 0),
            lengths_to_track // self.chunk_size,
            torch.full_like(lengths_to_track, -1),
        )
        return forward_metadata.mamba_track_indices, boundary_steps

    def prepare_target_verify(self, *, forward_batch, cache_indices: torch.Tensor):
        """Resolve graph-stable request, boundary, tail, and padding metadata once."""

        draft_token_num = forward_batch.spec_info.draft_token_num
        batch_size = forward_batch.input_ids.shape[0] // draft_token_num
        request_rows = forward_batch.req_pool_indices[:batch_size].to(
            device=cache_indices.device, dtype=torch.long
        )
        valid_mask = cache_indices[:batch_size] != PAD_SLOT_ID
        request_rows = torch.where(
            valid_mask, request_rows, torch.zeros_like(request_rows)
        )
        boundary_slots = self.verify_boundary_slots[request_rows]
        boundary_slots = torch.where(
            valid_mask, boundary_slots, torch.zeros_like(boundary_slots)
        )
        verify_conv_slots = torch.where(
            valid_mask,
            cache_indices[:batch_size],
            torch.zeros_like(cache_indices[:batch_size]),
        )
        # TARGET_VERIFY keeps seq_lens at the accepted endpoint while its
        # speculative cache rows are appended separately.
        accepted_tail_lens = forward_batch.seq_lens[:batch_size].to(
            device=cache_indices.device, dtype=torch.long
        ) % self.chunk_size
        accepted_tail_lens = torch.where(
            valid_mask,
            accepted_tail_lens,
            torch.zeros_like(accepted_tail_lens),
        )
        boundary_state_steps = torch.where(
            valid_mask & (accepted_tail_lens + draft_token_num >= self.chunk_size),
            torch.ones_like(accepted_tail_lens),
            torch.full_like(accepted_tail_lens, -1),
        )
        self._verify_plan = (
            boundary_slots,
            request_rows,
            verify_conv_slots,
            accepted_tail_lens,
            valid_mask,
            boundary_state_steps,
        )

    def verify_conv_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return target conv slots and request-owned intermediate rows."""

        if self._verify_plan is None:
            raise RuntimeError("DVR target-verify metadata was not prepared.")
        return self._verify_plan[2], self._verify_plan[1]

    def forward_target_verify(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Replay GDN state from cached transitions and candidate q/k/v/g/beta."""

        if self._verify_plan is None:
            raise RuntimeError("DVR target-verify metadata was not prepared.")
        if not query.is_cuda:
            raise RuntimeError("DVR GDN target verify requires CUDA tensors.")
        (
            boundary_slots,
            request_rows,
            _,
            accepted_tail_lens,
            valid_mask,
            boundary_state_steps,
        ) = self._verify_plan
        batch_size = boundary_slots.shape[0]
        draft_token_num = query.shape[1] // batch_size
        candidate_inputs = (
            query.reshape(batch_size, draft_token_num, *query.shape[2:]),
            key.reshape(batch_size, draft_token_num, *key.shape[2:]),
            value.reshape(batch_size, draft_token_num, *value.shape[2:]),
            g.reshape(batch_size, draft_token_num, g.shape[-1]),
            beta.reshape(batch_size, draft_token_num, beta.shape[-1]),
        )
        k_cache, v_cache, g_cache, beta_cache = (
            cache[layer_idx] for cache in self.transition_windows
        )
        # The fixed window may retain values after accepted_tail_lens + draft_token_num.
        # They are causally after every returned logit, a boundary is exported
        # only after all chunk rows are valid, and state rebuilds use token_count.
        with profile_range("verify_state_pack"):
            # Cached q rows would produce only discarded prefix outputs.
            # Keep them zero while packing candidate q and persistent k in
            # the same launch.
            q, k = _pack_verify_window_pair(
                k_cache,
                candidate_inputs[0],
                cache1=k_cache,
                candidate1=candidate_inputs[1],
                request_rows=request_rows,
                accepted_tail_lens=accepted_tail_lens,
                valid_mask=valid_mask,
                read_cache0=False,
                persist_cache0=False,
            )
            v, _ = _pack_verify_window_pair(
                v_cache,
                candidate_inputs[2],
                request_rows=request_rows,
                accepted_tail_lens=accepted_tail_lens,
                valid_mask=valid_mask,
            )
            cached_g, cached_beta = _pack_verify_window_pair(
                g_cache,
                candidate_inputs[3],
                cache1=beta_cache,
                candidate1=candidate_inputs[4],
                request_rows=request_rows,
                accepted_tail_lens=accepted_tail_lens,
                valid_mask=valid_mask,
            )
        core_attn_out, _, _ = self.kernel_dispatcher.extend(
            q=q,
            k=k,
            v=v,
            g=cached_g,
            beta=cached_beta,
            ssm_states=self.state_cache.temporal[layer_idx],
            cache_indices=boundary_slots,
            query_start_loc=None,
            inplace_update=False,
            boundary_state=self.state_cache.intermediate_ssm[layer_idx, :, 0],
            boundary_state_indices=request_rows,
            boundary_state_steps=boundary_state_steps,
        )

        with profile_range("verify_output_gather"):
            return _gather_verify_output(
                core_attn_out,
                accepted_tail_lens=accepted_tail_lens,
                draft_tokens=draft_token_num,
            )

    def commit_accepted_state(
        self,
        *,
        request_rows: torch.Tensor,
        endpoint_slots: torch.Tensor,
        boundary_slots: torch.Tensor,
        alternate_boundary_slots: torch.Tensor,
        tail_lens_before: torch.Tensor,
        accepted_token_counts: torch.Tensor,
    ) -> torch.Tensor:
        tail_lens_before = tail_lens_before.to(
            device=endpoint_slots.device, dtype=torch.long
        )
        accepted_token_counts = accepted_token_counts.to(
            device=endpoint_slots.device, dtype=torch.long
        )
        tail_lens_after = tail_lens_before + accepted_token_counts
        crosses_chunk_boundary = tail_lens_after >= self.chunk_size
        no_commit_step = torch.full_like(tail_lens_before, -1)
        # accept_lens includes the bonus token, so every non-idle request has a
        # valid accepted step. The fused scatter already handles the empty batch.
        fused_conv_window_scatter_with_mask(
            self.state_cache.conv[0],
            self.state_cache.intermediate_conv_window[0],
            endpoint_slots,
            accepted_token_counts - 1,
            src_indices_raw=request_rows,
        )
        # Convolution state is authoritative at the accepted endpoint; unlike
        # temporal state, it is cheap and necessary to resume the next conv.
        if self.self_draft_conv_state is not None:
            fused_conv_window_scatter_with_mask(
                self.self_draft_conv_state[0],
                self.state_cache.intermediate_conv_window[0],
                request_rows,
                accepted_token_counts - 1,
                src_indices_raw=request_rows,
            )

        # Commit to the pool-selected next checkpoint slot. Radix execution uses
        # two slots so finish can publish either overlap boundary; without Radix,
        # the sole request-local slot is updated in place.
        with profile_range("boundary_publish"):
            fused_mamba_state_scatter_with_mask(
                self.state_cache.temporal,
                self.state_cache.intermediate_ssm,
                alternate_boundary_slots,
                torch.where(
                    crosses_chunk_boundary,
                    torch.zeros_like(tail_lens_before),
                    no_commit_step,
                ),
                src_indices_raw=request_rows,
            )
            fused_conv_window_scatter_with_mask(
                self.state_cache.conv[0],
                self.state_cache.intermediate_conv_window[0],
                alternate_boundary_slots,
                torch.where(
                    crosses_chunk_boundary,
                    self.chunk_size - 1 - tail_lens_before,
                    no_commit_step,
                ),
                src_indices_raw=request_rows,
            )
        new_tail_lens = tail_lens_after - self.chunk_size
        tail_lens_after = torch.where(
            crosses_chunk_boundary, new_tail_lens, tail_lens_after
        )
        with profile_range("state_window_compact"):
            _compact_gdn_transition_windows(
                self.transition_windows,
                indices=request_rows,
                crosses_chunk_boundary=crosses_chunk_boundary,
                chunk_size=self.chunk_size,
                accepted_tail_lens=tail_lens_after,
            )
        if self.self_draft_conv_state is not None:
            # Reconstruct only self-draft's private endpoint. Target recurrent
            # state remains boundary-owned and target verify reads it directly.
            with profile_range("draft_state_rebuild"):
                _rebuild_gdn_self_draft_state(
                    self.transition_windows,
                    state_cache=self.state_cache,
                    request_rows=request_rows,
                    boundary_slots=torch.where(
                        crosses_chunk_boundary,
                        alternate_boundary_slots,
                        boundary_slots,
                    ),
                    token_count=tail_lens_after,
                )
        # EAGLE/MTP owns separate draft state and skips this reconstruction.

        return crosses_chunk_boundary
