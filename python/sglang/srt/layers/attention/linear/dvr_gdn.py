"""GDN-specific DVR state allocation, replay kernels, and lifecycle adapter."""

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.fla.op import exp
from sglang.srt.layers.attention.linear.dvr_state import DVRStateInputCache
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
    "dvr_gdn_state_input_bytes_per_request",
]

if not is_cpu():
    import triton
    import triton.language as tl


if not is_cpu():

    @triton.jit
    def _dvr_pack_verify_window_kernel(
        cache0,
        candidate0,
        output0,
        cache1,
        candidate1,
        output1,
        state_input_indices,
        tail_lens,
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
        BLOCK_SIZE: tl.constexpr,
    ):
        i_req = tl.program_id(0).to(tl.int64)
        i_token = tl.program_id(1).to(tl.int64)
        offsets = tl.program_id(2).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        slot = tl.load(state_input_indices + i_req).to(tl.int64)
        tail = tl.load(tail_lens + i_req).to(tl.int64)
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
        cache0_value = tl.load(cache0_ptr, mask=offsets < E0, other=0)
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
        tail_lens,
        source_batch_stride,
        source_token_stride,
        output_batch_stride,
        output_token_stride,
        DRAFT_TOKENS: tl.constexpr,
        E: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        i_req = tl.program_id(0).to(tl.int64)
        i_token = tl.program_id(1).to(tl.int64)
        offsets = tl.program_id(2).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        source_token = tl.load(tail_lens + i_req).to(tl.int64) + i_token
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
        state_input_indices,
        boundary_indices,
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

        state_input_idx = tl.load(state_input_indices + i_n).to(tl.int64)
        boundary_idx = tl.load(boundary_indices + i_n).to(tl.int64)
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
    state_input_indices: torch.Tensor,
    tail_lens: torch.Tensor,
    valid_mask: torch.Tensor,
    cache1: Optional[torch.Tensor] = None,
    candidate1: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Materialize one or two verify windows while persisting candidate rows."""

    if not cache0.is_cuda:
        raise RuntimeError("The fused DVR verify-window pack is CUDA-only.")
    has_second = cache1 is not None
    if has_second != (candidate1 is not None):
        raise ValueError(
            "A second cache and candidate tensor must be provided together."
        )

    batch_size, draft_tokens = candidate0.shape[:2]
    output0 = cache0.new_empty((batch_size, *cache0.shape[1:]))
    output1 = cache1.new_empty((batch_size, *cache1.shape[1:])) if has_second else None
    tensors = (cache0, candidate0, output0)
    if has_second:
        tensors += (cache1, candidate1, output1)
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise RuntimeError("DVR verify-window tensors must be contiguous.")

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
        state_input_indices,
        tail_lens,
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
        BLOCK_SIZE=block_size,
    )
    return output0, output1


def _gather_verify_output(
    source: torch.Tensor,
    *,
    tail_lens: torch.Tensor,
    draft_tokens: int,
) -> torch.Tensor:
    """Gather logical candidate rows directly into the target output layout."""

    batch_size = tail_lens.shape[0]
    source = source.view(batch_size, -1, *source.shape[-2:])
    output = source.new_empty((batch_size, draft_tokens, *source.shape[2:]))
    elements = math.prod(source.shape[2:])
    block_size = 256
    _dvr_gather_verify_output_kernel[
        (batch_size, draft_tokens, triton.cdiv(elements, block_size))
    ](
        source,
        output,
        tail_lens,
        source.stride(0),
        source.stride(1),
        output.stride(0),
        output.stride(1),
        DRAFT_TOKENS=draft_tokens,
        E=elements,
        BLOCK_SIZE=block_size,
    )
    return output.reshape(1, batch_size * draft_tokens, *source.shape[2:])


def _local_gdn_dimensions(state_shape):
    local_value_heads, value_dim, key_dim = state_shape.temporal
    key_group_width = 2 * state_shape.state_size
    assert key_group_width > 0
    assert state_shape.conv_dim >= state_shape.intermediate_size
    assert (
        state_shape.conv_dim - state_shape.intermediate_size
    ) % key_group_width == 0
    padded_key_groups = (
        state_shape.conv_dim - state_shape.intermediate_size
    ) // key_group_width
    assert local_value_heads > 0
    assert state_shape.num_heads % local_value_heads == 0
    tp_world_size = state_shape.num_heads // local_value_heads
    assert padded_key_groups % tp_world_size == 0
    local_key_heads = padded_key_groups // tp_world_size
    return local_key_heads, local_value_heads, key_dim, value_dim


def dvr_gdn_state_input_bytes_per_request(cache_params, num_draft_tokens: int) -> int:
    """Return persistent q/k/v/g/beta window bytes for one request slot."""

    local_key_heads, local_value_heads, key_dim, value_dim = _local_gdn_dimensions(
        cache_params.shape
    )
    window_len = FLA_CHUNK_SIZE + num_draft_tokens
    conv_bytes = cache_params.dtype.conv.itemsize
    values_per_token = (
        2 * local_key_heads * key_dim * conv_bytes
        + local_value_heads * value_dim * conv_bytes
        + 2 * local_value_heads * 4  # g and beta are fp32.
    )
    return len(cache_params.layers) * window_len * values_per_token + 4


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
        + math.prod(cache_params.shape.temporal)
        * cache_params.dtype.temporal.itemsize
    )
    if draft_reuses_target_state:
        total += (
            num_layers
            * sum(math.prod(shape) for shape in cache_params.shape.conv)
            * cache_params.dtype.conv.itemsize
        )
    return total + dvr_gdn_state_input_bytes_per_request(
        cache_params, num_draft_tokens
    )


def _rebuild_gdn_self_draft_state(
    state_window: DVRStateInputCache,
    *,
    state_cache,
    state_input_indices: torch.Tensor,
    boundary_indices: torch.Tensor,
    token_count: torch.Tensor,
) -> None:
    """Rebuild request-owned self-draft state from an exact boundary and tail."""

    _, k, v, g, beta = state_window.tensors
    if is_cpu():
        raise NotImplementedError("DVR GDN self-draft state rebuild is GPU-only.")
    num_layers, num_slots, num_tokens, num_key_heads, key_dim = k.shape
    _, _, _, num_value_heads, value_dim = v.shape
    num_reqs = state_input_indices.numel()
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
        state_input_indices=state_input_indices,
        boundary_indices=boundary_indices,
        destination_indices=state_input_indices,
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


@dataclass
class DVRGDNStateAdapter:
    """Adapter for DVR state replay in gated linear-state layers.

    The adapter owns the rolling-window and commit mechanics so model backends
    do not need to know the verify/post-verify lifecycle details.
    """

    kernel_dispatcher: Any
    chunk_size: int = FLA_CHUNK_SIZE
    draft_reuses_target_state: bool = False
    state_input_cache: Optional[DVRStateInputCache] = None
    verify_boundary_indices: Optional[torch.Tensor] = None
    _verify_metadata: Optional[tuple[torch.Tensor, ...]] = field(
        default=None, init=False, repr=False
    )
    draft_conv_state: Optional[tuple[torch.Tensor, ...]] = field(
        default=None, init=False, repr=False
    )
    _layer_state_input_caches: Optional[tuple[DVRStateInputCache, ...]] = field(
        default=None, init=False, repr=False
    )
    _self_draft_decode_active: bool = field(default=False, init=False, repr=False)

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
        q = torch.zeros(
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
            draft_reuses_target_state=model_runner.spec_algorithm.is_dvr_self_draft(),
            state_input_cache=DVRStateInputCache(
                tensors=(q, torch.zeros_like(q), v, gate, torch.zeros_like(gate)),
                # Slot 0 is the padded-row dummy; real rows use req_pool_idx.
                tail_lens=torch.zeros(
                    num_slots, dtype=torch.int32, device=model_runner.device
                ),
            ),
            verify_boundary_indices=torch.zeros(
                num_slots, dtype=torch.long, device=model_runner.device
            ),
        )

    def batch_state(self, *, batch):
        """Resolve physical state and request-owned rows for one target batch."""

        pool = batch.req_to_token_pool
        live_indices = pool.get_mamba_indices(batch.req_pool_indices).to(torch.long)
        state_input_indices = batch.req_pool_indices.to(
            device=live_indices.device, dtype=torch.long
        )
        return (
            pool.get_speculative_mamba2_params_all_layers(),
            state_input_indices,
            live_indices,
        )

    def state_input_window(
        self, *, layer_idx: Optional[int] = None
    ) -> DVRStateInputCache:
        cache = self.state_input_cache
        if cache is None:
            raise RuntimeError("DVR linear state-input cache is not initialized.")
        if layer_idx is not None:
            if self._layer_state_input_caches is None:
                self._layer_state_input_caches = tuple(
                    cache[layer] for layer in range(cache.tensors[0].shape[0])
                )
            cache = self._layer_state_input_caches[layer_idx]
        return cache

    def zero_recurrent_state(self, *, state_cache, indices: torch.Tensor):
        indices = indices.to(device=state_cache.temporal.device, dtype=torch.long)
        for conv in state_cache.conv:
            conv[:, indices] = 0
        state_cache.temporal[:, indices] = 0

    def allocate_self_draft_state(self, *, state_cache, num_slots: int) -> None:
        """Allocate the small convolution half of self-draft's private state."""

        if not self.draft_reuses_target_state:
            return
        self.draft_conv_state = tuple(
            tensor.new_zeros((tensor.shape[0], num_slots, *tensor.shape[2:]))
            for tensor in state_cache.conv
        )

    def initialize_self_draft_state(
        self,
        *,
        state_cache,
        state_input_indices: torch.Tensor,
        live_indices: torch.Tensor,
    ) -> None:
        """Seed private self-draft state after target EXTEND.

        This full-state copy runs only when EXTEND establishes a new accepted
        endpoint. Steady-state decode reconstructs the same workspace directly
        from the authoritative boundary plus accepted transition rows.
        """

        if not self.draft_reuses_target_state:
            return
        if self.draft_conv_state is None:
            raise RuntimeError("DVR self draft has no private convolution state.")
        live_indices = live_indices.to(
            device=state_cache.temporal.device, dtype=torch.long
        )
        state_input_indices = state_input_indices.to(
            device=state_cache.temporal.device, dtype=torch.long
        )
        with profile_range("draft_state_copy"):
            state_cache.intermediate_ssm[:, state_input_indices, 0] = (
                state_cache.temporal[:, live_indices]
            )
            for target_conv, draft_conv in zip(
                state_cache.conv, self.draft_conv_state, strict=True
            ):
                draft_conv[:, state_input_indices] = target_conv[:, live_indices]

    @contextmanager
    def self_draft_decode(self):
        """Route only self-draft GDN decode to request-owned private state."""

        if not self.draft_reuses_target_state:
            yield
            return
        if self.draft_conv_state is None:
            raise RuntimeError("DVR self draft state is not initialized.")
        previous = self._self_draft_decode_active
        self._self_draft_decode_active = True
        try:
            yield
        finally:
            self._self_draft_decode_active = previous

    def decode_state(self, *, layer_cache, forward_batch, layer_idx: int):
        """Return private GDN state only while self-draft decode is captured/eager."""

        if not self._self_draft_decode_active:
            return None
        state_input_indices = forward_batch.req_pool_indices.to(
            device=layer_cache.temporal.device, dtype=torch.long
        )
        return (
            self.draft_conv_state[0][layer_idx],
            layer_cache.intermediate_ssm[:, 0],
            state_input_indices,
        )

    def set_verify_boundaries(
        self, *, state_input_indices: torch.Tensor, boundary_indices: torch.Tensor
    ) -> None:
        """Bind logical request rows to the physical exact boundary for verify."""

        if self.verify_boundary_indices is None:
            raise RuntimeError("DVR verify boundary table is not initialized.")
        state_input_indices = state_input_indices.to(
            device=self.verify_boundary_indices.device, dtype=torch.long
        )
        self.verify_boundary_indices[state_input_indices] = boundary_indices.to(
            device=self.verify_boundary_indices.device, dtype=torch.long
        )

    def cache_extend_tail(
        self,
        *,
        forward_batch,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        layer_idx: int,
    ):
        state_window = self.state_input_window(layer_idx=layer_idx)
        if (
            forward_batch.extend_prefix_lens_cpu is None
            or forward_batch.extend_seq_lens_cpu is None
        ):
            return

        state_inputs = (
            q.reshape(q.shape[1], q.shape[2], q.shape[3]),
            k.reshape(k.shape[1], k.shape[2], k.shape[3]),
            v.reshape(v.shape[1], v.shape[2], v.shape[3]),
            g.reshape(-1, g.shape[-1]),
            beta.reshape(-1, beta.shape[-1]),
        )
        state_input_indices = forward_batch.req_pool_indices.to(
            device=q.device, dtype=torch.long
        )
        state_window.write_extend_tail(
            values=state_inputs,
            indices=state_input_indices,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            chunk_size=self.chunk_size,
        )

    def prepare_target_verify(self, *, forward_batch, cache_indices: torch.Tensor):
        """Resolve graph-stable request, boundary, tail, and padding metadata once."""

        draft_token_num = forward_batch.spec_info.draft_token_num
        batch_size = forward_batch.input_ids.shape[0] // draft_token_num
        state_input_indices = forward_batch.req_pool_indices[:batch_size].to(
            device=cache_indices.device, dtype=torch.long
        )
        valid_mask = cache_indices[:batch_size] != PAD_SLOT_ID
        state_input_indices = torch.where(
            valid_mask, state_input_indices, torch.zeros_like(state_input_indices)
        )
        if self.verify_boundary_indices is None:
            raise RuntimeError("DVR verify boundary table is not initialized.")
        boundary_indices = self.verify_boundary_indices[state_input_indices]
        boundary_indices = torch.where(
            valid_mask, boundary_indices, torch.zeros_like(boundary_indices)
        )
        verify_cache_indices = torch.where(
            valid_mask,
            cache_indices[:batch_size],
            torch.zeros_like(cache_indices[:batch_size]),
        )
        tail_lens = (
            self.state_input_window()
            .get_tail_lens(indices=state_input_indices)
            .to(torch.long)
        )
        tail_lens = torch.where(
            valid_mask,
            tail_lens.clamp(min=0, max=self.chunk_size),
            torch.zeros_like(tail_lens),
        )
        boundary_state_steps = torch.where(
            valid_mask & (tail_lens + draft_token_num >= self.chunk_size),
            torch.ones_like(tail_lens),
            torch.full_like(tail_lens, -1),
        )
        self._verify_metadata = (
            boundary_indices,
            state_input_indices,
            verify_cache_indices,
            tail_lens,
            valid_mask,
            boundary_state_steps,
        )

    def target_verify_metadata(self) -> tuple[torch.Tensor, ...]:
        if self._verify_metadata is None:
            raise RuntimeError("DVR target-verify metadata was not prepared.")
        return self._verify_metadata

    def forward_target_verify(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_cache,
        boundary_indices: torch.Tensor,
        state_input_indices: torch.Tensor,
        tail_lens: torch.Tensor,
        valid_mask: torch.Tensor,
        boundary_state_steps: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Replay GDN state from cached prefill inputs and draft q/k/v/g/beta."""

        assert self.state_input_cache is not None
        batch_size = boundary_indices.shape[0]
        draft_token_num = query.shape[1] // batch_size
        draft_state_inputs = (
            query.reshape(batch_size, draft_token_num, *query.shape[2:]),
            key.reshape(batch_size, draft_token_num, *key.shape[2:]),
            value.reshape(batch_size, draft_token_num, *value.shape[2:]),
            g.reshape(batch_size, draft_token_num, g.shape[-1]),
            beta.reshape(batch_size, draft_token_num, beta.shape[-1]),
        )
        state_window = self.state_input_window(layer_idx=layer_idx)
        # The fixed window may retain values after tail_lens + draft_token_num.
        # They are causally after every returned logit, a boundary is exported
        # only after all chunk rows are valid, and state rebuilds use token_count.
        if query.is_cuda:
            q_cache, k_cache, v_cache, g_cache, beta_cache = state_window.tensors
            with profile_range("verify_state_pack"):
                q, k = _pack_verify_window_pair(
                    q_cache,
                    draft_state_inputs[0],
                    cache1=k_cache,
                    candidate1=draft_state_inputs[1],
                    state_input_indices=state_input_indices,
                    tail_lens=tail_lens,
                    valid_mask=valid_mask,
                )
                v, _ = _pack_verify_window_pair(
                    v_cache,
                    draft_state_inputs[2],
                    state_input_indices=state_input_indices,
                    tail_lens=tail_lens,
                    valid_mask=valid_mask,
                )
                cached_g, cached_beta = _pack_verify_window_pair(
                    g_cache,
                    draft_state_inputs[3],
                    cache1=beta_cache,
                    candidate1=draft_state_inputs[4],
                    state_input_indices=state_input_indices,
                    tail_lens=tail_lens,
                    valid_mask=valid_mask,
                )
        else:
            cols = torch.arange(draft_token_num).unsqueeze(0) + tail_lens.unsqueeze(1)
            state_window.write_rows(
                indices=state_input_indices.unsqueeze(1).expand(-1, draft_token_num),
                cols=cols,
                values=draft_state_inputs,
            )
            q, k, v, cached_g, cached_beta = state_window.read(
                indices=state_input_indices
            )
        core_attn_out, _, h = self.kernel_dispatcher.extend(
            q=q,
            k=k,
            v=v,
            g=cached_g,
            beta=cached_beta,
            ssm_states=state_cache.temporal,
            cache_indices=boundary_indices,
            query_start_loc=None,
            inplace_update=False,
        )
        if h is None or h.shape[1] <= 1:
            raise RuntimeError(
                "DVR GDN verify requires a linear-attention prefill backend "
                "that exports exact chunk-boundary states; use Triton for "
                "linear-attention prefill."
            )
        # Draft has finished, so the one-state workspace may now stage h64.
        # Rollback publishes it only for accepted crossings, then rebuilds the
        # self-draft endpoint in the same request-owned row.
        with profile_range("verify_boundary_stage"):
            if h.is_cuda:
                fused_mamba_state_scatter_with_mask(
                    state_cache.intermediate_ssm[:, 0].unsqueeze(0),
                    h[:batch_size].unsqueeze(0),
                    state_input_indices,
                    boundary_state_steps,
                )
            else:
                may_cross = boundary_state_steps >= 0
                state_cache.intermediate_ssm[state_input_indices[may_cross], 0] = h[
                    :batch_size, 1
                ][may_cross].to(state_cache.intermediate_ssm.dtype)

        with profile_range("verify_output_gather"):
            if core_attn_out.is_cuda:
                return _gather_verify_output(
                    core_attn_out,
                    tail_lens=tail_lens,
                    draft_tokens=draft_token_num,
                )
            value_shape = core_attn_out.shape[-2:]
            core_attn_out = core_attn_out.view(
                batch_size, self.chunk_size + draft_token_num, *value_shape
            )
            rows = torch.arange(batch_size).unsqueeze(1)
            cols = torch.arange(draft_token_num).unsqueeze(0) + tail_lens.unsqueeze(1)
            return core_attn_out[rows.expand(-1, draft_token_num), cols].reshape(
                1, batch_size * draft_token_num, *value_shape
            )

    def commit_after_verify(
        self,
        *,
        state_cache,
        state_input_indices: torch.Tensor,
        live_indices: torch.Tensor,
        boundary_indices: torch.Tensor,
        next_boundary_indices: torch.Tensor,
        accepted_token_counts: torch.Tensor,
    ) -> torch.Tensor:
        state_window = self.state_input_window()
        tail_lens_before = state_window.get_tail_lens(indices=state_input_indices).to(
            device=live_indices.device, dtype=torch.long
        )
        accepted_token_counts = accepted_token_counts.to(
            device=live_indices.device, dtype=torch.long
        )
        tail_lens_after = tail_lens_before + accepted_token_counts
        crosses_chunk_boundary = tail_lens_after >= self.chunk_size
        no_commit_step = torch.full_like(tail_lens_before, -1)
        # accept_lens includes the bonus token, so every non-idle request has a
        # valid accepted step. The fused scatter already handles the empty batch.
        fused_conv_window_scatter_with_mask(
            state_cache.conv[0],
            state_cache.intermediate_conv_window[0],
            live_indices,
            accepted_token_counts - 1,
        )
        if self.draft_reuses_target_state:
            if self.draft_conv_state is None:
                raise RuntimeError("DVR self draft has no private convolution state.")
            fused_conv_window_scatter_with_mask(
                self.draft_conv_state[0],
                state_cache.intermediate_conv_window[0],
                state_input_indices,
                accepted_token_counts - 1,
            )

        # Commit to the pool-selected next checkpoint slot. Radix execution uses
        # two slots so finish can publish either overlap boundary; without Radix,
        # the sole request-local slot is updated in place.
        with profile_range("boundary_publish"):
            fused_mamba_state_scatter_with_mask(
                state_cache.temporal,
                state_cache.intermediate_ssm,
                next_boundary_indices,
                torch.where(
                    crosses_chunk_boundary,
                    torch.zeros_like(tail_lens_before),
                    no_commit_step,
                ),
                src_indices_raw=state_input_indices,
            )
            fused_conv_window_scatter_with_mask(
                state_cache.conv[0],
                state_cache.intermediate_conv_window[0],
                next_boundary_indices,
                torch.where(
                    crosses_chunk_boundary,
                    self.chunk_size - 1 - tail_lens_before,
                    no_commit_step,
                ),
            )
        new_tail_lens = tail_lens_after - self.chunk_size
        tail_lens_after = torch.where(
            crosses_chunk_boundary, new_tail_lens, tail_lens_after
        )
        with profile_range("state_window_compact"):
            state_window.shift_after_boundary(
                indices=state_input_indices,
                crosses_chunk_boundary=crosses_chunk_boundary,
                chunk_size=self.chunk_size,
                tail_lens=tail_lens_after,
            )
        if self.draft_reuses_target_state:
            # Reconstruct only self-draft's private endpoint. Target recurrent
            # state remains boundary-owned and target verify reads it directly.
            with profile_range("draft_state_rebuild"):
                _rebuild_gdn_self_draft_state(
                    state_window,
                    state_cache=state_cache,
                    state_input_indices=state_input_indices,
                    boundary_indices=torch.where(
                        crosses_chunk_boundary,
                        next_boundary_indices,
                        boundary_indices,
                    ),
                    token_count=tail_lens_after,
                )
        # EAGLE/MTP owns separate draft state and skips this reconstruction.

        state_window.set_tail_lens(
            indices=state_input_indices, value=tail_lens_after.to(torch.int32)
        )
        return crosses_chunk_boundary
