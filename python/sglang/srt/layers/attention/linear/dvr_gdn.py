"""GDN-specific DVR state allocation, replay kernels, and lifecycle adapter."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.fla.op import exp
from sglang.srt.layers.attention.linear.dvr_state import DVRStateInputCache
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_conv_window_scatter_with_mask,
    fused_mamba_state_scatter_with_mask,
)
from sglang.srt.utils import is_cpu

__all__ = ["DVRGDNStateAdapter", "dvr_gdn_state_input_bytes_per_request"]

if not is_cpu():
    import triton
    import triton.language as tl


if not is_cpu():

    @triton.jit
    def _dvr_gdn_rebuild_live_state_kernel(
        k,
        v,
        g,
        beta,
        state,
        state_input_indices,
        boundary_indices,
        live_indices,
        token_count,
        N: tl.constexpr,
        S: tl.constexpr,
        C: tl.constexpr,
        T: tl.constexpr,
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
        live_idx = tl.load(live_indices + i_n).to(tl.int64)
        state_offset = (i_l * C + boundary_idx) * HV * V * K + i_hv * V * K
        p_h0 = state + state_offset + o_v[:, None] * K + o_k[None, :]
        recurrent_state = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        steps = tl.load(token_count + i_n).to(tl.int64)
        p_k = k + (((i_l * S + state_input_idx) * T * H + i_h) * K + o_k)
        p_v = v + (((i_l * S + state_input_idx) * T * HV + i_hv) * V + o_v)
        p_g = g + ((i_l * S + state_input_idx) * T * HV + i_hv)
        p_beta = beta + ((i_l * S + state_input_idx) * T * HV + i_hv)

        for step in range(0, T):
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

        state_offset = (i_l * C + live_idx) * HV * V * K + i_hv * V * K
        p_ht = state + state_offset + o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, recurrent_state.to(p_ht.dtype.element_ty), mask=mask_h)


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


def _rebuild_gdn_live_state_for_self_draft(
    state_window: DVRStateInputCache,
    *,
    state_cache,
    state_input_indices: torch.Tensor,
    boundary_indices: torch.Tensor,
    live_indices: torch.Tensor,
    token_count: Optional[Union[int, torch.Tensor]] = None,
) -> None:
    """Rebuild the live state directly from request-local window slots."""

    _, k, v, g, beta = state_window.tensors
    if is_cpu():
        raise NotImplementedError("DVR GDN self-draft state rebuild is GPU-only.")
    if token_count is None:
        token_count = k.shape[2]
    if isinstance(token_count, int):
        token_count = torch.full(
            (state_input_indices.numel(),),
            token_count,
            dtype=torch.int32,
            device=k.device,
        )

    num_layers, num_slots, num_tokens, num_key_heads, key_dim = k.shape
    _, _, _, num_value_heads, value_dim = v.shape
    num_reqs = state_input_indices.numel()
    if num_reqs == 0:
        return
    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 8)
    _dvr_gdn_rebuild_live_state_kernel[
        (
            triton.cdiv(value_dim, block_v),
            num_layers * num_reqs * num_value_heads,
        )
    ](
        k=k,
        v=v,
        g=g,
        beta=beta,
        state=state_cache.temporal,
        state_input_indices=state_input_indices,
        boundary_indices=boundary_indices,
        live_indices=live_indices,
        token_count=token_count.contiguous(),
        N=num_reqs,
        S=num_slots,
        C=state_cache.temporal.shape[1],
        T=num_tokens,
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
        if (
            state_cache.temporal.dtype != torch.float32
            or state_cache.intermediate_ssm.dtype != torch.float32
        ):
            raise RuntimeError(
                "DVR GDN verify requires fp32 recurrent and intermediate states."
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
        )

    def get_state_cache(self, *, batch):
        return batch.req_to_token_pool.get_speculative_mamba2_params_all_layers()

    def state_input_window(
        self, *, layer_idx: Optional[int] = None
    ) -> DVRStateInputCache:
        cache = self.state_input_cache
        if cache is None:
            raise RuntimeError("DVR linear state-input cache is not initialized.")
        if layer_idx is not None:
            cache = cache[layer_idx]
        return cache

    def get_live_indices(self, *, batch) -> torch.Tensor:
        return batch.req_to_token_pool.get_mamba_indices(batch.req_pool_indices).to(
            torch.long
        )

    def get_state_input_indices(self, *, batch, device: torch.device) -> torch.Tensor:
        return batch.req_pool_indices.to(device=device, dtype=torch.long)

    def zero_recurrent_state(self, *, state_cache, indices: torch.Tensor):
        indices = indices.to(device=state_cache.temporal.device, dtype=torch.long)
        for conv in state_cache.conv:
            conv[:, indices] = 0
        state_cache.temporal[:, indices] = 0

    def backup_draft_state(
        self,
        *,
        state_cache,
        indices: torch.Tensor,
        backup_indices: torch.Tensor,
        backup_size: int,
        out: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> tuple[torch.Tensor, ...]:
        indices = indices.to(device=state_cache.temporal.device, dtype=torch.long)
        backup_indices = backup_indices.to(
            device=state_cache.temporal.device, dtype=torch.long
        )
        if out is None:
            out = self.allocate_draft_state_backup(
                state_cache=state_cache, backup_size=backup_size
            )
        for tensor, saved_tensor in zip(state_cache.conv, out, strict=True):
            saved_tensor[:, backup_indices] = tensor[:, indices]
        return out

    @staticmethod
    def allocate_draft_state_backup(
        *, state_cache, backup_size: int
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor.new_empty((tensor.shape[0], backup_size, *tensor.shape[2:]))
            for tensor in state_cache.conv
        )

    def prepare_recurrent_state_for_verify(
        self,
        *,
        state_cache,
        live_indices: torch.Tensor,
        boundary_indices: torch.Tensor,
        draft_state_backup: Optional[tuple[torch.Tensor, ...]],
        backup_indices: Optional[torch.Tensor] = None,
    ):
        state_cache.temporal[:, live_indices] = state_cache.temporal[
            :, boundary_indices
        ]
        if draft_state_backup is not None:
            if backup_indices is None:
                raise RuntimeError("DVR live-state backup indices are missing.")
            backup_indices = backup_indices.to(
                device=state_cache.temporal.device, dtype=torch.long
            )
            for conv, saved_conv in zip(
                state_cache.conv, draft_state_backup, strict=True
            ):
                conv[:, live_indices] = saved_conv[:, backup_indices].to(
                    conv.dtype, copy=False
                )

    def capture_extend_prefix_boundary(
        self,
        *,
        forward_batch,
        state_cache,
        cache_indices: torch.Tensor,
    ) -> None:
        """Preserve an exact cached-prefix boundary before target EXTEND mutates it."""

        if forward_batch.mamba_track_indices is None:
            return

        prefix_lens = forward_batch.extend_prefix_lens
        if prefix_lens is None:
            return
        batch_size = prefix_lens.numel()
        seq_lens = forward_batch.seq_lens[:batch_size]
        boundaries = (
            torch.div(seq_lens, self.chunk_size, rounding_mode="floor")
            * self.chunk_size
        )
        capture_mask = (boundaries > 0) & (boundaries == prefix_lens)
        if forward_batch.mamba_track_mask is not None:
            capture_mask &= ~forward_batch.mamba_track_mask[:batch_size]

        src = cache_indices[:batch_size][capture_mask].to(torch.long)
        dst = forward_batch.mamba_track_indices[:batch_size][capture_mask].to(
            torch.long
        )
        for conv in state_cache.conv:
            conv[dst] = conv[src]
        state_cache.temporal[dst] = state_cache.temporal[src]

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

    def target_verify_indices(self, *, forward_batch, cache_indices: torch.Tensor):
        """Map padded target-verify rows to DVR live and input-window slots."""

        draft_token_num = forward_batch.spec_info.draft_token_num
        batch_size = forward_batch.input_ids.shape[0] // draft_token_num
        # Slot 0 is the shared dummy row used by padded CUDA graph requests.
        state_input_indices = forward_batch.req_pool_indices[:batch_size].to(
            device=cache_indices.device, dtype=torch.long
        )
        valid_mask = state_input_indices != 0
        dvr_indices = cache_indices[:batch_size].to(torch.long)
        dvr_indices = torch.where(
            valid_mask, dvr_indices, torch.zeros_like(dvr_indices)
        )
        return dvr_indices, state_input_indices, valid_mask

    def forward_target_verify(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_cache,
        dvr_indices: torch.Tensor,
        state_input_indices: torch.Tensor,
        valid_mask: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Replay GDN state from cached prefill inputs and draft q/k/v/g/beta."""

        assert self.state_input_cache is not None
        batch_size = dvr_indices.shape[0]
        draft_token_num = query.shape[1] // batch_size
        draft_state_inputs = (
            query.reshape(batch_size, draft_token_num, *query.shape[2:]),
            key.reshape(batch_size, draft_token_num, *key.shape[2:]),
            value.reshape(batch_size, draft_token_num, *value.shape[2:]),
            g.reshape(batch_size, draft_token_num, g.shape[-1]),
            beta.reshape(batch_size, draft_token_num, beta.shape[-1]),
        )
        state_window = self.state_input_window(layer_idx=layer_idx)
        tail_lens = state_window.get_tail_lens(indices=state_input_indices).to(
            torch.long
        )
        tail_lens = torch.where(
            valid_mask,
            tail_lens.clamp(min=0, max=self.chunk_size),
            torch.zeros_like(tail_lens),
        )
        # The fixed window may retain values after tail_lens + draft_token_num.
        # They are causally after every returned logit, a boundary is exported
        # only after all chunk rows are valid, and state rebuilds use token_count.
        state_window.write_rows(
            indices=state_input_indices.unsqueeze(1).expand(-1, draft_token_num),
            cols=(
                torch.arange(
                    draft_token_num,
                    dtype=torch.long,
                    device=state_input_indices.device,
                ).unsqueeze(0)
                + tail_lens.unsqueeze(1)
            ),
            values=draft_state_inputs,
        )
        q, k, v, cached_g, cached_beta = state_window.read(indices=state_input_indices)
        core_attn_out, _, h = self.kernel_dispatcher.extend(
            q=q,
            k=k,
            v=v,
            g=cached_g,
            beta=cached_beta,
            ssm_states=state_cache.temporal,
            cache_indices=dvr_indices,
            query_start_loc=None,
        )
        if h is None or h.shape[1] <= 1:
            raise RuntimeError(
                "DVR GDN verify requires a linear-attention prefill backend "
                "that exports exact chunk-boundary states; use Triton for "
                "linear-attention prefill."
            )
        state_cache.intermediate_ssm[
            intermediate_state_indices[:batch_size].to(torch.long), 0
        ] = h[:batch_size, 1].to(state_cache.intermediate_ssm.dtype)

        value_shape = core_attn_out.shape[-2:]
        core_attn_out = core_attn_out.view(
            batch_size, self.chunk_size + draft_token_num, *value_shape
        )
        rows = torch.arange(
            batch_size, dtype=torch.long, device=core_attn_out.device
        ).unsqueeze(1)
        cols = torch.arange(
            draft_token_num, dtype=torch.long, device=core_attn_out.device
        ).unsqueeze(0) + tail_lens.unsqueeze(1)
        return (
            core_attn_out[rows.expand(-1, draft_token_num), cols]
            .reshape(1, batch_size * draft_token_num, *value_shape)
            .contiguous()
        )

    def commit_after_verify(
        self,
        *,
        state_cache,
        state_input_indices: torch.Tensor,
        live_indices: torch.Tensor,
        boundary_indices: torch.Tensor,
        previous_boundary_indices: torch.Tensor,
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

        # Commit to the pool-selected next checkpoint slot. Radix execution uses
        # two slots so finish can publish either overlap boundary; without Radix,
        # the sole request-local slot is updated in place.
        fused_mamba_state_scatter_with_mask(
            state_cache.temporal,
            state_cache.intermediate_ssm,
            previous_boundary_indices,
            torch.where(
                crosses_chunk_boundary,
                torch.zeros_like(tail_lens_before),
                no_commit_step,
            ),
        )
        fused_conv_window_scatter_with_mask(
            state_cache.conv[0],
            state_cache.intermediate_conv_window[0],
            previous_boundary_indices,
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
        state_window.shift_after_boundary(
            indices=state_input_indices,
            crosses_chunk_boundary=crosses_chunk_boundary,
            chunk_size=self.chunk_size,
        )
        if self.draft_reuses_target_state:
            # Self draft consumes the target model's live recurrent slot. Keep
            # this path host-sync free and rebuild every accepted suffix.
            _rebuild_gdn_live_state_for_self_draft(
                state_window,
                state_cache=state_cache,
                state_input_indices=state_input_indices,
                boundary_indices=torch.where(
                    crosses_chunk_boundary,
                    previous_boundary_indices,
                    boundary_indices,
                ),
                live_indices=live_indices,
                token_count=tail_lens_after,
            )
        # EAGLE/MTP owns a separate draft recurrent cache. Its next target
        # verify restores target temporal state from boundary_indices, so
        # rebuilding the target live temporal slot here is dead work. The live
        # convolution state above remains necessary for accepted target tokens.

        state_window.set_tail_lens(
            indices=state_input_indices, value=tail_lens_after.to(torch.int32)
        )
        return crosses_chunk_boundary
