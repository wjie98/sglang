"""GDN-specific DVR state allocation, replay kernels, and lifecycle adapter."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch

from sglang.srt.configs.mamba_utils import Mamba2StateShape
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.fla.op import exp
from sglang.srt.layers.attention.linear.dvr_state import (
    DVRRecurrentStateBackup,
    DVRStateInputCache,
    rebuild_dvr_live_state_grouped,
)
from sglang.srt.utils import is_cpu

__all__ = ["DVRGDNStateAdapter", "create_gdn_state_input_cache"]

if not is_cpu():
    import triton
    import triton.language as tl


def _infer_gdn_state_input_shapes(
    state_shape: Mamba2StateShape,
) -> Tuple[int, int, int, int]:
    """Infer local q/k and v dimensions from existing Mamba2 metadata."""

    local_value_heads, value_head_dim, key_head_dim = state_shape.temporal
    key_group_width = 2 * state_shape.state_size
    assert key_group_width > 0
    assert state_shape.conv_dim >= state_shape.intermediate_size
    assert (
        state_shape.conv_dim - state_shape.intermediate_size
    ) % key_group_width == 0

    padded_num_key_groups = (
        state_shape.conv_dim - state_shape.intermediate_size
    ) // key_group_width
    assert local_value_heads > 0
    assert state_shape.num_heads % local_value_heads == 0
    tp_world_size = state_shape.num_heads // local_value_heads
    assert padded_num_key_groups % tp_world_size == 0
    local_key_heads = padded_num_key_groups // tp_world_size
    return local_key_heads, key_head_dim, local_value_heads, value_head_dim


if not is_cpu():

    @triton.jit
    def _dvr_gdn_final_state_kernel(
        k,
        v,
        g,
        beta,
        initial_state,
        final_state,
        token_count,
        T: tl.constexpr,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        i_v, i_nh = tl.program_id(0), tl.program_id(1)
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        state_offset = i_n * HV * V * K + i_hv * V * K
        p_h0 = initial_state + state_offset + o_v[:, None] * K + o_k[None, :]
        state = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        steps = tl.load(token_count + i_n).to(tl.int64)
        p_k = k + ((i_n * T * H + i_h) * K + o_k)
        p_v = v + ((i_n * T * HV + i_hv) * V + o_v)
        p_g = g + (i_n * T * HV + i_hv)
        p_beta = beta + (i_n * T * HV + i_hv)

        for step in range(0, T):
            if step < steps:
                key = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
                value = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
                key /= tl.sqrt(tl.sum(key * key) + 1e-6)
                state *= exp(tl.load(p_g).to(tl.float32))
                value -= tl.sum(state * key[None, :], 1)
                value *= tl.load(p_beta).to(tl.float32)
                state += value[:, None] * key[None, :]

            p_k += H * K
            p_v += HV * V
            p_g += HV
            p_beta += HV

        p_ht = final_state + state_offset + o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, state.to(p_ht.dtype.element_ty), mask=mask_h)


def _zero_gdn_rows_after_token_count(
    tensor: torch.Tensor, token_count: torch.Tensor
) -> torch.Tensor:
    token_count = token_count.to(device=tensor.device, dtype=torch.long).clamp(
        min=0, max=tensor.shape[1]
    )
    rows = torch.arange(tensor.shape[1], device=tensor.device).unsqueeze(0)
    keep = rows < token_count.unsqueeze(1)
    view_shape = keep.shape + (1,) * (tensor.dim() - 2)
    return torch.where(keep.view(view_shape), tensor, torch.zeros_like(tensor))


def _rebuild_gdn_state_from_qkvg_beta_chunkwise(
    state_inputs: Tuple[torch.Tensor, ...],
    *,
    initial_state: torch.Tensor,
    token_count: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor:
    """Rebuild partial acceptance with the deterministic prefill kernel."""

    q, k, v, g, beta = state_inputs
    if token_count is None:
        token_count = q.shape[1]
    if isinstance(token_count, int):
        token_count = torch.full(
            (q.shape[0],), token_count, dtype=torch.long, device=q.device
        )

    q = _zero_gdn_rows_after_token_count(q, token_count)
    k = _zero_gdn_rows_after_token_count(k, token_count)
    v = _zero_gdn_rows_after_token_count(v, token_count)
    g = _zero_gdn_rows_after_token_count(g, token_count)
    beta = _zero_gdn_rows_after_token_count(beta, token_count)

    # FLA exports the final boundary in h rather than updating initial_state.
    cache_indices = torch.arange(q.shape[0], dtype=torch.int32, device=q.device)
    _, _, h = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        initial_state_indices=cache_indices,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    return h[:, -1]


def _rebuild_gdn_state_for_self_draft(
    state_inputs: Tuple[torch.Tensor, ...],
    *,
    initial_state: torch.Tensor,
    token_count: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor:
    """Rebuild only the final recurrent state consumed by self draft."""

    q, k, v, g, beta = state_inputs
    if is_cpu():
        raise NotImplementedError("DVR GDN self-draft state rebuild is GPU-only.")
    if token_count is None:
        token_count = q.shape[1]
    if isinstance(token_count, int):
        token_count = torch.full(
            (q.shape[0],), token_count, dtype=torch.long, device=q.device
        )

    _, num_tokens, num_key_heads, key_dim = q.shape
    batch_size, _, num_value_heads, value_dim = v.shape
    token_count = token_count.to(device=k.device, dtype=torch.long).clamp(
        min=0, max=num_tokens
    )
    final_state = torch.empty_like(initial_state)
    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 8)
    _dvr_gdn_final_state_kernel[
        (triton.cdiv(value_dim, block_v), batch_size * num_value_heads)
    ](
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        final_state=final_state,
        token_count=token_count,
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
    return final_state


def create_gdn_state_input_cache(
    *,
    num_layers: int,
    num_slots: int,
    num_draft_tokens: int,
    state_shape: Mamba2StateShape,
    dtype: torch.dtype,
    device: str,
) -> DVRStateInputCache:
    """Allocate the q/k/v/g/beta rolling window used by GDN DVR."""

    local_key_heads, key_dim, local_value_heads, value_dim = (
        _infer_gdn_state_input_shapes(state_shape)
    )
    window_len = FLA_CHUNK_SIZE + num_draft_tokens
    q = torch.zeros(
        num_layers,
        num_slots,
        window_len,
        local_key_heads,
        key_dim,
        dtype=dtype,
        device=device,
    )
    v = torch.zeros(
        num_layers,
        num_slots,
        window_len,
        local_value_heads,
        value_dim,
        dtype=dtype,
        device=device,
    )
    gate = torch.zeros(
        num_layers,
        num_slots,
        window_len,
        local_value_heads,
        dtype=torch.float32,
        device=device,
    )
    return DVRStateInputCache(
        tensors=(q, torch.zeros_like(q), v, gate, torch.zeros_like(gate)),
        tail_lens=torch.zeros(
            num_layers,
            num_slots,
            dtype=torch.int32,
            device=device,
        ),
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
    _verify_exports_boundary_state: Optional[bool] = None

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
        spec_state_size = state_cache.intermediate_ssm.shape[1] - 1
        num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        if num_draft_tokens is None:
            raise RuntimeError("DVR requires speculative_num_draft_tokens.")

        return cls(
            kernel_dispatcher,
            draft_reuses_target_state=model_runner.spec_algorithm.is_dvr_self_draft(),
            state_input_cache=create_gdn_state_input_cache(
                num_layers=num_layers,
                # Slot 0 is the padded-row dummy; real DVR rows use req_pool_idx + 1.
                num_slots=spec_state_size + 2,
                num_draft_tokens=num_draft_tokens,
                state_shape=mamba_cache_params.shape,
                dtype=mamba_cache_params.dtype.conv,
                device=model_runner.device,
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

    def get_state_input_indices(
        self, *, batch, device: torch.device
    ) -> torch.Tensor:
        return batch.req_pool_indices.to(device=device, dtype=torch.long) + 1

    def _scatter_state(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        dst_indices: torch.Tensor,
        step_indices: torch.Tensor,
    ) -> None:
        # The fused scatter kernel flat-copies source rows and requires a
        # contiguous source buffer.  DVR sometimes receives layer views from the
        # speculative Mamba cache, so normalize layout at the DVR commit edge.
        if not src.is_contiguous():
            src = src.contiguous()
        from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
            fused_mamba_state_scatter_with_mask,
        )

        fused_mamba_state_scatter_with_mask(
            dst, src, dst_indices, step_indices
        )

    def zero_recurrent_state(self, *, state_cache, indices: torch.Tensor):
        indices = indices.to(device=state_cache.temporal.device, dtype=torch.long)
        for conv in state_cache.conv:
            conv[:, indices] = 0
        state_cache.temporal[:, indices] = 0

    def backup_recurrent_state(
        self,
        *,
        state_cache,
        indices: torch.Tensor,
        include_temporal: bool = True,
    ) -> DVRRecurrentStateBackup:
        indices = indices.to(device=state_cache.temporal.device, dtype=torch.long)
        return DVRRecurrentStateBackup(
            conv=tuple(conv[:, indices].clone() for conv in state_cache.conv),
            temporal=(
                state_cache.temporal[:, indices].clone()
                if include_temporal
                else None
            ),
        )

    def prepare_recurrent_state_for_verify(
        self,
        *,
        state_cache,
        live_indices: torch.Tensor,
        boundary_indices: torch.Tensor,
        boundary_backup: Optional[DVRRecurrentStateBackup],
        live_backup: Optional[DVRRecurrentStateBackup],
    ):
        if boundary_backup is None:
            if self.draft_reuses_target_state:
                raise RuntimeError(
                    "DVR self-draft target verify is missing its recurrent-state snapshot."
                )
            state_cache.temporal[:, live_indices] = state_cache.temporal[
                :, boundary_indices
            ]
            return

        # Draft decode mutates the live recurrent slot. DVR target verify needs
        # the chunk-boundary SSM state for chunkwise scan, but the draft-start
        # conv state for producing the draft suffix inputs.
        assert boundary_backup.temporal is not None
        for conv, saved_conv in zip(
            state_cache.conv, boundary_backup.conv, strict=True
        ):
            conv[:, boundary_indices] = saved_conv.to(conv.dtype, copy=False)
        state_cache.temporal[:, boundary_indices] = boundary_backup.temporal.to(
            state_cache.temporal.dtype, copy=False
        )
        state_cache.temporal[:, live_indices] = boundary_backup.temporal.to(
            state_cache.temporal.dtype, copy=False
        )
        if live_backup is not None:
            for conv, saved_conv in zip(state_cache.conv, live_backup.conv, strict=True):
                conv[:, live_indices] = saved_conv.to(conv.dtype, copy=False)

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
        boundaries = torch.div(
            seq_lens, self.chunk_size, rounding_mode="floor"
        ) * self.chunk_size
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
            indices=state_input_indices + 1,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            chunk_size=self.chunk_size,
        )
        state_input_indices = state_input_indices + 1
        state_window.zero_after_lens(
            indices=state_input_indices,
            keep_lens=state_window.get_tail_lens(indices=state_input_indices),
        )

    def target_verify_indices(self, *, forward_batch, cache_indices: torch.Tensor):
        """Map padded target-verify rows to DVR live and input-window slots."""

        draft_token_num = forward_batch.spec_info.draft_token_num
        batch_size = forward_batch.input_ids.shape[0] // draft_token_num
        rows = torch.arange(batch_size, dtype=torch.long, device=cache_indices.device)
        num_token_non_padded = forward_batch.num_token_non_padded
        if num_token_non_padded is None:
            valid_mask = torch.ones(
                batch_size, dtype=torch.bool, device=cache_indices.device
            )
        else:
            if torch.is_tensor(num_token_non_padded):
                num_token_non_padded = num_token_non_padded.to(
                    device=cache_indices.device, dtype=torch.long
                )
            valid_mask = rows * draft_token_num < num_token_non_padded

        # Slot 0 is the shared dummy row used by padded CUDA graph requests.
        dvr_indices = cache_indices[:batch_size].to(torch.long)
        dvr_indices = torch.where(valid_mask, dvr_indices, torch.zeros_like(dvr_indices))
        state_input_indices = forward_batch.req_pool_indices[:batch_size].to(
            device=cache_indices.device, dtype=torch.long
        )
        state_input_indices = state_input_indices + 1
        state_input_indices = torch.where(
            valid_mask, state_input_indices, torch.zeros_like(state_input_indices)
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
            cache_indices=dvr_indices,
            query_start_loc=None,
        )
        exports_boundary_state = h is not None
        if self._verify_exports_boundary_state is None:
            self._verify_exports_boundary_state = exports_boundary_state
        elif self._verify_exports_boundary_state != exports_boundary_state:
            raise RuntimeError("GDN verify backend changed its boundary-state contract.")
        if exports_boundary_state and h.shape[1] > 1:
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
        return core_attn_out[
            rows.expand(-1, draft_token_num), cols
        ].reshape(1, batch_size * draft_token_num, *value_shape).contiguous()

    def commit_after_verify(
        self,
        *,
        state_cache,
        state_input_indices: torch.Tensor,
        live_indices: torch.Tensor,
        boundary_indices: torch.Tensor,
        verified_tail_lens: torch.Tensor,
        accepted_token_counts: torch.Tensor,
        accepted_steps: torch.Tensor,
    ) -> None:
        state_window = self.state_input_window()
        tail_lens_before = verified_tail_lens.to(
            device=live_indices.device, dtype=torch.long
        )
        tail_lens_after = tail_lens_before + accepted_token_counts
        crosses_chunk_boundary = tail_lens_after >= self.chunk_size
        no_commit_step = torch.full_like(tail_lens_before, -1)
        req_indices = torch.arange(
            live_indices.shape[0],
            dtype=torch.long,
            device=live_indices.device,
        )
        # accept_lens includes the bonus token, so every non-idle request has a
        # valid accepted step. The fused scatter already handles the empty batch.
        self._scatter_state(
            state_cache.conv[0],
            state_cache.intermediate_conv_window[0],
            live_indices,
            accepted_steps,
        )

        if self._verify_exports_boundary_state is None:
            raise RuntimeError("GDN verify committed before its state scan completed.")
        if self._verify_exports_boundary_state:
            # Target verify stores the first crossed boundary in step 0 when
            # the selected linear backend exports FLA's `h`.
            self._scatter_state(
                state_cache.temporal,
                state_cache.intermediate_ssm,
                boundary_indices,
                torch.where(
                    crosses_chunk_boundary,
                    torch.zeros_like(tail_lens_before),
                    no_commit_step,
                ),
            )
        else:
            # Some linear backends return only final state. Keep this fallback
            # shape-static: zero-token rows reproduce their initial boundary,
            # while crossed rows replay exactly one chunk. Dynamic nonzero()
            # compaction would synchronize the host before every fallback.
            rebuild_dvr_live_state_grouped(
                state_input_cache=state_window,
                temporal_state=state_cache.temporal,
                state_input_indices=state_input_indices,
                live_indices=boundary_indices,
                boundary_indices=boundary_indices,
                req_indices=req_indices,
                token_count=torch.where(
                    crosses_chunk_boundary,
                    torch.full_like(tail_lens_before, self.chunk_size),
                    torch.zeros_like(tail_lens_before),
                ),
                rebuild_fn=_rebuild_gdn_state_from_qkvg_beta_chunkwise,
            )
        self._scatter_state(
            state_cache.conv[0],
            state_cache.intermediate_conv_window[0],
            boundary_indices,
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
        # Keep the fixed verify window clean once per batch, not once per GDN
        # layer inside the captured target forward.
        state_window.zero_after_lens(
            indices=state_input_indices, keep_lens=tail_lens_after
        )
        if self.draft_reuses_target_state:
            # Self draft consumes the target model's live recurrent slot. Keep
            # this path host-sync free and rebuild every accepted suffix.
            rebuild_dvr_live_state_grouped(
                state_input_cache=state_window,
                temporal_state=state_cache.temporal,
                state_input_indices=state_input_indices,
                live_indices=live_indices,
                boundary_indices=boundary_indices,
                req_indices=req_indices,
                token_count=tail_lens_after,
                rebuild_fn=_rebuild_gdn_state_for_self_draft,
            )
        # EAGLE/MTP owns a separate draft recurrent cache. Its next target
        # verify restores target temporal state from boundary_indices, so
        # rebuilding the target live temporal slot here is dead work. The live
        # convolution state above remains necessary for accepted target tokens.

        state_window.set_tail_lens(
            indices=state_input_indices, value=tail_lens_after.to(torch.int32)
        )
