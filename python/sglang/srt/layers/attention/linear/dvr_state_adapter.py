"""Adapter boundary between model backends and DVR linear-state lifecycle."""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.linear.dvr_gdn_state import (
    _rebuild_gdn_state_from_qkvg_beta_chunkwise,
    create_gdn_state_input_cache,
)
from sglang.srt.layers.attention.linear.dvr_state import (
    DVRRecurrentStateBackup,
    DVRStateInputCache,
    DVRStateInputs,
    rebuild_dvr_live_state_grouped,
    run_dvr_chunkwise_verify,
    write_dvr_conv_windows,
)

__all__ = ["DVRGatedStateAdapter"]


@dataclass
class DVRGatedStateAdapter:
    """Adapter for DVR state replay in gated linear-state layers.

    The backend turns model-family tensors into DVRStateInputs. This adapter
    owns the rolling-window and commit mechanics so model backends do not need
    to know the verify/post-verify lifecycle details.
    """

    kernel_dispatcher: Any
    chunk_size: int = FLA_CHUNK_SIZE
    is_draft_worker: bool = False
    state_shape: Any = None
    conv_dtype: Optional[torch.dtype] = None
    device: Optional[str] = None
    state_input_cache: Optional[DVRStateInputCache] = None
    _verify_exports_boundary_state: bool = False

    @classmethod
    def for_gdn(
        cls,
        kernel_dispatcher,
        *,
        model_runner: Any,
        is_draft_worker: bool = False,
    ) -> "DVRGatedStateAdapter":
        mamba_cache_params = model_runner.mambaish_config.mamba2_cache_params

        return cls(
            kernel_dispatcher,
            is_draft_worker=is_draft_worker,
            state_shape=mamba_cache_params.shape,
            conv_dtype=mamba_cache_params.dtype.conv,
            device=model_runner.device,
        )

    def has_dvr_state(self, *, batch) -> bool:
        req_to_token_pool = batch.req_to_token_pool
        return (
            batch.batch_size() > 0
            and hasattr(req_to_token_pool, "get_mamba_indices")
            and hasattr(req_to_token_pool, "get_speculative_mamba2_params_all_layers")
            and hasattr(
                getattr(
                    getattr(req_to_token_pool, "mamba_pool", None),
                    "mamba_cache",
                    None,
                ),
                "intermediate_ssm",
            )
            and all(
                getattr(req, "mamba_ping_pong_track_buffer", None) is not None
                for req in batch.reqs
            )
        )

    def get_state_cache(self, *, batch):
        self.get_or_create_state_input_cache(req_to_token_pool=batch.req_to_token_pool)
        return batch.req_to_token_pool.get_speculative_mamba2_params_all_layers()

    def get_or_create_state_input_cache(
        self, *, req_to_token_pool
    ) -> DVRStateInputCache:
        cache = self.state_input_cache
        if cache is not None:
            return cache
        if self.state_shape is None or self.conv_dtype is None or self.device is None:
            raise RuntimeError("DVR GDN state-input cache metadata is not initialized.")

        state_cache = req_to_token_pool.get_speculative_mamba2_params_all_layers()
        num_layers = state_cache.intermediate_ssm.shape[0]
        spec_state_size = state_cache.intermediate_ssm.shape[1] - 1
        num_draft_tokens = state_cache.intermediate_ssm.shape[2]
        cache = create_gdn_state_input_cache(
            num_layers=num_layers,
            # Slot 0 is the padded-row dummy; real DVR rows use req_pool_idx + 1
            # to preserve the v5 hot-path indexing convention.
            num_slots=spec_state_size + 2,
            num_draft_tokens=num_draft_tokens,
            state_shape=self.state_shape,
            dtype=self.conv_dtype,
            device=self.device,
        )
        self.state_input_cache = cache
        return cache

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

    def scan_chunkwise(self, *, state_inputs: DVRStateInputs, **kwargs) -> tuple:
        q, k, v, g, beta = state_inputs.tensors()
        result = self.kernel_dispatcher.extend(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            **kwargs,
        )
        self._verify_exports_boundary_state = result[2] is not None
        return result

    def rebuild_recurrent_state(
        self,
        state_inputs: DVRStateInputs,
        *,
        initial_state: torch.Tensor,
        token_count=None,
    ) -> torch.Tensor:
        q, k, v, g, beta = state_inputs.tensors()
        return _rebuild_gdn_state_from_qkvg_beta_chunkwise(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            token_count=token_count,
        )

    def validate_state_cache(self, *, state_cache):
        assert state_cache.temporal.dtype == torch.float32, (
            "DVR linear-state verify requires fp32 temporal state checkpoints. "
            "bf16/fp16 checkpoints round the chunkwise scan state and can "
            "diverge from full prefill across chunks."
        )
        assert state_cache.intermediate_ssm.dtype == torch.float32, (
            "DVR linear-state verify requires fp32 intermediate prefill states."
        )

    def zero_recurrent_state(self, *, state_cache, indices: torch.Tensor):
        indices = indices.to(device=state_cache.temporal.device, dtype=torch.long)
        for conv in state_cache.conv:
            conv[:, indices] = 0
        state_cache.temporal[:, indices] = 0

    def backup_recurrent_state(
        self, *, state_cache, indices: torch.Tensor
    ) -> DVRRecurrentStateBackup:
        indices = indices.to(device=state_cache.temporal.device, dtype=torch.long)
        return DVRRecurrentStateBackup(
            conv=tuple(conv[:, indices].clone() for conv in state_cache.conv),
            temporal=state_cache.temporal[:, indices].clone(),
            indices=indices.clone(),
        )

    def restore_recurrent_state(
        self,
        *,
        state_cache,
        backup: DVRRecurrentStateBackup,
        indices: Optional[torch.Tensor] = None,
    ):
        dst_indices = backup.indices if indices is None else indices
        dst_indices = dst_indices.to(device=state_cache.temporal.device, dtype=torch.long)
        for conv, saved_conv in zip(state_cache.conv, backup.conv, strict=True):
            conv[:, dst_indices] = saved_conv.to(conv.dtype, copy=False)
        state_cache.temporal[:, dst_indices] = backup.temporal.to(
            state_cache.temporal.dtype, copy=False
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
            state_cache.temporal[:, live_indices] = state_cache.temporal[
                :, boundary_indices
            ]
            return

        # Draft decode mutates the live recurrent slot. DVR target verify needs
        # the chunk-boundary SSM state for chunkwise scan, but the draft-start
        # conv state for producing the draft suffix inputs.
        self.restore_recurrent_state(
            state_cache=state_cache,
            backup=boundary_backup,
            indices=boundary_indices,
        )
        state_cache.temporal[:, live_indices] = boundary_backup.temporal.to(
            state_cache.temporal.dtype, copy=False
        )
        if live_backup is not None:
            for conv, saved_conv in zip(state_cache.conv, live_backup.conv, strict=True):
                conv[:, live_indices] = saved_conv.to(conv.dtype, copy=False)

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
        if self.is_draft_worker:
            # DVR state-input windows are target-model prefill oracles.  EAGLE
            # and MTP draft workers may share request slots with the target
            # worker, so draft-model state inputs must never overwrite them.
            return
        state_window = self.state_input_window(layer_idx=layer_idx)
        if (
            forward_batch.extend_prefix_lens_cpu is None
            or forward_batch.extend_seq_lens_cpu is None
        ):
            return

        state_inputs = DVRStateInputs.from_tensors(
            (
                q.reshape(q.shape[1], q.shape[2], q.shape[3]),
                k.reshape(k.shape[1], k.shape[2], k.shape[3]),
                v.reshape(v.shape[1], v.shape[2], v.shape[3]),
                g.reshape(-1, g.shape[-1]),
                beta.reshape(-1, beta.shape[-1]),
            )
        )
        state_input_indices = forward_batch.req_pool_indices.to(
            device=q.device, dtype=torch.long
        )
        state_inputs.write_extend_tail(
            state_window,
            indices=state_input_indices + 1,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            chunk_size=self.chunk_size,
        )

    def forward_gdn_target_verify(
        self,
        *,
        layer,
        forward_batch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        state_cache,
        cache_indices: torch.Tensor,
        query_start_loc: Optional[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """Run GDN target verify using DVR's prefill-equivalent state replay."""

        assert forward_batch.forward_mode.is_target_verify()
        assert self.state_input_cache is not None
        draft_token_num = forward_batch.spec_info.draft_token_num
        batch_size = mixed_qkv.shape[0] // draft_token_num
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

        has_initial_states = (forward_batch.seq_lens[:batch_size] > 0).to(
            dtype=torch.bool,
            device=forward_batch.input_ids.device,
        )
        has_initial_states = has_initial_states & valid_mask.to(
            device=has_initial_states.device
        )
        conv_input_reshaped = mixed_qkv.view(
            batch_size, draft_token_num, -1
        ).transpose(1, 2)
        initial_conv_windows = state_cache.conv[0][dvr_indices].clone()
        from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn

        conv_output = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=state_cache.conv[0],
            has_initial_state=has_initial_states,
            cache_indices=dvr_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=[draft_token_num] * batch_size,
        ).transpose(0, 1)[: mixed_qkv.shape[0]]

        write_dvr_conv_windows(
            intermediate_conv_window_cache=state_cache.intermediate_conv_window[0],
            intermediate_state_indices=torch.arange(
                cache_indices.shape[0],
                dtype=torch.int32,
                device=cache_indices.device,
            ),
            initial_conv_windows=initial_conv_windows,
            conv_input_reshaped=conv_input_reshaped,
            num_draft_tokens=draft_token_num,
        )

        query, key, value = torch.split(
            conv_output,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        draft_state_inputs = DVRStateInputs.from_tensors(
            (
                query.reshape(
                    batch_size,
                    draft_token_num,
                    layer.num_q_heads,
                    layer.head_q_dim,
                ),
                key.reshape(
                    batch_size,
                    draft_token_num,
                    layer.num_k_heads,
                    layer.head_k_dim,
                ),
                value.reshape(
                    batch_size,
                    draft_token_num,
                    layer.num_v_heads,
                    layer.head_v_dim,
                ),
                g.reshape(batch_size, draft_token_num, layer.num_v_heads),
                beta.reshape(batch_size, draft_token_num, layer.num_v_heads),
            )
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
        return run_dvr_chunkwise_verify(
            state_ops=self,
            state_input_cache=state_window,
            draft_state_inputs=draft_state_inputs,
            ssm_states=state_cache.temporal,
            cache_indices=dvr_indices,
            state_input_indices=state_input_indices,
            tail_lens=tail_lens,
            intermediate_state_cache=state_cache.intermediate_ssm,
            intermediate_state_indices=torch.arange(
                cache_indices.shape[0],
                dtype=torch.int32,
                device=cache_indices.device,
            ),
            batch_size=batch_size,
            draft_token_num=draft_token_num,
            chunk_size=self.chunk_size,
        )

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
    ) -> torch.Tensor:
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
        boundary_needs_rebuild = crosses_chunk_boundary

        has_live_conv_commit = accepted_token_counts > 0
        live_conv_req_indices = torch.nonzero(has_live_conv_commit).flatten()
        if live_conv_req_indices.numel() > 0:
            self._scatter_state(
                state_cache.conv[0],
                state_cache.intermediate_conv_window[0],
                live_indices[live_conv_req_indices],
                accepted_steps[live_conv_req_indices],
            )

        crossing_req_indices = torch.nonzero(boundary_needs_rebuild).flatten()
        if crossing_req_indices.numel() > 0:
            if self._verify_exports_boundary_state:
                # run_dvr_chunkwise_verify stores the first crossed boundary in
                # step 0 when the selected linear backend exports FLA's `h`.
                self._scatter_state(
                    state_cache.temporal,
                    state_cache.intermediate_ssm,
                    boundary_indices,
                    torch.where(
                        boundary_needs_rebuild,
                        torch.zeros_like(tail_lens_before),
                        no_commit_step,
                    ),
                )
            else:
                # Some prefill kernels return only final state. Rebuild only the
                # crossed boundary; attention backend selection remains free.
                rebuild_dvr_live_state_grouped(
                    state_ops=self,
                    state_input_cache=state_window,
                    temporal_state=state_cache.temporal,
                    state_input_indices=state_input_indices,
                    live_indices=boundary_indices,
                    boundary_indices=boundary_indices,
                    req_indices=crossing_req_indices,
                    token_count=torch.full(
                        (crossing_req_indices.numel(),),
                        self.chunk_size,
                        dtype=torch.long,
                        device=tail_lens_before.device,
                    ),
                )
        self._scatter_state(
            state_cache.conv[0],
            state_cache.intermediate_conv_window[0],
            boundary_indices,
            torch.where(
                boundary_needs_rebuild,
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
        state_window.zero_after_lens(
            indices=state_input_indices,
            keep_lens=tail_lens_after,
        )
        draft_token_num = state_cache.intermediate_conv_window[0].shape[2]
        partial_accept = accepted_token_counts < draft_token_num
        partial_accept_req_indices = req_indices[partial_accept]
        if partial_accept_req_indices.numel() > 0:
            # Target verify already left the full accepted chain in the live
            # slot. Only rejection shortens that chain and requires a rebuild.
            rebuild_dvr_live_state_grouped(
                state_ops=self,
                state_input_cache=state_window,
                temporal_state=state_cache.temporal,
                state_input_indices=state_input_indices,
                live_indices=live_indices,
                boundary_indices=boundary_indices,
                req_indices=partial_accept_req_indices,
                token_count=tail_lens_after[partial_accept_req_indices],
            )

        state_window.set_tail_lens(
            indices=state_input_indices, value=tail_lens_after.to(torch.int32)
        )
        return crosses_chunk_boundary
