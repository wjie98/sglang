"""Adapter boundary between model backends and DVR linear-state lifecycle.

External callers:
- gdn_backend enters process_target_verify_* during TARGET_VERIFY.
- dvr_linear_state uses backup/restore/commit methods around draft and verify.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.linear.dvr_state import (
    DVRRecurrentStateBackup,
    DVRStateInputWindow,
    DVRStateOps,
)
from sglang.srt.layers.attention.linear.dvr_state_verify import (
    rebuild_dvr_live_state_grouped,
    run_dvr_chunkwise_verify,
    write_dvr_conv_windows,
)

__all__ = ["DVRGatedStateAdapter"]


@dataclass(frozen=True)
class DVRGatedForwardContext:
    """Layer-local DVR state context for one gated linear-state forward."""

    layer: Any
    forward_batch: Any
    state_cache: Any
    cache_indices: torch.Tensor
    query_start_loc: Optional[torch.Tensor]
    conv_states: torch.Tensor
    ssm_states: torch.Tensor
    seq_len: int
    is_target_verify: bool

    @property
    def spec_info(self):
        return self.forward_batch.spec_info

    @property
    def draft_token_num(self) -> int:
        return self.spec_info.draft_token_num

    @property
    def graph_batch_size(self) -> int:
        return self.seq_len // self.draft_token_num

    def valid_request_mask(self) -> torch.Tensor:
        batch_size = self.graph_batch_size
        device = self.cache_indices.device
        rows = torch.arange(batch_size, dtype=torch.long, device=device)
        num_token_non_padded = self.forward_batch.num_token_non_padded
        if num_token_non_padded is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if torch.is_tensor(num_token_non_padded):
            num_token_non_padded = num_token_non_padded.to(
                device=device, dtype=torch.long
            )
        return rows * self.draft_token_num < num_token_non_padded

    def safe_cache_indices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.cache_indices[: self.graph_batch_size].to(torch.long)
        valid_mask = self.valid_request_mask()
        # Slot 0 is the shared dummy mamba slot used by padded graph rows.
        return torch.where(valid_mask, indices, torch.zeros_like(indices)), valid_mask

    def safe_state_input_indices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.forward_batch.req_pool_indices[: self.graph_batch_size].to(
            device=self.cache_indices.device, dtype=torch.long
        )
        indices = indices + 1
        valid_mask = self.valid_request_mask()
        # Slot 0 is the shared dummy DVR state-input slot used by padded graph rows.
        return torch.where(valid_mask, indices, torch.zeros_like(indices)), valid_mask


@dataclass
class DVRGatedStateAdapter:
    """Adapter for DVR state replay in gated linear-state layers.

    The backend still produces layer-specific tensors such as q/k/v/g/beta and
    conv windows. This adapter owns the DVR rolling-window and commit mechanics
    so a model backend can opt in through a small set of calls.
    """

    ops: DVRStateOps
    chunk_size: int = FLA_CHUNK_SIZE

    @classmethod
    def for_gdn(cls, kernel_dispatcher) -> "DVRGatedStateAdapter":
        return cls(DVRStateOps.for_gdn(kernel_dispatcher))

    def is_dvr_target_verify(self, *, state_cache, is_target_verify: bool) -> bool:
        return is_target_verify and DVRStateInputWindow.from_cache(state_cache).enabled

    def state_input_tail_lens(
        self, *, state_cache, state_input_indices: torch.Tensor
    ) -> Optional[torch.Tensor]:
        state_window = DVRStateInputWindow.from_cache(state_cache)
        if not state_window.enabled:
            return None
        return state_window.tail_lens(indices=state_input_indices)

    def set_state_input_tail_lens(
        self,
        *,
        state_cache,
        state_input_indices: torch.Tensor,
        tail_lens: torch.Tensor,
    ):
        state_window = DVRStateInputWindow.from_cache(state_cache)
        if not state_window.enabled:
            return
        state_window.set_tail_lens(indices=state_input_indices, value=tail_lens)

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

    def _backup_recurrent_state(
        self, *, state_cache, indices: torch.Tensor
    ) -> DVRRecurrentStateBackup:
        indices = indices.to(device=state_cache.temporal.device, dtype=torch.long)
        return DVRRecurrentStateBackup(
            conv=tuple(conv[:, indices].clone() for conv in state_cache.conv),
            temporal=state_cache.temporal[:, indices].clone(),
            indices=indices.clone(),
        )

    def _restore_recurrent_state(
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

    def backup_verify_recurrent_states(
        self,
        *,
        state_cache,
        boundary_indices: torch.Tensor,
        live_indices: torch.Tensor,
    ) -> Tuple[DVRRecurrentStateBackup, DVRRecurrentStateBackup]:
        return (
            self._backup_recurrent_state(
                state_cache=state_cache, indices=boundary_indices
            ),
            self._backup_recurrent_state(state_cache=state_cache, indices=live_indices),
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
        # conv state for producing q/k/v on the draft suffix.
        self._restore_recurrent_state(
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

    def make_forward_context(
        self,
        *,
        layer,
        forward_batch,
        state_cache,
        cache_indices: torch.Tensor,
        query_start_loc: Optional[torch.Tensor],
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        seq_len: int,
    ) -> DVRGatedForwardContext:
        return DVRGatedForwardContext(
            layer=layer,
            forward_batch=forward_batch,
            state_cache=state_cache,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            conv_states=conv_states,
            ssm_states=ssm_states,
            seq_len=seq_len,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
        )

    def cache_extend_tail_from_forward(
        self,
        *,
        forward_batch,
        state_cache,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ):
        state_window = DVRStateInputWindow.from_cache(state_cache)
        if not state_window.enabled:
            return
        if (
            forward_batch.extend_prefix_lens_cpu is None
            or forward_batch.extend_seq_lens_cpu is None
        ):
            return

        q = q.reshape(q.shape[1], q.shape[2], q.shape[3])
        k = k.reshape(k.shape[1], k.shape[2], k.shape[3])
        v = v.reshape(v.shape[1], v.shape[2], v.shape[3])
        g = g.reshape(-1, g.shape[-1])
        beta = beta.reshape(-1, beta.shape[-1])
        state_input_indices = forward_batch.req_pool_indices.to(
            device=q.device, dtype=torch.long
        )
        state_input_indices = state_input_indices + 1

        state_window.write_extend_tail(
            indices=state_input_indices,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            chunk_size=self.chunk_size,
        )

    def process_target_verify_conv(
        self,
        *,
        context: DVRGatedForwardContext,
        mixed_qkv: torch.Tensor,
    ) -> torch.Tensor:
        """Run DVR draft conv and export absolute-offset conv windows."""

        assert self.is_dvr_target_verify(
            state_cache=context.state_cache, is_target_verify=context.is_target_verify
        )

        draft_token_num = context.draft_token_num
        batch_size = context.graph_batch_size
        forward_batch = context.forward_batch
        dvr_indices, valid_mask = context.safe_cache_indices()
        has_initial_states = (forward_batch.seq_lens[:batch_size] > 0).to(
            dtype=torch.bool,
            device=forward_batch.input_ids.device,
        )
        has_initial_states = has_initial_states & valid_mask.to(
            device=has_initial_states.device
        )
        mixed_qkv_linear = mixed_qkv
        mixed_qkv_reshaped = mixed_qkv_linear.view(
            batch_size, draft_token_num, -1
        ).transpose(1, 2)
        initial_conv_windows = context.conv_states[dvr_indices].clone()
        mixed_qkv = self.ops.run_verify_conv(
            mixed_qkv_linear.transpose(0, 1),
            context.layer.conv_weights,
            context.layer.bias,
            activation=context.layer.activation,
            conv_states=context.conv_states,
            has_initial_state=has_initial_states,
            cache_indices=dvr_indices,
            query_start_loc=context.query_start_loc,
            seq_lens_cpu=[draft_token_num] * batch_size,
        ).transpose(0, 1)[: mixed_qkv.shape[0]]

        write_dvr_conv_windows(
            intermediate_conv_window_cache=context.state_cache.intermediate_conv_window[
                0
            ],
            intermediate_state_indices=torch.arange(
                context.cache_indices.shape[0],
                dtype=torch.int32,
                device=context.cache_indices.device,
            ),
            initial_conv_windows=initial_conv_windows,
            mixed_qkv_reshaped=mixed_qkv_reshaped,
            verify_window_size=draft_token_num,
        )
        return mixed_qkv

    def process_target_verify_state(
        self,
        *,
        context: DVRGatedForwardContext,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        assert self.is_dvr_target_verify(
            state_cache=context.state_cache, is_target_verify=context.is_target_verify
        )

        draft_token_num = context.draft_token_num
        batch_size = context.graph_batch_size
        dvr_indices, valid_mask = context.safe_cache_indices()
        state_input_indices, _ = context.safe_state_input_indices()
        state_window = DVRStateInputWindow.from_cache(context.state_cache)
        tail_lens = state_window.tail_lens(indices=state_input_indices).to(torch.long)
        tail_lens = torch.where(
            valid_mask,
            tail_lens.clamp(min=0, max=self.chunk_size),
            torch.zeros_like(tail_lens),
        )
        return run_dvr_chunkwise_verify(
            state_ops=self.ops,
            state_window=state_window,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=context.ssm_states,
            cache_indices=dvr_indices,
            state_input_indices=state_input_indices,
            tail_lens=tail_lens,
            intermediate_state_cache=context.state_cache.intermediate_ssm,
            intermediate_state_indices=torch.arange(
                context.cache_indices.shape[0],
                dtype=torch.int32,
                device=context.cache_indices.device,
            ),
            batch_size=batch_size,
            draft_token_num=draft_token_num,
            num_q_heads=context.layer.num_q_heads,
            head_q_dim=context.layer.head_q_dim,
            num_k_heads=context.layer.num_k_heads,
            head_k_dim=context.layer.head_k_dim,
            num_v_heads=context.layer.num_v_heads,
            head_v_dim=context.layer.head_v_dim,
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
        accepted_tokens: torch.Tensor,
        accepted_steps: torch.Tensor,
    ) -> torch.Tensor:
        state_window = DVRStateInputWindow.from_cache(state_cache)
        pos_before = verified_tail_lens.to(device=live_indices.device, dtype=torch.long)
        pos_after = pos_before + accepted_tokens
        crossing = pos_after >= self.chunk_size

        self.ops.scatter_state(
            state_cache.conv[0],
            state_cache.intermediate_conv_window[0],
            live_indices,
            accepted_steps,
        )

        boundary_state_step = (
            0 if state_cache.intermediate_ssm.shape[2] == 1 else self.chunk_size - 1
        )
        no_commit_step = torch.full_like(pos_before, -1)
        commit_step = torch.where(
            crossing,
            torch.full_like(pos_before, boundary_state_step),
            no_commit_step,
        )
        self.ops.scatter_state(
            state_cache.temporal,
            state_cache.intermediate_ssm,
            boundary_indices,
            commit_step,
        )
        self.ops.scatter_state(
            state_cache.conv[0],
            state_cache.intermediate_conv_window[0],
            boundary_indices,
            torch.where(crossing, self.chunk_size - 1 - pos_before, no_commit_step),
        )

        new_pos = pos_after - self.chunk_size
        pos_after = torch.where(crossing, new_pos, pos_after)
        state_window.shift_after_boundary(
            indices=state_input_indices,
            crossing=crossing,
            chunk_size=self.chunk_size,
        )
        rebuild_dvr_live_state_grouped(
            state_ops=self.ops,
            state_window=state_window,
            temporal_state=state_cache.temporal,
            state_input_indices=state_input_indices,
            live_indices=live_indices,
            boundary_indices=boundary_indices,
            req_indices=torch.arange(
                live_indices.shape[0],
                dtype=torch.long,
                device=live_indices.device,
            ),
            token_count=pos_after,
        )

        state_window.set_tail_lens(
            indices=state_input_indices, value=pos_after.to(torch.int32)
        )
        return crossing
