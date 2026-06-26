"""Adapter boundary between model backends and DVR linear-state lifecycle.

External callers:
- gdn_backend enters process_target_verify_* during TARGET_VERIFY.
- dvr_linear_state uses backup/restore/commit methods around draft and verify.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.linear.dvr_state import (
    DVRRecurrentStateBackup,
    DVRStateInputs,
    DVRStateInputWindow,
    DVRStateOps,
)
from sglang.srt.layers.attention.linear.dvr_state_verify import (
    rebuild_dvr_live_state_grouped,
    run_dvr_chunkwise_verify,
    write_dvr_conv_windows,
)

__all__ = ["DVRGatedStateAdapter"]

_DVR_DEBUG_VERIFY_STATE_PRINTS = 0


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
    def verify_batch_size(self) -> int:
        return self.seq_len // self.draft_token_num

    def valid_request_mask(self) -> torch.Tensor:
        batch_size = self.verify_batch_size
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

    def padded_cache_indices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.cache_indices[: self.verify_batch_size].to(torch.long)
        valid_mask = self.valid_request_mask()
        # Slot 0 is the shared dummy mamba slot used by padded graph rows.
        return torch.where(valid_mask, indices, torch.zeros_like(indices)), valid_mask

    def padded_state_input_indices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.forward_batch.req_pool_indices[: self.verify_batch_size].to(
            device=self.cache_indices.device, dtype=torch.long
        )
        valid_mask = self.valid_request_mask()
        # Slot 0 is the shared dummy DVR state-input slot used by padded graph rows.
        return torch.where(valid_mask, indices, torch.zeros_like(indices)), valid_mask


@dataclass
class DVRGatedStateAdapter:
    """Adapter for DVR state replay in gated linear-state layers.

    The backend turns model-family tensors into DVRStateInputs. This adapter
    owns the rolling-window and commit mechanics so model backends do not need
    to know the verify/post-verify lifecycle details.
    """

    ops: DVRStateOps
    chunk_size: int = FLA_CHUNK_SIZE

    @classmethod
    def for_gdn(cls, kernel_dispatcher) -> "DVRGatedStateAdapter":
        from sglang.srt.layers.attention.linear.dvr_gdn_state import DVRGDNStateOps

        return cls(DVRGDNStateOps.create(kernel_dispatcher))

    def has_dvr_state(self, *, batch) -> bool:
        req_to_token_pool = batch.req_to_token_pool
        return (
            batch.batch_size() > 0
            and hasattr(req_to_token_pool, "get_mamba_indices")
            and hasattr(req_to_token_pool, "get_speculative_mamba2_params_all_layers")
            and all(
                getattr(req, "mamba_ping_pong_track_buffer", None) is not None
                for req in batch.reqs
            )
        )

    def get_state_cache(self, *, batch):
        return batch.req_to_token_pool.get_speculative_mamba2_params_all_layers()

    def get_live_indices(self, *, batch) -> torch.Tensor:
        return batch.req_to_token_pool.get_mamba_indices(batch.req_pool_indices).to(
            torch.long
        )

    def get_state_input_indices(
        self, *, batch, device: torch.device
    ) -> torch.Tensor:
        return batch.req_pool_indices.to(device=device, dtype=torch.long)

    def get_boundary_indices(
        self,
        *,
        batch,
        boundary_track_idx_by_rid,
        device: torch.device,
    ) -> torch.Tensor:
        return self.get_boundary_indices_for_reqs(
            reqs=batch.reqs,
            track_indices=[
                boundary_track_idx_by_rid[req.rid] for req in batch.reqs
            ],
            device=device,
        )

    def get_boundary_indices_for_reqs(
        self,
        *,
        reqs,
        track_indices,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.stack(
            [
                req.mamba_ping_pong_track_buffer[track_idx]
                for req, track_idx in zip(reqs, track_indices, strict=True)
            ]
        ).to(device=device, dtype=torch.long)

    def get_other_track_idx(self, *, batch, track_idx: int) -> int:
        return batch.req_to_token_pool.get_mamba_ping_pong_other_idx(track_idx)

    def get_current_prefill_checkpoint_track_idx(
        self, *, batch, req, boundary_seqlen: int
    ) -> Optional[int]:
        last_track_seqlen = req.mamba_last_track_seqlen
        if last_track_seqlen is not None and last_track_seqlen > 0:
            assert last_track_seqlen % self.chunk_size == 0, (
                "DVR linear-state verify must not reuse non-chunk-boundary checkpoints."
            )
        if boundary_seqlen <= 0 or last_track_seqlen != boundary_seqlen:
            return None
        return self.get_other_track_idx(batch=batch, track_idx=req.mamba_next_track_idx)

    def reserve_boundary_checkpoint(self, *, req) -> Tuple[int, torch.Tensor]:
        track_idx = req.mamba_next_track_idx
        return track_idx, req.mamba_ping_pong_track_buffer[track_idx]

    def set_request_boundary_checkpoint(
        self, *, batch, req, track_idx: int, boundary_seqlen: int
    ):
        req.mamba_last_track_seqlen = boundary_seqlen
        req.mamba_next_track_idx = self.get_other_track_idx(
            batch=batch, track_idx=track_idx
        )

    def copy_state_indices(
        self, *, batch, src_indices: torch.Tensor, dst_indices: torch.Tensor
    ):
        batch.req_to_token_pool.mamba_pool.copy_from(
            src_indices.reshape(-1), dst_indices.reshape(-1)
        )

    def copy_boundary_state_from_radix_node(
        self, *, batch, node, dst_indices: torch.Tensor
    ) -> bool:
        state_indices = self.radix_node_state_indices(node)
        if state_indices is None:
            return False
        self.copy_state_indices(
            batch=batch,
            src_indices=state_indices,
            dst_indices=dst_indices,
        )
        return True

    @staticmethod
    def radix_node_state_indices(node) -> Optional[torch.Tensor]:
        return None if node is None else getattr(node, "mamba_value", None)

    @staticmethod
    def radix_node_seqlen(node) -> int:
        seqlen = 0
        while node is not None:
            key = getattr(node, "key", None)
            if key is not None:
                seqlen += len(key)
            node = getattr(node, "parent", None)
        return seqlen

    def find_exact_radix_boundary_node(self, *, req, boundary_seqlen: int):
        node = getattr(req, "last_node", None)
        while node is not None:
            if (
                self.radix_node_state_indices(node) is not None
                and self.radix_node_seqlen(node) == boundary_seqlen
            ):
                return node
            node = getattr(node, "parent", None)
        return None

    def find_nearest_radix_state_node(self, *, req, boundary_seqlen: int):
        node = getattr(req, "last_node", None)
        while node is not None:
            node_seqlen = self.radix_node_seqlen(node)
            if (
                self.radix_node_state_indices(node) is not None
                and node_seqlen < boundary_seqlen
                and node_seqlen % self.chunk_size == 0
            ):
                return node, node_seqlen
            node = getattr(node, "parent", None)
        return None, 0

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

    def backup_recurrent_state(
        self, *, state_cache, indices: torch.Tensor
    ) -> DVRRecurrentStateBackup:
        return self._backup_recurrent_state(state_cache=state_cache, indices=indices)

    def restore_recurrent_state(
        self,
        *,
        state_cache,
        backup: DVRRecurrentStateBackup,
        indices: Optional[torch.Tensor] = None,
    ):
        self._restore_recurrent_state(
            state_cache=state_cache,
            backup=backup,
            indices=indices,
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
        # conv state for producing the draft suffix inputs.
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
            if os.environ.get("SGLANG_DVR_DEBUG_LIVE_VERIFY") == "1" or os.environ.get(
                "SGLANG_DVR_DEBUG_STEP_DECODE_VERIFY"
            ) == "1":
                state_cache.temporal[:, live_indices] = live_backup.temporal.to(
                    state_cache.temporal.dtype, copy=False
                )

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

    def cache_extend_tail_from_state_inputs(
        self,
        *,
        forward_batch,
        state_cache,
        state_inputs: DVRStateInputs,
    ):
        state_window = DVRStateInputWindow.from_cache(state_cache)
        if not state_window.enabled:
            return
        if (
            forward_batch.extend_prefix_lens_cpu is None
            or forward_batch.extend_seq_lens_cpu is None
        ):
            return

        input_tensors = state_inputs.tensors()
        assert input_tensors
        state_input_indices = forward_batch.req_pool_indices.to(
            device=input_tensors[0].device, dtype=torch.long
        )

        state_inputs.write_extend_tail(
            state_window,
            indices=state_input_indices,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            chunk_size=self.chunk_size,
        )

    def process_target_verify_conv(
        self,
        *,
        context: DVRGatedForwardContext,
        conv_input: torch.Tensor,
    ) -> torch.Tensor:
        """Run DVR draft conv and export absolute-offset conv windows."""

        assert self.is_dvr_target_verify(
            state_cache=context.state_cache, is_target_verify=context.is_target_verify
        )

        draft_token_num = context.draft_token_num
        batch_size = context.verify_batch_size
        forward_batch = context.forward_batch
        dvr_indices, valid_mask = context.padded_cache_indices()
        has_initial_states = (forward_batch.seq_lens[:batch_size] > 0).to(
            dtype=torch.bool,
            device=forward_batch.input_ids.device,
        )
        has_initial_states = has_initial_states & valid_mask.to(
            device=has_initial_states.device
        )
        conv_input_linear = conv_input
        conv_input_reshaped = conv_input_linear.view(
            batch_size, draft_token_num, -1
        ).transpose(1, 2)
        initial_conv_windows = context.conv_states[dvr_indices].clone()
        conv_output = self.ops.run_verify_conv(
            conv_input_linear.transpose(0, 1),
            context.layer.conv_weights,
            context.layer.bias,
            activation=context.layer.activation,
            conv_states=context.conv_states,
            has_initial_state=has_initial_states,
            cache_indices=dvr_indices,
            query_start_loc=context.query_start_loc,
            seq_lens_cpu=[draft_token_num] * batch_size,
        ).transpose(0, 1)[: conv_input.shape[0]]

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
            conv_input_reshaped=conv_input_reshaped,
            num_draft_tokens=draft_token_num,
        )
        return conv_output

    def process_target_verify_state(
        self,
        *,
        context: DVRGatedForwardContext,
        draft_state_inputs: DVRStateInputs,
    ) -> torch.Tensor:
        assert self.is_dvr_target_verify(
            state_cache=context.state_cache, is_target_verify=context.is_target_verify
        )

        draft_token_num = context.draft_token_num
        batch_size = context.verify_batch_size
        dvr_indices, valid_mask = context.padded_cache_indices()
        state_input_indices, _ = context.padded_state_input_indices()
        state_window = DVRStateInputWindow.from_cache(context.state_cache)
        tail_lens = state_window.tail_lens(indices=state_input_indices).to(torch.long)
        tail_lens = torch.where(
            valid_mask,
            tail_lens.clamp(min=0, max=self.chunk_size),
            torch.zeros_like(tail_lens),
        )
        global _DVR_DEBUG_VERIFY_STATE_PRINTS
        if (
            os.environ.get("SGLANG_DVR_DEBUG_GDN_STATE") == "1"
            and _DVR_DEBUG_VERIFY_STATE_PRINTS < 200
            and not torch.cuda.is_current_stream_capturing()
        ):
            print(
                "[DVR_GDN_VERIFY_STATE]",
                {
                    "draft_token_num": int(draft_token_num),
                    "batch_size": int(batch_size),
                    "cache_indices": dvr_indices.detach().cpu().tolist(),
                    "state_input_indices": state_input_indices.detach().cpu().tolist(),
                    "tail_lens": tail_lens.detach().cpu().tolist(),
                },
                flush=True,
            )
            _DVR_DEBUG_VERIFY_STATE_PRINTS += 1
        return run_dvr_chunkwise_verify(
            state_ops=self.ops,
            state_window=state_window,
            draft_state_inputs=draft_state_inputs,
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
        state_window = DVRStateInputWindow.from_cache(state_cache)
        tail_lens_before = verified_tail_lens.to(
            device=live_indices.device, dtype=torch.long
        )
        tail_lens_after = tail_lens_before + accepted_token_counts
        crosses_chunk_boundary = tail_lens_after >= self.chunk_size
        if (
            os.environ.get("SGLANG_DVR_DEBUG_COMMIT") == "1"
            and crosses_chunk_boundary.any()
            and not torch.cuda.is_current_stream_capturing()
        ):
            print(
                "[DVR_COMMIT_DEBUG]",
                {
                    "tail_lens_before": tail_lens_before.detach().cpu().tolist(),
                    "accepted_token_counts": accepted_token_counts.detach()
                    .cpu()
                    .tolist(),
                    "accepted_steps": accepted_steps.detach().cpu().tolist(),
                    "crosses_chunk_boundary": crosses_chunk_boundary.detach()
                    .cpu()
                    .tolist(),
                },
                flush=True,
            )

        self.ops.scatter_state(
            state_cache.conv[0],
            state_cache.intermediate_conv_window[0],
            live_indices,
            accepted_steps,
        )

        no_commit_step = torch.full_like(tail_lens_before, -1)
        use_oracle_boundary = (
            os.environ.get("SGLANG_DVR_TRACK_ORACLE_BOUNDARY") == "1"
        )
        crossing_req_indices = torch.nonzero(crosses_chunk_boundary).flatten()
        if crossing_req_indices.numel() > 0 and not use_oracle_boundary:
            # Build the chunk checkpoint from DVR's prefill-equivalent
            # state-input window.  The target-verify intermediate buffer is
            # row-indexed for accepted draft steps; rebuilding from the rolling
            # window keeps boundary checkpoints aligned with full prefill.
            rebuild_dvr_live_state_grouped(
                state_ops=self.ops,
                state_window=state_window,
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
        if not use_oracle_boundary:
            self.ops.scatter_state(
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
            token_count=tail_lens_after,
        )

        state_window.set_tail_lens(
            indices=state_input_indices, value=tail_lens_after.to(torch.int32)
        )
        return crosses_chunk_boundary
