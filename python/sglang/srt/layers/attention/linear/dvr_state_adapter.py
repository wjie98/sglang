"""Adapter boundary between model backends and DVR linear-state lifecycle.

External callers:
- gdn_backend enters process_target_verify_* during TARGET_VERIFY.
- dvr_linear_state uses backup/restore/commit methods around draft and verify.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.linear.dvr_gdn_state import DVRGDNStateInputs
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
        indices = indices + 1
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
    is_draft_worker: bool = False

    @classmethod
    def for_gdn(
        cls, kernel_dispatcher, *, is_draft_worker: bool = False
    ) -> "DVRGatedStateAdapter":
        from sglang.srt.layers.attention.linear.dvr_gdn_state import DVRGDNStateOps

        return cls(
            DVRGDNStateOps.create(kernel_dispatcher),
            is_draft_worker=is_draft_worker,
        )

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
        return batch.req_pool_indices.to(device=device, dtype=torch.long) + 1

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

    def backup_state_input_window(
        self, *, state_cache, state_input_indices: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        state_window = DVRStateInputWindow.from_cache(state_cache)
        return state_window.backup_rows(indices=state_input_indices)

    def restore_state_input_window(
        self,
        *,
        state_cache,
        state_input_indices: torch.Tensor,
        backup: Optional[Tuple[torch.Tensor, ...]],
    ):
        state_window = DVRStateInputWindow.from_cache(state_cache)
        state_window.restore_rows(indices=state_input_indices, backup=backup)

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

    def zero_state_input_after_lens(
        self,
        *,
        state_cache,
        state_input_indices: torch.Tensor,
        keep_lens: torch.Tensor,
    ):
        state_window = DVRStateInputWindow.from_cache(state_cache)
        if not state_window.enabled:
            return
        state_window.zero_after_lens(indices=state_input_indices, keep_lens=keep_lens)

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
        if self.is_draft_worker:
            # DVR state-input windows are target-model prefill oracles.  EAGLE
            # and MTP draft workers may share request slots with the target
            # worker, so draft-model state inputs must never overwrite them.
            return
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
        state_input_indices = state_input_indices + 1
        state_inputs.write_extend_tail(
            state_window,
            indices=state_input_indices,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            chunk_size=self.chunk_size,
        )

    def cache_gdn_extend_tail(
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
        self.cache_extend_tail_from_state_inputs(
            forward_batch=forward_batch,
            state_cache=state_cache,
            state_inputs=DVRGDNStateInputs.from_extend_forward(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
            ),
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
    ) -> torch.Tensor:
        """Run GDN target verify using DVR's prefill-equivalent state replay."""

        context = self.make_forward_context(
            layer=layer,
            forward_batch=forward_batch,
            state_cache=state_cache,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            conv_states=state_cache.conv[0],
            ssm_states=state_cache.temporal,
            seq_len=mixed_qkv.shape[0],
        )
        mixed_qkv = self.process_target_verify_conv(
            context=context,
            conv_input=mixed_qkv,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        draft_state_inputs = DVRGDNStateInputs.from_draft_rows(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            batch_size=context.verify_batch_size,
            draft_token_num=context.draft_token_num,
            num_q_heads=layer.num_q_heads,
            head_q_dim=layer.head_q_dim,
            num_k_heads=layer.num_k_heads,
            head_k_dim=layer.head_k_dim,
            num_v_heads=layer.num_v_heads,
            head_v_dim=layer.head_v_dim,
        )
        return self.process_target_verify_state(
            context=context,
            draft_state_inputs=draft_state_inputs,
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
        boundary_already_tracked: Optional[torch.Tensor] = None,
        live_state_already_replayed: Optional[torch.Tensor] = None,
        use_fast_self_draft_commit: bool = False,
    ) -> torch.Tensor:
        state_window = DVRStateInputWindow.from_cache(state_cache)
        tail_lens_before = verified_tail_lens.to(
            device=live_indices.device, dtype=torch.long
        )
        tail_lens_after = tail_lens_before + accepted_token_counts
        crosses_chunk_boundary = tail_lens_after >= self.chunk_size
        if use_fast_self_draft_commit:
            # Self-DVR uses the target model as its own draft model, so the
            # target-verify intermediate state is already the v5 hot-path state
            # for the accepted chain.  EAGLE keeps the exact replay path below
            # because rejected rows may commit target-generated tokens that did
            # not come from the draft model.
            self.ops.scatter_state(
                state_cache.conv[0],
                state_cache.intermediate_conv_window[0],
                live_indices,
                accepted_steps,
            )

            boundary_state_step = (
                0 if state_cache.intermediate_ssm.shape[2] == 1 else self.chunk_size - 1
            )
            no_commit_step = torch.full_like(tail_lens_before, -1)
            commit_step = torch.where(
                crosses_chunk_boundary,
                torch.full_like(tail_lens_before, boundary_state_step),
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

        if boundary_already_tracked is None:
            boundary_already_tracked = torch.zeros_like(crosses_chunk_boundary)
        else:
            boundary_already_tracked = boundary_already_tracked.to(
                device=live_indices.device, dtype=torch.bool
            )
        if live_state_already_replayed is None:
            live_state_already_replayed = torch.zeros_like(crosses_chunk_boundary)
        else:
            live_state_already_replayed = live_state_already_replayed.to(
                device=live_indices.device, dtype=torch.bool
            )
        # Suffix replay can pre-materialize the next boundary through the
        # backend's normal EXTEND tracking path.  Do not rebuild or scatter that
        # same boundary from TARGET_VERIFY intermediates, whose rows are indexed
        # by accepted draft steps rather than prefill positions.
        boundary_needs_rebuild = crosses_chunk_boundary & ~boundary_already_tracked

        has_live_conv_commit = (accepted_token_counts > 0) & ~live_state_already_replayed
        live_conv_req_indices = torch.nonzero(has_live_conv_commit).flatten()
        if live_conv_req_indices.numel() > 0:
            self.ops.scatter_state(
                state_cache.conv[0],
                state_cache.intermediate_conv_window[0],
                live_indices[live_conv_req_indices],
                accepted_steps[live_conv_req_indices],
            )

        no_commit_step = torch.full_like(tail_lens_before, -1)
        crossing_req_indices = torch.nonzero(boundary_needs_rebuild).flatten()
        if crossing_req_indices.numel() > 0:
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
                use_chunkwise_rebuild=True,
            )
        self.ops.scatter_state(
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
        # When suffix EXTEND replay already tracked the new chunk boundary, it
        # also wrote the post-boundary tail into columns starting at zero.  Do
        # not shift the old physical window over those freshly replayed rows.
        shift_window_mask = crosses_chunk_boundary & ~boundary_already_tracked
        state_window.shift_after_boundary(
            indices=state_input_indices,
            crosses_chunk_boundary=shift_window_mask,
            chunk_size=self.chunk_size,
        )
        state_window.zero_after_lens(
            indices=state_input_indices,
            keep_lens=tail_lens_after,
        )
        req_indices = torch.arange(
            live_indices.shape[0],
            dtype=torch.long,
            device=live_indices.device,
        )
        draft_token_num = state_cache.intermediate_conv_window[0].shape[2]
        partial_accept = accepted_token_counts < draft_token_num
        needs_live_rebuild = ~live_state_already_replayed
        full_accept_req_indices = req_indices[(~partial_accept) & needs_live_rebuild]
        full_accept_crossing_req_indices = full_accept_req_indices[
            crosses_chunk_boundary[full_accept_req_indices]
        ]
        if full_accept_crossing_req_indices.numel() > 0:
            rebuild_dvr_live_state_grouped(
                state_ops=self.ops,
                state_window=state_window,
                temporal_state=state_cache.temporal,
                state_input_indices=state_input_indices,
                live_indices=live_indices,
                boundary_indices=boundary_indices,
                req_indices=full_accept_crossing_req_indices,
                token_count=tail_lens_after[full_accept_crossing_req_indices],
                use_chunkwise_rebuild=True,
            )
        full_accept_fast_req_indices = full_accept_req_indices[
            ~crosses_chunk_boundary[full_accept_req_indices]
        ]
        if full_accept_fast_req_indices.numel() > 0:
            rebuild_dvr_live_state_grouped(
                state_ops=self.ops,
                state_window=state_window,
                temporal_state=state_cache.temporal,
                state_input_indices=state_input_indices,
                live_indices=live_indices,
                boundary_indices=boundary_indices,
                req_indices=full_accept_fast_req_indices,
                token_count=tail_lens_after[full_accept_fast_req_indices],
            )
        partial_accept_req_indices = req_indices[partial_accept & needs_live_rebuild]
        if partial_accept_req_indices.numel() > 0:
            # Partial accept is the only time DVR must start the next draft from
            # a shortened suffix.  Use the same chunkwise GDN math as prefill
            # here; the all-accepted hot path keeps the faster recurrent rebuild.
            rebuild_dvr_live_state_grouped(
                state_ops=self.ops,
                state_window=state_window,
                temporal_state=state_cache.temporal,
                state_input_indices=state_input_indices,
                live_indices=live_indices,
                boundary_indices=boundary_indices,
                req_indices=partial_accept_req_indices,
                token_count=tail_lens_after[partial_accept_req_indices],
                use_chunkwise_rebuild=True,
            )

        state_window.set_tail_lens(
            indices=state_input_indices, value=tail_lens_after.to(torch.int32)
        )
        return crosses_chunk_boundary
