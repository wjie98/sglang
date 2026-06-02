from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclass
class DVRLinearStateContext:
    state_cache: Any
    state_adapter: Any
    state_input_indices: torch.Tensor
    live_indices: torch.Tensor
    boundary_indices: Optional[torch.Tensor] = None


@dataclass
class DVRBoundaryReplayTask:
    req: Any
    source_seqlen: int
    boundary_seqlen: int
    source_state_indices: Optional[torch.Tensor]
    boundary_track_idx: int
    live_idx: torch.Tensor


class DVRLinearStateLifecycle:
    """Manage chunk-boundary state for DVR linear-state layers.

    The current implementation is backed by SGLang's linear-state cache and
    ping-pong prefill checkpoints. Keeping it outside `dvr_worker.py` prevents
    the speculative control flow from depending on those backend details.
    """

    def __init__(self, *, server_args, model_runner):
        self.server_args = server_args
        self.model_runner = model_runner
        self.boundary_seqlen = {}
        self.boundary_track_idx = {}
        self.boundary_backup = None
        self.live_backup = None
        self.validate_args()

    def validate_args(self):
        if self.state_adapter() is None:
            return
        if self.server_args.mamba_track_interval != FLA_CHUNK_SIZE:
            raise ValueError(
                "DVR linear-state verify requires mamba_track_interval to match "
                f"FLA_CHUNK_SIZE={FLA_CHUNK_SIZE}, got "
                f"{self.server_args.mamba_track_interval}. Multiples larger than "
                "FLA_CHUNK_SIZE can miss the latest chunk boundary from the "
                "first prefill because the current extra_buffer path stores "
                "only one tracked prefill checkpoint."
            )
        if self.server_args.mamba_ssm_dtype != "float32":
            raise ValueError(
                "DVR linear-state verify requires fp32 recurrent state storage."
            )

    def prepare_for_draft(self, batch: ScheduleBatch) -> List[DVRBoundaryReplayTask]:
        self.sync_active_reqs(batch)
        return self.ensure_boundary_state(batch)

    def finish_prepare_for_draft(self, batch: ScheduleBatch):
        self.backup_boundary_state(batch)

    def restore_for_verify(
        self, batch: ScheduleBatch
    ) -> Optional[DVRLinearStateContext]:
        replay_tasks = self.ensure_boundary_state(batch)
        if replay_tasks:
            raise RuntimeError(
                "DVR boundary replay tasks must be materialized before target verify."
            )
        ctx = self.state_context(batch, require_boundary=True)
        if ctx is None:
            return None
        assert ctx.boundary_indices is not None
        ctx.state_adapter.prepare_recurrent_state_for_verify(
            state_cache=ctx.state_cache,
            live_indices=ctx.live_indices,
            boundary_indices=ctx.boundary_indices,
            boundary_backup=self.boundary_backup,
            live_backup=self.live_backup,
        )
        return ctx

    def commit_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        accepted_token_counts: torch.Tensor,
        accepted_steps: torch.Tensor,
        accepted_token_counts_cpu,
        ctx: Optional[DVRLinearStateContext] = None,
    ):
        ctx = ctx or self.state_context(batch, require_boundary=True)
        if ctx is None:
            return
        assert ctx.boundary_indices is not None
        if accepted_token_counts.numel() == 0:
            return

        # EAGLE verify has already appended accepted tokens to req.output_ids.
        # Recover the pre-verify tail length from CPU request metadata instead
        # of reading the GPU state-window position tensor back to CPU.
        verified_tail_lens_cpu = [
            req.seqlen - accepted_token_num - 1 - self.boundary_seqlen[req.rid]
            for req, accepted_token_num in zip(
                batch.reqs, accepted_token_counts_cpu, strict=True
            )
        ]
        verified_tail_lens = ctx.state_adapter.state_input_tail_lens(
            state_cache=ctx.state_cache,
            state_input_indices=ctx.state_input_indices,
        )
        if verified_tail_lens is None:
            verified_tail_lens = torch.tensor(
                verified_tail_lens_cpu,
                dtype=torch.long,
                device=batch.seq_lens.device,
            )
        verified_tail_lens = verified_tail_lens.to(
            device=ctx.live_indices.device, dtype=torch.long
        )
        ctx.state_adapter.commit_after_verify(
            state_cache=ctx.state_cache,
            state_input_indices=ctx.state_input_indices,
            live_indices=ctx.live_indices,
            boundary_indices=ctx.boundary_indices,
            verified_tail_lens=verified_tail_lens,
            accepted_token_counts=accepted_token_counts,
            accepted_steps=accepted_steps,
        )

        for req, verified_tail_len, accepted_token_num in zip(
            batch.reqs, verified_tail_lens_cpu, accepted_token_counts_cpu, strict=True
        ):
            if verified_tail_len + accepted_token_num >= FLA_CHUNK_SIZE:
                new_boundary_seqlen = self.boundary_seqlen[req.rid] + FLA_CHUNK_SIZE
                self.boundary_seqlen[req.rid] = new_boundary_seqlen
                ctx.state_adapter.set_request_boundary_checkpoint(
                    batch=batch,
                    req=req,
                    track_idx=self.boundary_track_idx[req.rid],
                    boundary_seqlen=new_boundary_seqlen,
                )
        self.boundary_backup = None
        self.live_backup = None

    def has_dvr_state(self, batch: ScheduleBatch) -> bool:
        state_adapter = self.state_adapter()
        return state_adapter is not None and state_adapter.has_dvr_state(batch=batch)

    def sync_active_reqs(self, batch: ScheduleBatch):
        active_rids = {req.rid for req in batch.reqs}
        for rid in list(self.boundary_seqlen):
            if rid not in active_rids:
                self.boundary_seqlen.pop(rid, None)
                self.boundary_track_idx.pop(rid, None)

        for req in batch.reqs:
            recorded_boundary = self.boundary_seqlen.get(req.rid)
            if recorded_boundary is None:
                continue
            current_boundary, _ = self.boundary_and_tail(req)
            if (
                req.rid not in self.boundary_track_idx
                or recorded_boundary != current_boundary
                or recorded_boundary % FLA_CHUNK_SIZE != 0
            ):
                self.boundary_seqlen.pop(req.rid, None)
                self.boundary_track_idx.pop(req.rid, None)

    def state_context(
        self, batch: ScheduleBatch, require_boundary: bool = False
    ) -> Optional[DVRLinearStateContext]:
        state_adapter = self.state_adapter()
        if state_adapter is None or not state_adapter.has_dvr_state(batch=batch):
            return None
        assert self.server_args.mamba_track_interval == FLA_CHUNK_SIZE, (
            "DVR linear-state target verify must start from FLA chunk boundaries. "
            "The current prefill tracker only guarantees the latest boundary "
            "when mamba_track_interval equals FLA_CHUNK_SIZE."
        )
        live_indices = state_adapter.get_live_indices(batch=batch)
        state_input_indices = state_adapter.get_state_input_indices(
            batch=batch, device=live_indices.device
        )
        state_cache = state_adapter.get_state_cache(batch=batch)
        state_adapter.validate_state_cache(state_cache=state_cache)
        boundary_indices = None
        if require_boundary:
            boundary_indices = state_adapter.get_boundary_indices(
                batch=batch,
                boundary_track_idx_by_rid=self.boundary_track_idx,
                device=live_indices.device,
            )
        return DVRLinearStateContext(
            state_cache=state_cache,
            state_adapter=state_adapter,
            state_input_indices=state_input_indices,
            live_indices=live_indices,
            boundary_indices=boundary_indices,
        )

    @staticmethod
    def boundary_and_tail(req) -> Tuple[int, int]:
        boundary_seqlen = ((req.seqlen - 1) // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
        verified_tail_len = (req.seqlen - 1) - boundary_seqlen
        return boundary_seqlen, verified_tail_len

    def current_prefill_checkpoint_track_idx(
        self, batch: ScheduleBatch, req, state_adapter
    ) -> Optional[int]:
        boundary_seqlen, _ = self.boundary_and_tail(req)
        assert boundary_seqlen % FLA_CHUNK_SIZE == 0
        return state_adapter.get_current_prefill_checkpoint_track_idx(
            batch=batch, req=req, boundary_seqlen=boundary_seqlen
        )

    def state_adapter(self):
        linear_backend = getattr(
            self.model_runner.attn_backend, "linear_attn_backend", None
        )
        if linear_backend is None:
            return None
        return getattr(linear_backend, "dvr_state_adapter", None)

    def set_boundary_checkpoint(
        self,
        batch: ScheduleBatch,
        req,
        track_idx: int,
        state_adapter,
        boundary_seqlen: Optional[int] = None,
    ):
        self.boundary_track_idx[req.rid] = track_idx
        if boundary_seqlen is None:
            boundary_seqlen, _ = self.boundary_and_tail(req)
        self.boundary_seqlen[req.rid] = boundary_seqlen
        state_adapter.set_request_boundary_checkpoint(
            batch=batch,
            req=req,
            track_idx=track_idx,
            boundary_seqlen=boundary_seqlen,
        )

    def materialize_radix_boundary_for_req(
        self,
        batch: ScheduleBatch,
        req,
        boundary_seqlen: int,
        dst: torch.Tensor,
        state_adapter,
    ) -> bool:
        node = state_adapter.find_exact_radix_boundary_node(
            req=req, boundary_seqlen=boundary_seqlen
        )
        return state_adapter.copy_boundary_state_from_radix_node(
            batch=batch, node=node, dst_indices=dst
        )

    def init_boundary_for_req(
        self,
        batch: ScheduleBatch,
        req,
        boundary_seqlen: int,
        live_idx: torch.Tensor,
        state_adapter,
    ) -> Tuple[Optional[torch.Tensor], Optional[DVRBoundaryReplayTask]]:
        assert boundary_seqlen % FLA_CHUNK_SIZE == 0
        checkpoint_track_idx = self.current_prefill_checkpoint_track_idx(
            batch, req, state_adapter
        )
        if checkpoint_track_idx is not None:
            # Normal prefill already wrote the chunk-aligned state into the
            # ping-pong checkpoint buffer. DVR verify mutates its boundary
            # slot after every accepted chunk, so copy-on-write the prefill
            # checkpoint into the request's next writable ping-pong slot before
            # registering it as the DVR boundary.
            boundary_track_idx, dst = state_adapter.reserve_boundary_checkpoint(
                req=req
            )
            src = req.mamba_ping_pong_track_buffer[checkpoint_track_idx]
            state_adapter.copy_state_indices(
                batch=batch,
                src_indices=src.unsqueeze(0),
                dst_indices=dst.unsqueeze(0),
            )
            self.set_boundary_checkpoint(
                batch,
                req,
                boundary_track_idx,
                state_adapter,
                boundary_seqlen,
            )
            return None, None

        boundary_track_idx, dst = state_adapter.reserve_boundary_checkpoint(req=req)
        if boundary_seqlen == 0:
            self.set_boundary_checkpoint(
                batch, req, boundary_track_idx, state_adapter, boundary_seqlen
            )
            return dst, None
        if self.materialize_radix_boundary_for_req(
            batch, req, boundary_seqlen, dst, state_adapter
        ):
            self.set_boundary_checkpoint(
                batch, req, boundary_track_idx, state_adapter, boundary_seqlen
            )
            return None, None

        source_node, source_seqlen = state_adapter.find_nearest_radix_state_node(
            req=req, boundary_seqlen=boundary_seqlen
        )
        self.set_boundary_checkpoint(
            batch, req, boundary_track_idx, state_adapter, boundary_seqlen
        )
        return None, DVRBoundaryReplayTask(
            req=req,
            source_seqlen=source_seqlen,
            boundary_seqlen=boundary_seqlen,
            source_state_indices=state_adapter.radix_node_state_indices(source_node),
            boundary_track_idx=boundary_track_idx,
            live_idx=live_idx,
        )

    def ensure_boundary_state(
        self, batch: ScheduleBatch, ctx: Optional[DVRLinearStateContext] = None
    ) -> List[DVRBoundaryReplayTask]:
        ctx = ctx or self.state_context(batch)
        if ctx is None:
            return []
        replay_tasks = []
        zero_boundary_indices = []
        reset_pos_indices = []
        reset_pos_values = []
        for i, req in enumerate(batch.reqs):
            if req.rid not in self.boundary_seqlen:
                boundary_seqlen, verified_tail_len = self.boundary_and_tail(req)
                reset_pos_indices.append(ctx.state_input_indices[i])
                reset_pos_values.append(verified_tail_len)
                zero_boundary_idx, replay_task = self.init_boundary_for_req(
                    batch,
                    req,
                    boundary_seqlen,
                    ctx.live_indices[i],
                    ctx.state_adapter,
                )
                if zero_boundary_idx is not None:
                    zero_boundary_indices.append(zero_boundary_idx)
                if replay_task is not None:
                    replay_tasks.append(replay_task)
        if zero_boundary_indices:
            boundary_indices_to_zero = torch.stack(zero_boundary_indices).to(
                device=ctx.live_indices.device, dtype=torch.long
            )
            ctx.state_adapter.zero_recurrent_state(
                state_cache=ctx.state_cache, indices=boundary_indices_to_zero
            )
        if reset_pos_indices:
            ctx.state_adapter.set_state_input_tail_lens(
                state_cache=ctx.state_cache,
                state_input_indices=torch.stack(reset_pos_indices),
                tail_lens=torch.tensor(reset_pos_values, device=ctx.live_indices.device),
            )
        return replay_tasks

    def restore_tail_lens_after_replay(
        self, batch: ScheduleBatch, tasks: List[DVRBoundaryReplayTask]
    ):
        if not tasks:
            return
        ctx = self.state_context(batch)
        if ctx is None:
            return
        task_rids = {task.req.rid for task in tasks}
        state_input_indices = []
        tail_lens = []
        for i, req in enumerate(batch.reqs):
            if req.rid not in task_rids:
                continue
            _, verified_tail_len = self.boundary_and_tail(req)
            state_input_indices.append(ctx.state_input_indices[i])
            tail_lens.append(verified_tail_len)
        if not state_input_indices:
            return
        ctx.state_adapter.set_state_input_tail_lens(
            state_cache=ctx.state_cache,
            state_input_indices=torch.stack(state_input_indices),
            tail_lens=torch.tensor(tail_lens, device=ctx.live_indices.device),
        )

    def backup_boundary_state(self, batch: ScheduleBatch):
        ctx = self.state_context(batch, require_boundary=True)
        if ctx is None:
            self.boundary_backup = None
            self.live_backup = None
            return
        assert ctx.boundary_indices is not None
        self.boundary_backup, self.live_backup = (
            ctx.state_adapter.backup_verify_recurrent_states(
                state_cache=ctx.state_cache,
                boundary_indices=ctx.boundary_indices,
                live_indices=ctx.live_indices,
            )
        )
