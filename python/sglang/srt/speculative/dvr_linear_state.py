from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.linear.dvr_state import DVRRecurrentStateBackup
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.dvr_replay import (
    build_boundary_replay_batch,
)
from sglang.srt.speculative.dvr_server_args import is_dvr_eagle_enabled


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
        # Identifies the worker-local target boundary snapshot.  Separate
        # EAGLE/MTP draft phases can mutate shared linear-state slots before
        # the next target verify, so DVR-EAGLE preserves this snapshot when the
        # request and chunk boundary are unchanged.
        self.boundary_backup_keys = None
        self.live_backup = None
        self.suffix_replay_boundary_track_mask = None
        self.validate_args()

    def clear_cache_state(self):
        self.boundary_seqlen.clear()
        self.boundary_track_idx.clear()
        self.boundary_backup = None
        self.boundary_backup_keys = None
        self.live_backup = None
        self.suffix_replay_boundary_track_mask = None

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

    def prepare_for_draft(
        self,
        batch: ScheduleBatch,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
        use_request_seqlen: bool = False,
    ) -> List[DVRBoundaryReplayTask]:
        self.sync_active_reqs(
            batch,
            seq_lens_cpu=seq_lens_cpu,
            use_request_seqlen=use_request_seqlen,
        )
        return self.ensure_boundary_state(
            batch,
            seq_lens_cpu=seq_lens_cpu,
            use_request_seqlen=use_request_seqlen,
        )

    def replay_boundary_tasks(
        self,
        batch: ScheduleBatch,
        tasks: List[DVRBoundaryReplayTask],
        *,
        request_token_ids_for_replay: Callable,
    ):
        if not tasks:
            return

        ctx = self.state_context(batch)
        if ctx is None:
            return

        live_indices = torch.stack([task.live_idx for task in tasks]).to(
            device=ctx.live_indices.device, dtype=torch.long
        )
        live_backup = ctx.state_adapter.backup_recurrent_state(
            state_cache=ctx.state_cache,
            indices=live_indices,
        )

        try:
            zero_live_indices = [
                task.live_idx
                for task in tasks
                if task.source_state_indices is None or task.source_seqlen == 0
            ]
            if zero_live_indices:
                ctx.state_adapter.zero_recurrent_state(
                    state_cache=ctx.state_cache,
                    indices=torch.stack(zero_live_indices).to(
                        device=ctx.live_indices.device, dtype=torch.long
                    ),
                )

            replay_source_indices = [
                task.source_state_indices.reshape(-1)
                for task in tasks
                if task.source_state_indices is not None and task.source_seqlen > 0
            ]
            replay_live_indices = [
                task.live_idx
                for task in tasks
                if task.source_state_indices is not None and task.source_seqlen > 0
            ]
            if replay_source_indices:
                self.copy_state_indices(
                    batch=batch,
                    src_indices=torch.cat(replay_source_indices).to(
                        device=ctx.live_indices.device, dtype=torch.long
                    ),
                    dst_indices=torch.stack(replay_live_indices).to(
                        device=ctx.live_indices.device, dtype=torch.long
                    ),
                )
            replay_batch = build_boundary_replay_batch(
                batch=batch,
                tasks=tasks,
                boundary_indices=self.boundary_indices_for_reqs(
                    reqs=[task.req for task in tasks],
                    track_indices=[task.boundary_track_idx for task in tasks],
                    device=batch.device,
                ),
                request_token_ids_for_replay=request_token_ids_for_replay,
            )
            if replay_batch is None:
                return

            forward_batch = ForwardBatch.init_new(replay_batch, self.model_runner)
            self.model_runner.forward(forward_batch)
        finally:
            ctx.state_adapter.restore_recurrent_state(
                state_cache=ctx.state_cache,
                backup=live_backup,
                indices=live_indices,
            )

    def finish_prepare_for_draft(self, batch: ScheduleBatch):
        self.refresh_boundary_backup(batch)

    def finish_prepare_for_separate_draft(self, batch: ScheduleBatch):
        self.ensure_boundary_backup(batch)

    def restore_for_verify(
        self,
        batch: ScheduleBatch,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
    ) -> Optional[DVRLinearStateContext]:
        replay_tasks = self.ensure_boundary_state(batch, seq_lens_cpu=seq_lens_cpu)
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
        seq_lens_cpu: Optional[List[int]] = None,
        live_state_already_replayed: Optional[torch.Tensor] = None,
        use_fast_self_draft_commit: bool = False,
        publish_boundary_checkpoint: bool = True,
        return_pending_boundary: bool = False,
    ):
        pending_track_indices = [None] * len(batch.reqs)
        pending_track_seqlens = [None] * len(batch.reqs)
        ctx = ctx or self.state_context(batch, require_boundary=True)
        if ctx is None:
            return (
                (pending_track_indices, pending_track_seqlens)
                if return_pending_boundary
                else None
            )
        assert ctx.boundary_indices is not None
        if accepted_token_counts.numel() == 0:
            return (
                (pending_track_indices, pending_track_seqlens)
                if return_pending_boundary
                else None
            )

        if use_fast_self_draft_commit and seq_lens_cpu is None:
            # Self-DVR follows the original v5 lifecycle: verify has already
            # appended accepted tokens to Req, so recover the pre-verify tail
            # from request metadata. Spec-v2 overlap passes seq_lens_cpu
            # explicitly because Req materialization is delayed there, even
            # though it still uses the fast self-draft adapter commit path.
            verified_tail_lens_cpu = [
                req.seqlen - accepted_token_num - 1 - self.boundary_seqlen[req.rid]
                for req, accepted_token_num in zip(
                    batch.reqs, accepted_token_counts_cpu, strict=True
                )
            ]
        else:
            seq_lens_cpu = seq_lens_cpu or self.batch_seq_lens_cpu(batch)
            # Use the immutable pre-verify logical lengths from ScheduleBatch.
            # Request metadata can already reflect accepted tokens in some spec
            # paths, which is exactly the off-by-one/full-prefix replay dependency
            # that suffix replay must avoid.
            verified_tail_lens_cpu = [
                int(seq_len) - self.boundary_seqlen[req.rid]
                for req, seq_len in zip(batch.reqs, seq_lens_cpu, strict=True)
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
        boundary_already_tracked = self.prepare_suffix_replay_boundary_commit(
            ctx=ctx,
            verified_tail_lens_cpu=verified_tail_lens_cpu,
            accepted_token_counts_cpu=accepted_token_counts_cpu,
        )
        ctx.state_adapter.commit_after_verify(
            state_cache=ctx.state_cache,
            state_input_indices=ctx.state_input_indices,
            live_indices=ctx.live_indices,
            boundary_indices=ctx.boundary_indices,
            verified_tail_lens=verified_tail_lens,
            accepted_token_counts=accepted_token_counts,
            accepted_steps=accepted_steps,
            boundary_already_tracked=boundary_already_tracked,
            live_state_already_replayed=live_state_already_replayed,
            use_fast_self_draft_commit=use_fast_self_draft_commit,
        )

        for i, (req, verified_tail_len, accepted_token_num) in enumerate(
            zip(
                batch.reqs,
                verified_tail_lens_cpu,
                accepted_token_counts_cpu,
                strict=True,
            )
        ):
            if verified_tail_len + accepted_token_num >= FLA_CHUNK_SIZE:
                new_boundary_seqlen = self.boundary_seqlen[req.rid] + FLA_CHUNK_SIZE
                self.boundary_seqlen[req.rid] = new_boundary_seqlen
                track_idx = self.boundary_track_idx[req.rid]
                if publish_boundary_checkpoint:
                    self.publish_request_boundary_checkpoint(
                        batch=batch,
                        req=req,
                        track_idx=track_idx,
                        boundary_seqlen=new_boundary_seqlen,
                    )
                if return_pending_boundary:
                    # Overlap materializes accepted tokens after the worker
                    # returns. Keep the checkpoint pending until scheduler output
                    # processing has committed those tokens to Req.output_ids.
                    pending_track_indices[i] = track_idx
                    pending_track_seqlens[i] = new_boundary_seqlen
        self.boundary_backup = None
        self.boundary_backup_keys = None
        self.live_backup = None
        self.suffix_replay_boundary_track_mask = None
        return (
            (pending_track_indices, pending_track_seqlens)
            if return_pending_boundary
            else None
        )

    def commit_after_verify_v2(
        self,
        *,
        batch: ScheduleBatch,
        accepted_token_counts: torch.Tensor,
        accepted_steps: torch.Tensor,
        ctx: Optional[DVRLinearStateContext],
        live_state_already_replayed: Optional[torch.Tensor] = None,
        use_fast_self_draft_commit: bool = False,
    ):
        return self.commit_after_verify(
            batch=batch,
            accepted_token_counts=accepted_token_counts,
            accepted_steps=accepted_steps,
            accepted_token_counts_cpu=accepted_token_counts.cpu().tolist(),
            ctx=ctx,
            seq_lens_cpu=self.batch_seq_lens_cpu(batch),
            live_state_already_replayed=live_state_already_replayed,
            use_fast_self_draft_commit=use_fast_self_draft_commit,
            publish_boundary_checkpoint=False,
            return_pending_boundary=True,
        )

    def sync_active_reqs(
        self,
        batch: ScheduleBatch,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
        use_request_seqlen: bool = False,
    ):
        active_rids = {req.rid for req in batch.reqs}
        for rid in list(self.boundary_seqlen):
            if rid not in active_rids:
                self.boundary_seqlen.pop(rid, None)
                self.boundary_track_idx.pop(rid, None)

        seq_lens_cpu = (
            None
            if use_request_seqlen
            else (seq_lens_cpu or self.batch_seq_lens_cpu(batch))
        )
        for i, req in enumerate(batch.reqs):
            recorded_boundary = self.boundary_seqlen.get(req.rid)
            if recorded_boundary is None:
                continue
            current_boundary, _ = (
                self.boundary_and_tail(req)
                if use_request_seqlen
                else self.boundary_and_tail_for_seq_len(int(seq_lens_cpu[i]))
            )
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
            boundary_indices = self.boundary_indices_for_batch(
                batch=batch,
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

    @staticmethod
    def boundary_and_tail_for_seq_len(seq_len: int) -> Tuple[int, int]:
        boundary_seqlen = (seq_len // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
        verified_tail_len = seq_len - boundary_seqlen
        return boundary_seqlen, verified_tail_len

    @staticmethod
    def batch_seq_lens_cpu(batch: ScheduleBatch) -> List[int]:
        if batch.seq_lens_cpu is not None:
            return [int(x) for x in batch.seq_lens_cpu.tolist()]
        return [int(x) for x in batch.seq_lens.detach().cpu().tolist()]

    def current_prefill_checkpoint_track_idx(
        self,
        batch: ScheduleBatch,
        req,
        boundary_seqlen: Optional[int] = None,
    ) -> Optional[int]:
        if boundary_seqlen is None:
            boundary_seqlen, _ = self.boundary_and_tail(req)
        assert boundary_seqlen % FLA_CHUNK_SIZE == 0
        last_track_seqlen = req.mamba_last_track_seqlen
        if last_track_seqlen is not None and last_track_seqlen > 0:
            assert last_track_seqlen % FLA_CHUNK_SIZE == 0, (
                "DVR linear-state verify must not reuse non-chunk-boundary "
                "checkpoints."
            )
        if boundary_seqlen <= 0 or last_track_seqlen != boundary_seqlen:
            return None
        return self.other_track_idx(
            batch=batch, track_idx=req.mamba_next_track_idx
        )

    def state_adapter(self):
        # Scheduler constructs the DVR worker before attention backends are
        # initialized. Treat a missing backend as "not ready" and resolve the
        # adapter lazily when verify/draft state is actually used.
        attn_backend = getattr(self.model_runner, "attn_backend", None)
        linear_backend = getattr(attn_backend, "linear_attn_backend", None)
        if linear_backend is None:
            return None
        return getattr(linear_backend, "dvr_state_adapter", None)

    def set_boundary_checkpoint(
        self,
        batch: ScheduleBatch,
        req,
        track_idx: int,
        boundary_seqlen: Optional[int] = None,
    ):
        self.boundary_track_idx[req.rid] = track_idx
        if boundary_seqlen is None:
            boundary_seqlen, _ = self.boundary_and_tail(req)
        self.boundary_seqlen[req.rid] = boundary_seqlen
        self.publish_request_boundary_checkpoint(
            batch=batch, req=req, track_idx=track_idx, boundary_seqlen=boundary_seqlen
        )

    @staticmethod
    def other_track_idx(*, batch: ScheduleBatch, track_idx: int) -> int:
        return batch.req_to_token_pool.get_mamba_ping_pong_other_idx(track_idx)

    def publish_request_boundary_checkpoint(
        self, *, batch: ScheduleBatch, req, track_idx: int, boundary_seqlen: int
    ):
        req.mamba_last_track_seqlen = boundary_seqlen
        req.mamba_next_track_idx = self.other_track_idx(batch=batch, track_idx=track_idx)

    @staticmethod
    def reserve_boundary_checkpoint(*, req) -> Tuple[int, torch.Tensor]:
        track_idx = req.mamba_next_track_idx
        return track_idx, req.mamba_ping_pong_track_buffer[track_idx]

    @staticmethod
    def copy_state_indices(
        *, batch: ScheduleBatch, src_indices: torch.Tensor, dst_indices: torch.Tensor
    ):
        batch.req_to_token_pool.mamba_pool.copy_from(
            src_indices.reshape(-1), dst_indices.reshape(-1)
        )

    @staticmethod
    def boundary_indices_for_reqs(*, reqs, track_indices, device) -> torch.Tensor:
        return torch.stack(
            [
                req.mamba_ping_pong_track_buffer[track_idx]
                for req, track_idx in zip(reqs, track_indices, strict=True)
            ]
        ).to(device=device, dtype=torch.long)

    def boundary_indices_for_batch(
        self, *, batch: ScheduleBatch, device
    ) -> torch.Tensor:
        return self.boundary_indices_for_reqs(
            reqs=batch.reqs,
            track_indices=[self.boundary_track_idx[req.rid] for req in batch.reqs],
            device=device,
        )

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
                and node_seqlen % FLA_CHUNK_SIZE == 0
            ):
                return node, node_seqlen
            node = getattr(node, "parent", None)
        return None, 0

    def materialize_radix_boundary_for_req(
        self,
        batch: ScheduleBatch,
        req,
        boundary_seqlen: int,
        dst: torch.Tensor,
    ) -> bool:
        node = self.find_exact_radix_boundary_node(
            req=req, boundary_seqlen=boundary_seqlen
        )
        state_indices = self.radix_node_state_indices(node)
        if state_indices is None:
            return False
        self.copy_state_indices(
            batch=batch,
            src_indices=state_indices,
            dst_indices=dst,
        )
        return True

    def init_boundary_for_req(
        self,
        batch: ScheduleBatch,
        req,
        boundary_seqlen: int,
        live_idx: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[DVRBoundaryReplayTask]]:
        assert boundary_seqlen % FLA_CHUNK_SIZE == 0
        checkpoint_track_idx = self.current_prefill_checkpoint_track_idx(
            batch, req, boundary_seqlen
        )
        if checkpoint_track_idx is not None:
            # Normal prefill already wrote the chunk-aligned state into the
            # ping-pong checkpoint buffer. DVR verify mutates its boundary
            # slot after every accepted chunk, so copy-on-write the prefill
            # checkpoint into the request's next writable ping-pong slot before
            # registering it as the DVR boundary.
            boundary_track_idx, dst = self.reserve_boundary_checkpoint(req=req)
            src = req.mamba_ping_pong_track_buffer[checkpoint_track_idx]
            self.copy_state_indices(
                batch=batch,
                src_indices=src.unsqueeze(0),
                dst_indices=dst.unsqueeze(0),
            )
            self.set_boundary_checkpoint(
                batch,
                req,
                boundary_track_idx,
                boundary_seqlen,
            )
            return None, None
        boundary_track_idx, dst = self.reserve_boundary_checkpoint(req=req)
        if boundary_seqlen == 0:
            self.set_boundary_checkpoint(
                batch, req, boundary_track_idx, boundary_seqlen
            )
            return dst, None
        if (
            not is_dvr_eagle_enabled(self.server_args)
            and self.materialize_radix_boundary_for_req(
                batch, req, boundary_seqlen, dst
            )
        ):
            self.set_boundary_checkpoint(
                batch, req, boundary_track_idx, boundary_seqlen
            )
            return None, None

        source_node, source_seqlen = self.find_nearest_radix_state_node(
            req=req, boundary_seqlen=boundary_seqlen
        )
        self.set_boundary_checkpoint(
            batch, req, boundary_track_idx, boundary_seqlen
        )
        return None, DVRBoundaryReplayTask(
            req=req,
            source_seqlen=source_seqlen,
            boundary_seqlen=boundary_seqlen,
            source_state_indices=self.radix_node_state_indices(source_node),
            boundary_track_idx=boundary_track_idx,
            live_idx=live_idx,
        )

    def ensure_boundary_state(
        self,
        batch: ScheduleBatch,
        ctx: Optional[DVRLinearStateContext] = None,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
        use_request_seqlen: bool = False,
    ) -> List[DVRBoundaryReplayTask]:
        ctx = ctx or self.state_context(batch)
        if ctx is None:
            return []
        replay_tasks = []
        zero_boundary_indices = []
        reset_pos_indices = []
        reset_pos_values = []
        seq_lens_cpu = (
            None
            if use_request_seqlen
            else (seq_lens_cpu or self.batch_seq_lens_cpu(batch))
        )
        for i, req in enumerate(batch.reqs):
            if req.rid not in self.boundary_seqlen:
                boundary_seqlen, verified_tail_len = (
                    self.boundary_and_tail(req)
                    if use_request_seqlen
                    else self.boundary_and_tail_for_seq_len(int(seq_lens_cpu[i]))
                )
                reset_pos_indices.append(ctx.state_input_indices[i])
                reset_pos_values.append(verified_tail_len)
                zero_boundary_idx, replay_task = self.init_boundary_for_req(
                    batch,
                    req,
                    boundary_seqlen,
                    ctx.live_indices[i],
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
        self,
        batch: ScheduleBatch,
        tasks: List[DVRBoundaryReplayTask],
        *,
        seq_lens_cpu: Optional[List[int]] = None,
        use_request_seqlen: bool = False,
    ):
        if not tasks:
            return
        ctx = self.state_context(batch)
        if ctx is None:
            return
        task_rids = {task.req.rid for task in tasks}
        seq_lens_by_rid = None
        if not use_request_seqlen:
            seq_lens_cpu = seq_lens_cpu or self.batch_seq_lens_cpu(batch)
            seq_lens_by_rid = {
                req.rid: int(seq_len)
                for req, seq_len in zip(batch.reqs, seq_lens_cpu, strict=True)
            }
        state_input_indices = []
        tail_lens = []
        for i, req in enumerate(batch.reqs):
            if req.rid not in task_rids:
                continue
            if use_request_seqlen:
                _, verified_tail_len = self.boundary_and_tail(req)
            else:
                seq_len = seq_lens_by_rid[req.rid]
                boundary = self.boundary_seqlen[req.rid]
                verified_tail_len = seq_len - boundary
            state_input_indices.append(ctx.state_input_indices[i])
            tail_lens.append(verified_tail_len)
        if not state_input_indices:
            return
        ctx.state_adapter.set_state_input_tail_lens(
            state_cache=ctx.state_cache,
            state_input_indices=torch.stack(state_input_indices),
            tail_lens=torch.tensor(tail_lens, device=ctx.live_indices.device),
        )

    def refresh_boundary_backup(self, batch: ScheduleBatch):
        self._backup_boundary_state(batch, preserve_existing=False)

    def ensure_boundary_backup(self, batch: ScheduleBatch):
        self._backup_boundary_state(batch, preserve_existing=True)

    def _backup_boundary_state(self, batch: ScheduleBatch, *, preserve_existing: bool):
        ctx = self.state_context(batch, require_boundary=True)
        if ctx is None:
            self.boundary_backup = None
            self.boundary_backup_keys = None
            self.live_backup = None
            return
        assert ctx.boundary_indices is not None
        backup_keys = [
            (req.rid, int(self.boundary_seqlen.get(req.rid, -1)))
            for req in batch.reqs
        ]
        if (
            preserve_existing
            and self.boundary_backup is not None
            and self.boundary_backup_keys == backup_keys
        ):
            return
        self.boundary_backup, self.live_backup = (
            ctx.state_adapter.backup_verify_recurrent_states(
                state_cache=ctx.state_cache,
                boundary_indices=ctx.boundary_indices,
                live_indices=ctx.live_indices,
            )
        )
        self.boundary_backup_keys = backup_keys

    def boundary_lens_for_replay(self, batch: ScheduleBatch, seq_lens_cpu) -> List[int]:
        boundary_lens = []
        for req, seq_len in zip(batch.reqs, seq_lens_cpu, strict=True):
            boundary = self.boundary_seqlen.get(req.rid)
            if boundary is None:
                raise RuntimeError(
                    "DVR suffix replay requires a prepared chunk-boundary "
                    f"checkpoint: rid={req.rid}, seq_len={int(seq_len)}."
                )
            if boundary > int(seq_len):
                raise RuntimeError(
                    "DVR suffix replay boundary is ahead of the batch prefix: "
                    f"rid={req.rid}, boundary={boundary}, seq_len={int(seq_len)}."
                )
            boundary_lens.append(int(boundary))
        return boundary_lens

    def restore_boundary_state_for_suffix_replay(self, ctx: DVRLinearStateContext):
        """Make live recurrent slots start exactly at DVR chunk boundaries.

        Target verify uses the boundary SSM state together with the draft-start
        conv state.  A deterministic suffix EXTEND oracle instead replays the
        unclosed prefill tail, so both temporal and conv state must come from the
        chunk-boundary checkpoint before the EXTEND forward runs.
        """

        assert ctx.boundary_indices is not None
        boundary_backup = self.boundary_backup
        if boundary_backup is None:
            boundary_backup = ctx.state_adapter.backup_recurrent_state(
                state_cache=ctx.state_cache,
                indices=ctx.boundary_indices,
            )
        ctx.state_adapter.restore_recurrent_state(
            state_cache=ctx.state_cache,
            backup=boundary_backup,
            indices=ctx.live_indices,
        )

    def set_suffix_replay_boundary_track_mask(
        self, mask: Optional[torch.Tensor]
    ) -> None:
        self.suffix_replay_boundary_track_mask = None if mask is None else mask.detach()

    @staticmethod
    def _select_recurrent_backup(
        backup: DVRRecurrentStateBackup, indices: torch.Tensor
    ) -> DVRRecurrentStateBackup:
        data_indices = indices.to(device=backup.temporal.device, dtype=torch.long)
        meta_indices = indices.to(device=backup.indices.device, dtype=torch.long)
        return DVRRecurrentStateBackup(
            conv=tuple(conv[:, data_indices].clone() for conv in backup.conv),
            temporal=backup.temporal[:, data_indices].clone(),
            indices=backup.indices[meta_indices].clone(),
        )

    def restore_boundary_backup_for_mask(
        self, ctx: DVRLinearStateContext, mask: torch.Tensor
    ) -> None:
        if self.boundary_backup is None:
            return
        selected_cpu = torch.nonzero(mask.detach().cpu(), as_tuple=True)[0]
        if selected_cpu.numel() == 0:
            return
        backup = self._select_recurrent_backup(self.boundary_backup, selected_cpu)
        selected = selected_cpu.to(device=ctx.boundary_indices.device, dtype=torch.long)
        ctx.state_adapter.restore_recurrent_state(
            state_cache=ctx.state_cache,
            backup=backup,
            indices=ctx.boundary_indices[selected],
        )

    def prepare_suffix_replay_boundary_commit(
        self,
        *,
        ctx: DVRLinearStateContext,
        verified_tail_lens_cpu,
        accepted_token_counts_cpu,
    ) -> Optional[torch.Tensor]:
        tracked = self.suffix_replay_boundary_track_mask
        if tracked is None:
            return None
        tracked = tracked.to(device=ctx.live_indices.device, dtype=torch.bool)
        # Suffix EXTEND replay must decide whether to write the next boundary
        # before speculative sampling knows how many draft tokens will commit.
        # Undo replay-written checkpoints for requests that do not actually
        # cross the boundary; for real crossings, tell commit_after_verify not
        # to rebuild the checkpoint from the target-verify intermediate buffer.
        crosses = torch.tensor(
            [
                int(tail_len) + int(accepted_count) >= FLA_CHUNK_SIZE
                for tail_len, accepted_count in zip(
                    verified_tail_lens_cpu, accepted_token_counts_cpu, strict=True
                )
            ],
            dtype=torch.bool,
            device=ctx.live_indices.device,
        )
        self.restore_boundary_backup_for_mask(ctx, tracked & ~crosses)
        boundary_already_tracked = tracked & crosses
        return boundary_already_tracked if boundary_already_tracked.any() else None
