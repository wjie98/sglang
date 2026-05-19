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
        if self.model_runner.hybrid_gdn_config is None:
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
        accepted_tokens: torch.Tensor,
        accepted_steps: torch.Tensor,
        accepted_tokens_cpu,
        ctx: Optional[DVRLinearStateContext] = None,
    ):
        ctx = ctx or self.state_context(batch, require_boundary=True)
        if ctx is None:
            return
        assert ctx.boundary_indices is not None
        if accepted_tokens.numel() == 0:
            return

        # EAGLE verify has already appended accepted tokens to req.output_ids.
        # Recover the pre-verify tail length from CPU request metadata instead
        # of reading the GPU state-window position tensor back to CPU.
        verified_tail_lens_cpu = [
            req.seqlen - accepted_token_num - 1 - self.boundary_seqlen[req.rid]
            for req, accepted_token_num in zip(
                batch.reqs, accepted_tokens_cpu, strict=True
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
            accepted_tokens=accepted_tokens,
            accepted_steps=accepted_steps,
        )

        for req, verified_tail_len, accepted_token_num in zip(
            batch.reqs, verified_tail_lens_cpu, accepted_tokens_cpu, strict=True
        ):
            if verified_tail_len + accepted_token_num >= FLA_CHUNK_SIZE:
                self.boundary_seqlen[req.rid] += FLA_CHUNK_SIZE
                req.mamba_last_track_seqlen = self.boundary_seqlen[req.rid]
                req.mamba_next_track_idx = self.mamba_other_track_idx(
                    batch, self.boundary_track_idx[req.rid]
                )
        self.boundary_backup = None
        self.live_backup = None

    def has_dvr_state(self, batch: ScheduleBatch) -> bool:
        return (
            self.model_runner.hybrid_gdn_config is not None
            and hasattr(batch.req_to_token_pool, "get_mamba_indices")
            and batch.batch_size() > 0
            and all(req.mamba_ping_pong_track_buffer is not None for req in batch.reqs)
        )

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
        if not self.has_dvr_state(batch):
            return None
        state_adapter = self.state_adapter()
        if state_adapter is None:
            return None
        assert self.server_args.mamba_track_interval == FLA_CHUNK_SIZE, (
            "DVR linear-state target verify must start from FLA chunk boundaries. "
            "The current prefill tracker only guarantees the latest boundary "
            "when mamba_track_interval equals FLA_CHUNK_SIZE."
        )
        live_indices = batch.req_to_token_pool.get_mamba_indices(
            batch.req_pool_indices
        ).to(torch.long)
        state_input_indices = batch.req_pool_indices.to(
            device=live_indices.device, dtype=torch.long
        )
        state_input_indices = state_input_indices + 1
        state_cache = batch.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        state_adapter.validate_state_cache(state_cache=state_cache)
        boundary_indices = None
        if require_boundary:
            boundary_indices = torch.stack(
                [
                    req.mamba_ping_pong_track_buffer[
                        self.boundary_track_idx[req.rid]
                    ]
                    for req in batch.reqs
                ]
            ).to(device=live_indices.device, dtype=torch.long)
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

    def mamba_other_track_idx(self, batch: ScheduleBatch, track_idx: int) -> int:
        return batch.req_to_token_pool.get_mamba_ping_pong_other_idx(track_idx)

    def current_prefill_checkpoint_track_idx(
        self, batch: ScheduleBatch, req
    ) -> Optional[int]:
        boundary_seqlen, _ = self.boundary_and_tail(req)
        assert boundary_seqlen % FLA_CHUNK_SIZE == 0
        last_track_seqlen = req.mamba_last_track_seqlen
        if last_track_seqlen is not None and last_track_seqlen > 0:
            assert last_track_seqlen % FLA_CHUNK_SIZE == 0, (
                "DVR linear-state verify must not reuse non-chunk-boundary checkpoints."
            )
        if boundary_seqlen <= 0:
            return None
        if last_track_seqlen != boundary_seqlen:
            return None
        return self.mamba_other_track_idx(batch, req.mamba_next_track_idx)

    def state_adapter(self):
        linear_backend = getattr(
            self.model_runner.attn_backend, "linear_attn_backend", None
        )
        if linear_backend is None:
            return None
        return getattr(linear_backend, "dvr_state_adapter", None)

    def set_boundary_checkpoint(self, batch: ScheduleBatch, req, track_idx: int):
        self.boundary_track_idx[req.rid] = track_idx
        self.boundary_seqlen[req.rid], _ = self.boundary_and_tail(req)
        req.mamba_last_track_seqlen = self.boundary_seqlen[req.rid]
        req.mamba_next_track_idx = self.mamba_other_track_idx(batch, track_idx)

    def materialize_radix_boundary_for_req(
        self, batch: ScheduleBatch, req, boundary_seqlen: int, dst: torch.Tensor
    ) -> bool:
        node = self.find_exact_radix_boundary_node(req, boundary_seqlen)
        if node is None:
            return False
        mamba_value = node.mamba_value

        batch.req_to_token_pool.mamba_pool.copy_from(
            mamba_value.reshape(-1), dst.reshape(-1)
        )
        return True

    @staticmethod
    def radix_node_seqlen(node) -> int:
        seqlen = 0
        while node is not None:
            key = getattr(node, "key", None)
            if key is not None:
                seqlen += len(key)
            node = getattr(node, "parent", None)
        return seqlen

    def find_exact_radix_boundary_node(self, req, boundary_seqlen: int):
        node = getattr(req, "last_node", None)
        while node is not None:
            if (
                getattr(node, "mamba_value", None) is not None
                and self.radix_node_seqlen(node) == boundary_seqlen
            ):
                return node
            node = getattr(node, "parent", None)
        return None

    def find_nearest_radix_state_node(self, req, boundary_seqlen: int):
        node = getattr(req, "last_node", None)
        while node is not None:
            node_seqlen = self.radix_node_seqlen(node)
            if (
                getattr(node, "mamba_value", None) is not None
                and node_seqlen < boundary_seqlen
                and node_seqlen % FLA_CHUNK_SIZE == 0
            ):
                return node, node_seqlen
            node = getattr(node, "parent", None)
        return None, 0

    def init_boundary_for_req(
        self,
        batch: ScheduleBatch,
        req,
        boundary_seqlen: int,
        live_idx: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[DVRBoundaryReplayTask]]:
        assert boundary_seqlen % FLA_CHUNK_SIZE == 0
        checkpoint_track_idx = self.current_prefill_checkpoint_track_idx(batch, req)
        if checkpoint_track_idx is not None:
            # Normal prefill already wrote the chunk-aligned state into the
            # ping-pong checkpoint buffer. Reuse that slot instead of copying
            # from the live decode slot, which may no longer hold the
            # deterministic prefill checkpoint.
            self.set_boundary_checkpoint(batch, req, checkpoint_track_idx)
            return None, None

        boundary_track_idx = req.mamba_next_track_idx
        dst = req.mamba_ping_pong_track_buffer[boundary_track_idx]
        if boundary_seqlen == 0:
            self.set_boundary_checkpoint(batch, req, boundary_track_idx)
            return dst, None
        if self.materialize_radix_boundary_for_req(
            batch, req, boundary_seqlen, dst
        ):
            self.set_boundary_checkpoint(batch, req, boundary_track_idx)
            return None, None

        source_node, source_seqlen = self.find_nearest_radix_state_node(
            req, boundary_seqlen
        )
        self.set_boundary_checkpoint(batch, req, boundary_track_idx)
        return None, DVRBoundaryReplayTask(
            req=req,
            source_seqlen=source_seqlen,
            boundary_seqlen=boundary_seqlen,
            source_state_indices=(
                source_node.mamba_value if source_node is not None else None
            ),
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
                    batch, req, boundary_seqlen, ctx.live_indices[i]
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
