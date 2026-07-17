from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclass
class DVRLinearStateContext:
    state_cache: Any
    state_input_indices: torch.Tensor
    live_indices: torch.Tensor
    boundary_indices: Optional[torch.Tensor] = None


@dataclass
class _DVRBoundaryCheckpoint:
    """Logical owner of one request's physical ping-pong boundary slot."""

    request: Any = field(repr=False)
    retraction_count: int
    track_idx: int
    seq_len: int


@dataclass
class DVRRollbackActions:
    """Mark a decode result whose Mamba checkpoint is owned by DVR."""

    linear_state: Any = field(repr=False)

    def commit_checkpoint_after_decode(self, *, req, batch) -> bool:
        # The verify kernel has already updated the physical boundary slot. The
        # CPU result only publishes its materialized length and prevents generic
        # speculative tracking from rotating that DVR-owned slot.
        self.linear_state.publish_checkpoint(req, batch.req_to_token_pool)
        return True


class DVRLinearStateLifecycle:
    """Own DVR's target boundary and self-draft rollback state.

    The recurrent boundary itself stays in SGLang's existing Mamba state pools:
    an aligned live state is preferred, otherwise the latest ping-pong state is
    reused. A radix miss is rebuilt by the ordinary match-prefix + EXTEND path;
    DVR does not maintain a second temporal checkpoint store.
    """

    def __init__(self, *, server_args, model_runner):
        self.server_args = server_args
        self.model_runner = model_runner
        self._state_adapter = None
        self.boundaries: dict[int, _DVRBoundaryCheckpoint] = {}
        self.draft_state_backup = None

    def bind_state_adapter(self, state_adapter) -> None:
        self._state_adapter = state_adapter
        if state_adapter is None:
            if getattr(self.model_runner, "mambaish_config", None) is not None:
                raise RuntimeError(
                    "DVR does not support this hybrid linear-state backend: no "
                    "target state adapter was initialized."
                )
            return

        if self.server_args.mamba_track_interval != state_adapter.chunk_size:
            raise ValueError(
                "DVR linear-state verify requires mamba_track_interval to match "
                f"the adapter chunk size {state_adapter.chunk_size}, got "
                f"{self.server_args.mamba_track_interval}."
            )
        if state_adapter.draft_reuses_target_state:
            state_cache = self.model_runner.req_to_token_pool.get_speculative_mamba2_params_all_layers()
            # Self draft mutates only the target's live convolution state before
            # verify. Keep one request-pool-indexed copy; recurrent boundaries
            # remain in the existing ping-pong pool.
            self.draft_state_backup = state_adapter.allocate_draft_state_backup(
                state_cache=state_cache,
                backup_size=int(
                    self.model_runner.req_to_token_pool.req_to_token.shape[0]
                ),
            )

    @property
    def chunk_size(self) -> int:
        if self._state_adapter is None:
            raise RuntimeError("DVR linear-state adapter is not initialized.")
        return self._state_adapter.chunk_size

    def clear_cache_state(self) -> None:
        self.boundaries.clear()

    @staticmethod
    def _request_retraction_count(req) -> int:
        return int(getattr(req, "retraction_count", 0))

    def _drop_checkpoint(self, req) -> None:
        if req.req_pool_idx is not None:
            self.boundaries.pop(int(req.req_pool_idx), None)

    def _checkpoint(self, req) -> Optional[_DVRBoundaryCheckpoint]:
        if req.req_pool_idx is None:
            return None
        slot = int(req.req_pool_idx)
        checkpoint = self.boundaries.get(slot)
        if checkpoint is not None and (
            checkpoint.request is not req
            or checkpoint.retraction_count != self._request_retraction_count(req)
        ):
            self.boundaries.pop(slot, None)
            return None
        return checkpoint

    def publish_checkpoint(self, req, req_to_token_pool) -> None:
        checkpoint = self._checkpoint(req)
        if checkpoint is None:
            raise RuntimeError(f"DVR lost checkpoint ownership for {req.rid}.")

        # The bonus token is visible but has no committed target state until the
        # next verify. This host length may lag the physical slot by one overlap
        # iteration; execution uses the slot directly, not this cache metadata.
        visible_len = len(req.origin_input_ids) + len(req.output_ids_through_stop)
        committed_len = min(req.kv_committed_len, max(0, visible_len - 1))
        checkpoint.seq_len = max(
            checkpoint.seq_len,
            committed_len // self.chunk_size * self.chunk_size,
        )
        req.mamba_last_track_seqlen = checkpoint.seq_len
        req.mamba_next_track_idx = req_to_token_pool.get_mamba_ping_pong_other_idx(
            checkpoint.track_idx
        )

    def prepare_for_cache_release(self, req) -> None:
        if self._state_adapter is None:
            return
        if self._checkpoint(req) is not None:
            # An overlap iteration may already have advanced the physical slot
            # beyond the result that finished the request. Keep the older radix
            # checkpoint and let a later standard EXTEND rebuild the suffix.
            req.skip_radix_cache_insert = True
            self._drop_checkpoint(req)

    def prepare_for_draft(
        self,
        batch,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
        prefill_prefix_lens: Optional[List[int]] = None,
    ) -> Optional[DVRLinearStateContext]:
        if self._state_adapter is None:
            return None

        missing = []
        for i, req in enumerate(batch.reqs):
            checkpoint = self._checkpoint(req)
            if checkpoint is None:
                missing.append(i)
                continue

            if seq_lens_cpu is not None:
                boundary = int(seq_lens_cpu[i]) // self.chunk_size * self.chunk_size
                if boundary != checkpoint.seq_len:
                    self._drop_checkpoint(req)
                    missing.append(i)

        self._ensure_boundary_state(
            batch,
            missing=missing,
            seq_lens_cpu=seq_lens_cpu,
            prefill_prefix_lens=prefill_prefix_lens,
        )

        ctx = self.state_context(batch, require_boundary=True)
        if ctx is None:
            return None
        if self._state_adapter.draft_reuses_target_state:
            if self.draft_state_backup is None:
                raise RuntimeError("DVR self draft has no live-state backup storage.")
            self.draft_state_backup = self._state_adapter.backup_draft_state(
                state_cache=ctx.state_cache,
                indices=ctx.live_indices,
                backup_indices=ctx.state_input_indices,
                backup_size=int(batch.req_to_token_pool.req_to_token.shape[0]),
                out=self.draft_state_backup,
            )
        return ctx

    def restore_for_verify(
        self, ctx: Optional[DVRLinearStateContext]
    ) -> Optional[DVRLinearStateContext]:
        if ctx is None:
            return None
        assert ctx.boundary_indices is not None
        if (
            self._state_adapter.draft_reuses_target_state
            and self.draft_state_backup is None
        ):
            raise RuntimeError("DVR self draft is missing live-state backups.")
        self._state_adapter.prepare_recurrent_state_for_verify(
            state_cache=ctx.state_cache,
            live_indices=ctx.live_indices,
            boundary_indices=ctx.boundary_indices,
            draft_state_backup=self.draft_state_backup,
            backup_indices=ctx.state_input_indices,
        )
        return ctx

    def rollback_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        ctx: Optional[DVRLinearStateContext],
        accept_lens: torch.Tensor,
    ) -> Optional[DVRRollbackActions]:
        if ctx is None:
            return None
        assert ctx.boundary_indices is not None
        if accept_lens.numel() > 0:
            self._state_adapter.commit_after_verify(
                state_cache=ctx.state_cache,
                state_input_indices=ctx.state_input_indices,
                live_indices=ctx.live_indices,
                boundary_indices=ctx.boundary_indices,
                accepted_token_counts=accept_lens.to(torch.long),
            )
        return DVRRollbackActions(linear_state=self)

    def state_context(
        self, batch: ScheduleBatch, require_boundary: bool = False
    ) -> Optional[DVRLinearStateContext]:
        if self._state_adapter is None or batch.batch_size() == 0:
            return None
        live_indices = self._state_adapter.get_live_indices(batch=batch)
        state_input_indices = self._state_adapter.get_state_input_indices(
            batch=batch, device=live_indices.device
        )
        boundary_indices = None
        if require_boundary:
            checkpoints = [self._checkpoint(req) for req in batch.reqs]
            if any(checkpoint is None for checkpoint in checkpoints):
                raise RuntimeError(
                    "DVR target verify is missing a boundary checkpoint."
                )
            boundary_indices = torch.stack(
                [
                    req.mamba_ping_pong_track_buffer[checkpoint.track_idx]
                    for req, checkpoint in zip(batch.reqs, checkpoints, strict=True)
                ]
            ).to(device=live_indices.device, dtype=torch.long)
        return DVRLinearStateContext(
            state_cache=self._state_adapter.get_state_cache(batch=batch),
            state_input_indices=state_input_indices,
            live_indices=live_indices,
            boundary_indices=boundary_indices,
        )

    @staticmethod
    def batch_seq_lens_cpu(batch: ScheduleBatch) -> List[int]:
        if batch.seq_lens_cpu is not None:
            return [int(x) for x in batch.seq_lens_cpu.tolist()]
        return [int(x) for x in batch.seq_lens.detach().cpu().tolist()]

    def _ensure_boundary_state(
        self,
        batch: ScheduleBatch,
        *,
        missing: Optional[List[int]] = None,
        seq_lens_cpu: Optional[List[int]] = None,
        prefill_prefix_lens: Optional[List[int]] = None,
    ) -> None:
        if missing is None:
            missing = [
                i for i, req in enumerate(batch.reqs) if self._checkpoint(req) is None
            ]
        if not missing:
            return
        ctx = self.state_context(batch)
        if ctx is None:
            return

        seq_lens_cpu = seq_lens_cpu or self.batch_seq_lens_cpu(batch)
        zero_indices = []
        tail_indices = []
        tail_lens = []
        for i in missing:
            req = batch.reqs[i]
            seq_len = int(seq_lens_cpu[i])
            boundary_len = seq_len // self.chunk_size * self.chunk_size
            track_idx = req.mamba_next_track_idx
            if track_idx is None or req.mamba_ping_pong_track_buffer is None:
                raise RuntimeError(
                    f"DVR request {req.rid} has no ping-pong checkpoint slot."
                )
            boundary_index = req.mamba_ping_pong_track_buffer[track_idx]

            if boundary_len == seq_len and boundary_len > 0:
                # A completed target EXTEND leaves the exact aligned state live.
                batch.req_to_token_pool.mamba_pool.copy_from(
                    ctx.live_indices[i].reshape(1), boundary_index.reshape(1)
                )
            elif req.mamba_last_track_seqlen == boundary_len and boundary_len > 0:
                # Upstream prefill tracking stores only chunk-boundary states in
                # the keep slot. Copy-on-write keeps it private from radix donation.
                keep_idx = batch.req_to_token_pool.get_mamba_ping_pong_keep_idx(req)
                source = req.mamba_ping_pong_track_buffer[keep_idx]
                if int(keep_idx) != int(track_idx):
                    batch.req_to_token_pool.mamba_pool.copy_from(
                        source.reshape(1), boundary_index.reshape(1)
                    )
            elif boundary_len == 0:
                zero_indices.append(boundary_index)
            elif (
                prefill_prefix_lens is None
                or int(prefill_prefix_lens[i]) != boundary_len
            ):
                raise RuntimeError(
                    "DVR target EXTEND did not rebuild the latest chunk boundary "
                    "from the nearest radix checkpoint: "
                    f"rid={req.rid}, boundary={boundary_len}, "
                    f"last_track={req.mamba_last_track_seqlen}, "
                    "prefix="
                    f"{None if prefill_prefix_lens is None else int(prefill_prefix_lens[i])}."
                )
            # Otherwise capture_extend_prefix_boundary copied the radix-owned
            # prefix state before target EXTEND consumed the unclosed tail.

            checkpoint = _DVRBoundaryCheckpoint(
                request=req,
                retraction_count=self._request_retraction_count(req),
                track_idx=int(track_idx),
                seq_len=boundary_len,
            )
            self.boundaries[int(req.req_pool_idx)] = checkpoint
            tail_indices.append(ctx.state_input_indices[i])
            tail_lens.append(seq_len - boundary_len)

        if zero_indices:
            self._state_adapter.zero_recurrent_state(
                state_cache=ctx.state_cache,
                indices=torch.stack(zero_indices).to(
                    device=ctx.live_indices.device, dtype=torch.long
                ),
            )
        if tail_indices:
            self._state_adapter.state_input_window().set_tail_lens(
                indices=torch.stack(tail_indices),
                value=torch.tensor(tail_lens, device=ctx.live_indices.device),
            )
