from __future__ import annotations

from dataclasses import dataclass
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
    rid: str
    track_idx: int
    seq_len: int
    publish_pending: bool = False


@dataclass
class DVRRollbackActions:
    """DVR state/cache work deferred until verified tokens are materialized."""

    pending_checkpoints: Optional[list[tuple[int, int]]] = None
    cache_generated_prefix: Optional[list[bool]] = None
    result_process_ready_event: Optional[torch.cuda.Event] = None

    def cache_prefill_after_rollback(
        self,
        *,
        req,
        req_index: int,
        tree_cache,
        enable_hisparse: bool,
        hisparse_coordinator,
    ) -> bool:
        if (
            self.cache_generated_prefix is None
            or not self.cache_generated_prefix[req_index]
        ):
            return False

        from sglang.srt.mem_cache.common import maybe_cache_unfinished_req

        maybe_cache_unfinished_req(req, tree_cache)
        if enable_hisparse:
            hisparse_coordinator.admit_request_into_staging(req)
        return True

    def commit_checkpoint_after_decode(
        self, *, req, batch, req_index: int, tree_cache
    ) -> bool:
        if self.pending_checkpoints is None:
            if batch.spec_algorithm.is_dvr_eagle() or batch.enable_overlap:
                raise RuntimeError("DVR decode result is missing checkpoint actions.")
            return False
        if req_index >= len(self.pending_checkpoints):
            raise RuntimeError(
                "DVR checkpoints do not match the request batch: "
                f"req_index={req_index}, actions={len(self.pending_checkpoints)}."
            )
        track_idx, seqlen = self.pending_checkpoints[req_index]
        if seqlen <= 0:
            raise RuntimeError(f"DVR produced invalid checkpoint length {seqlen}.")
        if req.mamba_last_track_seqlen is not None and (
            seqlen <= req.mamba_last_track_seqlen
        ):
            return True
        if seqlen > req.kv_committed_len:
            return True

        buffer = req.mamba_ping_pong_track_buffer
        if buffer is None or track_idx < 0 or track_idx >= buffer.numel():
            raise RuntimeError(
                "DVR checkpoint references an invalid tracking slot: "
                f"track_idx={track_idx}, slots={0 if buffer is None else buffer.numel()}."
            )
        # Pool ownership and the host-side slot bound above are authoritative.
        # Reading the GPU slot value here only duplicates that contract and
        # forces a device-wide synchronization in every overlap result.
        page_size = getattr(tree_cache, "page_size", 1)
        if page_size != 1 and seqlen % page_size != 0:
            raise RuntimeError(
                f"DVR checkpoint is not page aligned: {seqlen=}, {page_size=}."
            )

        req.mamba_last_track_seqlen = seqlen
        req.mamba_next_track_idx = batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
            track_idx
        )
        return True


class DVRLinearStateLifecycle:
    """Manage chunk-boundary state for DVR linear-state layers.

    The current implementation is backed by SGLang's linear-state cache and
    ping-pong prefill checkpoints. Keeping it outside `dvr_worker.py` prevents
    the speculative control flow from depending on those backend details.
    """

    def __init__(self, *, server_args, model_runner):
        self.server_args = server_args
        self.model_runner = model_runner
        self._state_adapter = None
        # Request-pool slots are bounded and reused. Pair each checkpoint with
        # its rid so stale state cannot survive slot reuse.
        self.boundaries: dict[int, _DVRBoundaryCheckpoint] = {}
        # (logical request/slot keys, boundary state, draft-start live state).
        # Keep the snapshot atomic when radix rebinds physical request slots.
        self.state_backup = None

    def bind_state_adapter(self, state_adapter) -> None:
        self._state_adapter = state_adapter
        if self._state_adapter is None:
            if getattr(self.model_runner, "mambaish_config", None) is not None:
                raise RuntimeError(
                    "DVR does not support this hybrid linear-state backend: no "
                    "target state adapter was initialized."
                )
            return
        chunk_size = self._state_adapter.chunk_size
        if self.server_args.mamba_track_interval != chunk_size:
            raise ValueError(
                "DVR linear-state verify requires mamba_track_interval to match "
                f"the adapter chunk size {chunk_size}, got "
                f"{self.server_args.mamba_track_interval}. Multiples larger than "
                "the chunk size can miss the latest boundary from the "
                "first prefill because the current extra_buffer path stores "
                "only one tracked prefill checkpoint."
            )

    def clear_cache_state(self):
        self.boundaries.clear()
        self.state_backup = None

    def _checkpoint(self, req) -> Optional[_DVRBoundaryCheckpoint]:
        slot = req.req_pool_idx
        if slot is None:
            return None
        checkpoint = self.boundaries.get(slot)
        if checkpoint is not None and checkpoint.rid != req.rid:
            del self.boundaries[slot]
            return None
        return checkpoint

    def prepare_for_draft(
        self,
        batch,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
        prefill_prefix_lens: Optional[List[int]] = None,
        defer_request_publish: bool = False,
    ) -> None:
        if self._state_adapter is None:
            return
        chunk_size = self._state_adapter.chunk_size
        if not defer_request_publish:
            for req in batch.reqs:
                checkpoint = self._checkpoint(req)
                if checkpoint is None or not checkpoint.publish_pending:
                    continue
                req.mamba_last_track_seqlen = checkpoint.seq_len
                req.mamba_next_track_idx = (
                    batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        checkpoint.track_idx
                    )
                )
                checkpoint.publish_pending = False

        for i, req in enumerate(batch.reqs):
            checkpoint = self._checkpoint(req)
            if checkpoint is None:
                continue
            if seq_lens_cpu is not None:
                current_boundary = int(seq_lens_cpu[i]) // chunk_size * chunk_size
                if current_boundary == checkpoint.seq_len + chunk_size:
                    checkpoint.seq_len = current_boundary
                elif checkpoint.seq_len != current_boundary:
                    del self.boundaries[req.req_pool_idx]
                continue

            # The overlap result processor publishes committed checkpoint
            # ownership one iteration behind GPU execution. Consume that host
            # authority without pulling accepted lengths back in the draft path.
            published_boundary = req.mamba_last_track_seqlen
            if (
                published_boundary is None
                or published_boundary <= checkpoint.seq_len
            ):
                continue
            if published_boundary % chunk_size != 0:
                raise RuntimeError(
                    "DVR received a non-chunk-boundary checkpoint from result "
                    f"processing: rid={req.rid}, boundary={published_boundary}."
                )
            checkpoint.seq_len = published_boundary

        self._ensure_boundary_state(
            batch,
            seq_lens_cpu=seq_lens_cpu,
            prefill_prefix_lens=prefill_prefix_lens,
            publish_to_request=not defer_request_publish,
        )

    def backup_boundary_state(
        self,
        batch: ScheduleBatch,
        *,
        preserve_existing: bool = False,
        ctx: Optional[DVRLinearStateContext] = None,
    ):
        if (
            not self._state_adapter.draft_reuses_target_state
            and not batch.enable_overlap
        ):
            # A separate synchronous draft model cannot mutate or race the
            # target checkpoint, so target verify can read the ping-pong slot
            # directly without copying a recurrent snapshot every iteration.
            self.state_backup = None
            return
        # Host logical lengths advance after overlap result processing, while
        # the target verify has already updated the physical boundary slot.
        # Key the snapshot by request-local boundary ownership. The supplied
        # verify context pins the physical slot used by this commit, while the
        # key remains valid if radix later rebinds that logical ping-pong slot.
        backup_keys = [
            (req.rid, self._checkpoint(req).track_idx) for req in batch.reqs
        ]
        if (
            preserve_existing
            and self.state_backup is not None
            and self.state_backup[0] == backup_keys
        ):
            return
        ctx = ctx or self.state_context(batch, require_boundary=True)
        if ctx is None:
            self.state_backup = None
            return
        assert ctx.boundary_indices is not None
        existing_boundary = existing_live = None
        if self.state_backup is not None and self.state_backup[0] == backup_keys:
            _, existing_boundary, existing_live = self.state_backup
        boundary_backup = self._state_adapter.backup_recurrent_state(
            state_cache=ctx.state_cache,
            indices=ctx.boundary_indices,
            out=existing_boundary,
        )
        # Only self-draft mutates the target live slot. Its temporal state is
        # rebuilt from the boundary oracle, so preserve only convolution state.
        live_backup = None
        if self._state_adapter.draft_reuses_target_state:
            live_backup = self._state_adapter.backup_recurrent_state(
                state_cache=ctx.state_cache,
                indices=ctx.live_indices,
                include_temporal=False,
                out=existing_live,
            )
        self.state_backup = (backup_keys, boundary_backup, live_backup)

    def restore_for_verify(
        self,
        batch,
    ) -> Optional[DVRLinearStateContext]:
        self._ensure_boundary_state(batch)
        ctx = self.state_context(batch, require_boundary=True)
        if ctx is None:
            return None
        assert ctx.boundary_indices is not None
        boundary_backup = live_backup = None
        if self.state_backup is not None:
            _, boundary_backup, live_backup = self.state_backup
        self._state_adapter.prepare_recurrent_state_for_verify(
            state_cache=ctx.state_cache,
            live_indices=ctx.live_indices,
            boundary_indices=ctx.boundary_indices,
            boundary_backup=boundary_backup,
            live_backup=live_backup,
        )
        return ctx

    def rollback_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        ctx: Optional[DVRLinearStateContext],
        accept_lens: torch.Tensor,
    ) -> DVRRollbackActions:
        pending_checkpoints = None
        if ctx is not None:
            chunk_size = self._state_adapter.chunk_size
            pending_checkpoints = []
            for req in batch.reqs:
                checkpoint = self._checkpoint(req)
                if checkpoint is None:
                    raise RuntimeError(
                        f"DVR lost the boundary checkpoint for {req.rid}."
                    )
                seq_len = checkpoint.seq_len
                if seq_len <= (req.mamba_last_track_seqlen or 0):
                    seq_len += chunk_size
                pending_checkpoints.append((checkpoint.track_idx, seq_len))

            assert ctx.boundary_indices is not None
            if accept_lens.numel() > 0:
                verified_tail_lens = self._state_adapter.state_input_window().get_tail_lens(
                    indices=ctx.state_input_indices
                )
                self._state_adapter.commit_after_verify(
                    state_cache=ctx.state_cache,
                    state_input_indices=ctx.state_input_indices,
                    live_indices=ctx.live_indices,
                    boundary_indices=ctx.boundary_indices,
                    verified_tail_lens=verified_tail_lens.to(
                        device=ctx.live_indices.device, dtype=torch.long
                    ),
                    accepted_token_counts=accept_lens.to(torch.long),
                    accepted_steps=(accept_lens - 1).to(torch.long),
                )

        deferred = batch.spec_algorithm.is_dvr_eagle() or batch.enable_overlap
        cache_generated_prefix = None
        if deferred and batch.decoding_reqs:
            cache_generated_prefix = [
                req in batch.decoding_reqs
                and int(
                    batch.extend_lens[i]
                    if batch.extend_lens is not None
                    else req.extend_input_len
                )
                > 1
                for i, req in enumerate(batch.reqs)
            ]
            if not any(cache_generated_prefix):
                cache_generated_prefix = None

        return DVRRollbackActions(
            pending_checkpoints=pending_checkpoints if deferred else None,
            cache_generated_prefix=cache_generated_prefix,
        )

    def state_context(
        self, batch: ScheduleBatch, require_boundary: bool = False
    ) -> Optional[DVRLinearStateContext]:
        state_adapter = self._state_adapter
        if state_adapter is None or batch.batch_size() == 0:
            return None
        assert self.server_args.mamba_track_interval == state_adapter.chunk_size, (
            "DVR linear-state target verify must start from adapter chunk boundaries. "
            "The current prefill tracker only guarantees the latest boundary "
            "when mamba_track_interval equals the adapter chunk size."
        )
        live_indices = state_adapter.get_live_indices(batch=batch)
        state_input_indices = state_adapter.get_state_input_indices(
            batch=batch, device=live_indices.device
        )
        state_cache = state_adapter.get_state_cache(batch=batch)
        boundary_indices = None
        if require_boundary:
            checkpoints = [self._checkpoint(req) for req in batch.reqs]
            if any(checkpoint is None for checkpoint in checkpoints):
                raise RuntimeError("DVR target verify is missing a boundary checkpoint.")
            boundary_indices = torch.stack(
                [
                    req.mamba_ping_pong_track_buffer[checkpoint.track_idx]
                    for req, checkpoint in zip(
                        batch.reqs, checkpoints, strict=True
                    )
                ]
            ).to(device=live_indices.device, dtype=torch.long)
        return DVRLinearStateContext(
            state_cache=state_cache,
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
        ctx: Optional[DVRLinearStateContext] = None,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
        prefill_prefix_lens: Optional[List[int]] = None,
        publish_to_request: bool = True,
    ) -> None:
        missing = [
            i for i, req in enumerate(batch.reqs) if self._checkpoint(req) is None
        ]
        if not missing:
            return
        ctx = ctx or self.state_context(batch)
        if ctx is None:
            return
        chunk_size = self._state_adapter.chunk_size
        zero_boundary_indices = []
        reset_pos_indices = []
        reset_pos_values = []
        seq_lens_cpu = seq_lens_cpu or self.batch_seq_lens_cpu(batch)
        for i in missing:
            req = batch.reqs[i]
            if self._checkpoint(req) is None:
                seq_len = int(seq_lens_cpu[i])
                boundary_seqlen = seq_len // chunk_size * chunk_size
                verified_tail_len = seq_len - boundary_seqlen
                reset_pos_indices.append(ctx.state_input_indices[i])
                reset_pos_values.append(verified_tail_len)
                last_track_seqlen = req.mamba_last_track_seqlen
                if last_track_seqlen is not None and last_track_seqlen > 0:
                    assert last_track_seqlen % chunk_size == 0, (
                        "DVR linear-state verify must not reuse non-chunk-boundary "
                        "checkpoints."
                    )

                boundary_track_idx = req.mamba_next_track_idx
                if boundary_seqlen > 0 and last_track_seqlen == boundary_seqlen:
                    # Normal prefill owns the keep slot. DVR copy-on-writes it
                    # into the request's next writable slot before verify starts.
                    keep_idx = batch.req_to_token_pool.get_mamba_ping_pong_keep_idx(
                        req
                    )
                    src = req.mamba_ping_pong_track_buffer[keep_idx]
                    dst = req.mamba_ping_pong_track_buffer[boundary_track_idx]
                    batch.req_to_token_pool.mamba_pool.copy_from(
                        src.reshape(-1), dst.reshape(-1)
                    )
                elif boundary_seqlen == 0:
                    zero_boundary_indices.append(
                        req.mamba_ping_pong_track_buffer[boundary_track_idx]
                    )
                elif (
                    prefill_prefix_lens is None
                    or int(prefill_prefix_lens[i]) != boundary_seqlen
                ):
                    raise RuntimeError(
                        "DVR target prefill did not publish the required "
                        "request-local linear-state checkpoint: "
                        f"rid={req.rid}, boundary={boundary_seqlen}, "
                        f"last_track={last_track_seqlen}, "
                        "prefix="
                        f"{None if prefill_prefix_lens is None else int(prefill_prefix_lens[i])}."
                    )
                # Otherwise GDN EXTEND copied the initial recurrent state into
                # this writable slot before consuming the unclosed prefix tail.
                slot = req.req_pool_idx
                if slot is None:
                    raise RuntimeError(
                        "DVR cannot own a checkpoint without a request slot."
                    )
                checkpoint = _DVRBoundaryCheckpoint(
                    rid=req.rid,
                    track_idx=boundary_track_idx,
                    seq_len=boundary_seqlen,
                    publish_pending=not publish_to_request,
                )
                self.boundaries[slot] = checkpoint
                if publish_to_request:
                    req.mamba_last_track_seqlen = boundary_seqlen
                    req.mamba_next_track_idx = (
                        batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                            boundary_track_idx
                        )
                    )
        if zero_boundary_indices:
            boundary_indices_to_zero = torch.stack(zero_boundary_indices).to(
                device=ctx.live_indices.device, dtype=torch.long
            )
            self._state_adapter.zero_recurrent_state(
                state_cache=ctx.state_cache, indices=boundary_indices_to_zero
            )
        if reset_pos_indices:
            self._state_adapter.state_input_window().set_tail_lens(
                indices=torch.stack(reset_pos_indices),
                value=torch.tensor(reset_pos_values, device=ctx.live_indices.device),
            )
