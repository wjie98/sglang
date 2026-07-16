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
class _DVRStateBackupStore:
    slot_owners: dict[int, tuple[str, int]]
    boundary: Any = None
    live: Any = None


@dataclass
class DVRRollbackActions:
    """DVR state/cache work deferred until verified tokens are materialized."""

    pending_checkpoints: Optional[list[tuple[int, int]]] = None
    cache_generated_prefix: Optional[list[bool]] = None

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

        # Track ownership is known when verify commits the GPU state, even when
        # result processing has not advanced the host-visible boundary length.
        # Publish it independently so radix always keeps the committed slot.
        req.mamba_next_track_idx = batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
            track_idx
        )
        if req.mamba_last_track_seqlen is not None and (
            seqlen <= req.mamba_last_track_seqlen
        ):
            return True
        if seqlen > req.kv_committed_len:
            return True

        req.mamba_last_track_seqlen = seqlen
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
        # Radix may rebind a request's physical ping-pong slot while its logical
        # checkpoint remains live. Store snapshots by request-pool slot so an
        # interleaved batch cannot overwrite another request's state.
        self.state_backup: Optional[_DVRStateBackupStore] = None

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

    @property
    def has_state_adapter(self) -> bool:
        return self._state_adapter is not None

    def clear_cache_state(self):
        self.boundaries.clear()
        self.state_backup = None

    def restore_for_cache_release(self, req, req_to_token_pool) -> None:
        if self._state_adapter is None:
            return
        checkpoint = self._checkpoint(req)
        request_slot = req.req_pool_idx
        if checkpoint is None:
            raise RuntimeError(
                f"DVR cache release is missing the checkpoint for {req.rid}."
            )
        if self.server_args.disable_radix_cache:
            self.boundaries.pop(request_slot, None)
            if self.state_backup is not None:
                self.state_backup.slot_owners.pop(int(request_slot), None)
            return
        if self.state_backup is None or self.state_backup.boundary is None:
            raise RuntimeError(
                f"DVR cache release is missing the boundary snapshot for {req.rid}."
            )
        request_slot = int(request_slot)
        owner = (req.rid, checkpoint.track_idx)
        if self.state_backup.slot_owners.get(request_slot) != owner:
            raise RuntimeError(
                "DVR cache release is missing the request-local checkpoint "
                f"snapshot for {req.rid}."
            )

        checkpoint_index = req.mamba_ping_pong_track_buffer[
            checkpoint.track_idx
        ].reshape(1)
        backup_index = torch.tensor(
            [request_slot],
            device=checkpoint_index.device,
            dtype=torch.long,
        )
        self._state_adapter.restore_recurrent_state_backup(
            state_cache=req_to_token_pool.get_speculative_mamba2_params_all_layers(),
            indices=checkpoint_index,
            backup=self.state_backup.boundary,
            backup_indices=backup_index,
        )
        req.mamba_next_track_idx = (
            req_to_token_pool.get_mamba_ping_pong_other_idx(checkpoint.track_idx)
        )
        self.boundaries.pop(request_slot, None)
        self.state_backup.slot_owners.pop(request_slot, None)

    def _invalidate_backup(self, slot: int) -> None:
        if self.state_backup is not None:
            self.state_backup.slot_owners.pop(slot, None)

    def _checkpoint(self, req) -> Optional[_DVRBoundaryCheckpoint]:
        slot = req.req_pool_idx
        if slot is None:
            return None
        checkpoint = self.boundaries.get(slot)
        if checkpoint is not None and checkpoint.rid != req.rid:
            del self.boundaries[slot]
            self._invalidate_backup(slot)
            return None
        return checkpoint

    def prepare_for_draft(
        self,
        batch,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
        prefill_prefix_lens: Optional[List[int]] = None,
    ) -> None:
        if self._state_adapter is None:
            return
        chunk_size = self._state_adapter.chunk_size
        publish_to_request = prefill_prefix_lens is None
        missing = []
        for i, req in enumerate(batch.reqs):
            checkpoint = self._checkpoint(req)
            if checkpoint is None:
                missing.append(i)
                continue

            if publish_to_request and checkpoint.publish_pending:
                req.mamba_last_track_seqlen = checkpoint.seq_len
                req.mamba_next_track_idx = (
                    batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        checkpoint.track_idx
                    )
                )
                checkpoint.publish_pending = False

            if seq_lens_cpu is not None:
                current_boundary = int(seq_lens_cpu[i]) // chunk_size * chunk_size
                if current_boundary == checkpoint.seq_len + chunk_size:
                    checkpoint.seq_len = current_boundary
                    # Target EXTEND replaced this boundary without passing a
                    # post-verify context, so the prior snapshot is obsolete.
                    self._invalidate_backup(req.req_pool_idx)
                elif checkpoint.seq_len != current_boundary:
                    del self.boundaries[req.req_pool_idx]
                    self._invalidate_backup(req.req_pool_idx)
                    missing.append(i)
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
            missing=missing,
            seq_lens_cpu=seq_lens_cpu,
            prefill_prefix_lens=prefill_prefix_lens,
            publish_to_request=publish_to_request,
        )

    def backup_boundary_state(
        self,
        batch: ScheduleBatch,
        *,
        ctx: Optional[DVRLinearStateContext] = None,
    ):
        if self._state_adapter is None:
            self.state_backup = None
            return
        backup_boundary = not self.server_args.disable_radix_cache
        backup_live = self._state_adapter.draft_reuses_target_state
        if not backup_boundary and not backup_live:
            # With radix disabled there is no checkpoint donation. A separate
            # draft model also cannot mutate the target live slot.
            self.state_backup = None
            return

        checkpoints = [self._checkpoint(req) for req in batch.reqs]
        if any(checkpoint is None for checkpoint in checkpoints):
            raise RuntimeError("DVR cannot back up a missing boundary checkpoint.")
        owner_keys = [
            (req.rid, checkpoint.track_idx)
            for req, checkpoint in zip(batch.reqs, checkpoints, strict=True)
        ]
        request_slots = [int(req.req_pool_idx) for req in batch.reqs]
        if self.state_backup is None:
            self.state_backup = _DVRStateBackupStore(slot_owners={})

        # Draft preparation must preserve a prior snapshot when radix has
        # rebound the physical slot. Post-verify supplies ctx and refreshes all
        # participating requests from the exact slots used by target verify.
        update_positions = (
            list(range(len(batch.reqs)))
            if ctx is not None
            else [
                i
                for i, (slot, owner) in enumerate(
                    zip(request_slots, owner_keys, strict=True)
                )
                if self.state_backup.slot_owners.get(slot) != owner
            ]
        )
        if not update_positions:
            return

        ctx = ctx or self.state_context(batch, require_boundary=True)
        if ctx is None:
            return
        assert ctx.boundary_indices is not None
        if len(update_positions) == len(batch.reqs):
            boundary_indices = ctx.boundary_indices
            live_indices = ctx.live_indices
            backup_indices = ctx.state_input_indices
        else:
            positions = torch.tensor(
                update_positions, device=ctx.live_indices.device, dtype=torch.long
            )
            boundary_indices = ctx.boundary_indices[positions]
            live_indices = ctx.live_indices[positions]
            backup_indices = ctx.state_input_indices[positions]
        # Match ReqToTokenPool directly: row 0 is the existing graph-padding
        # dummy, and real request slots retain their native indices.
        backup_size = int(batch.req_to_token_pool.req_to_token.shape[0])
        if backup_boundary:
            self.state_backup.boundary = self._state_adapter.backup_recurrent_state(
                state_cache=ctx.state_cache,
                indices=boundary_indices,
                backup_indices=backup_indices,
                backup_size=backup_size,
                out=self.state_backup.boundary,
            )
        # Only self-draft mutates the target live slot. Its temporal state is
        # rebuilt from the boundary oracle, so preserve only convolution state.
        if backup_live:
            self.state_backup.live = self._state_adapter.backup_recurrent_state(
                state_cache=ctx.state_cache,
                indices=live_indices,
                backup_indices=backup_indices,
                backup_size=backup_size,
                include_temporal=False,
                out=self.state_backup.live,
            )
        for position in update_positions:
            self.state_backup.slot_owners[request_slots[position]] = owner_keys[
                position
            ]

    def restore_for_verify(
        self,
        batch,
    ) -> Optional[DVRLinearStateContext]:
        self._ensure_boundary_state(batch)
        ctx = self.state_context(batch, require_boundary=True)
        if ctx is None:
            return None
        assert ctx.boundary_indices is not None
        checkpoints = [self._checkpoint(req) for req in batch.reqs]
        if any(checkpoint is None for checkpoint in checkpoints):
            raise RuntimeError("DVR target verify is missing a boundary checkpoint.")
        needs_boundary = not self.server_args.disable_radix_cache
        needs_live = self._state_adapter.draft_reuses_target_state
        boundary_backup = live_backup = None
        backup_indices = None
        if needs_boundary or needs_live:
            if self.state_backup is None:
                raise RuntimeError("DVR target verify is missing state backups.")
            missing = [
                req.rid
                for req, checkpoint in zip(batch.reqs, checkpoints, strict=True)
                if self.state_backup.slot_owners.get(int(req.req_pool_idx))
                != (req.rid, checkpoint.track_idx)
            ]
            if missing:
                raise RuntimeError(
                    "DVR request-local state backup is missing for "
                    f"requests {missing}."
                )
            if needs_boundary:
                boundary_backup = self.state_backup.boundary
                if boundary_backup is None:
                    raise RuntimeError(
                        "DVR target verify is missing boundary-state backups."
                    )
            if needs_live:
                live_backup = self.state_backup.live
                if live_backup is None:
                    raise RuntimeError(
                        "DVR self draft is missing live-state backups."
                    )
            backup_indices = ctx.state_input_indices
        self._state_adapter.prepare_recurrent_state_for_verify(
            state_cache=ctx.state_cache,
            live_indices=ctx.live_indices,
            boundary_indices=ctx.boundary_indices,
            boundary_backup=boundary_backup,
            live_backup=live_backup,
            backup_indices=backup_indices,
        )
        return ctx

    def rollback_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        ctx: Optional[DVRLinearStateContext],
        accept_lens: torch.Tensor,
    ) -> Optional[DVRRollbackActions]:
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
                self._state_adapter.commit_after_verify(
                    state_cache=ctx.state_cache,
                    state_input_indices=ctx.state_input_indices,
                    live_indices=ctx.live_indices,
                    boundary_indices=ctx.boundary_indices,
                    accepted_token_counts=accept_lens.to(torch.long),
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

        if pending_checkpoints is None and cache_generated_prefix is None:
            return None
        return DVRRollbackActions(
            pending_checkpoints=pending_checkpoints,
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
        missing: Optional[List[int]] = None,
        seq_lens_cpu: Optional[List[int]] = None,
        prefill_prefix_lens: Optional[List[int]] = None,
        publish_to_request: bool = True,
    ) -> None:
        if missing is None:
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
                self._invalidate_backup(slot)
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
