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
    state_input_cache: Any
    state_input_indices: torch.Tensor
    live_indices: torch.Tensor
    boundary_indices: Optional[torch.Tensor] = None


@dataclass
class DVRRollbackActions:
    """DVR state/cache work deferred until verified tokens are materialized."""

    pending_checkpoints: Optional[list[Optional[tuple[int, int]]]] = None
    cache_generated_prefix: Optional[list[bool]] = None

    def cache_prefill_after_rollback(
        self,
        *,
        req,
        batch,
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
        checkpoint = self.pending_checkpoints[req_index]
        if checkpoint is None:
            return True
        track_idx, seqlen = checkpoint
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
        if buffer[track_idx].item() == -1:
            raise RuntimeError(f"DVR checkpoint slot {track_idx} is unallocated.")
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
        self.boundary_seqlen = {}
        self.boundary_track_idx = {}
        self.pending_boundary_publish = set()
        self.slot_owner = {}
        self.boundary_backup = None
        # Identifies the worker-local target boundary snapshot.  Separate
        # EAGLE/MTP draft phases can mutate shared linear-state slots before
        # the next target verify, so DVR-EAGLE preserves this snapshot when the
        # request and chunk boundary are unchanged.
        self.boundary_backup_keys = None
        self.live_backup = None
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

    def clear_cache_state(self):
        self.boundary_seqlen.clear()
        self.boundary_track_idx.clear()
        self.pending_boundary_publish.clear()
        self.slot_owner.clear()
        self.boundary_backup = None
        self.boundary_backup_keys = None
        self.live_backup = None

    def prepare_for_draft(
        self,
        batch,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
        prefill_prefix_lens: Optional[List[int]] = None,
        defer_request_publish: bool = False,
    ) -> None:
        # Batches contain only a subset of running requests. Prune ownership
        # when a bounded request-pool slot is reused, not merely when a request
        # is absent from the current batch.
        for req in batch.reqs:
            slot = getattr(req, "req_pool_idx", None)
            if slot is None:
                continue
            previous_rid = self.slot_owner.get(slot)
            if previous_rid is not None and previous_rid != req.rid:
                self.boundary_seqlen.pop(previous_rid, None)
                self.boundary_track_idx.pop(previous_rid, None)
                self.pending_boundary_publish.discard(previous_rid)
            self.slot_owner[slot] = req.rid

        if not defer_request_publish:
            for req in batch.reqs:
                if req.rid not in self.pending_boundary_publish:
                    continue
                self._publish_boundary_checkpoint(batch, req)
                self.pending_boundary_publish.remove(req.rid)

        seq_lens_cpu = seq_lens_cpu or self.batch_seq_lens_cpu(batch)
        for i, req in enumerate(batch.reqs):
            recorded_boundary = self.boundary_seqlen.get(req.rid)
            if recorded_boundary is None:
                continue
            current_boundary, _ = self.boundary_and_tail_for_seq_len(
                int(seq_lens_cpu[i])
            )
            if (
                req.rid in self.boundary_track_idx
                and current_boundary == recorded_boundary + FLA_CHUNK_SIZE
            ):
                # The preceding target verify already committed this boundary
                # into the same request-local slot. Advance host metadata from
                # the scheduler's normal seq_lens mirror instead of synchronously
                # copying accept_lens in the worker hot path.
                self.boundary_seqlen[req.rid] = current_boundary
            elif (
                req.rid not in self.boundary_track_idx
                or recorded_boundary != current_boundary
                or recorded_boundary % FLA_CHUNK_SIZE != 0
            ):
                self.boundary_seqlen.pop(req.rid, None)
                self.boundary_track_idx.pop(req.rid, None)

        self.ensure_boundary_state(
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
        # Host logical lengths advance after overlap result processing, while
        # the target verify has already updated the physical boundary slot.
        # Key the snapshot by request-local boundary ownership. The supplied
        # verify context pins the physical slot used by this commit, while the
        # key remains valid if radix later rebinds that logical ping-pong slot.
        backup_keys = [
            (req.rid, int(self.boundary_track_idx.get(req.rid, -1)))
            for req in batch.reqs
        ]
        if (
            preserve_existing
            and self.boundary_backup is not None
            and self.boundary_backup_keys == backup_keys
        ):
            return
        ctx = ctx or self.state_context(batch, require_boundary=True)
        if ctx is None:
            self.boundary_backup = None
            self.boundary_backup_keys = None
            self.live_backup = None
            return
        assert ctx.boundary_indices is not None
        self.boundary_backup = ctx.state_adapter.backup_recurrent_state(
            state_cache=ctx.state_cache,
            indices=ctx.boundary_indices,
        )
        self.live_backup = ctx.state_adapter.backup_recurrent_state(
            state_cache=ctx.state_cache,
            indices=ctx.live_indices,
        )
        self.boundary_backup_keys = backup_keys

    def restore_for_verify(
        self,
        batch,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
    ) -> Optional[DVRLinearStateContext]:
        self.ensure_boundary_state(batch, seq_lens_cpu=seq_lens_cpu)
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
        ctx: DVRLinearStateContext,
    ):
        pending_checkpoints = [
            (
                self.boundary_track_idx[req.rid],
                self.boundary_seqlen[req.rid]
                if self.boundary_seqlen[req.rid]
                > (req.mamba_last_track_seqlen or 0)
                else self.boundary_seqlen[req.rid] + FLA_CHUNK_SIZE
            )
            for req in batch.reqs
        ]
        assert ctx.boundary_indices is not None
        if accepted_token_counts.numel() == 0:
            return pending_checkpoints

        verified_tail_lens = ctx.state_input_cache.get_tail_lens(
            indices=ctx.state_input_indices
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
        self.boundary_backup = None
        self.boundary_backup_keys = None
        self.live_backup = None
        return pending_checkpoints

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
            boundary_indices = self.boundary_indices_for_reqs(
                reqs=batch.reqs,
                track_indices=[
                    self.boundary_track_idx[req.rid] for req in batch.reqs
                ],
                device=live_indices.device,
            )
        return DVRLinearStateContext(
            state_cache=state_cache,
            state_adapter=state_adapter,
            state_input_cache=state_adapter.state_input_window(),
            state_input_indices=state_input_indices,
            live_indices=live_indices,
            boundary_indices=boundary_indices,
        )

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
        boundary_seqlen: int,
        publish_to_request: bool,
    ):
        self.boundary_track_idx[req.rid] = track_idx
        self.boundary_seqlen[req.rid] = boundary_seqlen
        if not publish_to_request:
            self.pending_boundary_publish.add(req.rid)
            return
        self._publish_boundary_checkpoint(batch, req)

    def _publish_boundary_checkpoint(self, batch: ScheduleBatch, req) -> None:
        track_idx = self.boundary_track_idx[req.rid]
        boundary_seqlen = self.boundary_seqlen[req.rid]
        req.mamba_last_track_seqlen = boundary_seqlen
        req.mamba_next_track_idx = batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
            track_idx
        )

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

    def init_boundary_for_req(
        self,
        batch: ScheduleBatch,
        req,
        boundary_seqlen: int,
        prefill_prefix_len: Optional[int],
        publish_to_request: bool,
    ) -> Optional[torch.Tensor]:
        assert boundary_seqlen % FLA_CHUNK_SIZE == 0
        last_track_seqlen = req.mamba_last_track_seqlen
        if last_track_seqlen is not None and last_track_seqlen > 0:
            assert last_track_seqlen % FLA_CHUNK_SIZE == 0, (
                "DVR linear-state verify must not reuse non-chunk-boundary "
                "checkpoints."
            )
        checkpoint_track_idx = (
            batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                req.mamba_next_track_idx
            )
            if boundary_seqlen > 0 and last_track_seqlen == boundary_seqlen
            else None
        )
        if checkpoint_track_idx is not None:
            # Normal prefill already wrote the chunk-aligned state into the
            # ping-pong checkpoint buffer. DVR verify mutates its boundary
            # slot after every accepted chunk, so copy-on-write the prefill
            # checkpoint into the request's next writable ping-pong slot before
            # registering it as the DVR boundary.
            boundary_track_idx = req.mamba_next_track_idx
            dst = req.mamba_ping_pong_track_buffer[boundary_track_idx]
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
                publish_to_request,
            )
            return None

        boundary_track_idx = req.mamba_next_track_idx
        dst = req.mamba_ping_pong_track_buffer[boundary_track_idx]
        if boundary_seqlen == 0:
            self.set_boundary_checkpoint(
                batch,
                req,
                boundary_track_idx,
                boundary_seqlen,
                publish_to_request,
            )
            return dst

        if prefill_prefix_len == boundary_seqlen:
            # GDN EXTEND copied the initial recurrent state into this request's
            # writable track slot before consuming the unclosed prefix tail.
            self.set_boundary_checkpoint(
                batch,
                req,
                boundary_track_idx,
                boundary_seqlen,
                publish_to_request,
            )
            return None

        raise RuntimeError(
            "DVR target prefill did not publish the required request-local "
            "linear-state checkpoint: "
            f"rid={req.rid}, boundary={boundary_seqlen}, "
            f"last_track={last_track_seqlen}, prefix={prefill_prefix_len}."
        )

    def ensure_boundary_state(
        self,
        batch: ScheduleBatch,
        ctx: Optional[DVRLinearStateContext] = None,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
        prefill_prefix_lens: Optional[List[int]] = None,
        publish_to_request: bool = True,
    ) -> None:
        ctx = ctx or self.state_context(batch)
        if ctx is None:
            return
        zero_boundary_indices = []
        reset_pos_indices = []
        reset_pos_values = []
        seq_lens_cpu = seq_lens_cpu or self.batch_seq_lens_cpu(batch)
        for i, req in enumerate(batch.reqs):
            if req.rid not in self.boundary_seqlen:
                boundary_seqlen, verified_tail_len = (
                    self.boundary_and_tail_for_seq_len(int(seq_lens_cpu[i]))
                )
                reset_pos_indices.append(ctx.state_input_indices[i])
                reset_pos_values.append(verified_tail_len)
                zero_boundary_idx = self.init_boundary_for_req(
                    batch,
                    req,
                    boundary_seqlen,
                    (
                        None
                        if prefill_prefix_lens is None
                        else int(prefill_prefix_lens[i])
                    ),
                    publish_to_request,
                )
                if zero_boundary_idx is not None:
                    zero_boundary_indices.append(zero_boundary_idx)
        if zero_boundary_indices:
            boundary_indices_to_zero = torch.stack(zero_boundary_indices).to(
                device=ctx.live_indices.device, dtype=torch.long
            )
            ctx.state_adapter.zero_recurrent_state(
                state_cache=ctx.state_cache, indices=boundary_indices_to_zero
            )
        if reset_pos_indices:
            ctx.state_input_cache.set_tail_lens(
                indices=torch.stack(reset_pos_indices),
                value=torch.tensor(reset_pos_values, device=ctx.live_indices.device),
            )


def rollback_dvr_verify(
    *,
    batch,
    linear_state: DVRLinearStateLifecycle,
    linear_state_ctx: Optional[DVRLinearStateContext],
    accept_lens: torch.Tensor,
) -> DVRRollbackActions:
    pending_checkpoints = None
    if linear_state_ctx is not None:
        pending_checkpoints = linear_state.commit_after_verify(
            batch=batch,
            accepted_token_counts=accept_lens.to(torch.long),
            accepted_steps=(accept_lens - 1).to(torch.long),
            ctx=linear_state_ctx,
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
