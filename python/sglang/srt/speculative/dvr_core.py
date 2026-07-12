from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


DVRMambaCheckpoint = tuple[int, int]


@dataclass
class DVRRollbackActions:
    """DVR work deferred until the scheduler materializes verified tokens."""

    pending_mamba_checkpoints: Optional[list[Optional[DVRMambaCheckpoint]]] = None

    def cache_prefill_after_rollback(
        self,
        *,
        req: Any,
        batch: Any,
        req_index: int,
        tree_cache: Any,
        enable_hisparse: bool,
        hisparse_coordinator: Any,
    ) -> bool:
        should_cache_unfinished = (
            not batch.decoding_reqs or req not in batch.decoding_reqs
        )
        is_dvr_spec_v2 = batch.spec_algorithm.is_dvr_eagle() or getattr(
            batch, "enable_overlap", False
        )
        if is_dvr_spec_v2 and not should_cache_unfinished:
            scheduled_extend_len = (
                batch.extend_lens[req_index]
                if batch.extend_lens is not None
                else req.extend_input_len
            )
            should_cache_unfinished = scheduled_extend_len > 1

        if not should_cache_unfinished:
            return False

        from sglang.srt.mem_cache.common import maybe_cache_unfinished_req

        maybe_cache_unfinished_req(req, tree_cache)
        if enable_hisparse:
            hisparse_coordinator.admit_request_into_staging(req)
        return True

    def commit_checkpoint_after_decode(
        self,
        *,
        req: Any,
        batch: Any,
        req_index: int,
        tree_cache: Any,
    ) -> bool:
        """Commit a DVR-owned decode checkpoint after its tokens are visible."""

        if not (
            batch.spec_algorithm.is_dvr_eagle()
            or getattr(batch, "enable_overlap", False)
        ):
            return False

        if self.pending_mamba_checkpoints is None:
            raise RuntimeError("DVR decode result is missing Mamba checkpoint actions.")
        if req_index >= len(self.pending_mamba_checkpoints):
            raise RuntimeError(
                "DVR Mamba checkpoint actions do not match the request batch: "
                f"req_index={req_index}, actions={len(self.pending_mamba_checkpoints)}."
            )
        checkpoint = self.pending_mamba_checkpoints[req_index]
        if checkpoint is None:
            return True
        track_idx, seqlen = checkpoint
        if seqlen <= 0:
            raise RuntimeError(f"DVR produced invalid Mamba checkpoint length {seqlen}.")

        last_track_seqlen = getattr(req, "mamba_last_track_seqlen", None)
        if last_track_seqlen is not None and seqlen <= last_track_seqlen:
            return True

        # The worker records the next possible chunk boundary without pulling
        # accept_lens back to the host. The newest output token is materialized
        # before it has a committed KV row, so only publish a recurrent checkpoint
        # once KV ownership reaches the same boundary.
        if seqlen > req.kv_committed_len:
            return True

        buffer = getattr(req, "mamba_ping_pong_track_buffer", None)
        if buffer is None or track_idx < 0 or track_idx >= buffer.numel():
            raise RuntimeError(
                "DVR Mamba checkpoint references an invalid tracking slot: "
                f"track_idx={track_idx}, slots={0 if buffer is None else buffer.numel()}."
            )
        if buffer[track_idx].item() == -1:
            raise RuntimeError(
                f"DVR Mamba checkpoint tracking slot {track_idx} is unallocated."
            )

        page_size = getattr(tree_cache, "page_size", 1)
        if page_size != 1 and seqlen % page_size != 0:
            raise RuntimeError(
                "DVR Mamba checkpoint is not page aligned: "
                f"checkpoint={seqlen}, page_size={page_size}."
            )

        req.mamba_last_track_seqlen = seqlen
        req.mamba_next_track_idx = batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
            track_idx
        )
        return True

def rollback_dvr_verify(
    *,
    batch: Any,
    linear_state: Any,
    linear_state_ctx: Any,
    accept_lens: torch.Tensor,
) -> DVRRollbackActions:
    """Rollback target linear state after one verified speculative step."""

    pending_track_indices = None
    pending_track_seqlens = None
    if linear_state_ctx is not None:
        pending_track_indices, pending_track_seqlens = linear_state.commit_after_verify(
            batch=batch,
            accepted_token_counts=accept_lens.to(torch.long),
            accepted_steps=(accept_lens - 1).to(torch.long),
            ctx=linear_state_ctx,
        )

    actions = DVRRollbackActions()
    if pending_track_indices is None and pending_track_seqlens is None:
        return actions

    if pending_track_indices is None:
        pending_track_indices = [None] * len(pending_track_seqlens)
    if pending_track_seqlens is None:
        pending_track_seqlens = [None] * len(pending_track_indices)
    actions.pending_mamba_checkpoints = [
        (
            (int(track_idx), int(seqlen))
            if track_idx is not None and seqlen is not None
            else None
        )
        for track_idx, seqlen in zip(
            pending_track_indices, pending_track_seqlens, strict=True
        )
    ] or None
    return actions
