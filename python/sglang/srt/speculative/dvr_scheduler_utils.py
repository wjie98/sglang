from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.mem_cache.mamba_radix_cache_policy import (
    clear_req_mamba_radix_insert_snapshot,
    mark_req_skip_mamba_radix_finished_insert,
    set_req_mamba_radix_insert_snapshot,
)


@dataclass
class DVRMambaCheckpoint:
    track_idx: Optional[int] = None
    seqlen: Optional[int] = None

    @property
    def valid(self) -> bool:
        return self.track_idx is not None and self.seqlen is not None


@dataclass
class DVRSpecResultAux:
    """Worker-to-scheduler DVR metadata carried by GenerationBatchResult."""

    pending_mamba_checkpoints: Optional[list[Optional[DVRMambaCheckpoint]]] = None

    @classmethod
    def from_pending_mamba_checkpoint_lists(
        cls,
        track_indices: Optional[list[Optional[int]]],
        seqlens: Optional[list[Optional[int]]],
    ) -> Optional["DVRSpecResultAux"]:
        if track_indices is None and seqlens is None:
            return None

        if track_indices is None:
            track_indices = [None] * len(seqlens)
        if seqlens is None:
            seqlens = [None] * len(track_indices)

        checkpoints = [
            (
                DVRMambaCheckpoint(track_idx=track_idx, seqlen=seqlen)
                if track_idx is not None and seqlen is not None
                else None
            )
            for track_idx, seqlen in zip(track_indices, seqlens, strict=True)
        ]
        return cls(pending_mamba_checkpoints=checkpoints)


class DVRReplayPrefixTracker:
    """Per-worker replay prefix stream for spec-v2 overlap DVR.

    Spec-v2 can start the next DVR replay before the scheduler has materialized
    the just-verified tokens into ``Req.output_ids``.  Self-DVR and DVR-EAGLE
    advance that logical prefix from different token sources, but both need the
    same request-id keyed stream and fallback reconstruction rules.
    """

    def __init__(self) -> None:
        self._output_ids_by_rid: dict[Any, list[int]] = {}

    def prune_to_batch(self, batch: Any) -> None:
        if batch.reqs is None:
            self._output_ids_by_rid.clear()
            return

        active_rids = {req.rid for req in batch.reqs}
        for rid in list(self._output_ids_by_rid):
            if rid not in active_rids:
                self._output_ids_by_rid.pop(rid, None)

    def stream_for_req(
        self,
        req: Any,
        *,
        initialize_from_req_output: bool,
    ) -> list[int]:
        initial_output_ids = list(req.output_ids) if initialize_from_req_output else []
        stream = self._output_ids_by_rid.setdefault(req.rid, initial_output_ids)
        if initialize_from_req_output and len(stream) < len(req.output_ids):
            stream[:] = list(req.output_ids)
        return stream

    def request_token_ids(
        self,
        req: Any,
        seq_len: int,
        *,
        initialize_from_req_output: bool,
        include_full_untruncated_fill_ids: bool = False,
        error_prefix: str,
    ) -> list[int]:
        origin_input_ids = list(req.origin_input_ids)
        output_len = seq_len - len(origin_input_ids)
        if output_len <= 0:
            return origin_input_ids[:seq_len]

        stream = self.stream_for_req(
            req, initialize_from_req_output=initialize_from_req_output
        )
        if len(stream) >= output_len:
            return origin_input_ids + stream[:output_len]

        token_ids = origin_input_ids + list(req.output_ids)
        if len(token_ids) >= seq_len:
            return token_ids

        if hasattr(req, "get_fill_ids"):
            fill_ids = list(req.get_fill_ids())
            if len(fill_ids) >= seq_len:
                return fill_ids

        if include_full_untruncated_fill_ids:
            fill_ids = getattr(req, "full_untruncated_fill_ids", None)
            if fill_ids is not None:
                fill_ids = list(fill_ids)
                if len(fill_ids) >= seq_len:
                    return fill_ids

        raise RuntimeError(
            f"{error_prefix} replay cannot reconstruct the verified prefix: "
            f"rid={req.rid}, origin_tokens={len(origin_input_ids)}, "
            f"req_output_tokens={len(req.output_ids)}, "
            f"tracked_output_tokens={len(stream)}, seq_len={seq_len}."
        )

    def append_output_tokens(
        self,
        req: Any,
        token_ids,
        *,
        initialize_from_req_output: bool,
    ) -> None:
        stream = self.stream_for_req(
            req, initialize_from_req_output=initialize_from_req_output
        )
        stream.extend(int(token_id) for token_id in token_ids)

    def append_batch_output_tokens(
        self,
        batch: Any,
        tokens_per_req,
        *,
        initialize_from_req_output: bool,
    ) -> None:
        """Advance all active replay streams from committed output-token rows."""

        self.prune_to_batch(batch)
        for req, token_ids in zip(batch.reqs, tokens_per_req, strict=True):
            self.append_output_tokens(
                req,
                token_ids,
                initialize_from_req_output=initialize_from_req_output,
            )

    def align_req_to_output_len(
        self,
        req: Any,
        output_len: int,
        *,
        error_prefix: str,
    ) -> None:
        stream = self.stream_for_req(req, initialize_from_req_output=False)
        if len(stream) > output_len:
            del stream[output_len:]
            return

        if len(stream) == output_len:
            return

        missing = output_len - len(stream)
        real_output_ids = req.output_ids[len(stream) : output_len]
        if len(real_output_ids) != missing:
            raise RuntimeError(
                f"{error_prefix} is behind the batch logical length: "
                f"rid={req.rid}, tracked={len(stream)}, required={output_len}."
            )
        stream.extend(int(token_id) for token_id in real_output_ids)


def pending_mamba_checkpoint_for_result(result: Any, i: int) -> DVRMambaCheckpoint:
    aux = getattr(result, "spec_aux", None)
    checkpoints = getattr(aux, "pending_mamba_checkpoints", None)
    if checkpoints is None or i >= len(checkpoints) or checkpoints[i] is None:
        return DVRMambaCheckpoint()
    return checkpoints[i]


def commit_pending_mamba_checkpoint_from_result(
    *,
    req: Any,
    batch: Any,
    result: Any,
    req_index: int,
    tree_cache: Any,
) -> None:
    checkpoint = pending_mamba_checkpoint_for_result(result, req_index)
    if not checkpoint.valid:
        return

    materialized_len = len(req.origin_input_ids) + len(req.output_ids)
    page_size = getattr(tree_cache, "page_size", 1)
    if checkpoint.seqlen > materialized_len or (
        page_size != 1 and checkpoint.seqlen % page_size != 0
    ):
        return

    req.mamba_last_track_seqlen = checkpoint.seqlen
    req.mamba_next_track_idx = batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
        checkpoint.track_idx
    )


def maybe_handle_dvr_mamba_checkpoint_after_decode(
    *,
    req: Any,
    batch: Any,
    result: Any,
    req_index: int,
    tree_cache: Any,
) -> bool:
    """Handle DVR's decode-time mamba checkpoint commit if applicable."""

    if not batch.spec_algorithm.is_decode_verify_rollback():
        return False

    mark_req_skip_mamba_radix_finished_insert(req)
    if batch.is_spec_v2:
        commit_pending_mamba_checkpoint_from_result(
            req=req,
            batch=batch,
            result=result,
            req_index=req_index,
            tree_cache=tree_cache,
        )
    return True


def cache_unfinished_prefill_req_with_dvr_mamba_snapshot(
    *,
    req: Any,
    batch: Any,
    req_index: int,
    tree_cache: Any,
    enable_hisparse: bool,
    hisparse_coordinator: Any,
) -> None:
    """Cache unfinished prefill reqs while preserving DVR mamba checkpoints.

    Normal prefill caching only inserts requests that are not already in the
    decode set. DVR spec-v2 overlap can materialize a generated-prefix
    checkpoint from an overlap prefill; if that prefill advanced more than one
    token, the radix-cache insert must carry the matching mamba track snapshot.
    """

    should_cache_unfinished = not batch.decoding_reqs or req not in batch.decoding_reqs
    is_dvr_spec_v2 = (
        batch.is_spec_v2
        and batch.spec_algorithm.needs_mamba_radix_snapshot_for_spec_v2()
    )
    if is_dvr_spec_v2 and not should_cache_unfinished:
        scheduled_extend_len = (
            batch.extend_lens[req_index]
            if batch.extend_lens is not None
            else req.extend_input_len
        )
        should_cache_unfinished = scheduled_extend_len > 1

    if not should_cache_unfinished:
        return

    if is_dvr_spec_v2:
        _set_req_radix_insert_snapshot_from_batch(
            req=req,
            batch=batch,
            req_index=req_index,
        )
    maybe_cache_unfinished_req(req, tree_cache)
    if enable_hisparse:
        hisparse_coordinator.admit_request_into_staging(req)


def _set_req_radix_insert_snapshot_from_batch(
    *,
    req: Any,
    batch: Any,
    req_index: int,
) -> None:
    clear_req_mamba_radix_insert_snapshot(req)
    if (
        batch.mamba_track_mask is None
        or batch.mamba_track_indices is None
        or batch.mamba_track_cache_seqlens is None
    ):
        return
    if not bool(batch.mamba_track_mask[req_index].item()):
        return
    set_req_mamba_radix_insert_snapshot(
        req,
        indices=batch.mamba_track_indices[req_index].reshape(1),
        seqlen=int(batch.mamba_track_cache_seqlens[req_index].item()),
    )
