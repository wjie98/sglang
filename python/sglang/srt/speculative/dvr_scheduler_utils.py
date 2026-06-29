from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sglang.srt.mem_cache.common import maybe_cache_unfinished_req


@dataclass
class DVRMambaCheckpoint:
    track_idx: Optional[int] = None
    seqlen: Optional[int] = None

    @property
    def valid(self) -> bool:
        return self.track_idx is not None and self.seqlen is not None


@dataclass
class DVRRadixInsertSnapshot:
    indices: Any
    seqlen: int


@dataclass
class DVRRequestState:
    """Per-request DVR scheduler state kept behind one Req field."""

    skip_mamba_radix_finished_insert: bool = False
    radix_insert_snapshot: Optional[DVRRadixInsertSnapshot] = None
    defer_non_streaming_logprob_output: bool = False


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


def get_req_dvr_state(req: Any, *, create: bool = False) -> Optional[DVRRequestState]:
    state = getattr(req, "dvr_runtime_state", None)
    if state is None and create:
        state = DVRRequestState()
        req.dvr_runtime_state = state
    return state


def reset_req_dvr_state(req: Any) -> None:
    req.dvr_runtime_state = None


def mark_req_skip_mamba_radix_finished_insert(req: Any) -> None:
    get_req_dvr_state(req, create=True).skip_mamba_radix_finished_insert = True


def should_skip_mamba_radix_finished_insert(req: Any) -> bool:
    state = get_req_dvr_state(req)
    return bool(state and state.skip_mamba_radix_finished_insert)


def set_req_radix_insert_snapshot(req: Any, *, indices: Any, seqlen: int) -> None:
    get_req_dvr_state(req, create=True).radix_insert_snapshot = (
        DVRRadixInsertSnapshot(indices=indices, seqlen=seqlen)
    )


def get_req_radix_insert_snapshot(req: Any) -> Optional[DVRRadixInsertSnapshot]:
    state = get_req_dvr_state(req)
    return None if state is None else state.radix_insert_snapshot


def clear_req_radix_insert_snapshot(req: Any) -> None:
    state = get_req_dvr_state(req)
    if state is not None:
        state.radix_insert_snapshot = None


def defer_req_non_streaming_logprob_output(req: Any) -> None:
    get_req_dvr_state(req, create=True).defer_non_streaming_logprob_output = True


def _defer_non_streaming_logprob_output(req: Any) -> bool:
    state = get_req_dvr_state(req)
    return bool(state and state.defer_non_streaming_logprob_output)


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

    should_cache_unfinished = (
        not batch.decoding_reqs or req not in batch.decoding_reqs
    )
    is_dvr_spec_v2 = (
        batch.is_spec_v2 and batch.spec_algorithm.is_decode_verify_rollback()
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


def should_emit_non_streaming_output_chunk(
    *,
    req: Any,
    return_logprob: bool,
    force_stream_interval: int,
) -> bool:
    """Return whether a non-streaming generation chunk can be sent now.

    Algorithms that repair output logprobs, such as DVR-EAGLE exact logprob
    replay, set a request-level defer flag. The streamer only follows that
    flag and keeps the algorithm-specific repair details out of output code.
    """

    if (
        return_logprob
        and req.return_logprob
        and not req.stream
        and _defer_non_streaming_logprob_output(req)
    ):
        return False
    return len(req.output_ids) % force_stream_interval == 0


def _set_req_radix_insert_snapshot_from_batch(
    *,
    req: Any,
    batch: Any,
    req_index: int,
) -> None:
    clear_req_radix_insert_snapshot(req)
    if (
        batch.mamba_track_mask is None
        or batch.mamba_track_indices is None
        or batch.mamba_track_cache_seqlens is None
    ):
        return
    if not bool(batch.mamba_track_mask[req_index].item()):
        return
    set_req_radix_insert_snapshot(
        req,
        indices=batch.mamba_track_indices[req_index].reshape(1),
        seqlen=int(batch.mamba_track_cache_seqlens[req_index].item()),
    )
