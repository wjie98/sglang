from __future__ import annotations

from typing import Any

from sglang.srt.mem_cache.common import maybe_cache_unfinished_req


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
        _set_mamba_radix_cache_insert_snapshot(
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
        and req.defer_non_streaming_logprob_output
    ):
        return False
    return len(req.output_ids) % force_stream_interval == 0


def _set_mamba_radix_cache_insert_snapshot(
    *,
    req: Any,
    batch: Any,
    req_index: int,
) -> None:
    req.mamba_radix_cache_insert_indices = None
    req.mamba_radix_cache_insert_seqlen = None
    if (
        batch.mamba_track_mask is None
        or batch.mamba_track_indices is None
        or batch.mamba_track_cache_seqlens is None
    ):
        return
    if not bool(batch.mamba_track_mask[req_index].item()):
        return
    req.mamba_radix_cache_insert_indices = batch.mamba_track_indices[
        req_index
    ].reshape(1)
    req.mamba_radix_cache_insert_seqlen = int(
        batch.mamba_track_cache_seqlens[req_index].item()
    )
