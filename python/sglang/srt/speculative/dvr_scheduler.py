from __future__ import annotations

from typing import Any, Optional

from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.mem_cache.dvr_mamba_radix_cache_policy import (
    clear_req_mamba_radix_insert_snapshot,
    mark_req_skip_mamba_radix_finished_insert,
    set_req_mamba_radix_insert_snapshot,
)
from sglang.srt.managers.scheduler_components.output_policy import (
    allow_req_non_streaming_logprob_output,
)
from sglang.srt.speculative.dvr_info import (
    DVRFinalLogprobRepair,
    DVRMambaCheckpoint,
)


def apply_dvr_final_logprob_repairs(
    batch: Any,
    repairs: Optional[list[Optional[DVRFinalLogprobRepair]]],
) -> None:
    """Apply exact final DVR output logprobs after output tokens materialize."""

    if repairs is None:
        return

    for req_i, req in enumerate(batch.reqs):
        if req_i >= len(repairs):
            break
        repair = repairs[req_i]
        if repair is None or not req.return_logprob:
            continue

        output_len = len(repair.output_ids)
        if output_len != len(repair.output_logprobs):
            raise RuntimeError(
                "DVR final logprob repair has inconsistent ids/logprobs length: "
                f"rid={req.rid}, output_len={output_len}, "
                f"logprob_len={len(repair.output_logprobs)}."
            )

        # Spec-v2 applies accepted tokens after the worker returns.  Repair only
        # the exact materialized prefix; a mismatch here means worker replay and
        # scheduler-owned output have diverged.
        materialized_output_ids = list(req.output_ids[:output_len])
        if materialized_output_ids != repair.output_ids:
            raise RuntimeError(
                "DVR final logprob repair no longer matches materialized output ids: "
                f"rid={req.rid}, materialized_tail={materialized_output_ids[-8:]}, "
                f"repair_tail={repair.output_ids[-8:]}, repair_len={output_len}, "
                f"req_output_len={len(req.output_ids)}."
            )

        if req.logprob.output_token_logprobs_val is not None:
            req.logprob.output_token_logprobs_val[:] = repair.output_logprobs
            req.logprob.output_token_logprobs_idx[:] = repair.output_ids
        allow_req_non_streaming_logprob_output(req)


def _commit_pending_mamba_checkpoint_from_result(
    *,
    req: Any,
    batch: Any,
    result: Any,
    req_index: int,
    tree_cache: Any,
) -> None:
    aux = getattr(result, "spec_aux", None)
    checkpoints = getattr(aux, "pending_mamba_checkpoints", None)
    checkpoint = (
        checkpoints[req_index]
        if checkpoints is not None
        and req_index < len(checkpoints)
        and checkpoints[req_index] is not None
        else DVRMambaCheckpoint()
    )
    if not _pending_mamba_checkpoint_is_committable(
        checkpoint=checkpoint,
        req=req,
        tree_cache=tree_cache,
    ):
        return

    req.mamba_last_track_seqlen = checkpoint.seqlen
    req.mamba_next_track_idx = batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
        checkpoint.track_idx
    )


def _pending_mamba_checkpoint_is_committable(
    *,
    checkpoint: DVRMambaCheckpoint,
    req: Any,
    tree_cache: Any,
) -> bool:
    if not checkpoint.valid:
        return False

    if checkpoint.seqlen <= 0:
        return False

    last_track_seqlen = getattr(req, "mamba_last_track_seqlen", None)
    if last_track_seqlen is not None and checkpoint.seqlen <= last_track_seqlen:
        return False

    materialized_len = len(req.origin_input_ids) + len(req.output_ids)
    if checkpoint.seqlen > materialized_len:
        return False

    if not _pending_mamba_checkpoint_track_idx_is_valid(
        checkpoint=checkpoint,
        req=req,
    ):
        return False

    page_size = getattr(tree_cache, "page_size", 1)
    return page_size == 1 or checkpoint.seqlen % page_size == 0


def _pending_mamba_checkpoint_track_idx_is_valid(
    *,
    checkpoint: DVRMambaCheckpoint,
    req: Any,
) -> bool:
    buffer = getattr(req, "mamba_ping_pong_track_buffer", None)
    if buffer is None:
        return False
    if checkpoint.track_idx < 0 or checkpoint.track_idx >= buffer.numel():
        return False

    # A pending DVR boundary names the request-local ping-pong slot that holds
    # the just-verified GDN state.  A freed lazy slot is marked as -1 and must
    # never become the scheduler-owned checkpoint for future radix inserts.
    return buffer[checkpoint.track_idx].item() != -1


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


def _is_dvr_spec_v2(batch: Any) -> bool:
    spec_algorithm = batch.spec_algorithm
    if not spec_algorithm.is_dvr():
        return False
    return spec_algorithm.is_dvr_eagle() or getattr(batch, "enable_overlap", False)


def maybe_filter_running_batch_with_spec_state(
    *,
    batch: Any,
    future_map: Any,
    enable_overlap: bool,
) -> bool:
    """Filter running batch with DVR's spec-v2 logical-finish state if needed."""

    spec_algorithm = batch.spec_algorithm
    if not enable_overlap or not spec_algorithm.is_dvr():
        return False
    if spec_algorithm.is_dvr_self_draft() and not batch.enable_overlap:
        return False

    future_map.resolve_seq_lens_cpu(batch)
    keep_indices = [
        i
        for i, req in enumerate(batch.reqs)
        if not req.finished()
        and not _dvr_is_finished_by_published_seq_len(batch=batch, req_index=i)
    ]
    batch.filter_batch(keep_indices=keep_indices)
    return True


def _dvr_is_finished_by_published_seq_len(*, batch: Any, req_index: int) -> bool:
    req = batch.reqs[req_index]
    max_new_tokens = req.sampling_params.max_new_tokens
    if max_new_tokens is None:
        return False
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens <= 0:
        return False

    if batch.seq_lens_cpu is not None:
        seq_len = int(batch.seq_lens_cpu[req_index].item())
    elif batch.seq_lens is not None:
        seq_len = int(batch.seq_lens[req_index].item())
    else:
        return False

    # Decode seq_lens includes KV-visible generated tokens; the newest sampled
    # bonus token is materialized into Req.output_ids one result-processing step
    # later, hence the final visible token corresponds to max_new_tokens - 1.
    return seq_len - len(req.origin_input_ids) >= max_new_tokens - 1


def apply_spec_final_logprob_repairs_from_result(batch: Any, result: Any) -> None:
    """Apply DVR final-response logprob repairs, if any."""

    if not batch.spec_algorithm.is_dvr():
        return

    aux = getattr(result, "spec_aux", None)
    repairs: Optional[list[Optional[DVRFinalLogprobRepair]]] = getattr(
        aux, "final_logprob_repairs", None
    )
    apply_dvr_final_logprob_repairs(batch, repairs)


def _is_dvr_self_draft_spec_v1(batch: Any) -> bool:
    return batch.spec_algorithm.is_dvr_self_draft() and not getattr(
        batch, "enable_overlap", False
    )


def should_skip_dvr_spec_v1_decode_logprob_append(*, batch: Any, result: Any) -> bool:
    """DVR spec-v1 computes request logprobs inside the worker compatibility path."""

    return _is_dvr_self_draft_spec_v1(batch) and getattr(
        result, "num_correct_drafts_per_req_cpu", None
    ) is not None


def maybe_cache_unfinished_prefill_req_with_spec_state(
    *,
    req: Any,
    batch: Any,
    req_index: int,
    tree_cache: Any,
    enable_hisparse: bool,
    hisparse_coordinator: Any,
) -> bool:
    """Cache unfinished prefill reqs while preserving DVR mamba checkpoints."""

    if not batch.spec_algorithm.is_dvr():
        return False

    # Normal prefill caching only inserts requests that are not already in the
    # decode set. DVR spec-v2 overlap can materialize a generated-prefix
    # checkpoint from an overlap prefill; if that prefill advanced more than one
    # token, the radix-cache insert must carry the matching mamba snapshot.
    should_cache_unfinished = not batch.decoding_reqs or req not in batch.decoding_reqs
    is_dvr_spec_v2 = _is_dvr_spec_v2(batch)
    if is_dvr_spec_v2 and not should_cache_unfinished:
        scheduled_extend_len = (
            batch.extend_lens[req_index]
            if batch.extend_lens is not None
            else req.extend_input_len
        )
        should_cache_unfinished = scheduled_extend_len > 1

    if not should_cache_unfinished:
        return False

    if is_dvr_spec_v2:
        _set_req_radix_insert_snapshot_from_batch(
            req=req,
            batch=batch,
            req_index=req_index,
        )
    maybe_cache_unfinished_req(req, tree_cache)
    if enable_hisparse:
        hisparse_coordinator.admit_request_into_staging(req)
    return True


def maybe_handle_spec_mamba_checkpoint_after_decode(
    *,
    req: Any,
    batch: Any,
    result: Any,
    req_index: int,
    tree_cache: Any,
) -> bool:
    """Handle DVR's decode-time mamba checkpoint commit if applicable."""

    if not batch.spec_algorithm.is_dvr():
        return False

    mark_req_skip_mamba_radix_finished_insert(req)
    if _is_dvr_spec_v2(batch):
        _commit_pending_mamba_checkpoint_from_result(
            req=req,
            batch=batch,
            result=result,
            req_index=req_index,
            tree_cache=tree_cache,
        )
    return True
