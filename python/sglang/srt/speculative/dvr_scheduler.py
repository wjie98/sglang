from __future__ import annotations

from typing import Any

from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.speculative.dvr_info import (
    DVRMambaCheckpoint,
    allow_dvr_non_streaming_logprob_output,
)


def _commit_pending_mamba_checkpoint_from_result(
    *,
    req: Any,
    batch: Any,
    result: Any,
    req_index: int,
    tree_cache: Any,
) -> None:
    aux = getattr(result, "dvr_aux", None)
    checkpoints = getattr(aux, "pending_mamba_checkpoints", None)
    checkpoint = (
        checkpoints[req_index]
        if checkpoints is not None
        and req_index < len(checkpoints)
        and checkpoints[req_index] is not None
        else DVRMambaCheckpoint()
    )
    if not checkpoint.valid or checkpoint.seqlen <= 0:
        return

    last_track_seqlen = getattr(req, "mamba_last_track_seqlen", None)
    if last_track_seqlen is not None and checkpoint.seqlen <= last_track_seqlen:
        return

    materialized_len = len(req.origin_input_ids) + len(req.output_ids)
    if checkpoint.seqlen > materialized_len:
        return

    buffer = getattr(req, "mamba_ping_pong_track_buffer", None)
    if buffer is None:
        return
    if checkpoint.track_idx < 0 or checkpoint.track_idx >= buffer.numel():
        return
    # A pending DVR boundary names the request-local ping-pong slot that holds
    # the just-verified GDN state.  A freed lazy slot is marked as -1 and must
    # never become the scheduler-owned checkpoint for future radix inserts.
    if buffer[checkpoint.track_idx].item() == -1:
        return

    page_size = getattr(tree_cache, "page_size", 1)
    if page_size != 1 and checkpoint.seqlen % page_size != 0:
        return

    req.mamba_last_track_seqlen = checkpoint.seqlen
    req.mamba_next_track_idx = batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
        checkpoint.track_idx
    )


def maybe_filter_running_batch_with_dvr_state(
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
    keep_indices = []
    for i, req in enumerate(batch.reqs):
        if req.finished():
            continue

        max_new_tokens = req.sampling_params.max_new_tokens
        dvr_finished = False
        if max_new_tokens is not None:
            max_new_tokens = int(max_new_tokens)
            if batch.seq_lens_cpu is not None:
                seq_len = int(batch.seq_lens_cpu[i].item())
            elif batch.seq_lens is not None:
                seq_len = int(batch.seq_lens[i].item())
            else:
                seq_len = None
            if max_new_tokens > 0 and seq_len is not None:
                # Decode seq_lens includes KV-visible generated tokens; the
                # newest sampled bonus token is materialized into Req.output_ids
                # one result-processing step later, hence the final visible
                # token corresponds to max_new_tokens - 1.
                dvr_finished = (
                    seq_len - len(req.origin_input_ids) >= max_new_tokens - 1
                )
        if not dvr_finished:
            keep_indices.append(i)
    batch.filter_batch(keep_indices=keep_indices)
    return True


def apply_dvr_deferred_output_from_result(batch: Any, result: Any) -> None:
    """Apply DVR-owned output work after scheduler materializes tokens."""

    if not batch.spec_algorithm.is_dvr():
        return

    aux = getattr(result, "dvr_aux", None)
    output = None if aux is None else getattr(aux, "output", None)
    repairs = None if output is None else output.final_logprob_repairs
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
        allow_dvr_non_streaming_logprob_output(req)


def maybe_cache_unfinished_prefill_req_with_dvr_state(
    *,
    req: Any,
    batch: Any,
    req_index: int,
    tree_cache: Any,
    enable_hisparse: bool,
    hisparse_coordinator: Any,
) -> bool:
    """Cache unfinished prefill reqs that DVR spec-v2 overlap materialized."""

    if not batch.spec_algorithm.is_dvr():
        return False

    should_cache_unfinished = not batch.decoding_reqs or req not in batch.decoding_reqs
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

    maybe_cache_unfinished_req(req, tree_cache)
    if enable_hisparse:
        hisparse_coordinator.admit_request_into_staging(req)
    return True


def maybe_handle_dvr_mamba_checkpoint_after_decode(
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

    if batch.spec_algorithm.is_dvr_eagle() or getattr(batch, "enable_overlap", False):
        _commit_pending_mamba_checkpoint_from_result(
            req=req,
            batch=batch,
            result=result,
            req_index=req_index,
            tree_cache=tree_cache,
        )
    return True
