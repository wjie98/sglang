from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, Callable, Optional

from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.mem_cache.dvr_mamba_radix_cache_policy import (
    clear_req_mamba_radix_insert_snapshot,
    mark_req_skip_mamba_radix_finished_insert,
    set_req_mamba_radix_insert_snapshot,
)
from sglang.srt.managers.scheduler_components.output_policy import (
    allow_req_non_streaming_logprob_output,
)


@dataclass
class DVRMambaCheckpoint:
    track_idx: Optional[int] = None
    seqlen: Optional[int] = None

    @property
    def valid(self) -> bool:
        return self.track_idx is not None and self.seqlen is not None


@dataclass
class DVRFinalLogprobRepair:
    """Exact final output logprobs produced by a DVR worker.

    Spec-v2 overlap materializes accepted tokens into ``Req.output_ids`` after
    the worker returns.  DVR workers may still compute a full-prefix logprob
    oracle on the forward stream; carry the final rows here and apply them only
    after scheduler result processing owns the final output token list.
    """

    output_ids: list[int]
    output_logprobs: list[float]


@dataclass
class DVRSpecResultAux:
    """Worker-to-scheduler DVR metadata carried by GenerationBatchResult."""

    pending_mamba_checkpoints: Optional[list[Optional[DVRMambaCheckpoint]]] = None
    final_logprob_repairs: Optional[
        list[Optional[DVRFinalLogprobRepair]]
    ] = None


def build_dvr_spec_result_aux(
    *,
    track_indices: Optional[list[Optional[int]]],
    seqlens: Optional[list[Optional[int]]],
    final_logprob_repairs: Optional[list[Optional[DVRFinalLogprobRepair]]] = None,
) -> Optional[DVRSpecResultAux]:
    if track_indices is None and seqlens is None and final_logprob_repairs is None:
        return None

    if track_indices is None:
        track_indices = [None] * (len(seqlens) if seqlens is not None else 0)
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
    return DVRSpecResultAux(
        pending_mamba_checkpoints=checkpoints or None,
        final_logprob_repairs=final_logprob_repairs,
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


class DVRPendingOutputPrefix:
    """Worker-owned output prefix journal for DVR overlap paths.

    Spec-v2 can start the next replay before result processing appends accepted
    tokens into ``Req.output_ids``.  DVR therefore records the real
    client-visible accepted tokens produced by each verify step.  Replay
    prefixes are built only from ``origin_input_ids`` + materialized
    ``Req.output_ids`` + this pending output journal; draft-token streams and
    broad fallback reconstruction are deliberately excluded.
    """

    def __init__(self) -> None:
        self._tokens_by_req: weakref.WeakKeyDictionary[Any, list[int]] = (
            weakref.WeakKeyDictionary()
        )

    def clear(self) -> None:
        self._tokens_by_req.clear()

    def prune_to_batch(self, batch: Any) -> None:
        if batch.reqs is None:
            self._tokens_by_req.clear()
            return

        active_reqs = set(batch.reqs)
        for req in list(self._tokens_by_req):
            if req not in active_reqs:
                self._tokens_by_req.pop(req, None)

    def _stream_for_req(
        self, req: Any, *, error_prefix: Optional[str] = None
    ) -> list[int]:
        stream = self._tokens_by_req.setdefault(req, [])
        output_ids = list(req.output_ids)
        if output_ids:
            common_len = min(len(stream), len(output_ids))
            if stream[:common_len] != output_ids[:common_len]:
                prefix = error_prefix or "DVR output prefix"
                raise RuntimeError(
                    f"{prefix} diverged from materialized output ids: "
                    f"rid={req.rid}, tracked_tail={stream[-8:]}, "
                    f"req_tail={output_ids[-8:]}, tracked_len={len(stream)}, "
                    f"req_output_len={len(output_ids)}."
                )
            if len(output_ids) > len(stream):
                stream.extend(int(token_id) for token_id in output_ids[len(stream) :])
        return stream

    def observed_output_len(self, req: Any) -> int:
        """Return the best known client-visible output length for final repair."""

        return len(self._stream_for_req(req))

    def _prefix_token_ids(
        self,
        req: Any,
        seq_len: int,
        *,
        error_prefix: Optional[str] = None,
    ) -> Optional[list[int]]:
        """Return an explicitly owned DVR replay prefix."""

        origin_input_ids = list(req.origin_input_ids)
        output_len = seq_len - len(origin_input_ids)
        if output_len <= 0:
            return origin_input_ids[:seq_len]

        stream = self._stream_for_req(req, error_prefix=error_prefix)
        if len(stream) >= output_len:
            return origin_input_ids + stream[:output_len]

        if error_prefix is not None:
            raise RuntimeError(
                f"{error_prefix} replay prefix is not yet owned by DVR: "
                f"rid={req.rid}, origin_tokens={len(req.origin_input_ids)}, "
                f"req_output_tokens={len(req.output_ids)}, "
                f"tracked_output_tokens={len(stream)}, seq_len={seq_len}."
            )
        return None

    def request_output_prefix_token_ids(
        self,
        req: Any,
        seq_len: int,
        *,
        error_prefix: str,
    ) -> list[int]:
        """Return a client-visible output prefix for DVR replay."""

        return self._prefix_token_ids(
            req,
            seq_len,
            error_prefix=error_prefix,
        )

    def try_request_output_prefix_token_ids(
        self,
        req: Any,
        seq_len: int,
    ) -> Optional[list[int]]:
        """Best-effort client-visible output prefix reconstruction."""

        return self._prefix_token_ids(
            req,
            seq_len,
        )

    def _append_tokens(
        self,
        req: Any,
        token_ids,
        *,
        error_prefix: str,
    ) -> None:
        stream = self._stream_for_req(req, error_prefix=error_prefix)
        stream.extend(int(token_id) for token_id in token_ids)

    def append_batch_output_tokens(
        self,
        batch: Any,
        tokens_per_req,
        *,
        base_seq_lens_cpu: Optional[list[int]] = None,
        error_prefix: str = "DVR output prefix",
    ) -> None:
        """Advance the client-visible output stream for all active requests."""

        self.prune_to_batch(batch)
        if base_seq_lens_cpu is None and getattr(batch, "seq_lens", None) is not None:
            base_seq_lens_cpu = (
                batch.seq_lens_cpu.tolist()
                if getattr(batch, "seq_lens_cpu", None) is not None
                else batch.seq_lens.detach().cpu().tolist()
            )

        if base_seq_lens_cpu is None:
            base_seq_lens_cpu = [None] * len(batch.reqs)

        for req, base_seq_len, token_ids in zip(
            batch.reqs, base_seq_lens_cpu, tokens_per_req, strict=True
        ):
            stream = self._stream_for_req(req, error_prefix=error_prefix)
            if base_seq_len is not None:
                required_len = max(0, int(base_seq_len) - len(req.origin_input_ids))
                if len(stream) < required_len:
                    raise RuntimeError(
                        f"{error_prefix} is behind the batch logical length: "
                        f"rid={req.rid}, tracked={len(stream)}, "
                        f"required={required_len}."
                    )
            stream.extend(int(token_id) for token_id in token_ids)


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


def _is_dvr_spec_v1_result(batch: Any, result: Any) -> bool:
    """Return whether result uses DVR self-draft's legacy spec-v1 shape."""

    return (
        batch.spec_algorithm.is_dvr_self_draft()
        and getattr(result, "accept_lens", None) is None
        and getattr(result, "num_correct_drafts_per_req_cpu", None) is not None
    )


def maybe_resolve_dvr_spec_v1_decode_tokens_from_result(
    *,
    batch: Any,
    result: Any,
    model_worker: Any,
    accept_grammar_tokens: Callable[[Any, Any], Any],
) -> Optional[list[list[int]]]:
    """Resolve DVR spec-v1 flat accepted tokens into per-request token runs."""

    if not _is_dvr_spec_v1_result(batch, result):
        return None

    assert result.next_token_ids.is_cpu
    flat_tokens = result.next_token_ids.tolist()
    accept_lens = [x + 1 for x in result.num_correct_drafts_per_req_cpu]
    if len(accept_lens) != len(batch.reqs):
        raise RuntimeError(
            "DVR spec-v1 result length mismatch: "
            f"{len(accept_lens)=}, {len(batch.reqs)=}."
        )
    if sum(accept_lens) != len(flat_tokens):
        raise RuntimeError(
            "DVR spec-v1 accepted-token length mismatch: "
            f"{sum(accept_lens)=}, {len(flat_tokens)=}."
        )

    result.num_correct_drafts = sum(result.num_correct_drafts_per_req_cpu)
    on_verify_complete_cpu = getattr(model_worker, "on_verify_complete_cpu", None)
    if callable(on_verify_complete_cpu):
        on_verify_complete_cpu(
            result.num_correct_drafts_per_req_cpu, batch_size=len(batch.reqs)
        )

    default_proposed_per_verify = max(
        0, int(getattr(result, "speculative_num_draft_tokens", 1) or 1) - 1
    )
    proposed_drafts_per_req = getattr(result, "num_proposed_drafts_per_req_cpu", None)

    predict_tokens = []
    offset = 0
    for i, (req, accept_len) in enumerate(zip(batch.reqs, accept_lens, strict=True)):
        accept_tokens = flat_tokens[offset : offset + accept_len]
        offset += accept_len

        if req.is_retracted:
            pass
        elif req.finished():
            # Spec prepare_for_decode pre-claims one bonus slot.
            req.kv_committed_len -= 1
        else:
            if req.grammar is not None:
                accept_tokens = accept_grammar_tokens(req, accept_tokens)

            num_accept_tokens = len(accept_tokens)
            # Spec prepare_for_decode already committed the bonus slot.
            req.kv_committed_len += num_accept_tokens - 1
            req.record_spec_verify_metrics(
                num_correct_drafts=result.num_correct_drafts_per_req_cpu[i],
                num_proposed_drafts=(
                    proposed_drafts_per_req[i]
                    if proposed_drafts_per_req is not None
                    else req.useful_spec_proposed_drafts(default_proposed_per_verify)
                ),
            )

        predict_tokens.append(accept_tokens)

    return predict_tokens


def should_skip_dvr_spec_v1_decode_logprob_append(*, batch: Any, result: Any) -> bool:
    """DVR spec-v1 computes request logprobs inside the worker compatibility path."""

    return _is_dvr_spec_v1_result(batch, result)


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
