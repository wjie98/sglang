from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.mem_cache.dvr_mamba_radix_cache_policy import (
    clear_req_mamba_radix_insert_snapshot,
    mark_req_skip_mamba_radix_finished_insert,
    set_req_mamba_radix_insert_snapshot,
)
from sglang.srt.speculative.dvr_output_policy import (
    allow_req_non_streaming_logprob_output,
)
from sglang.srt.speculative.spec_policy import get_spec_algorithm_policy


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


def compact_output_token_rows(
    output_tokens: Any,
    output_lens: Any,
) -> Optional[list[list[int]]]:
    """Return per-request output token rows without padded verify tail tokens."""

    if output_tokens is None:
        return None

    output_lens_cpu = output_lens.detach().cpu().tolist()
    output_tokens_cpu = output_tokens.detach().cpu().tolist()
    return [
        [int(token_id) for token_id in token_row[: int(output_len)]]
        for token_row, output_len in zip(
            output_tokens_cpu, output_lens_cpu, strict=True
        )
    ]


def apply_dvr_final_logprob_repairs_from_result(batch: Any, result: Any) -> None:
    """Apply DVR exact final logprob repairs after Spec-v2 output materializes."""

    aux = getattr(result, "spec_aux", None)
    repairs: Optional[list[Optional[DVRFinalLogprobRepair]]] = getattr(
        aux, "final_logprob_repairs", None
    )
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


class DVRReplayPrefixTracker:
    """Per-worker replay prefix stream for DVR overlap paths.

    Spec-v2 can start the next DVR replay before the scheduler has materialized
    the just-verified tokens into ``Req.output_ids``.  Each tracker instance
    owns exactly one logical stream: either verifier tokens or client-visible
    output tokens.  Public methods name that stream explicitly so callers do not
    pass semantic booleans at DVR worker call sites.
    """

    def __init__(self) -> None:
        self._tokens_by_rid: dict[Any, list[int]] = {}

    def clear(self) -> None:
        self._tokens_by_rid.clear()

    def prune_to_batch(self, batch: Any) -> None:
        if batch.reqs is None:
            self._tokens_by_rid.clear()
            return

        active_rids = {req.rid for req in batch.reqs}
        for rid in list(self._tokens_by_rid):
            if rid not in active_rids:
                self._tokens_by_rid.pop(rid, None)

    def _stream_for_req(
        self,
        req: Any,
        *,
        seed_from_req_output: bool,
    ) -> list[int]:
        initial_output_ids = list(req.output_ids) if seed_from_req_output else []
        stream = self._tokens_by_rid.setdefault(req.rid, initial_output_ids)
        if seed_from_req_output and len(stream) < len(req.output_ids):
            stream[:] = list(req.output_ids)
        return stream

    def observed_output_len(self, req: Any) -> int:
        """Return the best known client-visible output length for final repair."""

        return max(
            len(req.output_ids),
            len(self._stream_for_req(req, seed_from_req_output=True)),
        )

    def _prefix_token_ids(
        self,
        req: Any,
        seq_len: int,
        *,
        seed_from_req_output: bool,
        include_full_untruncated_fill_ids: bool = False,
        error_prefix: Optional[str] = None,
    ) -> Optional[list[int]]:
        """Reconstruct a DVR replay prefix, optionally raising on misses."""

        origin_input_ids = list(req.origin_input_ids)
        output_len = seq_len - len(origin_input_ids)
        if output_len <= 0:
            return origin_input_ids[:seq_len]

        stream = self._stream_for_req(req, seed_from_req_output=seed_from_req_output)
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

        if error_prefix is not None:
            raise RuntimeError(
                f"{error_prefix} replay cannot reconstruct the verified prefix: "
                f"rid={req.rid}, origin_tokens={len(req.origin_input_ids)}, "
                f"req_output_tokens={len(req.output_ids)}, "
                f"tracked_output_tokens={len(stream)}, seq_len={seq_len}."
            )
        return None

    def request_verifier_prefix_token_ids(
        self,
        req: Any,
        seq_len: int,
        *,
        error_prefix: str,
    ) -> list[int]:
        """Reconstruct the verifier-side token prefix.

        Self-DVR and DVR-EAGLE verifier replay can run before accepted tokens
        are materialized in ``Req.output_ids``.  This stream is therefore owned
        by the DVR worker and only falls back to scheduler fields when it has
        not yet observed a token.
        """

        return self._prefix_token_ids(
            req,
            seq_len,
            seed_from_req_output=False,
            include_full_untruncated_fill_ids=True,
            error_prefix=error_prefix,
        )

    def request_output_prefix_token_ids(
        self,
        req: Any,
        seq_len: int,
        *,
        error_prefix: str,
    ) -> list[int]:
        """Reconstruct the client-visible output prefix.

        EAGLE verification has a verifier prefix and a client output stream
        with different token semantics.  Final logprob repair must score the
        latter, so this helper intentionally seeds from ``Req.output_ids`` when
        the scheduler has already materialized those tokens.
        """

        return self._prefix_token_ids(
            req,
            seq_len,
            seed_from_req_output=True,
            include_full_untruncated_fill_ids=True,
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
            seed_from_req_output=True,
            include_full_untruncated_fill_ids=True,
        )

    def _append_tokens(
        self,
        req: Any,
        token_ids,
        *,
        seed_from_req_output: bool,
    ) -> None:
        stream = self._stream_for_req(req, seed_from_req_output=seed_from_req_output)
        stream.extend(int(token_id) for token_id in token_ids)

    def append_batch_output_tokens(
        self,
        batch: Any,
        tokens_per_req,
    ) -> None:
        """Advance the client-visible output stream for all active requests."""

        self.prune_to_batch(batch)
        for req, token_ids in zip(batch.reqs, tokens_per_req, strict=True):
            self._append_tokens(req, token_ids, seed_from_req_output=True)

    def append_batch_verifier_tokens(
        self,
        batch: Any,
        tokens_per_req,
    ) -> None:
        """Advance the verifier-prefix stream for all active requests."""

        self.prune_to_batch(batch)
        for req, token_ids in zip(batch.reqs, tokens_per_req, strict=True):
            self._append_tokens(req, token_ids, seed_from_req_output=False)

    def advance_output_stream_from_compact_rows(
        self,
        *,
        batch: Any,
        compact_output_token_ids_per_req: list[list[int]],
        error_prefix: str,
    ) -> None:
        """Append compact client-visible output rows for DVR-EAGLE repair."""

        if batch.forward_mode.is_idle() or batch.reqs is None:
            return

        self.prune_to_batch(batch)

        seq_lens_cpu = (
            batch.seq_lens_cpu.tolist()
            if batch.seq_lens_cpu is not None
            else batch.seq_lens.detach().cpu().tolist()
        )

        for req, seq_len, compact_output_token_ids in zip(
            batch.reqs,
            seq_lens_cpu,
            compact_output_token_ids_per_req,
            strict=True,
        ):
            prompt_len = len(req.origin_input_ids)
            stream = self._stream_for_req(req, seed_from_req_output=True)
            # In spec-v2 overlap, model-side seq_len can lag the
            # client-visible output stream by one result. Do not truncate a
            # prefix already learned from Req.output_ids/tracker state.
            prefix_output_len = max(0, int(seq_len) - prompt_len, len(stream))
            prefix_ids = self.request_output_prefix_token_ids(
                req,
                prompt_len + prefix_output_len,
                error_prefix=error_prefix,
            )
            # Overlap can compute final repair before the prefill/previous
            # decode result is materialized into Req.output_ids. Seed from the
            # best known prefix, then append this verify's compact output rows.
            stream[:] = prefix_ids[prompt_len:]
            stream.extend(compact_output_token_ids)

    def advance_eagle_verifier_stream_from_draft_rows(
        self,
        *,
        batch: Any,
        draft_token: Any,
        draft_token_num: int,
        accept_lens: Any,
        error_prefix: str,
    ) -> None:
        """Append accepted EAGLE/MTP draft rows to the verifier stream.

        DVR-EAGLE has two token streams: client output uses target predictions,
        while the next deterministic verifier prefix uses the accepted draft
        tokens whose target KV/GDN state was just verified.  Keeping this update
        here prevents the worker verify loop from duplicating replay-prefix
        ownership details.
        """

        if batch.forward_mode.is_idle() or batch.reqs is None:
            return

        self.prune_to_batch(batch)

        bs = len(batch.seq_lens)
        seq_lens_cpu = (
            batch.seq_lens_cpu.tolist()
            if batch.seq_lens_cpu is not None
            else batch.seq_lens.detach().cpu().tolist()
        )
        accept_lens_cpu = accept_lens.detach().cpu().tolist()
        draft_tokens_cpu = (
            draft_token.reshape(bs, draft_token_num).detach().cpu().tolist()
        )

        for req, seq_len, accepted_len, draft_tokens in zip(
            batch.reqs,
            seq_lens_cpu,
            accept_lens_cpu,
            draft_tokens_cpu,
            strict=True,
        ):
            prefix_output_len = max(0, int(seq_len) - len(req.origin_input_ids))
            self._align_req_to_output_len(
                req,
                prefix_output_len,
                error_prefix=error_prefix,
            )

            self._append_tokens(
                req,
                [int(token_id) for token_id in draft_tokens[: int(accepted_len)]],
                seed_from_req_output=False,
            )

    def _align_req_to_output_len(
        self,
        req: Any,
        output_len: int,
        *,
        error_prefix: str,
    ) -> None:
        stream = self._stream_for_req(req, seed_from_req_output=False)
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


def commit_pending_mamba_checkpoint_from_result(
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


def maybe_handle_dvr_mamba_checkpoint_after_decode(
    *,
    req: Any,
    batch: Any,
    result: Any,
    req_index: int,
    tree_cache: Any,
) -> bool:
    """Handle DVR's decode-time mamba checkpoint commit if applicable."""

    if not get_spec_algorithm_policy(batch.spec_algorithm).is_dvr():
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
        and get_spec_algorithm_policy(
            batch.spec_algorithm
        ).needs_mamba_radix_snapshot_for_spec_v2()
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
