from __future__ import annotations

import weakref
from dataclasses import dataclass, fields
from typing import Any, Optional

import torch

from sglang.srt.speculative.eagle_info import EagleVerifyInput


DVRMambaCheckpoint = tuple[int, int]
"""Request-local Mamba checkpoint as ``(track_idx, seqlen)``."""

DVRFinalLogprobRepair = tuple[list[int], list[float]]
"""Exact final output ids/logprobs carried until Req.output_ids is materialized."""


_DVR_LOGPROB_DEFERRED = 1
_DVR_LOGPROB_REPAIR_CLAIMED = 2


@dataclass
class DVRDeferredActions:
    """DVR postprocess work carried by GenerationBatchResult.dvr_aux."""

    pending_mamba_checkpoints: Optional[list[Optional[DVRMambaCheckpoint]]] = None
    final_logprob_repairs: Optional[list[Optional[DVRFinalLogprobRepair]]] = None

    def cache_unfinished_prefill_req(
        self,
        *,
        req: Any,
        batch: Any,
        req_index: int,
        tree_cache: Any,
        enable_hisparse: bool,
        hisparse_coordinator: Any,
    ) -> bool:
        """Cache unfinished prefill reqs that DVR spec-v2 materialized."""

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

    def apply_output_after_materialize(self, *, batch: Any) -> None:
        """Apply DVR-owned output work after scheduler materializes tokens."""

        repairs = self.final_logprob_repairs
        if repairs is None:
            return

        for req_i, req in enumerate(batch.reqs):
            if req_i >= len(repairs):
                break
            repair = repairs[req_i]
            if repair is None or not req.return_logprob:
                continue

            output_ids, output_logprobs = repair
            output_len = len(output_ids)
            if output_len != len(output_logprobs):
                raise RuntimeError(
                    "DVR final logprob repair has inconsistent ids/logprobs "
                    f"length: rid={req.rid}, output_len={output_len}, "
                    f"logprob_len={len(output_logprobs)}."
                )

            # Spec-v2 applies accepted tokens after the worker returns.  Repair
            # only the exact materialized prefix; a mismatch here means worker
            # replay and scheduler-owned output have diverged.
            materialized_output_ids = list(req.output_ids[:output_len])
            if materialized_output_ids != output_ids:
                raise RuntimeError(
                    "DVR final logprob repair no longer matches materialized "
                    f"output ids: rid={req.rid}, "
                    f"materialized_tail={materialized_output_ids[-8:]}, "
                    f"repair_tail={output_ids[-8:]}, "
                    f"repair_len={output_len}, req_output_len={len(req.output_ids)}."
                )

            if req.logprob.output_token_logprobs_val is not None:
                req.logprob.output_token_logprobs_val[:] = output_logprobs
                req.logprob.output_token_logprobs_idx[:] = output_ids
            allow_dvr_non_streaming_logprob_output(req)

    def commit_mamba_checkpoint_after_decode(
        self,
        *,
        req: Any,
        batch: Any,
        req_index: int,
        tree_cache: Any,
    ) -> bool:
        """Commit DVR-owned decode-time Mamba checkpoint if this path owns it."""

        if not (
            batch.spec_algorithm.is_dvr_eagle()
            or getattr(batch, "enable_overlap", False)
        ):
            return False

        checkpoint = None
        if (
            self.pending_mamba_checkpoints is not None
            and req_index < len(self.pending_mamba_checkpoints)
        ):
            checkpoint = self.pending_mamba_checkpoints[req_index]
        if checkpoint is None:
            return True
        track_idx, seqlen = checkpoint
        if seqlen <= 0:
            return True

        last_track_seqlen = getattr(req, "mamba_last_track_seqlen", None)
        if last_track_seqlen is not None and seqlen <= last_track_seqlen:
            return True

        materialized_len = len(req.origin_input_ids) + len(req.output_ids)
        if seqlen > materialized_len:
            return True

        buffer = getattr(req, "mamba_ping_pong_track_buffer", None)
        if buffer is None:
            return True
        if track_idx < 0 or track_idx >= buffer.numel():
            return True
        # A pending DVR boundary names the request-local ping-pong slot that
        # holds the just-verified GDN state.  A freed lazy slot is marked as -1
        # and must never become the scheduler-owned checkpoint for radix insert.
        if buffer[track_idx].item() == -1:
            return True

        page_size = getattr(tree_cache, "page_size", 1)
        if page_size != 1 and seqlen % page_size != 0:
            return True

        req.mamba_last_track_seqlen = seqlen
        req.mamba_next_track_idx = batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
            track_idx
        )
        return True


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
        self._logprobs_by_req: weakref.WeakKeyDictionary[
            Any, list[Optional[float]]
        ] = weakref.WeakKeyDictionary()

    def clear(self) -> None:
        self._tokens_by_req.clear()
        self._logprobs_by_req.clear()

    def prune_to_batch(self, batch: Any) -> None:
        if batch.reqs is None:
            self._tokens_by_req.clear()
            self._logprobs_by_req.clear()
            return

        active_reqs = set(batch.reqs)
        for req in list(self._tokens_by_req):
            if req not in active_reqs:
                self._tokens_by_req.pop(req, None)
                self._logprobs_by_req.pop(req, None)

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

    def _logprob_stream_for_req(self, req: Any) -> list[Optional[float]]:
        logprob_stream = self._logprobs_by_req.setdefault(req, [])
        req_logprobs = getattr(
            getattr(req, "logprob", None), "output_token_logprobs_val", None
        )
        if not req_logprobs:
            return logprob_stream

        for i, value in enumerate(req_logprobs):
            if i < len(logprob_stream):
                if logprob_stream[i] is None:
                    logprob_stream[i] = float(value)
            else:
                logprob_stream.append(float(value))
        return logprob_stream

    def observed_output_len(self, req: Any) -> int:
        """Return the best known client-visible output length for final repair."""

        return len(self._stream_for_req(req))

    def request_output_prefix_token_ids(
        self,
        req: Any,
        seq_len: int,
        *,
        error_prefix: str,
    ) -> list[int]:
        """Return a client-visible output prefix for DVR replay."""

        origin_input_ids = list(req.origin_input_ids)
        output_len = seq_len - len(origin_input_ids)
        if output_len <= 0:
            return origin_input_ids[:seq_len]

        stream = self._stream_for_req(req, error_prefix=error_prefix)
        if len(stream) >= output_len:
            return origin_input_ids + stream[:output_len]

        raise RuntimeError(
            f"{error_prefix} replay prefix is not yet owned by DVR: "
            f"rid={req.rid}, origin_tokens={len(req.origin_input_ids)}, "
            f"req_output_tokens={len(req.output_ids)}, "
            f"tracked_output_tokens={len(stream)}, seq_len={seq_len}."
        )

    def append_batch_output_tokens(
        self,
        batch: Any,
        tokens_per_req,
        *,
        token_logprobs_per_req: Optional[list[Optional[list[float]]]] = None,
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
        if token_logprobs_per_req is None:
            token_logprobs_per_req = [None] * len(batch.reqs)

        for req, base_seq_len, token_ids, token_logprobs in zip(
            batch.reqs,
            base_seq_lens_cpu,
            tokens_per_req,
            token_logprobs_per_req,
            strict=True,
        ):
            stream = self._stream_for_req(req, error_prefix=error_prefix)
            logprob_stream = self._logprob_stream_for_req(req)
            if len(logprob_stream) < len(stream):
                # Tokens can be learned from Req.output_ids before their
                # logprobs are visible to the worker.  Keep positional
                # alignment and fill the holes once result processing exposes
                # them through Req.logprob or the next verify journal append.
                logprob_stream.extend([None] * (len(stream) - len(logprob_stream)))
            if base_seq_len is not None:
                required_len = max(0, int(base_seq_len) - len(req.origin_input_ids))
                if len(stream) < required_len:
                    raise RuntimeError(
                        f"{error_prefix} is behind the batch logical length: "
                        f"rid={req.rid}, tracked={len(stream)}, "
                        f"required={required_len}."
                    )
            stream.extend(int(token_id) for token_id in token_ids)
            if token_logprobs is not None:
                if len(token_logprobs) != len(token_ids):
                    raise RuntimeError(
                        f"{error_prefix} has inconsistent token/logprob counts: "
                        f"rid={req.rid}, token_count={len(token_ids)}, "
                        f"logprob_count={len(token_logprobs)}."
                    )
                logprob_stream.extend(float(value) for value in token_logprobs)
            else:
                logprob_stream.extend([None] * len(token_ids))

    def final_logprob_repair(
        self,
        req: Any,
        output_len: int,
        *,
        error_prefix: str,
    ) -> DVRFinalLogprobRepair:
        """Build final repair rows from the verify-step output journal."""

        token_stream = self._stream_for_req(req, error_prefix=error_prefix)
        logprob_stream = self._logprob_stream_for_req(req)
        if len(token_stream) < output_len or len(logprob_stream) < output_len:
            raise RuntimeError(
                f"{error_prefix} is incomplete: rid={req.rid}, "
                f"output_len={output_len}, tracked_tokens={len(token_stream)}, "
                f"tracked_logprobs={len(logprob_stream)}."
            )

        repair_logprobs = logprob_stream[:output_len]
        if any(value is None for value in repair_logprobs):
            missing = [i for i, value in enumerate(repair_logprobs) if value is None]
            raise RuntimeError(
                f"{error_prefix} is missing verify logprobs: rid={req.rid}, "
                f"missing_positions={missing[:8]}, output_len={output_len}."
            )
        return token_stream[:output_len], [float(value) for value in repair_logprobs]


def defer_dvr_non_streaming_logprob_output(req: Any) -> None:
    """Hold DVR non-streaming logprob output until final repair overwrites it."""

    if getattr(req, "dvr_deferred_output", None) is None:
        req.dvr_deferred_output = _DVR_LOGPROB_DEFERRED


def allow_dvr_non_streaming_logprob_output(req: Any) -> None:
    """Release a DVR non-streaming logprob response after final repair."""

    req.dvr_deferred_output = None


def try_claim_dvr_final_logprob_repair(req: Any) -> bool:
    """Claim the request's DVR final logprob repair exactly once."""

    deferred_output = getattr(req, "dvr_deferred_output", None)
    if deferred_output is None:
        return False
    if deferred_output == _DVR_LOGPROB_REPAIR_CLAIMED:
        return False

    req.dvr_deferred_output = _DVR_LOGPROB_REPAIR_CLAIMED
    return True


def should_hold_dvr_non_streaming_logprob_output(
    *,
    req: Any,
    return_logprob: bool,
    require_final_repair: bool = False,
) -> bool:
    """Return whether DVR still owns this non-streaming logprob chunk."""

    deferred_output = getattr(req, "dvr_deferred_output", None)
    should_hold = return_logprob and req.return_logprob and not req.stream
    should_hold = should_hold and deferred_output is not None
    if not should_hold:
        return False
    return not require_final_repair or deferred_output == _DVR_LOGPROB_REPAIR_CLAIMED


def compact_dvr_output_rows(
    *,
    batch: Any,
    output_tokens: torch.Tensor,
    accept_lens,
    tokens_per_req: Optional[int] = None,
    base_seq_lens_cpu: Optional[list[int]] = None,
) -> tuple[Optional[list[int]], list[int], list[list[int]]]:
    """Return accepted output rows in the same order scheduler materializes."""

    if base_seq_lens_cpu is None and getattr(batch, "seq_lens", None) is not None:
        base_seq_lens_cpu = (
            batch.seq_lens_cpu.tolist()
            if getattr(batch, "seq_lens_cpu", None) is not None
            else batch.seq_lens.detach().cpu().tolist()
        )
    if base_seq_lens_cpu is not None:
        base_seq_lens_cpu = [int(seq_len) for seq_len in base_seq_lens_cpu]

    if torch.is_tensor(accept_lens):
        accept_lens_cpu = [int(x) for x in accept_lens.detach().cpu().tolist()]
    else:
        accept_lens_cpu = [int(x) for x in accept_lens]
    token_ids = output_tokens.detach().cpu().reshape(-1).tolist()

    token_ids_per_req = []
    if tokens_per_req is None:
        offset = 0
        for accept_len in accept_lens_cpu:
            end = offset + accept_len
            token_ids_per_req.append([int(x) for x in token_ids[offset:end]])
            offset = end
    else:
        for req_i, accept_len in enumerate(accept_lens_cpu):
            start = req_i * tokens_per_req
            end = start + accept_len
            token_ids_per_req.append([int(x) for x in token_ids[start:end]])

    return base_seq_lens_cpu, accept_lens_cpu, token_ids_per_req


def dvr_compact_output_indices(
    *,
    accept_index: torch.Tensor,
    num_draft_tokens: int,
    max_accept: Optional[int] = None,
) -> torch.Tensor:
    """Return flat predict slots in the order DVR result processing emits."""

    if max_accept is None:
        max_accept = accept_index.shape[1]
    bs = accept_index.shape[0]
    device = accept_index.device
    base = (
        torch.arange(bs, dtype=torch.long, device=device).unsqueeze(1)
        * int(num_draft_tokens)
    )
    offsets = torch.arange(max_accept, dtype=torch.long, device=device).unsqueeze(0)
    return base + offsets


def compact_dvr_accepted_tokens_and_cache_locs(
    *,
    batch: Any,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    num_draft_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return accepted tokens/cache slots in scheduler output order.

    DVR replay must follow the same compact per-request output slices that
    result processing materializes, while ``accept_index`` still names the
    verifier KV/cache rows for the accepted path.
    """

    max_accept = accept_index.shape[1]
    valid_accept = torch.arange(
        max_accept, dtype=torch.long, device=accept_index.device
    ).unsqueeze(0) < accept_lens.to(torch.long).unsqueeze(1)
    compact_predict_indices = dvr_compact_output_indices(
        accept_index=accept_index,
        num_draft_tokens=num_draft_tokens,
        max_accept=max_accept,
    )
    return (
        predict[compact_predict_indices[valid_accept]],
        batch.out_cache_loc[accept_index.clamp_min(0).long()[valid_accept]],
    )


def maybe_filter_running_batch_with_dvr_state(
    *,
    batch: Any,
    future_map: Any,
    enable_overlap: bool,
) -> bool:
    """Filter a DVR running batch with spec-v2 logical-finish state if needed."""

    spec_algorithm = batch.spec_algorithm
    if not enable_overlap:
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


@dataclass
class DVRVerifyInput(EagleVerifyInput):
    """DVR target verify input for both self-draft and EAGLE/MTP draft.

    The verifier shape is EAGLE-compatible in both modes.  ``is_self_draft``
    selects the chain accept/reject sampler and fast self-draft state commit;
    EAGLE/MTP leaves it false and uses target-only sampling.
    """

    is_self_draft: bool = False

    @classmethod
    def from_eagle_verify_input(
        cls, verify_input: EagleVerifyInput, *, is_self_draft: bool = False
    ):
        """Preserve EAGLE draft metadata while adding DVR mode selection."""

        return cls(
            **{
                field.name: getattr(verify_input, field.name)
                for field in fields(EagleVerifyInput)
            },
            is_self_draft=is_self_draft,
        )
