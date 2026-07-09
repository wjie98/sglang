from __future__ import annotations

import weakref
from dataclasses import dataclass, fields
from typing import Any, Optional

import torch

from sglang.srt.speculative.eagle_info import EagleVerifyInput


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
    the worker returns.  DVR carries the final rows here and applies them only
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
        return DVRFinalLogprobRepair(
            output_ids=token_stream[:output_len],
            output_logprobs=[float(value) for value in repair_logprobs],
        )


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
