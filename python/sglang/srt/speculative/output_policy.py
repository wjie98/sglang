from __future__ import annotations

from typing import Any


_DEFER_NON_STREAMING_LOGPROB_OUTPUT_ATTR = "_defer_non_streaming_logprob_output"
_EXPECT_FINAL_LOGPROB_REPAIR_ATTR = "_expect_final_logprob_repair"
_FINAL_LOGPROB_REPAIR_PLANNED_ATTR = "_final_logprob_repair_planned"


def defer_req_non_streaming_logprob_output(req: Any) -> None:
    """Defer a non-streaming logprob chunk until final response repair."""

    setattr(req, _DEFER_NON_STREAMING_LOGPROB_OUTPUT_ATTR, True)


def allow_req_non_streaming_logprob_output(req: Any) -> None:
    """Allow a previously deferred non-streaming logprob response to emit."""

    setattr(req, _DEFER_NON_STREAMING_LOGPROB_OUTPUT_ATTR, False)
    setattr(req, _EXPECT_FINAL_LOGPROB_REPAIR_ATTR, False)


def try_expect_req_final_logprob_repair(req: Any) -> bool:
    """Claim the request's final logprob repair exactly once."""

    if getattr(req, _FINAL_LOGPROB_REPAIR_PLANNED_ATTR, False):
        return False

    setattr(req, _EXPECT_FINAL_LOGPROB_REPAIR_ATTR, True)
    setattr(req, _FINAL_LOGPROB_REPAIR_PLANNED_ATTR, True)
    return True


def should_defer_non_streaming_logprob_output(
    *,
    req: Any,
    return_logprob: bool,
) -> bool:
    """Return whether exact-logprob repair still owns this response."""

    return (
        return_logprob
        and req.return_logprob
        and not req.stream
        and getattr(req, _DEFER_NON_STREAMING_LOGPROB_OUTPUT_ATTR, False)
    )


def should_defer_finished_non_streaming_logprob_output(
    *,
    req: Any,
    return_logprob: bool,
) -> bool:
    """Return whether a finished response is waiting for final repair rows."""

    return should_defer_non_streaming_logprob_output(
        req=req,
        return_logprob=return_logprob,
    ) and getattr(req, _EXPECT_FINAL_LOGPROB_REPAIR_ATTR, False)


def should_emit_non_streaming_output_chunk(
    *,
    req: Any,
    return_logprob: bool,
    force_stream_interval: int,
) -> bool:
    """Return whether a non-streaming generation chunk can be sent now.

    Some speculative algorithms repair exact logprobs at the final response.
    They set a request-level defer flag; the streamer only follows that generic
    flag and does not depend on an algorithm-specific module.
    """

    if should_defer_non_streaming_logprob_output(
        req=req,
        return_logprob=return_logprob,
    ):
        return False
    return len(req.output_ids) % force_stream_interval == 0
