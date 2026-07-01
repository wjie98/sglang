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


def should_hold_non_streaming_logprob_output(
    *,
    req: Any,
    return_logprob: bool,
    require_final_repair: bool = False,
) -> bool:
    """Return whether exact-logprob repair still owns this response.

    This is intentionally request-level state, not a DVR/EAGLE branch in the
    streamer.  The producing worker owns when to set and clear the flag.
    """

    should_hold = (
        return_logprob
        and req.return_logprob
        and not req.stream
        and getattr(req, _DEFER_NON_STREAMING_LOGPROB_OUTPUT_ATTR, False)
    )
    if not should_hold:
        return False
    return not require_final_repair or getattr(
        req, _EXPECT_FINAL_LOGPROB_REPAIR_ATTR, False
    )
