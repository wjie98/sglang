from __future__ import annotations

from typing import Any


def defer_req_non_streaming_logprob_output(req: Any) -> None:
    """Defer a non-streaming logprob chunk until final response repair."""

    req.dvr_defer_non_streaming_logprob_output = True


def allow_req_non_streaming_logprob_output(req: Any) -> None:
    """Allow a previously deferred non-streaming logprob response to emit."""

    req.dvr_defer_non_streaming_logprob_output = False


def try_claim_req_final_logprob_repair(req: Any) -> bool:
    """Claim the request's final logprob repair exactly once."""

    if getattr(req, "dvr_final_logprob_repair_claimed", False):
        return False

    req.dvr_final_logprob_repair_claimed = True
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

    should_hold = return_logprob and req.return_logprob and not req.stream
    should_hold = (
        should_hold
        and getattr(req, "dvr_defer_non_streaming_logprob_output", False)
    )
    if not should_hold:
        return False
    return not require_final_repair or getattr(
        req, "dvr_final_logprob_repair_claimed", False
    )
