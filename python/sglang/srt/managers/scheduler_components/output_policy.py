from __future__ import annotations

from typing import Any


def defer_req_non_streaming_logprob_output(req: Any) -> None:
    """Defer a non-streaming logprob chunk until a producer repairs it."""

    req.defer_non_streaming_logprob_output = True


def allow_req_non_streaming_logprob_output(req: Any) -> None:
    """Allow a previously deferred non-streaming logprob response to emit."""

    req.defer_non_streaming_logprob_output = False


def try_claim_req_final_logprob_repair(req: Any) -> bool:
    """Claim the request's final logprob repair exactly once."""

    if getattr(req, "final_logprob_repair_claimed", False):
        return False

    req.final_logprob_repair_claimed = True
    return True


def should_hold_non_streaming_logprob_output(
    *,
    req: Any,
    return_logprob: bool,
    require_final_repair: bool = False,
) -> bool:
    """Return whether a producer still owns this non-streaming logprob chunk."""

    should_hold = (
        return_logprob
        and req.return_logprob
        and not req.stream
        and getattr(req, "defer_non_streaming_logprob_output", False)
    )
    if not should_hold:
        return False
    return not require_final_repair or getattr(
        req, "final_logprob_repair_claimed", False
    )
