from __future__ import annotations

from typing import Any


def defer_dvr_non_streaming_logprob_output(req: Any) -> None:
    """Hold DVR non-streaming logprob output until final repair overwrites it."""

    req.dvr_defer_non_streaming_logprob_output = True


def allow_dvr_non_streaming_logprob_output(req: Any) -> None:
    """Release a DVR non-streaming logprob response after final repair."""

    req.dvr_defer_non_streaming_logprob_output = False


def try_claim_dvr_final_logprob_repair(req: Any) -> bool:
    """Claim the request's DVR final logprob repair exactly once."""

    if getattr(req, "dvr_final_logprob_repair_claimed", False):
        return False

    req.dvr_final_logprob_repair_claimed = True
    return True


def should_hold_dvr_non_streaming_logprob_output(
    *,
    req: Any,
    return_logprob: bool,
    require_final_repair: bool = False,
) -> bool:
    """Return whether DVR still owns this non-streaming logprob chunk."""

    should_hold = (
        return_logprob
        and req.return_logprob
        and not req.stream
        and getattr(req, "dvr_defer_non_streaming_logprob_output", False)
    )
    if not should_hold:
        return False
    return not require_final_repair or getattr(
        req, "dvr_final_logprob_repair_claimed", False
    )
