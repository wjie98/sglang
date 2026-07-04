from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SpeculativeOutputPolicyState:
    """Request-local output policy state owned by speculative algorithms."""

    defer_non_streaming_logprob_output: bool = False
    expect_final_logprob_repair: bool = False
    final_logprob_repair_planned: bool = False


def _get_req_output_policy(req: Any, *, create: bool = False):
    policy = getattr(req, "speculative_output_policy", None)
    if policy is None and create:
        policy = SpeculativeOutputPolicyState()
        setattr(req, "speculative_output_policy", policy)
    return policy


def defer_req_non_streaming_logprob_output(req: Any) -> None:
    """Defer a non-streaming logprob chunk until final response repair."""

    _get_req_output_policy(req, create=True).defer_non_streaming_logprob_output = True


def allow_req_non_streaming_logprob_output(req: Any) -> None:
    """Allow a previously deferred non-streaming logprob response to emit."""

    policy = _get_req_output_policy(req, create=True)
    policy.defer_non_streaming_logprob_output = False
    policy.expect_final_logprob_repair = False


def try_expect_req_final_logprob_repair(req: Any) -> bool:
    """Claim the request's final logprob repair exactly once."""

    policy = _get_req_output_policy(req, create=True)
    if policy.final_logprob_repair_planned:
        return False

    policy.expect_final_logprob_repair = True
    policy.final_logprob_repair_planned = True
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

    policy = _get_req_output_policy(req)
    should_hold = return_logprob and req.return_logprob and not req.stream
    should_hold = (
        should_hold
        and policy is not None
        and policy.defer_non_streaming_logprob_output
    )
    if not should_hold:
        return False
    return not require_final_repair or policy.expect_final_logprob_repair
