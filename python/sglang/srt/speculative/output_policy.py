from __future__ import annotations

from typing import Any

_DEFER_NON_STREAMING_LOGPROB_OUTPUT_ATTR = "_defer_non_streaming_logprob_output"


def defer_req_non_streaming_logprob_output(req: Any) -> None:
    """Defer a non-streaming logprob chunk until final response repair."""

    setattr(req, _DEFER_NON_STREAMING_LOGPROB_OUTPUT_ATTR, True)


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

    if (
        return_logprob
        and req.return_logprob
        and not req.stream
        and getattr(req, _DEFER_NON_STREAMING_LOGPROB_OUTPUT_ATTR, False)
    ):
        return False
    return len(req.output_ids) % force_stream_interval == 0
