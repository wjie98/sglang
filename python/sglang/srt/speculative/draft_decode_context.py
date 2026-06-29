from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def draft_decode_performance_context(
    model_runner,
    *,
    graph_capture: bool = False,
    clear_kernel_config_caches: bool = False,
    attn_backends=(),
):
    """Apply algorithm-specific performance policy for provisional draft decode.

    Draft decode is always verified later, so algorithms such as DVR can use
    normal performance-first decode kernels without making the target prefill or
    verify path non-deterministic. Keep the public helper in speculative code so
    generic EAGLE workers do not import DVR graph-runner internals directly.
    """

    if model_runner.spec_algorithm.is_decode_verify_rollback_eagle():
        from sglang.srt.model_executor.dvr_draft_cuda_graph_runner import (
            dvr_eagle_draft_decode_context,
        )

        with dvr_eagle_draft_decode_context(
            model_runner,
            graph_capture=graph_capture,
            clear_kernel_config_caches=clear_kernel_config_caches,
            attn_backends=attn_backends,
        ):
            yield
        return

    yield
