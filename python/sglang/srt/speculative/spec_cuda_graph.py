from __future__ import annotations

from contextlib import nullcontext
from typing import Any


def prepare_spec_cuda_graph_replay_buffers(
    spec_info: Any, graph_runner: Any, raw_num_token: int
) -> None:
    """Run an optional spec-input replay-buffer hook.

    Most speculative inputs follow the normal CUDA graph path.  DVR target
    verify is the only current user that needs to copy graph-owned metadata
    before replay, so keep the default as an absent hook instead of extending
    the base SpecInput contract.
    """

    hook = getattr(spec_info, "prepare_cuda_graph_replay_buffers", None)
    if hook is not None:
        hook(graph_runner, raw_num_token)


def spec_cuda_graph_metadata_context(
    spec_info: Any,
    *,
    model_runner: Any,
    attn_backend: Any,
    forward_mode: Any,
    fallback_custom_mask: Any = None,
):
    """Return an optional metadata-build context for graph-only invariants."""

    hook = getattr(spec_info, "cuda_graph_metadata_context", None)
    if hook is None:
        return nullcontext()
    return hook(
        model_runner=model_runner,
        attn_backend=attn_backend,
        forward_mode=forward_mode,
        fallback_custom_mask=fallback_custom_mask,
    )
