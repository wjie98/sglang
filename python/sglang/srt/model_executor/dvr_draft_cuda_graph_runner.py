from __future__ import annotations

from contextlib import contextmanager

from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _iter_attention_backends(attn_backend):
    if attn_backend is None:
        return

    seen = set()
    stack = [attn_backend]
    while stack:
        backend = stack.pop()
        if backend is None or id(backend) in seen:
            continue
        seen.add(id(backend))
        yield backend

        for attr_name in ("decode_backend", "prefill_backend", "linear_attn_backend"):
            stack.append(getattr(backend, attr_name, None))

        for attr_name in ("attn_backends", "backends"):
            for child in getattr(backend, attr_name, None) or ():
                stack.append(child)


@contextmanager
def _maybe_disable_batch_invariant_ops(disable: bool):
    if not disable:
        yield
        return

    from sglang.srt.batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )

    was_enabled = is_batch_invariant_mode_enabled()
    if was_enabled:
        disable_batch_invariant_mode()
    try:
        yield
    finally:
        if was_enabled:
            enable_batch_invariant_mode()


def _clear_determinism_sensitive_kernel_caches():
    # These caches read global deterministic or batch-invariant state, but their
    # cache keys do not include that state. Clear them around self-draft graph
    # capture/fallback so target-verify deterministic choices cannot leak into
    # the non-deterministic draft path, and vice versa.
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
        get_moe_configs,
    )
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
        should_enable_swap_ab,
    )

    get_moe_configs.cache_clear()
    should_enable_swap_ab.cache_clear()


@contextmanager
def dvr_self_draft_nondeterministic_decode(
    model_runner,
    *,
    disable_batch_invariant_ops: bool = False,
    clear_kernel_config_caches: bool = False,
):
    """Run DVR self-draft forward with normal non-deterministic decode kernels.

    DVR verifies proposed draft tokens with the target path afterwards, so only
    TARGET_VERIFY needs batch-invariant deterministic kernels. Keeping this
    override local to self-draft capture/replay preserves the deterministic
    verifier while allowing the proposal step to use the faster decode kernels.
    """

    patched_attrs = []

    def patch_attr(obj, attr_name, value):
        if obj is None or not hasattr(obj, attr_name):
            return
        patched_attrs.append((obj, attr_name, getattr(obj, attr_name)))
        setattr(obj, attr_name, value)

    env_override = envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False)
    env_override.__enter__()

    try:
        if clear_kernel_config_caches:
            _clear_determinism_sensitive_kernel_caches()

        patch_attr(model_runner.server_args, "enable_deterministic_inference", False)
        patch_attr(get_global_server_args(), "enable_deterministic_inference", False)

        for backend in _iter_attention_backends(model_runner.attn_backend):
            patch_attr(backend, "enable_deterministic", False)

        with _maybe_disable_batch_invariant_ops(disable_batch_invariant_ops):
            yield
    finally:
        for obj, attr_name, original_value in reversed(patched_attrs):
            setattr(obj, attr_name, original_value)
        if clear_kernel_config_caches:
            _clear_determinism_sensitive_kernel_caches()
        env_override.__exit__(None, None, None)


@contextmanager
def _dvr_draft_decode_graph_capture(model_runner):
    original_spec_algorithm = model_runner.spec_algorithm
    original_init_cuda_graph_state = model_runner.attn_backend.init_cuda_graph_state
    original_mamba_scheduler_strategy = (
        model_runner.server_args.mamba_scheduler_strategy
    )
    global_server_args = get_global_server_args()
    original_global_mamba_scheduler_strategy = (
        global_server_args.mamba_scheduler_strategy
    )

    def skip_init_cuda_graph_state(*args, **kwargs):
        return None

    # The target model's main runner has already initialized attention backend
    # graph buffers for the larger TARGET_VERIFY graph. Capture the DVR self
    # draft graph as an ordinary decode graph without reinitializing those
    # shared backend buffers.
    model_runner.spec_algorithm = SpeculativeAlgorithm.NONE
    model_runner.attn_backend.init_cuda_graph_state = skip_init_cuda_graph_state
    # DVR self-draft tokens are provisional and must not update mamba prefix-cache
    # tracking slots. The verified state is committed by DVR after target verify.
    model_runner.server_args.mamba_scheduler_strategy = "no_buffer"
    global_server_args.mamba_scheduler_strategy = "no_buffer"
    try:
        yield
    finally:
        global_server_args.mamba_scheduler_strategy = (
            original_global_mamba_scheduler_strategy
        )
        model_runner.server_args.mamba_scheduler_strategy = (
            original_mamba_scheduler_strategy
        )
        model_runner.attn_backend.init_cuda_graph_state = original_init_cuda_graph_state
        model_runner.spec_algorithm = original_spec_algorithm


class DVRDraftDecodeCudaGraphRunner:
    """CUDA graph runner for DVR self-draft decode.

    DVR's normal target-model graph runner captures TARGET_VERIFY graphs. The
    self-draft path still runs ordinary one-token DECODE steps, so it needs a
    separate decode graph runner instead of reusing the target-verify graph.
    """

    def __init__(self, dvr_worker):
        self.dvr_worker = dvr_worker
        model_runner = dvr_worker.model_runner
        with _dvr_draft_decode_graph_capture(
            model_runner
        ), dvr_self_draft_nondeterministic_decode(
            model_runner,
            disable_batch_invariant_ops=True,
            clear_kernel_config_caches=True,
        ):
            self.runner = CudaGraphRunner(model_runner)

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        return self.runner.can_run(forward_batch)

    def replay(self, forward_batch: ForwardBatch):
        with dvr_self_draft_nondeterministic_decode(self.dvr_worker.model_runner):
            return self.runner.replay(forward_batch)
