from __future__ import annotations

from contextlib import contextmanager

from sglang.srt.distributed import get_moe_ep_group, get_moe_tp_group
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


_BACKEND_CHILD_ATTRS = (
    "decode_backend",
    "prefill_backend",
    "full_attn_backend",
    "linear_attn_backend",
    "primary",
)
_BACKEND_CHILD_LIST_ATTRS = (
    "attn_backend_list",
    "attn_backends",
    "backends",
    "children",
)


def _iter_attention_backends(attn_backend):
    """Yield a backend tree rooted at model_runner.attn_backend.

    Attention backends may be wrapped by HybridAttnBackend, HybridLinearAttnBackend,
    multi-step speculative containers, or TBO. DVR self-draft decode needs to patch
    the real full-attention backend inside those wrappers, not just the top-level
    object. PDMUX is intentionally excluded because its per-stream decode backends
    live on model_runner.decode_attn_backend_group, outside this object tree, and
    that graph-state path needs separate validation.
    """

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

        for attr_name in _BACKEND_CHILD_ATTRS:
            stack.append(getattr(backend, attr_name, None))

        for attr_name in _BACKEND_CHILD_LIST_ATTRS:
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


def _uses_init_time_deterministic_num_splits(backend) -> bool:
    # FA3 and NSA store deterministic split policy at backend init time. FA4 is
    # deliberately excluded because FA4 CUDA graph does not support num_splits=0.
    return (
        getattr(backend, "fa_impl_ver", None) == 3
        or backend.__class__.__name__ == "NativeSparseAttnBackend"
    )


def _patch_self_draft_decode_backend_defaults(backend, patch_attr):
    """Undo init-time deterministic decode knobs for DVR self-draft only.

    Prefill and verify still run through the deterministic target path. The
    self-draft decode graph is provisional and verified afterwards, so it should
    use the same decode heuristics as normal non-deterministic serving whenever
    the deterministic choice was baked into a backend field at init time.
    """

    if _uses_init_time_deterministic_num_splits(backend):
        patch_attr(backend, "num_splits", 0)

    if hasattr(backend, "decode_split_tile_size"):
        # FlashInfer deterministic mode fixes decode split size and disables
        # CUDA graph KV split. Restore normal decode planning for self-draft.
        patch_attr(backend, "decode_split_tile_size", None)
        patch_attr(backend, "disable_cuda_graph_kv_split", False)


def _custom_all_reduce_is_ready(ca_comm) -> bool:
    return hasattr(ca_comm, "world_size") and (
        hasattr(ca_comm, "_ptr") or hasattr(ca_comm, "obj")
    )


def _ensure_decode_custom_all_reduce_comm(group):
    ca_comm = getattr(group, "ca_comm", None)
    if ca_comm is None and getattr(group, "world_size", 1) > 1:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            dispatch_custom_allreduce,
        )

        try:
            ca_comm = dispatch_custom_allreduce()(
                group=group.cpu_group,
                device=group.device,
            )
        except Exception:
            ca_comm = None
        group.ca_comm = ca_comm

    if ca_comm is None or not _custom_all_reduce_is_ready(ca_comm):
        return None

    if hasattr(ca_comm, "full_nvlink") and not ca_comm.full_nvlink:
        ca_comm.disabled = True
        if hasattr(ca_comm, "original_disabled"):
            ca_comm.original_disabled = True
        return None

    ca_comm.disabled = True
    if hasattr(ca_comm, "original_disabled"):
        ca_comm.original_disabled = True
    return ca_comm


def _iter_decode_custom_all_reduce_comms(model_runner):
    candidate_groups = [getattr(model_runner, "tp_group", None)]
    for get_group in (get_moe_ep_group, get_moe_tp_group):
        try:
            candidate_groups.append(get_group())
        except AssertionError:
            pass

    seen = set()
    for group in candidate_groups:
        if group is None or id(group) in seen:
            continue
        seen.add(id(group))
        ca_comm = _ensure_decode_custom_all_reduce_comm(group)
        if ca_comm is not None:
            yield ca_comm


@contextmanager
def dvr_self_draft_decode_context(
    model_runner,
    *,
    graph_capture: bool = False,
    disable_batch_invariant_ops: bool = False,
    clear_kernel_config_caches: bool = False,
):
    """Temporarily switch the target runner into DVR self-draft decode mode.

    DVR verifies proposals with the deterministic target path afterwards, so the
    provisional self-draft path can use normal decode kernels, avoid
    batch-invariant linear ops, and skip mamba prefix tracking during graph capture.
    All mutations are restored before target verify runs.
    """

    patched_attrs = []
    global_server_args = get_global_server_args()

    def patch_attr(obj, attr_name, value):
        if obj is None or not hasattr(obj, attr_name):
            return
        patched_attrs.append((obj, attr_name, getattr(obj, attr_name)))
        setattr(obj, attr_name, value)

    def skip_init_cuda_graph_state(*args, **kwargs):
        return None

    env_override = envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False)
    env_override.__enter__()

    try:
        if clear_kernel_config_caches:
            _clear_determinism_sensitive_kernel_caches()

        patch_attr(model_runner.server_args, "enable_deterministic_inference", False)
        patch_attr(global_server_args, "enable_deterministic_inference", False)

        for backend in _iter_attention_backends(model_runner.attn_backend):
            patch_attr(backend, "enable_deterministic", False)
            _patch_self_draft_decode_backend_defaults(backend, patch_attr)

        if graph_capture:
            # Custom all-reduce is unsafe for deterministic DVR prefill/verify,
            # but the self-draft decode graph can capture it for speed.
            for ca_comm in _iter_decode_custom_all_reduce_comms(model_runner):
                patch_attr(ca_comm, "disabled", False)
            # The target runner already initialized attention backend graph
            # buffers for TARGET_VERIFY. Capture DVR self-draft as ordinary
            # DECODE without reinitializing those shared buffers.
            patch_attr(model_runner, "spec_algorithm", SpeculativeAlgorithm.NONE)
            patch_attr(
                model_runner.attn_backend,
                "init_cuda_graph_state",
                skip_init_cuda_graph_state,
            )
            # Provisional self-draft tokens must not update mamba prefix-cache
            # tracking slots. DVR commits verified recurrent state after verify.
            patch_attr(
                model_runner.server_args, "mamba_scheduler_strategy", "no_buffer"
            )
            patch_attr(global_server_args, "mamba_scheduler_strategy", "no_buffer")

        with _maybe_disable_batch_invariant_ops(disable_batch_invariant_ops):
            yield
    finally:
        for obj, attr_name, original_value in reversed(patched_attrs):
            setattr(obj, attr_name, original_value)
        if clear_kernel_config_caches:
            _clear_determinism_sensitive_kernel_caches()
        env_override.__exit__(None, None, None)


class DVRDraftDecodeCudaGraphRunner:
    """CUDA graph runner for DVR self-draft decode.

    DVR's normal target-model graph runner captures TARGET_VERIFY graphs. The
    self-draft path still runs ordinary one-token DECODE steps, so it needs a
    separate decode graph runner instead of reusing the target-verify graph.
    """

    def __init__(self, dvr_worker):
        self.dvr_worker = dvr_worker
        model_runner = dvr_worker.model_runner
        with dvr_self_draft_decode_context(
            model_runner,
            graph_capture=True,
            disable_batch_invariant_ops=True,
            clear_kernel_config_caches=True,
        ):
            self.runner = CudaGraphRunner(model_runner)

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        return self.runner.can_run(forward_batch)

    def replay(self, forward_batch: ForwardBatch):
        with dvr_self_draft_decode_context(self.dvr_worker.model_runner):
            return self.runner.replay(forward_batch)
