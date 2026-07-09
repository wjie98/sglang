from __future__ import annotations

from contextlib import contextmanager

from sglang.srt.distributed import get_moe_ep_group, get_moe_tp_group
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


_ATTN_BACKEND_CHILD_ATTRS = (
    "decode_backend",
    "prefill_backend",
    "full_attn_backend",
    "linear_attn_backend",
    "primary",
)
_ATTN_BACKEND_CHILD_LIST_ATTRS = (
    "attn_backend_list",
    "attn_backends",
    "backends",
    "children",
)


def iter_dvr_attention_backends(attn_backend):
    """Yield every backend wrapper that may carry decode determinism state."""

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

        for attr_name in _ATTN_BACKEND_CHILD_ATTRS:
            stack.append(getattr(backend, attr_name, None))

        for attr_name in _ATTN_BACKEND_CHILD_LIST_ATTRS:
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
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_moe_configs,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
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


def _skip_init_cuda_graph_state(*args, **kwargs):
    return None


def _min_seq_len_cpu(forward_batch: ForwardBatch) -> int:
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if seq_lens_cpu is not None:
        return int(seq_lens_cpu.min().item())
    return int(forward_batch.seq_lens.min().item())


def dvr_self_draft_graph_skip_reason(forward_batch: ForwardBatch) -> str | None:
    """Return the local reason a self-draft decode step cannot use its graph."""

    # The first self-draft decode after a one-token prompt reaches the graph as
    # seq_len=2 (prompt token + anchor token). On GDN models that graph replay
    # can hit an illegal access in the packed decode state-input window; eager
    # decode handles the same boundary and the next draft step can return to the
    # graph.
    if _min_seq_len_cpu(forward_batch) <= 2:
        return "seq_len<=2 initial GDN state-input graph boundary"
    return None


def _patch_draft_determinism_flags(
    model_runner,
    global_server_args,
    patch_attr,
    *,
    disable_model_runner_graph: bool,
):
    patch_attr(model_runner.server_args, "enable_deterministic_inference", False)
    patch_attr(global_server_args, "enable_deterministic_inference", False)
    if disable_model_runner_graph:
        patch_attr(model_runner, "decode_cuda_graph_runner", None)
        patch_attr(model_runner, "graph_runner", None)


def _patch_draft_decode_backend_tree(model_runner, extra_attn_backends, patch_attr):
    seen_backend_roots = set()
    for backend_root in (model_runner.attn_backend, *(extra_attn_backends or ())):
        if backend_root is None or id(backend_root) in seen_backend_roots:
            continue
        seen_backend_roots.add(id(backend_root))
        for backend in iter_dvr_attention_backends(backend_root):
            patch_attr(backend, "enable_deterministic", False)
            _patch_self_draft_decode_backend_defaults(backend, patch_attr)


def _patch_graph_capture_extras(
    model_runner,
    patch_attr,
    *,
    self_draft_graph: bool,
):
    # Custom all-reduce is unsafe for deterministic DVR prefill/verify, but the
    # self-draft decode graph can capture it for speed.
    for ca_comm in _iter_decode_custom_all_reduce_comms(model_runner):
        patch_attr(ca_comm, "disabled", False)
    if self_draft_graph:
        # The target runner already initialized attention backend graph buffers
        # for TARGET_VERIFY. Capture DVR self-draft as ordinary DECODE without
        # reinitializing those shared buffers.
        patch_attr(model_runner, "spec_algorithm", SpeculativeAlgorithm.NONE)
        patch_attr(
            model_runner.attn_backend,
            "init_cuda_graph_state",
            _skip_init_cuda_graph_state,
        )


def _patch_draft_mamba_tracking(model_runner, global_server_args, patch_attr):
    # Provisional draft tokens must not update mamba prefix-cache tracking
    # slots. DVR commits verified recurrent state after target verify.
    patch_attr(model_runner.server_args, "mamba_radix_cache_strategy", "no_buffer")
    patch_attr(global_server_args, "mamba_radix_cache_strategy", "no_buffer")


def _assert_draft_decode_performance_state(model_runner, extra_attn_backends):
    """Fail fast if deterministic target settings leak into draft decode."""

    # These checks deliberately run on every DVR draft decode context entry.
    # The draft path is only provisional, so running deterministic target
    # kernels here is pure overhead and was the source of earlier throughput
    # regressions. Raising early is preferable to silently falling back to the
    # deterministic verify profile.
    if getattr(model_runner.server_args, "enable_deterministic_inference", False):
        raise RuntimeError("DVR draft decode must run with deterministic inference off.")
    if envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get():
        raise RuntimeError("DVR draft decode env still enables deterministic inference.")

    for backend_root in (model_runner.attn_backend, *(extra_attn_backends or ())):
        for backend in iter_dvr_attention_backends(backend_root):
            if getattr(backend, "enable_deterministic", False):
                raise RuntimeError(
                    "DVR draft decode backend still has enable_deterministic=True."
                )
            if (
                _uses_init_time_deterministic_num_splits(backend)
                and hasattr(backend, "num_splits")
                and getattr(backend, "num_splits") != 0
            ):
                raise RuntimeError(
                    "DVR draft decode backend still uses deterministic num_splits."
                )
            if hasattr(backend, "decode_split_tile_size") and getattr(
                backend, "decode_split_tile_size"
            ) is not None:
                raise RuntimeError(
                    "DVR draft decode backend still uses deterministic split tile size."
                )
            if getattr(backend, "disable_cuda_graph_kv_split", False):
                raise RuntimeError(
                    "DVR draft decode backend still disables CUDA graph KV split."
                )


@contextmanager
def _dvr_draft_decode_context(
    model_runner,
    *,
    graph_capture: bool = False,
    self_draft_graph_capture: bool = False,
    disable_model_runner_graph: bool = False,
    disable_batch_invariant_ops: bool = False,
    clear_kernel_config_caches: bool = False,
    extra_attn_backends=(),
):
    """Temporarily switch a DVR draft runner into performance-first decode mode.

    This is intentionally private: callers should use one of the fixed semantic
    wrappers below instead of composing graph/determinism/mamba knobs manually.
    """

    patched_attrs = []
    global_server_args = get_global_server_args()

    def patch_attr(obj, attr_name, value):
        if obj is None or not hasattr(obj, attr_name):
            return
        patched_attrs.append((obj, attr_name, getattr(obj, attr_name)))
        setattr(obj, attr_name, value)

    with envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False):
        try:
            if clear_kernel_config_caches:
                _clear_determinism_sensitive_kernel_caches()

            _patch_draft_determinism_flags(
                model_runner,
                global_server_args,
                patch_attr,
                disable_model_runner_graph=disable_model_runner_graph,
            )
            _patch_draft_decode_backend_tree(
                model_runner, extra_attn_backends, patch_attr
            )

            if graph_capture:
                _patch_graph_capture_extras(
                    model_runner,
                    patch_attr,
                    self_draft_graph=self_draft_graph_capture,
                )

            _patch_draft_mamba_tracking(model_runner, global_server_args, patch_attr)

            _assert_draft_decode_performance_state(model_runner, extra_attn_backends)

            with _maybe_disable_batch_invariant_ops(disable_batch_invariant_ops):
                yield
        finally:
            for obj, attr_name, original_value in reversed(patched_attrs):
                setattr(obj, attr_name, original_value)
            if clear_kernel_config_caches:
                _clear_determinism_sensitive_kernel_caches()


def dvr_self_draft_graph_capture_context(model_runner):
    """Capture the dedicated DVR self-draft decode CUDA graph."""

    return _dvr_draft_decode_context(
        model_runner,
        graph_capture=True,
        self_draft_graph_capture=True,
        disable_batch_invariant_ops=True,
        clear_kernel_config_caches=True,
    )


def dvr_self_draft_graph_replay_context(model_runner):
    """Replay the DVR self-draft graph without mamba tracking side effects."""

    return _dvr_draft_decode_context(model_runner)


def dvr_self_draft_eager_context(model_runner):
    """Run the explicit DVR self-draft eager fallback for graph-unsafe edges."""

    return _dvr_draft_decode_context(
        model_runner,
        disable_model_runner_graph=True,
        disable_batch_invariant_ops=True,
        clear_kernel_config_caches=True,
    )


class DVRTargetVerifyCudaGraphRunner(DecodeCudaGraphRunner):
    """Target-verify graph runner for DVR self-draft and DVR-EAGLE.

    DVR target verify uses the standard EAGLE verifier shape, but the graph
    metadata must follow DVR's causal verifier and GDN state-input windows. Keep
    those rules on the graph runner instead of attaching execution hooks to the
    spec_info data object.
    """

    skip_prefill_only_deterministic_for_capture = False

    def __init__(
        self,
        model_runner,
        *,
        skip_prefill_only_deterministic_for_capture: bool = False,
        **kwargs,
    ):
        self.skip_prefill_only_deterministic_for_capture = (
            skip_prefill_only_deterministic_for_capture
        )
        super().__init__(model_runner, **kwargs)

    def get_spec_info(self, num_tokens: int):
        spec_info = super().get_spec_info(num_tokens)
        if spec_info is None:
            return None
        if self.model_runner.spec_algorithm.is_dvr_self_draft():
            from sglang.srt.speculative.dvr_info import DVRVerifyInput

            return DVRVerifyInput.from_eagle_verify_input(
                spec_info, is_self_draft=True
            )
        if self.model_runner.spec_algorithm.is_dvr_eagle():
            from sglang.srt.speculative.dvr_info import DVRVerifyInput

            return DVRVerifyInput.from_eagle_verify_input(spec_info)
        return spec_info

    def _fill_replay_side_buffers(
        self, forward_batch: ForwardBatch, raw_num_token: int
    ) -> None:
        if not self.capture_forward_mode.is_target_verify():
            return
        # The generic num_token_non_padded slot is enabled only for EP. GDN DVR
        # verify also needs the raw token count to mask padded graph rows.
        self.buffers.num_token_non_padded.fill_(raw_num_token)

    @contextmanager
    def _forward_metadata_out_graph_context(
        self,
        *,
        forward_batch: ForwardBatch,
        attn_backend,
        forward_mode,
        fallback_custom_mask=None,
    ):
        del fallback_custom_mask
        spec_info = getattr(forward_batch, "spec_info", None)
        old_custom_mask = getattr(spec_info, "custom_mask", None)
        should_clear_custom_mask = (
            self.model_runner.spec_algorithm.is_dvr()
            and forward_mode.is_target_verify()
            and spec_info is not None
        )
        if should_clear_custom_mask:
            spec_info.custom_mask = None
        try:
            yield
        finally:
            if not should_clear_custom_mask:
                return
            spec_info.custom_mask = old_custom_mask
            for backend in iter_dvr_attention_backends(attn_backend):
                metadata = getattr(backend, "forward_metadata", None)
                if metadata is None:
                    continue
                if hasattr(metadata, "custom_mask"):
                    metadata.custom_mask = None
                if hasattr(metadata, "mask_indptr"):
                    metadata.mask_indptr = None


class DVRDraftDecodeCudaGraphRunner:
    """CUDA graph runner for DVR self-draft decode.

    DVR's normal target-model graph runner captures TARGET_VERIFY graphs. The
    self-draft path still runs ordinary one-token DECODE steps, so it needs a
    separate decode graph runner instead of reusing the target-verify graph.
    """

    def __init__(self, dvr_worker):
        self.dvr_worker = dvr_worker
        model_runner = dvr_worker.model_runner
        with dvr_self_draft_graph_capture_context(model_runner):
            self.runner = DecodeCudaGraphRunner(model_runner)

    @property
    def capture_bs(self):
        return self.runner.capture_bs

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        if dvr_self_draft_graph_skip_reason(forward_batch) is not None:
            return False
        return self.runner.can_run_graph(forward_batch)

    def replay(self, forward_batch: ForwardBatch):
        with dvr_self_draft_graph_replay_context(self.dvr_worker.model_runner):
            return self.runner.execute(forward_batch)
