from __future__ import annotations

from contextlib import contextmanager

import torch

from sglang.srt.distributed import get_moe_ep_group, get_moe_tp_group
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_bool_env_var


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
    # capture/replay so target-verify deterministic choices cannot leak into
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


def _patch_draft_decode_backend_defaults(backend, server_args, patch_attr):
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

    if backend.__class__.__name__ == "TritonAttnBackend":
        split_tile_size = server_args.triton_attention_split_tile_size
        normal_max_splits = server_args.triton_attention_num_kv_splits
        if split_tile_size is not None:
            normal_max_splits = (
                backend.max_context_len + split_tile_size - 1
            ) // split_tile_size
        # The kernel requires max_kv_splits to exactly match its scratch view.
        # Reuse the deterministic allocation through a narrower view instead of
        # allocating a second graph workspace for draft decode.
        normal_max_splits = min(backend.max_kv_splits, normal_max_splits)
        for name in (
            "cuda_graph_attn_logits",
            "cuda_graph_attn_lse",
            "cuda_graph_swa_attn_logits",
        ):
            buffer = getattr(backend, name, None)
            if buffer is not None:
                patch_attr(backend, name, buffer[:, :, :normal_max_splits])
        for name in (
            "cuda_graph_num_kv_splits",
            "cuda_graph_window_num_kv_splits",
        ):
            buffer = getattr(backend, name, None)
            if buffer is not None:
                draft_name = f"_dvr_draft_{name}"
                draft_buffer = getattr(backend, draft_name, None)
                if draft_buffer is None:
                    draft_buffer = torch.full_like(buffer, normal_max_splits)
                    setattr(backend, draft_name, draft_buffer)
                patch_attr(backend, name, draft_buffer)
        patch_attr(
            backend,
            "max_kv_splits",
            normal_max_splits,
        )
        patch_attr(backend, "split_tile_size", split_tile_size)
        patch_attr(
            backend,
            "static_kv_splits",
            get_bool_env_var(
                "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
            ),
        )


def _ensure_decode_custom_all_reduce_comm(group):
    ca_comm = getattr(group, "ca_comm", None)
    if ca_comm is None and getattr(group, "world_size", 1) > 1:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            dispatch_custom_allreduce,
        )

        custom_allreduce_cls = dispatch_custom_allreduce(
            group=group.cpu_group,
            device=group.device,
        )
        ca_comm = custom_allreduce_cls(
            group=group.cpu_group,
            device=group.device,
        )
        group.ca_comm = ca_comm

    if ca_comm is None or not hasattr(ca_comm, "world_size") or not (
        hasattr(ca_comm, "_ptr") or hasattr(ca_comm, "obj")
    ):
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


def dvr_self_draft_graph_block_reason(forward_batch: ForwardBatch) -> str | None:
    """Return why a self-draft decode step is unsupported for graph replay."""

    # The first self-draft decode after a one-token prompt reaches the graph as
    # seq_len=2 (prompt token + anchor token).  Normal chat-template prompts are
    # much longer, so keep this synthetic edge out of the DVR serving path
    # instead of hiding it behind a slow eager branch.
    if _min_seq_len_cpu(forward_batch) <= 2:
        return "seq_len<=2 initial GDN state-input graph boundary"
    return None


@contextmanager
def _dvr_draft_decode_context(
    model_runner,
    *,
    graph_capture: bool = False,
    self_draft_graph_capture: bool = False,
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

            patch_attr(model_runner.server_args, "enable_deterministic_inference", False)
            patch_attr(global_server_args, "enable_deterministic_inference", False)

            seen_backend_roots = set()
            for backend_root in (
                model_runner.attn_backend,
                *(extra_attn_backends or ()),
            ):
                if backend_root is None or id(backend_root) in seen_backend_roots:
                    continue
                seen_backend_roots.add(id(backend_root))
                for backend in iter_dvr_attention_backends(backend_root):
                    patch_attr(backend, "enable_deterministic", False)
                    _patch_draft_decode_backend_defaults(
                        backend, model_runner.server_args, patch_attr
                    )

            if graph_capture:
                # Deterministic target prefill/verify keeps custom all-reduce
                # disabled. Only capture it into provisional draft graphs.
                for ca_comm in _iter_decode_custom_all_reduce_comms(model_runner):
                    patch_attr(ca_comm, "disabled", False)
                if self_draft_graph_capture:
                    # Target graph buffers are already initialized for
                    # TARGET_VERIFY; capture self draft as ordinary DECODE.
                    patch_attr(
                        model_runner, "spec_algorithm", SpeculativeAlgorithm.NONE
                    )
                    patch_attr(
                        model_runner.attn_backend,
                        "init_cuda_graph_state",
                        _skip_init_cuda_graph_state,
                    )

            # Provisional draft tokens must not update mamba prefix-cache
            # tracking slots. DVR commits verified recurrent state after target
            # verify.
            patch_attr(
                model_runner.server_args, "mamba_radix_cache_strategy", "no_buffer"
            )
            patch_attr(global_server_args, "mamba_radix_cache_strategy", "no_buffer")

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


def dvr_draft_graph_capture_context(model_runner, *, extra_attn_backends=()):
    """Capture a DVR draft-model graph with performance-first decode settings."""

    return _dvr_draft_decode_context(
        model_runner,
        graph_capture=True,
        disable_batch_invariant_ops=True,
        clear_kernel_config_caches=True,
        extra_attn_backends=extra_attn_backends,
    )


def dvr_draft_graph_replay_context(model_runner, *, extra_attn_backends=()):
    """Run one complete DVR draft phase without target determinism settings."""

    return _dvr_draft_decode_context(
        model_runner,
        extra_attn_backends=extra_attn_backends,
    )


class DVRTargetVerifyCudaGraphRunner(DecodeCudaGraphRunner):
    """Target-verify graph runner for DVR self-draft and DVR-EAGLE.

    DVR target verify uses the standard EAGLE verifier shape, but the graph
    metadata must follow DVR's causal verifier and GDN state-input windows. Keep
    those rules on the graph runner instead of attaching execution hooks to the
    spec_info data object.
    """

    def __init__(
        self,
        model_runner,
        *,
        dvr_target_verify_cuda_graph: bool = False,
        **kwargs,
    ):
        had_override = hasattr(model_runner, "enable_dvr_target_verify_cuda_graph")
        old_override = getattr(
            model_runner, "enable_dvr_target_verify_cuda_graph", False
        )
        model_runner.enable_dvr_target_verify_cuda_graph = (
            dvr_target_verify_cuda_graph
        )
        try:
            super().__init__(model_runner, **kwargs)
        finally:
            if had_override:
                model_runner.enable_dvr_target_verify_cuda_graph = old_override
            else:
                delattr(model_runner, "enable_dvr_target_verify_cuda_graph")

    def get_spec_info(self, num_tokens: int):
        capture_hidden_mode = (
            CaptureHiddenMode.FULL
            if self.model_runner.spec_algorithm.is_dvr_eagle()
            else CaptureHiddenMode.NULL
        )
        spec_info = EagleVerifyInput(
            draft_token=None,
            custom_mask=self.buffers.custom_mask,
            positions=None,
            retrieve_index=None,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.model_runner.server_args.speculative_eagle_topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=capture_hidden_mode,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )
        if self.model_runner.spec_algorithm.is_dvr_eagle():
            spec_info.hidden_states = torch.zeros(
                (num_tokens, self.model_runner.model_config.hidden_size),
                dtype=self.model_runner.dtype,
                device=self.model_runner.device,
            )
        return spec_info

    def load_batch(self, forward_batch, pp_proxy_tensors=None):
        # The generic num_token_non_padded slot is enabled only for EP. GDN DVR
        # verify also needs the raw token count to mask padded graph rows.
        self.buffers.num_token_non_padded.fill_(
            forward_batch.batch_size * self.num_tokens_per_bs
        )
        return super().load_batch(forward_batch, pp_proxy_tensors)

    @contextmanager
    def _forward_metadata_out_graph_context(
        self,
        *,
        forward_batch: ForwardBatch,
        attn_backend,
        forward_mode,
    ):
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
        if dvr_self_draft_graph_block_reason(forward_batch) is not None:
            return False
        return self.runner.can_run_graph(forward_batch)

    def replay(self, forward_batch: ForwardBatch):
        return self.runner.execute(forward_batch)
