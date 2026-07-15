from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext

import torch
from sglang.srt.distributed import get_moe_ep_group, get_moe_tp_group
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


def iter_dvr_attention_backends(*roots):
    """Yield each backend reachable through SGLang's attention wrappers once."""

    seen = set()
    stack = list(reversed(roots))
    while stack:
        backend = stack.pop()
        if backend is None or isinstance(backend, str) or id(backend) in seen:
            continue
        seen.add(id(backend))
        yield backend

        for attr_name in ("decode_backend", "prefill_backend", "primary"):
            stack.append(getattr(backend, attr_name, None))

        for attr_name in ("attn_backend_list", "attn_backends", "children"):
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


def _patch_draft_decode_backend_defaults(backend, model_runner, patch_attr):
    """Undo init-time deterministic decode knobs for DVR self-draft only.

    Prefill and verify still run through the deterministic target path. The
    self-draft decode graph is provisional and verified afterwards, so it should
    use the same decode heuristics as normal non-deterministic serving whenever
    the deterministic choice was baked into a backend field at init time.
    """

    server_args = model_runner.server_args

    # FA3 and NSA store deterministic split policy at backend init time. FA4 is
    # excluded because its CUDA graph does not support num_splits=0.
    if (
        getattr(backend, "fa_impl_ver", None) == 3
        or backend.__class__.__name__ == "NativeSparseAttnBackend"
    ):
        patch_attr(backend, "num_splits", 0)

    if hasattr(backend, "decode_split_tile_size"):
        # FlashInfer deterministic mode fixes decode split size and disables
        # CUDA graph KV split. Restore normal decode planning for self-draft.
        patch_attr(backend, "decode_split_tile_size", None)
        patch_attr(backend, "disable_cuda_graph_kv_split", False)
        if hasattr(backend, "decode_use_tensor_cores"):
            from sglang.srt.layers.attention.flashinfer_backend import (
                should_use_tensor_core,
            )

            patch_attr(
                backend,
                "decode_use_tensor_cores",
                should_use_tensor_core(
                    kv_cache_dtype=model_runner.kv_cache_dtype,
                    num_attention_heads=(
                        model_runner.model_config.num_attention_heads
                        // model_runner.tp_size
                    ),
                    num_kv_heads=model_runner.model_config.get_num_kv_heads(
                        model_runner.tp_size
                    ),
                ),
            )

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

        try:
            custom_allreduce_cls = dispatch_custom_allreduce(
                group=group.cpu_group,
                device=group.device,
            )
            ca_comm = custom_allreduce_cls(
                group=group.cpu_group,
                device=group.device,
            )
            group.ca_comm = ca_comm
        except Exception as exc:
            logger.warning("DVR draft custom all-reduce setup failed: %s", exc)
            return None

    if ca_comm is None or not hasattr(ca_comm, "world_size"):
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


@contextmanager
def dvr_draft_decode_context(
    model_runner,
    *,
    capture: bool = False,
    self_draft: bool = False,
    extra_attn_backends=(),
):
    """Temporarily switch a DVR draft runner into performance-first decode mode.

    Self-draft replay only patches backend fields consumed by its captured graph;
    EAGLE replay and both capture paths also switch global deterministic state.
    """

    patched_attrs = []
    global_server_args = get_global_server_args()
    patch_global_state = capture or not self_draft

    def patch_attr(obj, attr_name, value):
        if obj is None or not hasattr(obj, attr_name):
            return
        patched_attrs.append((obj, attr_name, getattr(obj, attr_name)))
        setattr(obj, attr_name, value)

    deterministic_env = (
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False)
        if patch_global_state
        else nullcontext()
    )
    with deterministic_env:
        try:
            if capture:
                _clear_determinism_sensitive_kernel_caches()

            if patch_global_state:
                for server_args in (model_runner.server_args, global_server_args):
                    patch_attr(server_args, "enable_deterministic_inference", False)

            for backend in iter_dvr_attention_backends(
                model_runner.attn_backend,
                *(extra_attn_backends or ()),
            ):
                patch_attr(backend, "enable_deterministic", False)
                _patch_draft_decode_backend_defaults(backend, model_runner, patch_attr)

            if capture:
                # Deterministic target prefill/verify keeps custom all-reduce
                # disabled. Only capture it into provisional draft graphs.
                if getattr(
                    model_runner.server_args,
                    "_dvr_enable_draft_custom_all_reduce",
                    False,
                ):
                    for ca_comm in _iter_decode_custom_all_reduce_comms(model_runner):
                        patch_attr(ca_comm, "disabled", False)
                if self_draft:
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
            if patch_global_state:
                for server_args in (model_runner.server_args, global_server_args):
                    patch_attr(server_args, "mamba_radix_cache_strategy", "no_buffer")

            with _maybe_disable_batch_invariant_ops(capture):
                yield
        finally:
            for obj, attr_name, original_value in reversed(patched_attrs):
                setattr(obj, attr_name, original_value)
            if capture:
                _clear_determinism_sensitive_kernel_caches()


class DVRDraftDecodeCudaGraphRunner(DecodeCudaGraphRunner):
    """Ordinary decode graph used only for provisional self-draft tokens."""

    # Target verify remains the final shared-pool reader. Publishing an event
    # after every provisional decode step allocates and records 15 events that
    # the DVR worker must discard before returning to Scheduler.
    record_war_fastpath_event = False


class DVRTargetVerifyCudaGraphRunner(DecodeCudaGraphRunner):
    """Target-verify graph runner for DVR self-draft and DVR-EAGLE.

    DVR target verify uses the standard EAGLE verifier shape, but the graph
    metadata must follow DVR's causal verifier and GDN state-input windows. Keep
    those rules on the graph runner instead of attaching execution hooks to the
    spec_info data object.
    """

    def __init__(self, model_runner):
        # Keep this capture policy on the dedicated runner rather than adding
        # transient DVR state to the shared ModelRunner.
        self.dvr_target_verify_cuda_graph = (
            model_runner.spec_algorithm.is_dvr_eagle()
        )
        super().__init__(model_runner)

    def get_spec_info(self, num_tokens: int):
        capture_hidden_mode = (
            CaptureHiddenMode.FULL
            if self.model_runner.spec_algorithm.is_dvr_eagle()
            else CaptureHiddenMode.NULL
        )
        spec_info = EagleVerifyInput(
            draft_token=None,
            custom_mask=None,
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
