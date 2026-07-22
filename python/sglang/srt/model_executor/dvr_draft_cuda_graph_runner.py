from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager

import torch

from sglang.srt.distributed import get_moe_ep_group, get_moe_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_bool_env_var

_OVERRIDE_PLAN_CACHE = object()
logger = logging.getLogger(__name__)


def _resolve_draft_flashinfer_allreduce_fusion(server_args):
    """Resolve the communication fusion ordinary decode would have selected."""

    request = getattr(server_args, "_dvr_draft_flashinfer_allreduce_fusion", None)
    if request is None:
        return None
    requested_backend, force_disabled = request
    if force_disabled:
        return None
    if requested_backend is not None:
        return requested_backend

    # Reuse the upstream eligibility policy after MoE/topology/model defaults
    # have settled instead of duplicating its architecture and hardware list.
    from sglang.srt.arg_groups.overrides import (
        _flashinfer_allreduce_fusion_auto_enable,
    )

    return _flashinfer_allreduce_fusion_auto_enable(server_args).get(
        "flashinfer_allreduce_fusion_backend"
    )


def _resolve_dvr_backends(backend, forward_mode=ForwardMode.DECODE):
    """Resolve full-attention leaves and the optional linear-state adapter."""

    leaves = []
    state_adapter = None

    def visit(current, *, collect_attention=True):
        nonlocal state_adapter
        if isinstance(current, HybridLinearAttnBackend):
            state_adapter = state_adapter or getattr(
                current.linear_attn_backend, "dvr_state_adapter", None
            )
            visit(current.full_attn_backend, collect_attention=collect_attention)
        elif isinstance(current, HybridAttnBackend):
            selected = current._select_backend(forward_mode)
            visit(selected, collect_attention=collect_attention)
            if state_adapter is None:
                for candidate in (current.decode_backend, current.prefill_backend):
                    if candidate is not selected:
                        visit(candidate, collect_attention=False)
        elif isinstance(current, TboAttnBackend):
            visit(current.primary, collect_attention=collect_attention)
            for child in current.children:
                visit(child, collect_attention=collect_attention)
        else:
            state_adapter = state_adapter or getattr(current, "dvr_state_adapter", None)
            if collect_attention:
                leaves.append(current)

    visit(backend)
    return leaves, state_adapter


def _validate_dvr_attention_backend(
    backend, forward_mode=ForwardMode.DECODE, *, phase="draft decode"
):
    """Return supported full-attention leaves after wrapper resolution."""

    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend,
    )
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    leaves, state_adapter = _resolve_dvr_backends(backend, forward_mode)
    if not leaves:
        raise RuntimeError(f"DVR {phase} did not resolve an attention backend.")
    for leaf in leaves:
        if isinstance(leaf, FlashAttentionBackend):
            if leaf.fa_impl_ver == 3:
                continue
            raise RuntimeError(
                f"DVR {phase} requires FlashAttention 3, got "
                f"fa_impl_ver={leaf.fa_impl_ver}."
            )
        if not isinstance(leaf, TritonAttnBackend):
            raise RuntimeError(
                f"DVR {phase} supports only Triton or FA3 full-attention "
                f"backends, got {type(leaf).__name__}."
            )
    return leaves, state_adapter


def _fast_decode_overrides(backend, model_runner, buffer_cache=None):
    """Return fast-decode state for the DVR-supported Triton and FA3 leaves."""

    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend,
    )

    plan_cache = None
    plan_key = None
    if buffer_cache is not None:
        plan_cache = buffer_cache.setdefault(_OVERRIDE_PLAN_CACHE, {})
        plan_key = (id(backend), id(model_runner))
        if plan_key in plan_cache:
            return plan_cache[plan_key]

    overrides = []
    leaves, _ = _validate_dvr_attention_backend(backend)
    for leaf in leaves:
        if isinstance(leaf, FlashAttentionBackend):
            overrides.append((leaf, "num_splits", 0))
            continue

        split_tile_size = model_runner.server_args.triton_attention_split_tile_size
        max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        if split_tile_size is not None:
            max_kv_splits = (
                leaf.max_context_len + split_tile_size - 1
            ) // split_tile_size
        max_kv_splits = min(leaf.max_kv_splits, max_kv_splits)
        overrides.append((leaf, "enable_deterministic", False))
        for name in (
            "cuda_graph_attn_logits",
            "cuda_graph_attn_lse",
            "cuda_graph_swa_attn_logits",
        ):
            buffer = getattr(leaf, name, None)
            if buffer is not None:
                overrides.append((leaf, name, buffer[:, :, :max_kv_splits]))

        if buffer_cache is None:
            buffer_cache = {}
        buffers = buffer_cache.setdefault(leaf, {})
        for name in (
            "cuda_graph_num_kv_splits",
            "cuda_graph_window_num_kv_splits",
        ):
            buffer = getattr(leaf, name, None)
            if buffer is not None:
                draft_buffer = buffers.get(name)
                if draft_buffer is None or draft_buffer.shape != buffer.shape:
                    draft_buffer = torch.full_like(buffer, max_kv_splits)
                    buffers[name] = draft_buffer
                overrides.append((leaf, name, draft_buffer))
        overrides.extend(
            (
                (leaf, "max_kv_splits", max_kv_splits),
                (leaf, "split_tile_size", split_tile_size),
                (
                    leaf,
                    "static_kv_splits",
                    get_bool_env_var(
                        "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
                    ),
                ),
            )
        )
    if plan_cache is not None:
        plan_cache[plan_key] = overrides
    return overrides


def _clear_moe_policy_caches():
    # These upstream caches read global determinism state without keying on it.
    # Draft graph capture is the only DVR phase that changes that state.
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_moe_configs,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
        should_enable_swap_ab,
    )

    get_moe_configs.cache_clear()
    should_enable_swap_ab.cache_clear()


def _iter_decode_custom_all_reduce_groups(model_runner):
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
        yield group


@contextmanager
def dvr_draft_decode_context(
    model_runner,
    buffer_cache,
    *,
    capture: bool = False,
    self_draft: bool = False,
    extra_attn_backends=(),
):
    """Temporarily switch a DVR draft runner into performance-first decode mode.

    Global determinism changes only while a draft graph is captured. Runtime
    replay temporarily switches attention metadata but leaves ordinary target,
    sampler, and MoE policy untouched.
    """

    patched_attrs = []
    patched_keys = set()
    global_server_args = get_global_server_args()
    if capture:
        # Attention graph buffers may be allocated inside this context. Never
        # retain a plan resolved before capture has finished initializing them.
        buffer_cache.pop(_OVERRIDE_PLAN_CACHE, None)

    def patch_attr(obj, attr_name, value):
        if obj is None or not hasattr(obj, attr_name):
            return
        key = (id(obj), attr_name)
        if key in patched_keys:
            setattr(obj, attr_name, value)
            return
        patched_keys.add(key)
        patched_attrs.append((obj, attr_name, getattr(obj, attr_name)))
        setattr(obj, attr_name, value)

    with ExitStack() as stack:
        try:
            if capture:
                stack.enter_context(
                    envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False)
                )
                _clear_moe_policy_caches()
                # Leave no fast-draft policy entry for deterministic target work.
                stack.callback(_clear_moe_policy_caches)
                for server_args in (model_runner.server_args, global_server_args):
                    patch_attr(server_args, "enable_deterministic_inference", False)

                if self_draft:
                    fusion_backend = _resolve_draft_flashinfer_allreduce_fusion(
                        model_runner.server_args
                    )
                    for server_args in (
                        model_runner.server_args,
                        global_server_args,
                    ):
                        patch_attr(
                            server_args,
                            "flashinfer_allreduce_fusion_backend",
                            fusion_backend,
                        )
                    if fusion_backend is not None:
                        # Workspace creation performs collectives and must
                        # precede both CUDA graph capture and custom-AR graph
                        # buffer registration. The captured graph then reuses
                        # these stable addresses without a runtime policy patch.
                        from sglang.srt.layers.communicator import (
                            FUSE_ALLREDUCE_MAX_BATCH_SIZE,
                        )
                        from sglang.srt.layers.flashinfer_comm_fusion import (
                            pre_initialize_workspaces,
                        )

                        pre_initialize_workspaces(
                            max_token_num=FUSE_ALLREDUCE_MAX_BATCH_SIZE,
                            hidden_dim=model_runner.model_config.hidden_size,
                            dtype=model_runner.dtype,
                        )
                    logger.info(
                        "DVR self-draft FlashInfer all-reduce fusion backend: %s",
                        fusion_backend or "disabled",
                    )

            resolved_backend = False
            draft_state_adapters = set()
            for backend in (
                model_runner.attn_backend,
                *(extra_attn_backends or ()),
            ):
                if backend is None:
                    continue
                if self_draft:
                    _, state_adapter = _resolve_dvr_backends(backend)
                    if (
                        state_adapter is not None
                        and id(state_adapter) not in draft_state_adapters
                    ):
                        draft_state_adapters.add(id(state_adapter))
                        stack.enter_context(state_adapter.self_draft_decode())
                overrides = _fast_decode_overrides(backend, model_runner, buffer_cache)
                resolved_backend = True
                for owner, name, value in overrides:
                    patch_attr(owner, name, value)

            if not resolved_backend:
                raise RuntimeError(
                    "DVR draft decode did not receive an attention backend."
                )

            if capture:
                # Deterministic target prefill/verify keeps custom all-reduce
                # disabled. Only capture it into provisional draft graphs.
                if getattr(
                    model_runner.server_args,
                    "_dvr_enable_draft_custom_all_reduce",
                    False,
                ):
                    for group in _iter_decode_custom_all_reduce_groups(model_runner):
                        stack.enter_context(
                            group.custom_allreduce_enabled(
                                create_if_missing=True,
                                require_full_nvlink=True,
                            )
                        )
                if self_draft:
                    # Target graph buffers are already initialized for
                    # TARGET_VERIFY; capture self draft as ordinary DECODE.
                    patch_attr(
                        model_runner, "spec_algorithm", SpeculativeAlgorithm.NONE
                    )
                from sglang.srt.batch_invariant_ops import (
                    disable_batch_invariant_mode,
                    enable_batch_invariant_mode,
                    is_batch_invariant_mode_enabled,
                )

                if is_batch_invariant_mode_enabled():
                    disable_batch_invariant_mode()
                    stack.callback(enable_batch_invariant_mode)

            yield
        finally:
            for obj, attr_name, original_value in reversed(patched_attrs):
                setattr(obj, attr_name, original_value)
            if capture:
                buffer_cache.pop(_OVERRIDE_PLAN_CACHE, None)


class DVRDraftDecodeCudaGraphRunner(DecodeCudaGraphRunner):
    """Ordinary decode graph used only for provisional self-draft tokens."""

    # Target verify remains the final shared-pool reader. Publishing an event
    # after every provisional decode step allocates and records 15 events that
    # the DVR worker must discard before returning to Scheduler.
    record_war_fastpath_event = False
    initialize_attention_backend_state = False
    enable_mamba_tracking = False


class DVRTargetVerifyCudaGraphRunner(DecodeCudaGraphRunner):
    """Target-verify graph runner for DVR self-draft and DVR-EAGLE.

    DVR target verify uses the standard EAGLE verifier shape, but the graph
    metadata must follow DVR's causal verifier and GDN state-input windows. Keep
    those rules on the graph runner instead of attaching execution hooks to the
    spec_info data object.
    """

    # Both self-draft and EAGLE use deterministic target verification. Prevent
    # prefill-only determinism from restoring ordinary decode split heuristics
    # while this dedicated graph is captured.
    dvr_target_verify_cuda_graph = True

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
