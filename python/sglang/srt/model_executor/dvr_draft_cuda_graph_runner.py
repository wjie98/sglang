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
    patched_keys = set()
    global_server_args = get_global_server_args()
    patch_global_state = capture or not self_draft

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

    deterministic_env = (
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False)
        if patch_global_state
        else nullcontext()
    )
    with deterministic_env:
        try:
            if patch_global_state:
                for server_args in (model_runner.server_args, global_server_args):
                    patch_attr(server_args, "enable_deterministic_inference", False)

            supported_backends = []
            for backend in iter_dvr_attention_backends(
                model_runner.attn_backend,
                *(extra_attn_backends or ()),
            ):
                get_overrides = getattr(
                    backend, "get_nondeterministic_decode_overrides", None
                )
                overrides = (
                    None if get_overrides is None else get_overrides(model_runner)
                )
                if overrides is None:
                    continue
                supported_backends.append(type(backend).__name__)
                patch_attr(backend, "enable_deterministic", False)
                for name, value in overrides:
                    patch_attr(backend, name, value)

            if not supported_backends:
                raise RuntimeError(
                    "DVR draft decode supports only Triton or FA3 full-attention "
                    "backends, but no supported backend was found under "
                    f"{type(model_runner.attn_backend).__name__}."
                )

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

            with _maybe_disable_batch_invariant_ops(capture):
                yield
        finally:
            for obj, attr_name, original_value in reversed(patched_attrs):
                setattr(obj, attr_name, original_value)


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
