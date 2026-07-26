from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.speculative.dvr_cuda_graph_runner as graph_module
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.dvr.gdn_backend import DVRGDNAttnBackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.speculative.dvr_cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
    _draft_custom_allreduce_enabled,
    _fast_decode_overrides,
    _resolve_dvr_backends,
    _resolve_draft_flashinfer_allreduce_fusion,
    validate_dvr_attention_backend,
    dvr_draft_decode_context,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


@pytest.mark.parametrize("draft_custom_all_reduce", [True, False])
def test_draft_capture_is_fast_and_restores_target_state(
    monkeypatch, draft_custom_all_reduce
):
    events = []

    backend = SimpleNamespace(enable_deterministic=True)
    server_args = SimpleNamespace(
        enable_deterministic_inference=True,
        dvr_enable_draft_custom_all_reduce=draft_custom_all_reduce,
    )
    global_server_args = SimpleNamespace(enable_deterministic_inference=True)
    model_runner = SimpleNamespace(
        attn_backend=backend,
        server_args=server_args,
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
    )

    class FakeEnv:
        @contextmanager
        def override(self, value):
            events.append(("deterministic_env", value))
            yield
            events.append(("deterministic_env", True))

    class FakeGroup:
        pass

    @contextmanager
    def custom_allreduce_enabled(_group, **kwargs):
        events.append(("custom_all_reduce", True, kwargs))
        yield True
        events.append(("custom_all_reduce", False, kwargs))

    monkeypatch.setattr(
        graph_module,
        "envs",
        SimpleNamespace(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=FakeEnv()),
    )
    monkeypatch.setattr(
        graph_module, "get_global_server_args", lambda: global_server_args
    )
    monkeypatch.setattr(graph_module, "_clear_moe_policy_caches", lambda: None)
    monkeypatch.setattr(
        graph_module,
        "_fast_decode_overrides",
        lambda *_args, **_kwargs: [(backend, "enable_deterministic", False)],
    )
    monkeypatch.setattr(
        graph_module, "_iter_decode_custom_all_reduce_groups", lambda _: [FakeGroup()]
    )
    monkeypatch.setattr(
        graph_module, "_draft_custom_allreduce_enabled", custom_allreduce_enabled
    )
    import sglang.srt.batch_invariant_ops as batch_invariant_ops

    monkeypatch.setattr(
        batch_invariant_ops, "is_batch_invariant_mode_enabled", lambda: False
    )

    with dvr_draft_decode_context(model_runner, {}, capture=True, self_draft=True):
        assert not backend.enable_deterministic
        assert not server_args.enable_deterministic_inference
        assert not global_server_args.enable_deterministic_inference
        assert (
            model_runner.spec_algorithm == SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
        )
        if draft_custom_all_reduce:
            assert events[-1][0] == "custom_all_reduce"

    assert backend.enable_deterministic
    assert server_args.enable_deterministic_inference
    assert global_server_args.enable_deterministic_inference
    assert model_runner.spec_algorithm == SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
    assert events[-1][0:2] == ("deterministic_env", True)
    assert (("custom_all_reduce", False) in [event[:2] for event in events]) == (
        draft_custom_all_reduce
    )


def test_triton_fast_decode_override_contract(monkeypatch):
    """Detect upstream Triton field changes at DVR's narrow graph boundary."""

    monkeypatch.setenv("SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false")
    model_runner = SimpleNamespace(
        server_args=SimpleNamespace(
            triton_attention_split_tile_size=None,
            triton_attention_num_kv_splits=8,
        ),
        kv_cache_dtype=torch.bfloat16,
        tp_size=2,
        model_config=SimpleNamespace(
            num_attention_heads=16,
            get_num_kv_heads=lambda tp_size: 8 // tp_size,
        ),
    )
    backend = object.__new__(TritonAttnBackend)
    backend.max_context_len = 8192
    backend.max_kv_splits = 32
    backend.split_tile_size = 256
    backend.static_kv_splits = False
    backend.enable_deterministic = True
    backend.cuda_graph_attn_logits = torch.zeros(4, 2, 32, 8)
    backend.cuda_graph_attn_lse = torch.zeros(4, 2, 32)
    backend.cuda_graph_swa_attn_logits = None
    backend.cuda_graph_num_kv_splits = torch.full((4,), 32, dtype=torch.int32)

    for owner, name, value in _fast_decode_overrides(backend, model_runner, {}):
        setattr(owner, name, value)

    assert not backend.enable_deterministic
    assert backend.max_kv_splits == 8
    assert backend.cuda_graph_attn_logits.shape[2] == 8
    assert torch.equal(
        backend.cuda_graph_num_kv_splits,
        torch.full((4,), 8, dtype=torch.int32),
    )


def test_backend_resolution_returns_attention_and_linear_state_once():
    class Backend:
        token_to_kv_pool = object()
        req_to_token_pool = object()
        needs_cpu_seq_lens = False

    adapter = object()
    linear_backend = object.__new__(DVRGDNAttnBackend)
    linear_backend.dvr_state_adapter = adapter
    linear_backend.needs_cpu_seq_lens = False
    full_attention = Backend()
    hybrid_linear = HybridLinearAttnBackend(
        full_attention, linear_backend, full_attn_layers=[]
    )
    children = [Backend(), Backend()]
    tbo = TboAttnBackend(primary=hybrid_linear, children=children)
    model_runner = SimpleNamespace(
        kv_cache_dtype=torch.bfloat16,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
    )
    hybrid = HybridAttnBackend(
        model_runner=model_runner,
        prefill_backend=Backend(),
        decode_backend=tbo,
    )

    leaves, resolved_adapter = _resolve_dvr_backends(hybrid)

    assert leaves == [full_attention, *children]
    assert resolved_adapter is adapter


def test_hybrid_backend_propagates_cpu_sequence_length_requirement():
    class Backend:
        token_to_kv_pool = object()
        req_to_token_pool = object()

        def __init__(self, needs_cpu_seq_lens):
            self.needs_cpu_seq_lens = needs_cpu_seq_lens

    model_runner = SimpleNamespace(
        kv_cache_dtype=torch.bfloat16,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
    )

    hybrid = HybridAttnBackend(
        model_runner=model_runner,
        prefill_backend=Backend(True),
        decode_backend=Backend(False),
    )

    assert hybrid.needs_cpu_seq_lens


def test_backend_resolution_rejects_distinct_linear_state_adapters():
    class Backend:
        token_to_kv_pool = object()
        req_to_token_pool = object()
        needs_cpu_seq_lens = False

    def hybrid_with_adapter(adapter):
        linear_backend = object.__new__(DVRGDNAttnBackend)
        linear_backend.dvr_state_adapter = adapter
        linear_backend.needs_cpu_seq_lens = False
        return HybridLinearAttnBackend(
            Backend(), linear_backend, full_attn_layers=[]
        )

    backend = TboAttnBackend(
        primary=hybrid_with_adapter(object()),
        children=[hybrid_with_adapter(object())],
    )

    with pytest.raises(RuntimeError, match="multiple linear-state adapters"):
        _resolve_dvr_backends(backend)


def test_backend_validation_accepts_triton_and_rejects_non_fa3():
    backend = object.__new__(TritonAttnBackend)
    leaves, adapter = validate_dvr_attention_backend(backend)
    assert leaves == [backend] and adapter is None

    fa4 = object.__new__(FlashAttentionBackend)
    fa4.fa_impl_ver = 4
    with pytest.raises(RuntimeError, match="requires FlashAttention 3"):
        validate_dvr_attention_backend(fa4)


def test_draft_fusion_policy_preserves_user_intent_and_reuses_auto_policy(
    monkeypatch,
):
    server_args = SimpleNamespace(
        dvr_draft_flashinfer_allreduce_fusion=("trtllm", False)
    )
    assert _resolve_draft_flashinfer_allreduce_fusion(server_args) == "trtllm"

    server_args.dvr_draft_flashinfer_allreduce_fusion = (None, True)
    assert _resolve_draft_flashinfer_allreduce_fusion(server_args) is None

    import sglang.srt.arg_groups.overrides as overrides

    server_args.dvr_draft_flashinfer_allreduce_fusion = (None, False)
    monkeypatch.setattr(
        overrides,
        "_flashinfer_allreduce_fusion_auto_enable",
        lambda _: {"flashinfer_allreduce_fusion_backend": "auto"},
    )
    assert _resolve_draft_flashinfer_allreduce_fusion(server_args) == "auto"


def test_self_draft_preinitializes_fusion_before_custom_all_reduce(monkeypatch):
    events = []
    backend = SimpleNamespace(enable_deterministic=True)
    server_args = SimpleNamespace(
        enable_deterministic_inference=True,
        flashinfer_allreduce_fusion_backend=None,
        dvr_enable_draft_custom_all_reduce=True,
        dvr_draft_flashinfer_allreduce_fusion=("trtllm", False),
    )
    global_server_args = SimpleNamespace(
        enable_deterministic_inference=True,
        flashinfer_allreduce_fusion_backend=None,
    )
    model_runner = SimpleNamespace(
        attn_backend=backend,
        server_args=server_args,
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
        model_config=SimpleNamespace(hidden_size=4096),
        dtype=torch.bfloat16,
    )

    class FakeEnv:
        @contextmanager
        def override(self, _value):
            yield

    class FakeGroup:
        pass

    @contextmanager
    def custom_allreduce_enabled(_group, **_kwargs):
        events.append("custom_ar_enter")
        yield
        events.append("custom_ar_exit")

    monkeypatch.setattr(
        graph_module,
        "envs",
        SimpleNamespace(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=FakeEnv()),
    )
    monkeypatch.setattr(
        graph_module, "get_global_server_args", lambda: global_server_args
    )
    monkeypatch.setattr(graph_module, "_clear_moe_policy_caches", lambda: None)
    monkeypatch.setattr(
        graph_module,
        "_fast_decode_overrides",
        lambda *_args, **_kwargs: [(backend, "enable_deterministic", False)],
    )
    monkeypatch.setattr(
        graph_module, "_iter_decode_custom_all_reduce_groups", lambda _: [FakeGroup()]
    )
    monkeypatch.setattr(
        graph_module, "_draft_custom_allreduce_enabled", custom_allreduce_enabled
    )

    import sglang.srt.batch_invariant_ops as batch_invariant_ops
    import sglang.srt.layers.flashinfer_comm_fusion as fusion

    monkeypatch.setattr(
        batch_invariant_ops, "is_batch_invariant_mode_enabled", lambda: False
    )

    def pre_initialize_workspaces(**kwargs):
        assert server_args.flashinfer_allreduce_fusion_backend == "trtllm"
        assert global_server_args.flashinfer_allreduce_fusion_backend == "trtllm"
        events.append(("workspace", kwargs))

    monkeypatch.setattr(fusion, "pre_initialize_workspaces", pre_initialize_workspaces)

    with dvr_draft_decode_context(model_runner, {}, capture=True, self_draft=True):
        assert events[0][0] == "workspace"
        assert events[1] == "custom_ar_enter"

    assert server_args.flashinfer_allreduce_fusion_backend is None
    assert global_server_args.flashinfer_allreduce_fusion_backend is None
    assert events[-1] == "custom_ar_exit"


def test_draft_custom_allreduce_context_restores_target_policy():
    communicator = SimpleNamespace(
        disabled=True,
        original_disabled=True,
        full_nvlink=True,
    )
    group = SimpleNamespace(ca_comm=communicator, world_size=2)

    with _draft_custom_allreduce_enabled(
        group, create_if_missing=False, require_full_nvlink=True
    ) as enabled:
        assert enabled
        assert not communicator.disabled

    assert communicator.disabled
    assert communicator.original_disabled

    communicator.full_nvlink = False
    with _draft_custom_allreduce_enabled(
        group, create_if_missing=False, require_full_nvlink=True
    ) as enabled:
        assert not enabled
        assert communicator.disabled


def test_dvr_draft_graph_uses_plain_decode_layout_and_shared_state():
    runner = object.__new__(DVRDraftDecodeCudaGraphRunner)

    assert runner._resolve_capture_layout() == (ForwardMode.DECODE, 1)
    assert not runner.owns_attention_graph_state


@pytest.mark.parametrize(
    ("forward_mode", "dvr_self_draft", "dflash", "records_event"),
    [
        (ForwardMode.DECODE, False, False, True),
        (ForwardMode.DECODE, True, False, False),
        (ForwardMode.TARGET_VERIFY, False, True, True),
        (ForwardMode.TARGET_VERIFY, True, False, False),
    ],
)
def test_cuda_graph_war_event_skips_provisional_self_draft(
    forward_mode, dvr_self_draft, dflash, records_event
):
    events = []

    class Event:
        def record(self):
            events.append("record")

    class Backend:
        @contextmanager
        def replay_session(self):
            yield

        def replay(self, _key, _forward_batch):
            return PPProxyTensors({})

    runner = object.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        device_timer=None,
        spec_algorithm=SimpleNamespace(
            is_dvr_self_draft=lambda: dvr_self_draft,
            is_dflash=lambda: dflash,
        ),
        war_fastpath_read_done_event=None,
    )
    runner.device_module = SimpleNamespace(Event=Event)
    runner.backend = Backend()
    runner.bs = 1

    def load_batch(_forward_batch, _pp_proxy_tensors):
        runner._replay_graph_key = 1

    runner.load_batch = load_batch
    runner.execute(SimpleNamespace(forward_mode=forward_mode))

    assert bool(events) is records_event
    assert (
        runner.model_runner.war_fastpath_read_done_event is not None
    ) is records_event
