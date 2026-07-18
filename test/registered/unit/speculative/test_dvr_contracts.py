from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.speculative.dvr_worker as dvr_worker_module
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.dvr_draft_cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
    DVRTargetVerifyCudaGraphRunner,
    _fast_decode_overrides,
    _resolve_dvr_backends,
    _validate_dvr_attention_backend,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dvr_state_flow import DVRLinearStateLifecycle
from sglang.srt.speculative.dvr_worker import (
    DecodeVerifyRollbackWorkerV2,
    _dvr_proposal_probs,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def test_dvr_spec_algorithm_contracts():
    self_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
    eagle_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE

    assert self_draft.is_dvr_self_draft() and not self_draft.is_dvr_eagle()
    assert eagle_draft.is_dvr_eagle() and not eagle_draft.is_eagle()
    assert not self_draft.has_draft_kv()
    assert eagle_draft.has_draft_kv()


def test_dvr_worker_and_graph_contracts():
    assert issubclass(DecodeVerifyRollbackWorkerV2, BaseSpecWorker)
    assert not DecodeVerifyRollbackWorkerV2.__abstractmethods__
    assert DVRTargetVerifyCudaGraphRunner.dvr_target_verify_cuda_graph
    assert not DVRDraftDecodeCudaGraphRunner.record_war_fastpath_event


def test_dvr_without_radix_reserves_one_active_boundary_per_request():
    server_args = SimpleNamespace(
        speculative_num_draft_tokens=16,
        speculative_eagle_topk=1,
        max_running_requests=8,
        max_mamba_cache_size=None,
        disable_radix_cache=True,
        dp_size=1,
        enable_dp_attention=False,
        enable_mamba_extra_buffer=lambda: True,
    )
    params = SimpleNamespace(
        mamba_cache_per_req=1024,
        layers=(0,),
        shape=SimpleNamespace(
            conv=((2, 4),),
            temporal=(2, 2, 2),
            state_size=2,
            conv_dim=6,
            intermediate_size=2,
            num_heads=2,
        ),
        dtype=SimpleNamespace(conv=torch.float32, temporal=torch.float32),
    )
    runner = SimpleNamespace(
        server_args=server_args,
        mambaish_config=SimpleNamespace(mamba2_cache_params=params),
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
        dp_size=1,
    )
    runner._calculate_mamba_ratio = lambda: (
        ModelRunnerKVCacheMixin._calculate_mamba_ratio(runner)
    )

    ModelRunnerKVCacheMixin.handle_max_mamba_cache(runner, total_rest_memory=1.0)

    assert runner._calculate_mamba_ratio() == 2
    assert server_args.max_mamba_cache_size == 16


def _sampling_info(top_ks, top_ps, min_ps):
    return SimpleNamespace(
        top_ks=torch.tensor(top_ks),
        top_ps=torch.tensor(top_ps),
        min_ps=torch.tensor(min_ps),
        need_top_k_sampling=any(value != 0 for value in top_ks),
        need_top_p_sampling=any(value != 1.0 for value in top_ps),
        need_min_p_sampling=any(value != 0.0 for value in min_ps),
    )


def test_dvr_proposal_filter_matches_target_distribution():
    proposal = _dvr_proposal_probs(
        torch.tensor([[0.6, 0.3, 0.1]]),
        _sampling_info([3], [1.0], [0.5]),
        "pytorch",
    )
    torch.testing.assert_close(proposal, torch.tensor([[2 / 3, 1 / 3, 0.0]]))

    joint = _dvr_proposal_probs(
        torch.tensor([[0.6, 0.25, 0.15]]),
        _sampling_info([2], [0.7], [0.0]),
        "pytorch",
    )
    torch.testing.assert_close(joint, torch.tensor([[12 / 17, 5 / 17, 0.0]]))


def test_dvr_proposal_filter_preserves_mixed_greedy_rows():
    probs = torch.tensor([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])
    filtered = _dvr_proposal_probs(
        probs,
        _sampling_info([1, 3], [1.0, 1.0], [0.0, 0.0]),
        "pytorch",
    )
    torch.testing.assert_close(filtered[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(filtered[1], probs[1])


def test_dvr_target_filter_repeats_per_request_sampling_rows():
    filtered = _dvr_proposal_probs(
        torch.tensor([[0.6, 0.4]] * 4),
        _sampling_info([1, 2], [1.0, 1.0], [0.0, 0.0]),
        "pytorch",
        repeat=2,
    )
    torch.testing.assert_close(filtered[:2], torch.tensor([[1.0, 0.0]] * 2))
    torch.testing.assert_close(filtered[2:], torch.tensor([[0.6, 0.4]] * 2))


def test_group_owns_lazy_custom_allreduce_state(monkeypatch):
    calls = []

    class FakeCustomAllReduce:
        def __init__(self, *, group, device):
            calls.append(("init", group, device))
            self.full_nvlink = True
            self.disabled = False
            self.original_disabled = False

    def fake_dispatch(*, group, device):
        calls.append(("dispatch", group, device))
        return FakeCustomAllReduce

    from sglang.srt.distributed.device_communicators import custom_all_reduce

    monkeypatch.setattr(custom_all_reduce, "dispatch_custom_allreduce", fake_dispatch)
    group = object.__new__(GroupCoordinator)
    group.ca_comm = None
    group.world_size = 2
    group.cpu_group = "cpu-group"
    group.device = "cuda:0"

    with group.custom_allreduce_enabled(
        create_if_missing=True, require_full_nvlink=True
    ) as active:
        assert active
        assert not group.ca_comm.disabled

    assert calls == [
        ("dispatch", "cpu-group", "cuda:0"),
        ("init", "cpu-group", "cuda:0"),
    ]
    assert group.ca_comm.disabled and group.ca_comm.original_disabled


def test_self_draft_runtime_does_not_patch_global_decode_state(monkeypatch):
    worker = object.__new__(DecodeVerifyRollbackWorkerV2)
    worker.is_dvr_eagle = False
    worker.model_runner = object()
    worker._draft_decode_buffers = {}
    calls = []
    monkeypatch.setattr(
        dvr_worker_module,
        "dvr_draft_decode_context",
        lambda *args, **kwargs: calls.append((args, kwargs)) or nullcontext(),
    )

    with worker._draft_context():
        pass

    assert calls == [
        ((worker.model_runner, worker._draft_decode_buffers), {"self_draft": True})
    ]


def test_fast_decode_overrides_are_dvr_local(monkeypatch):
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

    for owner, name, value in _fast_decode_overrides(backend, model_runner):
        setattr(owner, name, value)

    assert not backend.enable_deterministic
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

        def __init__(self, adapter=None):
            self.dvr_state_adapter = adapter

    adapter = object()
    full_attention = Backend()
    hybrid_linear = HybridLinearAttnBackend(
        full_attention, Backend(adapter), full_attn_layers=[]
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


def test_dvr_validates_resolved_target_backend():
    backend = object.__new__(TritonAttnBackend)
    leaves, adapter = _validate_dvr_attention_backend(backend)
    assert leaves == [backend] and adapter is None

    fa4 = object.__new__(FlashAttentionBackend)
    fa4.fa_impl_ver = 4
    with pytest.raises(RuntimeError, match="requires FlashAttention 3"):
        _validate_dvr_attention_backend(fa4)


@pytest.mark.parametrize("is_dvr_eagle", [False, True])
def test_short_prefix_uses_one_root_verify_sentinel(is_dvr_eagle):
    worker = object.__new__(DecodeVerifyRollbackWorkerV2)
    worker.is_dvr_eagle = is_dvr_eagle
    worker.device = "cpu"
    worker.num_draft_tokens = 4
    worker._chain_retrieve_index = torch.arange(8).view(2, 4)
    worker._chain_retrieve_sibling = torch.full((2, 4), -1)
    worker._chain_position_offsets = torch.arange(4)
    batch = SimpleNamespace(
        spec_info=dvr_worker_module.EagleDraftInput(bonus_tokens=torch.tensor([6, 7])),
        seq_lens=torch.tensor([1, 65]),
        seq_lens_cpu=torch.tensor([1, 65]),
        seq_lens_sum=66,
    )

    verify_input = worker._build_one_root_verify_input(batch)

    assert verify_input.draft_token.tolist() == [6] * 4 + [7] * 4
    assert verify_input.spec_steps == 0
    assert verify_input.retrieve_next_token.eq(-1).all()
    assert verify_input.positions.tolist() == [1, 2, 3, 4, 65, 66, 67, 68]


class _Pool:
    def __init__(self, slots, copies):
        self.slots = torch.tensor(slots)
        self.mamba_pool = SimpleNamespace(
            copy_from=lambda source, destination: copies.append(
                (source.tolist(), destination.tolist())
            )
        )

    def get_mamba_ping_pong_slots(self, _indices):
        return self.slots.unsqueeze(0)

    def get_mamba_ping_pong_other_idx(self, indices):
        return 1 - indices if self.slots.numel() == 2 else indices

    def get_mamba_ping_pong_keep_idx(self, req):
        return self.get_mamba_ping_pong_other_idx(req.mamba_next_track_idx)


class _Adapter:
    chunk_size = 64
    draft_reuses_target_state = False

    def __init__(self, zeroed, tail_updates, backups=None):
        self.zeroed = zeroed
        self.tail_updates = tail_updates
        self.backups = backups

    @staticmethod
    def batch_state(*, batch):
        return object(), batch.req_pool_indices.to(torch.long), torch.tensor([20])

    def zero_recurrent_state(self, *, state_cache, indices):
        self.zeroed.extend(indices.tolist())

    def state_input_window(self):
        return SimpleNamespace(
            set_tail_lens=lambda **kwargs: self.tail_updates.append(
                (kwargs["indices"].tolist(), kwargs["value"].tolist())
            )
        )


def _lifecycle_fixture(
    *, seq_len, last_track, prefix_len, disable_radix=False, slots=(10, 11)
):
    copies, zeroed, tail_updates = [], [], []
    adapter = _Adapter(zeroed, tail_updates)
    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle.server_args = SimpleNamespace(
        disable_radix_cache=disable_radix,
        strip_thinking_cache=False,
        enable_mamba_extra_buffer=lambda: not disable_radix,
    )
    lifecycle._state_adapter = adapter
    lifecycle.draft_state_backup = None
    lifecycle.boundary_lens = torch.full((8, len(slots)), -1, dtype=torch.int64)
    pool = _Pool(slots, copies)
    lifecycle.model_runner = SimpleNamespace(req_to_token_pool=pool)
    req = SimpleNamespace(
        rid="r0",
        req_pool_idx=1,
        extend_input_len=seq_len - prefix_len,
        prefix_indices=[0] * prefix_len,
        mamba_last_track_seqlen=last_track,
        mamba_next_track_idx=0,
        mamba_ping_pong_track_buffer=torch.tensor(slots),
    )
    batch = SimpleNamespace(
        reqs=[req],
        req_pool_indices=torch.tensor([1]),
        req_to_token_pool=pool,
        device="cpu",
        seq_lens=torch.tensor([seq_len]),
        seq_lens_cpu=torch.tensor([seq_len]),
        prefix_lens=[prefix_len],
        batch_size=lambda: 1,
    )
    return lifecycle, req, batch, copies, zeroed, tail_updates


@pytest.mark.parametrize(
    ("seq_len", "last_track", "prefix_len", "expected_copy", "expected_tail"),
    [
        (64, 64, 0, ([20], [10]), 0),
        (65, 64, 0, ([11], [10]), 1),
        (65, None, 64, None, 1),
    ],
)
def test_target_extend_publishes_exact_boundary(
    seq_len, last_track, prefix_len, expected_copy, expected_tail
):
    lifecycle, _, batch, copies, _, tails = _lifecycle_fixture(
        seq_len=seq_len, last_track=last_track, prefix_len=prefix_len
    )
    lifecycle.boundary_lens[1] = torch.tensor([999, 888])

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)

    assert copies == ([] if expected_copy is None else [expected_copy])
    assert tails == [([1], [expected_tail])]
    assert lifecycle.boundary_lens[1].tolist() == [64, -1]


def test_target_extend_replaces_stale_request_slot_state():
    lifecycle, _, batch, _, zeroed, _ = _lifecycle_fixture(
        seq_len=1, last_track=None, prefix_len=0
    )
    lifecycle.boundary_lens[1] = torch.tensor([128, 192])

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)

    assert zeroed == [10]
    assert lifecycle.boundary_lens[1].tolist() == [0, -1]


def test_target_extend_fails_when_standard_sources_have_no_boundary():
    lifecycle, _, batch, *_ = _lifecycle_fixture(
        seq_len=65, last_track=None, prefix_len=0
    )
    lifecycle.prepare_target_extend(batch)

    with pytest.raises(RuntimeError, match="did not restore"):
        lifecycle.finish_target_extend(batch)


def test_radix_off_uses_one_request_local_boundary_slot():
    lifecycle, req, batch, _, _, _ = _lifecycle_fixture(
        seq_len=65,
        last_track=64,
        prefix_len=0,
        disable_radix=True,
        slots=(10,),
    )

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)

    assert batch.mamba_track_indices.tolist() == [10]
    assert batch.mamba_track_mask.tolist() == [True]
    assert req.mamba_last_track_seqlen == 64
    assert lifecycle.boundary_lens[1].tolist() == [64]


def test_prepare_for_draft_uses_device_authoritative_boundary():
    lifecycle, req, batch, *_ = _lifecycle_fixture(
        seq_len=65, last_track=64, prefix_len=0
    )
    lifecycle.boundary_lens[1, 1] = 64

    ctx = lifecycle.prepare_for_draft(batch)

    assert ctx.boundary_indices.tolist() == [11]
    assert ctx.next_boundary_indices.tolist() == [10]
    assert ctx.boundary_track_indices.tolist() == [1]


def test_rollback_advances_only_the_next_boundary_slot():
    calls = []

    class Adapter:
        chunk_size = 64

        @staticmethod
        def commit_after_verify(**kwargs):
            calls.append(kwargs)
            return torch.tensor([True])

    lifecycle, _, batch, *_ = _lifecycle_fixture(
        seq_len=65, last_track=64, prefix_len=0
    )
    lifecycle._state_adapter = Adapter()
    lifecycle.boundary_lens[1, 0] = 64
    ctx = SimpleNamespace(
        state_cache=object(),
        state_input_indices=torch.tensor([1]),
        live_indices=torch.tensor([2]),
        boundary_indices=torch.tensor([10]),
        next_boundary_indices=torch.tensor([11]),
        boundary_track_indices=torch.tensor([0]),
    )

    lifecycle.rollback_after_verify(
        batch=batch, ctx=ctx, accept_lens=torch.tensor([5], dtype=torch.int32)
    )

    assert calls[0]["accepted_token_counts"].tolist() == [5]
    assert lifecycle.boundary_lens[1].tolist() == [64, 128]


def _finished_req(**overrides):
    values = dict(
        rid="r0",
        req_pool_idx=1,
        skip_radix_cache_insert=False,
        mamba_last_track_seqlen=64,
        mamba_next_track_idx=0,
        kv_committed_len=192,
        finished_reason=None,
        reasoning_tokens=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_release_publishes_latest_visible_boundary_and_clears_row():
    lifecycle, _, _, *_ = _lifecycle_fixture(
        seq_len=65, last_track=64, prefix_len=0
    )
    lifecycle.boundary_lens[1] = torch.tensor([128, 192])
    req = _finished_req()

    lifecycle.prepare_for_cache_release(req)

    assert req.mamba_last_track_seqlen == 192
    assert req.mamba_next_track_idx == 0
    assert lifecycle.boundary_lens[1].tolist() == [-1, -1]


def test_release_fails_if_committed_boundary_was_lost():
    lifecycle, _, _, *_ = _lifecycle_fixture(
        seq_len=65, last_track=64, prefix_len=0
    )
    lifecycle.boundary_lens[1] = torch.tensor([256, -1])

    with pytest.raises(RuntimeError, match="lost the recurrent checkpoint"):
        lifecycle.prepare_for_cache_release(_finished_req(kv_committed_len=128))

    assert lifecycle.boundary_lens[1].tolist() == [-1, -1]


def test_radix_off_release_drops_active_boundary_without_publication():
    lifecycle, _, _, *_ = _lifecycle_fixture(
        seq_len=65,
        last_track=64,
        prefix_len=0,
        disable_radix=True,
        slots=(10,),
    )
    lifecycle.boundary_lens[1, 0] = 64
    req = _finished_req()

    lifecycle.prepare_for_cache_release(req)

    assert not req.skip_radix_cache_insert
    assert lifecycle.boundary_lens[1].tolist() == [-1]
