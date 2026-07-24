from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.speculative.dvr_state_flow import (
    DVRStateLifecycle,
    copy_cached_prefix_boundary,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


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

    def __init__(self, zeroed, tail_updates):
        self.zeroed = zeroed
        self.tail_updates = tail_updates
        self.accepted_tail_lens = torch.zeros(8, dtype=torch.int32)
        self.verify_boundaries = []
        self.draft_initializations = []

    @staticmethod
    def resolve_request_slots(*, batch):
        return batch.req_pool_indices.to(torch.long), torch.tensor([20])

    def zero_boundary_state(self, *, indices):
        self.zeroed.extend(indices.tolist())

    def set_accepted_tail_lens(self, **kwargs):
        self.tail_updates.append((kwargs["indices"].tolist(), kwargs["value"].tolist()))

    def set_verify_boundaries(self, **kwargs):
        self.verify_boundaries.append(
            (
                kwargs["request_rows"].tolist(),
                kwargs["boundary_slots"].tolist(),
            )
        )

    def initialize_self_draft_state(self, **kwargs):
        self.draft_initializations.append(
            (
                kwargs["request_rows"].tolist(),
                kwargs["endpoint_slots"].tolist(),
            )
        )


def _lifecycle_fixture(
    *,
    seq_len,
    last_track,
    prefix_len,
    disable_radix=False,
    slots=(10, 11),
    next_track=None,
):
    copies, zeroed, tail_updates = [], [], []
    lifecycle = object.__new__(DVRStateLifecycle)
    lifecycle.server_args = SimpleNamespace(
        disable_radix_cache=disable_radix,
        strip_thinking_cache=False,
        enable_mamba_extra_buffer=lambda: not disable_radix,
    )
    lifecycle._state_adapter = _Adapter(zeroed, tail_updates)
    lifecycle.boundary_seq_lens = torch.full((8, len(slots)), -1, dtype=torch.int64)
    pool = _Pool(slots, copies)
    lifecycle.model_runner = SimpleNamespace(req_to_token_pool=pool)
    if next_track is None:
        next_track = (
            1
            if len(slots) == 2 and last_track is not None and seq_len - prefix_len >= 64
            else 0
        )
    req = SimpleNamespace(
        rid="r0",
        req_pool_idx=1,
        prefix_indices=[0] * prefix_len,
        mamba_last_track_seqlen=last_track,
        mamba_next_track_idx=next_track,
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
        extend_lens=[seq_len - prefix_len],
        batch_size=lambda: 1,
    )
    return lifecycle, req, batch, copies, zeroed, tail_updates


def test_radix_disabled_reserves_one_active_boundary_per_request():
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
            num_k_heads_per_tp=2,
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


def test_dvr_memory_budget_includes_cuda_graph_dummy_row(monkeypatch):
    intermediate_per_row = 1 << 20
    monkeypatch.setattr(
        "sglang.srt.layers.attention.linear.dvr_gdn."
        "dvr_gdn_intermediate_bytes_per_request",
        lambda *args, **kwargs: intermediate_per_row,
    )
    server_args = SimpleNamespace(
        speculative_num_draft_tokens=16,
        speculative_eagle_topk=1,
        max_running_requests=2,
        max_mamba_cache_size=8,
        disable_radix_cache=False,
        dp_size=1,
        enable_dp_attention=False,
    )
    runner = SimpleNamespace(
        server_args=server_args,
        mambaish_config=SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(mamba_cache_per_req=1024)
        ),
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
        dp_size=1,
        _calculate_mamba_ratio=lambda: 2,
    )

    remaining = ModelRunnerKVCacheMixin.handle_max_mamba_cache(
        runner, total_rest_memory=1.0
    )

    expected_bytes = 3 * intermediate_per_row + 8 * 1024
    assert remaining == pytest.approx(1.0 - expected_bytes / (1 << 30))


def test_radix_enabled_reserves_two_checkpoint_slots_per_request():
    server_args = SimpleNamespace(
        disable_radix_cache=False,
        disable_overlap_schedule=False,
        enable_mamba_extra_buffer=lambda: True,
        enable_mamba_extra_buffer_lazy=lambda: False,
    )
    runner = SimpleNamespace(
        server_args=server_args,
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
    )

    # Upstream's three live/radix slots plus two DVR checkpoint slots.
    assert ModelRunnerKVCacheMixin._calculate_mamba_ratio(runner) == 5


@pytest.mark.parametrize(
    "spec_algorithm",
    [
        SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
        SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE,
        SpeculativeAlgorithm.EAGLE,
    ],
)
def test_spec_overlap_schedules_before_processing_previous_result(spec_algorithm):
    events = []
    req = SimpleNamespace(finished=False, mamba_next_track_idx=0)
    scheduled_req_states = []
    pending_batch = SimpleNamespace(
        spec_algorithm=spec_algorithm,
        forward_mode=ForwardMode.EXTEND,
    )
    pending_result = object()
    scheduler = SimpleNamespace(
        gracefully_exit=False,
        _engine_paused=False,
        last_batch=pending_batch,
        server_args=SimpleNamespace(enable_unified_memory=False),
        is_generation=False,
    )

    def recv_requests():
        scheduler.result_queue.append((pending_batch, pending_result))
        return []

    def process_batch_result(batch, result):
        assert (batch, result) == (pending_batch, pending_result)
        events.append("process_result")
        req.finished = True
        req.mamba_next_track_idx = None

    def get_next_batch_to_run():
        events.append("schedule")
        scheduled_req_states.append((req.finished, req.mamba_next_track_idx))
        scheduler.gracefully_exit = True
        return None

    scheduler.request_receiver = SimpleNamespace(recv_requests=recv_requests)
    scheduler.process_input_requests = lambda _reqs: None
    scheduler._apply_war_barrier = lambda: None
    scheduler.process_batch_result = process_batch_result
    scheduler.get_next_batch_to_run = get_next_batch_to_run
    scheduler.is_disable_overlap_for_batch = lambda _batch: False

    Scheduler.event_loop_overlap.__wrapped__(scheduler)

    assert events == ["schedule", "process_result"]
    assert scheduled_req_states == [(False, 0)]
    assert not scheduler.result_queue


def test_warm_prefix_boundary_copy_uses_current_physical_slots():
    copies = []

    class BoundaryPool:
        mamba_pool = SimpleNamespace(
            copy_from=lambda source, destination: copies.append(
                (source.tolist(), destination.tolist())
            )
        )

        @staticmethod
        def get_mamba_indices(req_indices):
            return req_indices + 10

        @staticmethod
        def translate_mamba_indices(indices):
            return indices + 100

        @staticmethod
        def get_mamba_ping_pong_slots(req_indices):
            return torch.stack((req_indices + 20, req_indices + 30), dim=1)

    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=4,
        _original_batch_size=3,
        extend_prefix_lens=torch.tensor([64, 64, 63, 64]),
        seq_lens=torch.tensor([65, 128, 64, 65]),
        req_pool_indices=torch.tensor([1, 2, 3, 4]),
        mamba_track_indices=torch.tensor([21, 22, 23, 24]),
        mamba_track_mask=torch.tensor([False, True, False, False]),
    )

    copy_cached_prefix_boundary(
        forward_batch=batch, req_to_token_pool=BoundaryPool(), chunk_size=64
    )

    # Only row 0 is a warm exact-boundary hit not owned by ordinary tracking.
    # Row 3 is padding and must not participate in the copy.
    assert copies == [([111], [131])]


def test_copy_mamba_checkpoint_preserves_request_ping_pong_slots():
    copies = []
    pool = object.__new__(HybridReqToTokenPool)
    pool.enable_mamba_extra_buffer_lazy = False
    pool.preserve_mamba_request_slots = True
    pool.mamba_ping_pong_track_buffer_size = 2
    pool.mamba_pool = SimpleNamespace(
        copy_from=lambda source, destination: copies.append(
            (source.tolist(), destination.tolist())
        )
    )
    pool.translate_mamba_indices = lambda indices: indices + 100
    req = SimpleNamespace(
        mamba_next_track_idx=1,
        mamba_ping_pong_track_buffer=torch.tensor([10, 11]),
    )

    result = pool.export_mamba_ping_pong_checkpoint(req, torch.tensor([20]))

    assert result.tolist() == [20]
    assert req.mamba_ping_pong_track_buffer.tolist() == [10, 11]
    assert copies == [([110], [120])]


@pytest.mark.parametrize(
    (
        "seq_len",
        "last_track",
        "prefix_len",
        "expected_boundary",
        "expected_next_boundary",
        "expected_tail",
    ),
    [
        (64, 64, 0, 10, 11, 0),
        (65, 64, 0, 10, 11, 1),
        (65, None, 64, 11, 10, 1),
    ],
)
def test_target_extend_publishes_exact_boundary(
    seq_len,
    last_track,
    prefix_len,
    expected_boundary,
    expected_next_boundary,
    expected_tail,
):
    lifecycle, _, batch, copies, _, tails = _lifecycle_fixture(
        seq_len=seq_len, last_track=last_track, prefix_len=prefix_len
    )
    lifecycle.boundary_seq_lens[1] = torch.tensor([999, 888])

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    draft_ctx = lifecycle.prepare_for_draft(batch)

    assert copies == []
    assert tails == [([1], [expected_tail])]
    assert lifecycle._state_adapter.draft_initializations == [([1], [20])]
    assert lifecycle._state_adapter.verify_boundaries == [([1], [expected_boundary])]
    assert draft_ctx.boundary_slots.tolist() == [expected_boundary]
    assert draft_ctx.alternate_boundary_slots.tolist() == [expected_next_boundary]


def test_target_extend_replaces_stale_request_state():
    lifecycle, _, batch, _, zeroed, _ = _lifecycle_fixture(
        seq_len=1, last_track=None, prefix_len=0
    )
    lifecycle.boundary_seq_lens[1] = torch.tensor([128, 192])

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    draft_ctx = lifecycle.prepare_for_draft(batch)

    assert zeroed == [11]
    assert draft_ctx.boundary_slots.tolist() == [11]


def test_target_extend_fails_without_a_boundary_source():
    lifecycle, _, batch, *_ = _lifecycle_fixture(
        seq_len=65, last_track=None, prefix_len=0
    )
    lifecycle.prepare_target_extend(batch)

    with pytest.raises(RuntimeError, match="did not restore"):
        lifecycle.finish_target_extend(batch)


def test_target_extend_requires_host_sequence_lengths():
    lifecycle, _, batch, *_ = _lifecycle_fixture(
        seq_len=65, last_track=64, prefix_len=0
    )
    batch.seq_lens_cpu = None

    with pytest.raises(RuntimeError, match="requires seq_lens_cpu"):
        lifecycle.finish_target_extend(batch)


def test_radix_disabled_builds_tracking_from_schedule_batch_lengths():
    lifecycle, req, batch, _, _, _ = _lifecycle_fixture(
        seq_len=65,
        last_track=64,
        prefix_len=0,
        disable_radix=True,
        slots=(10,),
    )

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    draft_ctx = lifecycle.prepare_for_draft(batch)

    assert batch.mamba_track_indices.tolist() == [10]
    assert batch.mamba_track_mask.tolist() == [True]
    assert req.mamba_last_track_seqlen == 64
    assert draft_ctx.boundary_slots.tolist() == [10]


def test_prepare_for_draft_uses_device_authoritative_boundary():
    lifecycle, req, batch, *_ = _lifecycle_fixture(
        seq_len=65, last_track=64, prefix_len=0
    )
    req.mamba_next_track_idx = 0
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)

    ctx = lifecycle.prepare_for_draft(batch)

    assert ctx.boundary_slots.tolist() == [11]
    assert ctx.alternate_boundary_slots.tolist() == [10]
    assert ctx.boundary_lanes.tolist() == [1]


def test_prepare_for_draft_keeps_request_slots_stable_during_radix_insert():
    lifecycle, req, batch, *_ = _lifecycle_fixture(
        seq_len=65, last_track=64, prefix_len=0
    )
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)

    before = req.mamba_ping_pong_track_buffer.clone()
    ctx = lifecycle.prepare_for_draft(batch)

    assert torch.equal(req.mamba_ping_pong_track_buffer, before)
    assert ctx.boundary_slots.tolist() == [10]
    assert ctx.alternate_boundary_slots.tolist() == [11]


def test_rollback_advances_only_the_next_boundary_slot():
    calls = []

    lifecycle, _, batch, *_ = _lifecycle_fixture(
        seq_len=65, last_track=64, prefix_len=0
    )
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    ctx = lifecycle.prepare_for_draft(batch)

    def commit_accepted_state(**kwargs):
        calls.append(kwargs)
        return torch.tensor([True])

    lifecycle._state_adapter.commit_accepted_state = commit_accepted_state

    lifecycle.rollback_after_verify(
        batch=batch,
        transaction=ctx,
        accept_lens=torch.tensor([5], dtype=torch.int32),
    )
    next_ctx = lifecycle.prepare_for_draft(batch)

    assert calls[0]["accepted_token_counts"].tolist() == [5]
    assert next_ctx.boundary_slots.tolist() == [11]
    assert next_ctx.alternate_boundary_slots.tolist() == [10]


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
    req = SimpleNamespace(**values)
    req._cache_commit_len = lambda: req.kv_committed_len
    return req


def test_release_publishes_latest_visible_boundary():
    lifecycle, _, batch, *_ = _lifecycle_fixture(
        seq_len=128, last_track=128, prefix_len=0
    )
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    ctx = lifecycle.prepare_for_draft(batch)
    lifecycle._state_adapter.commit_accepted_state = lambda **_kwargs: torch.tensor(
        [True]
    )
    lifecycle.rollback_after_verify(
        batch=batch,
        transaction=ctx,
        accept_lens=torch.tensor([64], dtype=torch.int32),
    )
    req = _finished_req()

    lifecycle.prepare_for_cache_release(req)

    assert req.mamba_last_track_seqlen == 192
    assert req.mamba_next_track_idx == 0


def test_release_skips_insert_if_committed_boundary_was_lost():
    lifecycle, _, batch, *_ = _lifecycle_fixture(
        seq_len=256, last_track=256, prefix_len=0
    )
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    req = _finished_req(kv_committed_len=128)

    lifecycle.prepare_for_cache_release(req)

    assert req.skip_radix_cache_insert
    assert lifecycle.boundary_seq_lens[req.req_pool_idx].tolist() == [-1, -1]


def test_prepare_for_draft_fails_without_an_exact_boundary():
    lifecycle, _, batch, *_ = _lifecycle_fixture(
        seq_len=65, last_track=64, prefix_len=0
    )

    with pytest.raises(RuntimeError, match="exact recurrent checkpoint"):
        lifecycle.prepare_for_draft(batch)


def test_radix_disabled_release_drops_request_local_boundary():
    lifecycle, _, batch, *_ = _lifecycle_fixture(
        seq_len=65,
        last_track=64,
        prefix_len=0,
        disable_radix=True,
        slots=(10,),
    )
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    req = _finished_req()

    lifecycle.prepare_for_cache_release(req)

    assert not req.skip_radix_cache_insert
