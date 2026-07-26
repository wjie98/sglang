from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.speculative.dvr_state_flow import DVRStateLifecycle
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class FakePool:
    def __init__(self, track_slots=(10, 11)):
        self.mamba_ping_pong_track_buffer_size = len(track_slots)
        self.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
            8, len(track_slots), dtype=torch.int64
        )
        self.req_index_to_mamba_ping_pong_track_buffer_mapping[1] = torch.tensor(
            track_slots
        )

    def get_mamba_ping_pong_keep_idx(self, req):
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - req.mamba_next_track_idx
        return req.mamba_next_track_idx


class FakeAdapter:
    chunk_size = 64

    def __init__(self):
        self.draft_state = torch.empty(1, 8, 1)
        self.zeroed = []
        self.published = []
        self.initialized = []
        self.commits = []
        self.crosses_boundary = torch.tensor([False])

    @staticmethod
    def resolve_request_slots(*, batch):
        return batch.req_pool_indices.to(torch.long), batch.live_slots

    def zero_boundary_state(self, *, indices):
        self.zeroed.extend(indices.tolist())

    def publish_boundary_state(self, **kwargs):
        self.published.append({name: value.tolist() for name, value in kwargs.items()})

    def initialize_self_draft_state(self, **kwargs):
        self.initialized.append(
            {name: value.tolist() for name, value in kwargs.items()}
        )

    def commit_accepted_state(self, **kwargs):
        self.commits.append(kwargs)
        return self.crosses_boundary


def make_lifecycle(*, disable_radix=False, track_slots=(10, 11)):
    pool = FakePool(track_slots)
    runner = SimpleNamespace(req_to_token_pool=pool, mambaish_config=object())
    server_args = SimpleNamespace(
        disable_radix_cache=disable_radix,
        mamba_track_interval=64,
        mamba_cache_chunk_size=64,
    )
    lifecycle = DVRStateLifecycle(server_args=server_args, model_runner=runner)
    adapter = FakeAdapter()
    lifecycle.bind_state_adapter(adapter)
    return lifecycle, adapter, pool


def make_batch(*, seq_len, prefix_len=0, next_track=1, track_slots=(10, 11)):
    req = SimpleNamespace(
        rid="r0",
        req_pool_idx=1,
        mamba_next_track_idx=next_track,
        mamba_ping_pong_track_buffer=torch.tensor(track_slots),
        skip_radix_cache_insert=False,
    )
    batch = SimpleNamespace(
        reqs=[req],
        req_pool_indices=torch.tensor([1]),
        req_to_token_pool=None,
        live_slots=torch.tensor([20]),
        seq_lens=torch.tensor([seq_len]),
        seq_lens_cpu=torch.tensor([seq_len]),
        prefix_lens=[prefix_len],
        batch_size=lambda: 1,
    )
    return req, batch


def attach_pool(batch, pool):
    batch.req_to_token_pool = pool
    return batch


def test_no_radix_uses_only_the_live_mamba_slot():
    server_args = SimpleNamespace(
        speculative_num_draft_tokens=16,
        speculative_eagle_topk=1,
        max_running_requests=8,
        max_mamba_cache_size=None,
        disable_radix_cache=True,
        dp_size=1,
        enable_dp_attention=False,
        enable_mamba_extra_buffer=lambda: False,
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
        start_layer=0,
        end_layer=1,
    )
    runner._calculate_mamba_ratio = lambda: (
        ModelRunnerKVCacheMixin._calculate_mamba_ratio(runner)
    )

    ModelRunnerKVCacheMixin.handle_max_mamba_cache(runner, total_rest_memory=1.0)

    assert runner._calculate_mamba_ratio() == 1
    assert server_args.max_mamba_cache_size == 8


def test_dvr_memory_budget_includes_cuda_graph_dummy_row(monkeypatch):
    workspace_state_slots = 2
    helper_args = {}

    def state_slots(*args, **kwargs):
        helper_args.update(kwargs)
        return workspace_state_slots

    monkeypatch.setattr(
        "sglang.srt.layers.attention.dvr.gdn_backend."
        "dvr_gdn_workspace_state_slots",
        state_slots,
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
            mamba2_cache_params=SimpleNamespace(
                mamba_cache_per_req=1024,
                layers=(0, 1, 2, 3),
            )
        ),
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
        dp_size=1,
        start_layer=1,
        end_layer=3,
        _calculate_mamba_ratio=lambda: 3,
    )

    remaining = ModelRunnerKVCacheMixin.handle_max_mamba_cache(
        runner, total_rest_memory=1.0
    )

    expected_bytes = (2 + 1) * workspace_state_slots * 1024 + 8 * 1024
    assert helper_args == {"num_layers": 2}
    assert remaining == pytest.approx(1.0 - expected_bytes / (1 << 30))


@pytest.mark.parametrize(
    ("graph_max_bs", "max_running_requests", "expected_capacity"),
    [(4, 48, 48), (64, 48, 64)],
)
def test_self_dvr_proposal_budget_matches_chain_capacity(
    monkeypatch, graph_max_bs, max_running_requests, expected_capacity
):
    available_gb = 2
    monkeypatch.setattr(
        "sglang.srt.model_executor.model_runner_kv_cache_mixin."
        "get_available_gpu_memory",
        lambda *args, **kwargs: available_gb,
    )
    monkeypatch.setattr(
        "sglang.srt.model_executor.model_runner_kv_cache_mixin.get_world_group",
        lambda: SimpleNamespace(world_size=1, cpu_group=None),
    )
    num_draft_steps = 3
    vocab_size = 10
    runner = SimpleNamespace(
        device="cuda",
        gpu_id=0,
        mem_fraction_static=1.0,
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(max_bs=graph_max_bs)
            ),
            max_running_requests=max_running_requests,
            speculative_num_steps=num_draft_steps,
        ),
        model_config=SimpleNamespace(vocab_size=vocab_size),
        mambaish_config=None,
    )

    available_bytes = ModelRunnerKVCacheMixin._profile_available_bytes(
        runner, pre_model_load_memory=available_gb
    )

    proposal_bytes = (
        expected_capacity * num_draft_steps * vocab_size * torch.float32.itemsize
    )
    assert available_bytes == available_gb * (1 << 30) - proposal_bytes


def test_radix_uses_upstream_ping_pong_capacity():
    server_args = SimpleNamespace(
        disable_radix_cache=False,
        disable_overlap_schedule=False,
        enable_mamba_extra_buffer=lambda: True,
        enable_mamba_extra_buffer_lazy=lambda: False,
    )
    runner = SimpleNamespace(server_args=server_args)

    assert ModelRunnerKVCacheMixin._calculate_mamba_ratio(runner) == 5


@pytest.mark.parametrize(
    ("seq_len", "expected_boundary", "expected_tail"),
    [(1, 0, 1), (63, 0, 63), (64, 64, 0), (65, 64, 1)],
)
def test_target_extend_records_live_boundary(seq_len, expected_boundary, expected_tail):
    lifecycle, adapter, pool = make_lifecycle()
    _, batch = make_batch(seq_len=seq_len)
    attach_pool(batch, pool)

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    plan = lifecycle.prepare_for_draft(batch)

    assert lifecycle.boundary_seq_lens[1].item() == expected_boundary
    assert plan.live_boundary_slots.tolist() == [20]
    assert plan.accepted_tail_lens.tolist() == [expected_tail]
    assert adapter.initialized[-1]["accepted_tail_lens"] == [expected_tail]
    assert adapter.zeroed == ([20] if expected_boundary == 0 else [])


def test_new_prefill_boundary_is_copied_to_radix_tracking_slot():
    lifecycle, adapter, pool = make_lifecycle()
    _, batch = make_batch(seq_len=65, next_track=1)
    attach_pool(batch, pool)

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    plan = lifecycle.prepare_for_draft(batch)

    assert adapter.published == [
        {
            "source_slots": [20],
            "destination_slots": [10],
            "publish_mask": [True],
        }
    ]
    assert lifecycle.published_boundary_lens[1].tolist() == [64, -1]
    assert plan.publish_boundary_slots.tolist() == [11]
    assert plan.publish_boundary_lanes.tolist() == [1]


def test_warm_partial_extend_needs_no_new_radix_checkpoint():
    lifecycle, adapter, pool = make_lifecycle()
    _, batch = make_batch(seq_len=65, prefix_len=64, next_track=0)
    attach_pool(batch, pool)

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    plan = lifecycle.prepare_for_draft(batch)

    assert adapter.published[-1]["publish_mask"] == [False]
    assert lifecycle.published_boundary_lens[1].tolist() == [-1, -1]
    assert plan.live_boundary_slots.tolist() == [20]
    assert plan.publish_boundary_slots.tolist() == [10]


def test_no_radix_boundary_crossing_updates_only_live_state():
    lifecycle, adapter, pool = make_lifecycle(disable_radix=True, track_slots=(10,))
    _, batch = make_batch(seq_len=63, track_slots=(10,))
    attach_pool(batch, pool)
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    plan = lifecycle.prepare_for_draft(batch)
    adapter.crosses_boundary = torch.tensor([True])

    lifecycle.commit_verified_state(
        batch=batch, plan=plan, accept_lens=torch.tensor([2])
    )

    assert adapter.commits[-1]["publish_boundary_slots"] is None
    assert lifecycle.boundary_seq_lens[1].item() == 64


def test_radix_boundary_crossing_rotates_publication_lane():
    lifecycle, adapter, pool = make_lifecycle()
    _, batch = make_batch(seq_len=65, next_track=1)
    attach_pool(batch, pool)
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    plan = lifecycle.prepare_for_draft(batch)
    adapter.crosses_boundary = torch.tensor([True])

    lifecycle.commit_verified_state(
        batch=batch, plan=plan, accept_lens=torch.tensor([63])
    )

    assert adapter.commits[-1]["accepted_conv_slots"].tolist() == [20]
    assert adapter.commits[-1]["publish_boundary_slots"].tolist() == [11]
    assert lifecycle.boundary_seq_lens[1].item() == 128
    assert lifecycle.published_boundary_lens[1].tolist() == [64, 128]


def test_prepare_for_draft_rejects_stale_live_boundary():
    lifecycle, _, pool = make_lifecycle()
    _, batch = make_batch(seq_len=65)
    attach_pool(batch, pool)

    with pytest.raises(RuntimeError, match="latest exact recurrent boundary"):
        lifecycle.prepare_for_draft(batch)


def test_target_extend_requires_host_sequence_lengths():
    lifecycle, _, pool = make_lifecycle()
    _, batch = make_batch(seq_len=65)
    attach_pool(batch, pool)
    batch.seq_lens_cpu = None

    with pytest.raises(RuntimeError, match="requires seq_lens_cpu"):
        lifecycle.finish_target_extend(batch)


def test_release_selects_latest_visible_radix_boundary():
    lifecycle, adapter, pool = make_lifecycle()
    req, batch = make_batch(seq_len=65, next_track=1)
    attach_pool(batch, pool)
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    plan = lifecycle.prepare_for_draft(batch)
    adapter.crosses_boundary = torch.tensor([True])
    lifecycle.commit_verified_state(
        batch=batch, plan=plan, accept_lens=torch.tensor([63])
    )
    req.kv_committed_len = 128
    req.finished_reason = None
    req._cache_commit_len = lambda: req.kv_committed_len

    lifecycle.prepare_for_cache_release(req)

    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 0
    assert lifecycle.boundary_seq_lens[1].item() == -1
    assert lifecycle.published_boundary_lens[1].tolist() == [-1, -1]
