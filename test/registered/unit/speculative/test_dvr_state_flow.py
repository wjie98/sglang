from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.speculative.dvr_state_flow import DVRLinearStateLifecycle
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
    draft_reuses_target_state = False

    def __init__(self, zeroed, tail_updates):
        self.zeroed = zeroed
        self.tail_updates = tail_updates

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
    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle.server_args = SimpleNamespace(
        disable_radix_cache=disable_radix,
        strip_thinking_cache=False,
        enable_mamba_extra_buffer=lambda: not disable_radix,
    )
    lifecycle._state_adapter = _Adapter(zeroed, tail_updates)
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
    draft_ctx = lifecycle.prepare_for_draft(batch)

    assert copies == ([] if expected_copy is None else [expected_copy])
    assert tails == [([1], [expected_tail])]
    assert draft_ctx.boundary_indices.tolist() == [10]
    assert draft_ctx.next_boundary_indices.tolist() == [11]


def test_target_extend_replaces_stale_request_state():
    lifecycle, _, batch, _, zeroed, _ = _lifecycle_fixture(
        seq_len=1, last_track=None, prefix_len=0
    )
    lifecycle.boundary_lens[1] = torch.tensor([128, 192])

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    draft_ctx = lifecycle.prepare_for_draft(batch)

    assert zeroed == [10]
    assert draft_ctx.boundary_indices.tolist() == [10]


def test_target_extend_fails_without_a_boundary_source():
    lifecycle, _, batch, *_ = _lifecycle_fixture(
        seq_len=65, last_track=None, prefix_len=0
    )
    lifecycle.prepare_target_extend(batch)

    with pytest.raises(RuntimeError, match="did not restore"):
        lifecycle.finish_target_extend(batch)


def test_radix_disabled_uses_request_local_boundary():
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
    assert draft_ctx.boundary_indices.tolist() == [10]


def test_prepare_for_draft_uses_device_authoritative_boundary():
    lifecycle, req, batch, *_ = _lifecycle_fixture(
        seq_len=65, last_track=64, prefix_len=0
    )
    req.mamba_next_track_idx = 1
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)

    ctx = lifecycle.prepare_for_draft(batch)

    assert ctx.boundary_indices.tolist() == [11]
    assert ctx.next_boundary_indices.tolist() == [10]
    assert ctx.boundary_track_indices.tolist() == [1]


def test_rollback_advances_only_the_next_boundary_slot():
    calls = []

    lifecycle, _, batch, *_ = _lifecycle_fixture(
        seq_len=65, last_track=64, prefix_len=0
    )
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    ctx = lifecycle.prepare_for_draft(batch)

    def commit_after_verify(**kwargs):
        calls.append(kwargs)
        return torch.tensor([True])

    lifecycle._state_adapter.commit_after_verify = commit_after_verify

    lifecycle.rollback_after_verify(
        batch=batch, ctx=ctx, accept_lens=torch.tensor([5], dtype=torch.int32)
    )
    next_ctx = lifecycle.prepare_for_draft(batch)

    assert calls[0]["accepted_token_counts"].tolist() == [5]
    assert next_ctx.boundary_indices.tolist() == [11]
    assert next_ctx.next_boundary_indices.tolist() == [10]


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


def test_release_publishes_latest_visible_boundary():
    lifecycle, _, batch, *_ = _lifecycle_fixture(
        seq_len=128, last_track=128, prefix_len=0
    )
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    ctx = lifecycle.prepare_for_draft(batch)
    lifecycle._state_adapter.commit_after_verify = lambda **_kwargs: torch.tensor(
        [True]
    )
    lifecycle.rollback_after_verify(
        batch=batch, ctx=ctx, accept_lens=torch.tensor([64], dtype=torch.int32)
    )
    req = _finished_req()

    lifecycle.prepare_for_cache_release(req)

    assert req.mamba_last_track_seqlen == 192
    assert req.mamba_next_track_idx == 0


def test_release_fails_if_committed_boundary_was_lost():
    lifecycle, _, batch, *_ = _lifecycle_fixture(
        seq_len=256, last_track=256, prefix_len=0
    )
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)

    with pytest.raises(RuntimeError, match="lost the recurrent checkpoint"):
        lifecycle.prepare_for_cache_release(_finished_req(kv_committed_len=128))


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
