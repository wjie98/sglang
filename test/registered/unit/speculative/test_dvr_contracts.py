from types import SimpleNamespace

import pytest
import torch

import sglang.srt.speculative.dvr_worker as dvr_worker_module
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.linear.dvr_gdn import DVRGDNStateAdapter
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.managers.overlap_utils import decide_needs_cpu_seq_lens
from sglang.srt.managers.utils import EmbeddingBatchResult, GenerationBatchResult
from sglang.srt.model_executor.dvr_draft_cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
    DVRTargetVerifyCudaGraphRunner,
    _ensure_decode_custom_all_reduce_comm,
    iter_dvr_attention_backends,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dvr_state_flow import (
    DVRLinearStateLifecycle,
    DVRRollbackActions,
)
from sglang.srt.speculative.dvr_worker import (
    DecodeVerifyRollbackWorkerV2,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import renorm_draft_probs, renorm_sampling_probs


def test_dvr_spec_algorithm_contracts():
    self_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
    eagle_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE

    assert self_draft.is_dvr()
    assert self_draft.is_dvr_self_draft()
    assert not self_draft.is_dvr_eagle()
    assert eagle_draft.is_dvr()
    assert eagle_draft.is_dvr_eagle()
    assert not eagle_draft.is_eagle()


def test_dvr_worker_uses_base_spec_worker_contract():
    assert issubclass(DecodeVerifyRollbackWorkerV2, BaseSpecWorker)
    assert not DecodeVerifyRollbackWorkerV2.__abstractmethods__


def test_dvr_target_graph_keeps_verify_deterministic_for_both_draft_backends():
    assert DVRTargetVerifyCudaGraphRunner.dvr_target_verify_cuda_graph


def test_dvr_without_radix_allocates_recurrent_slots_per_request():
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
        shape=SimpleNamespace(conv=((2, 4),), temporal=(2, 2)),
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

    assert server_args.max_mamba_cache_size == 8 * runner._calculate_mamba_ratio()


def test_overlap_batch_results_default_to_no_result_process_fence():
    assert GenerationBatchResult().result_process_ready_event is None
    assert (
        EmbeddingBatchResult(
            embeddings=torch.empty((0,), dtype=torch.float32)
        ).result_process_ready_event
        is None
    )


@pytest.mark.parametrize(
    "algorithm",
    [
        "DECODE_VERIFY_ROLLBACK",
        "DECODE_VERIFY_ROLLBACK_EAGLE",
    ],
)
def test_dvr_future_map_respects_backend_cpu_seq_lens_policy(algorithm):
    server_args = SimpleNamespace(
        enable_two_batch_overlap=False,
        speculative_algorithm=algorithm,
    )
    backend = SimpleNamespace(needs_cpu_seq_lens=False)

    assert not decide_needs_cpu_seq_lens(server_args, [backend])


def test_dvr_draft_proposal_applies_min_p():
    sampling_info = SimpleNamespace(
        top_ks=torch.tensor([3]),
        top_ps=torch.tensor([1.0]),
        min_ps=torch.tensor([0.5]),
        need_min_p_sampling=True,
    )
    proposal = renorm_sampling_probs(torch.tensor([[0.6, 0.3, 0.1]]), sampling_info)

    torch.testing.assert_close(proposal, torch.tensor([[2 / 3, 1 / 3, 0.0]]))


def test_upstream_eagle_draft_proposal_keeps_unfiltered_distribution():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    sampling_info = SimpleNamespace(
        temperatures=torch.tensor([[0.5]]),
        top_ks=torch.tensor([1]),
        top_ps=torch.tensor([0.1]),
        min_ps=torch.tensor([0.9]),
    )

    proposal = renorm_draft_probs(logits, sampling_info, True)

    torch.testing.assert_close(
        proposal, torch.softmax(logits / sampling_info.temperatures, dim=-1)
    )


def test_dvr_seq_lens_uses_current_batch_when_host_mirror_is_absent():
    assert DVRLinearStateLifecycle.batch_seq_lens_cpu(
        SimpleNamespace(seq_lens_cpu=None, seq_lens=torch.tensor([8]))
    ) == [8]


def test_dvr_self_draft_weight_update_does_not_reload_target():
    worker = object.__new__(DecodeVerifyRollbackWorkerV2)
    worker.is_dvr_eagle = False
    assert worker.update_weights_from_disk(SimpleNamespace()) == (
        True,
        "Succeeded to update model weights.",
    )
    assert worker.update_weights_from_ipc(SimpleNamespace()) == (
        True,
        "Succeeded to update model weights.",
    )


def test_dvr_pending_mamba_checkpoint_commit_guards():
    class Pool:
        @staticmethod
        def get_mamba_ping_pong_other_idx(track_idx):
            return 1 - track_idx

    req = SimpleNamespace(
        rid="r0",
        origin_input_ids=[1] * 64,
        output_ids=[2] * 64,
        kv_committed_len=128,
        mamba_last_track_seqlen=64,
        mamba_next_track_idx=0,
        mamba_ping_pong_track_buffer=torch.tensor([10, 11]),
    )
    batch = SimpleNamespace(
        req_to_token_pool=Pool(),
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
        enable_overlap=True,
    )
    tree_cache = SimpleNamespace(page_size=64)

    dvr_rollback_actions = DVRRollbackActions(
        pending_checkpoints=[(0, 128)],
    )
    assert dvr_rollback_actions.commit_checkpoint_after_decode(
        req=req,
        batch=batch,
        req_index=0,
        tree_cache=tree_cache,
    )
    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 1

    dvr_rollback_actions.pending_checkpoints = [(1, 128)]
    assert dvr_rollback_actions.commit_checkpoint_after_decode(
        req=req,
        batch=batch,
        req_index=0,
        tree_cache=tree_cache,
    )
    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 0

    dvr_rollback_actions.pending_checkpoints = [(0, 192)]
    req.kv_committed_len = 127
    assert dvr_rollback_actions.commit_checkpoint_after_decode(
        req=req,
        batch=batch,
        req_index=0,
        tree_cache=tree_cache,
    )
    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 1
    req.kv_committed_len = 128

    dvr_rollback_actions.pending_checkpoints = None
    with pytest.raises(RuntimeError, match="missing checkpoint actions"):
        dvr_rollback_actions.commit_checkpoint_after_decode(
            req=req,
            batch=batch,
            req_index=0,
            tree_cache=tree_cache,
        )

    dvr_rollback_actions.pending_checkpoints = [(2, 128)]
    req.mamba_last_track_seqlen = 64
    with pytest.raises(RuntimeError, match="invalid tracking slot"):
        dvr_rollback_actions.commit_checkpoint_after_decode(
            req=req,
            batch=batch,
            req_index=0,
            tree_cache=tree_cache,
        )
    assert req.mamba_last_track_seqlen == 64
    assert req.mamba_next_track_idx == 1


def test_dvr_custom_all_reduce_uses_current_dispatch_contract(monkeypatch):
    calls = []

    class FakeCustomAllReduce:
        def __init__(self, *, group, device):
            calls.append(("init", group, device))
            self.world_size = 2
            self.full_nvlink = True
            self.disabled = False
            self.original_disabled = False

    def fake_dispatch(*, group, device):
        calls.append(("dispatch", group, device))
        return FakeCustomAllReduce

    from sglang.srt.distributed.device_communicators import custom_all_reduce

    monkeypatch.setattr(custom_all_reduce, "dispatch_custom_allreduce", fake_dispatch)
    group = SimpleNamespace(
        ca_comm=None,
        world_size=2,
        cpu_group="cpu-group",
        device="cuda:0",
    )

    ca_comm = _ensure_decode_custom_all_reduce_comm(group)

    assert calls == [
        ("dispatch", "cpu-group", "cuda:0"),
        ("init", "cpu-group", "cuda:0"),
    ]
    assert group.ca_comm is ca_comm
    assert ca_comm.disabled
    assert ca_comm.original_disabled


def test_dvr_draft_restores_backend_decode_defaults():
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

    def apply_overrides(backend, method):
        for name, value in method(backend, model_runner):
            setattr(backend, name, value)

    fa3 = SimpleNamespace()
    fa3.fa_impl_ver = 3
    fa3.num_splits = 1
    apply_overrides(fa3, FlashAttentionBackend.get_nondeterministic_decode_overrides)
    assert fa3.num_splits == 0

    flashinfer = SimpleNamespace()
    flashinfer.decode_split_tile_size = 2048
    flashinfer.disable_cuda_graph_kv_split = True
    flashinfer.decode_use_tensor_cores = True
    flashinfer._nondeterministic_decode_use_tensor_cores = False
    apply_overrides(
        flashinfer, FlashInferAttnBackend.get_nondeterministic_decode_overrides
    )
    assert flashinfer.decode_split_tile_size is None
    assert not flashinfer.disable_cuda_graph_kv_split
    assert not flashinfer.decode_use_tensor_cores

    triton_backend = SimpleNamespace()
    triton_backend.max_context_len = 8192
    triton_backend.max_kv_splits = 32
    triton_backend._nondeterministic_decode_config = (False, None, 8)
    triton_backend.split_tile_size = 256
    triton_backend.static_kv_splits = False
    triton_backend.cuda_graph_attn_logits = torch.zeros(4, 2, 32, 8)
    triton_backend.cuda_graph_attn_lse = torch.zeros(4, 2, 32)
    triton_backend.cuda_graph_swa_attn_logits = None
    triton_backend.cuda_graph_num_kv_splits = torch.full((4,), 32, dtype=torch.int32)
    apply_overrides(
        triton_backend, TritonAttnBackend.get_nondeterministic_decode_overrides
    )
    assert triton_backend.max_kv_splits == 8
    assert triton_backend.split_tile_size is None
    assert not triton_backend.static_kv_splits
    assert triton_backend.cuda_graph_attn_logits.shape[2] == 8
    assert triton_backend.cuda_graph_attn_lse.shape[2] == 8
    assert torch.equal(
        triton_backend.cuda_graph_num_kv_splits,
        torch.full((4,), 8, dtype=torch.int32),
    )


def test_dvr_draft_reaches_nested_hybrid_attention_backends():
    full_attention = SimpleNamespace()
    linear_attention = SimpleNamespace()
    hybrid = SimpleNamespace(
        attn_backend_list=[full_attention, linear_attention],
        full_attn_backend=full_attention,
    )

    backends = list(iter_dvr_attention_backends(hybrid))

    assert {id(backend) for backend in backends} == {
        id(hybrid),
        id(full_attention),
        id(linear_attention),
    }

    assert (
        list(iter_dvr_attention_backends(hybrid, full_attention, linear_attention))
        == backends
    )


@pytest.mark.parametrize("is_dvr_eagle", [False, True])
def test_dvr_short_prefix_uses_one_root_verify_sentinel(is_dvr_eagle):
    worker = object.__new__(DecodeVerifyRollbackWorkerV2)
    worker.is_dvr_eagle = is_dvr_eagle
    worker.device = "cpu"
    worker.num_draft_tokens = 4
    batch = SimpleNamespace(
        spec_info=dvr_worker_module.EagleDraftInput(bonus_tokens=torch.tensor([6, 7])),
        seq_lens=torch.tensor([1, 65]),
        seq_lens_cpu=torch.tensor([1, 65]),
        seq_lens_sum=66,
    )

    verify_input = worker._build_one_root_verify_input(batch)

    assert verify_input.draft_token.tolist() == [6, 6, 6, 6, 7, 7, 7, 7]
    assert verify_input.draft_token.dtype == torch.long
    assert verify_input.draft_token.is_contiguous()
    assert verify_input.draft_token_num == 4
    assert verify_input.spec_steps == 0
    assert verify_input.retrieve_index.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert verify_input.retrieve_next_token.tolist() == [
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ]
    assert verify_input.positions.tolist() == [1, 2, 3, 4, 65, 66, 67, 68]
    assert verify_input.capture_hidden_mode == (
        dvr_worker_module.CaptureHiddenMode.FULL
        if is_dvr_eagle
        else dvr_worker_module.CaptureHiddenMode.NULL
    )


def test_dvr_plain_transformer_self_draft_rejects_eager_decode():
    worker = object.__new__(DecodeVerifyRollbackWorkerV2)
    worker.cuda_graph_runner_for_draft_decode = None

    with pytest.raises(RuntimeError, match="requires the dedicated CUDA graph"):
        worker._draft_decode_forward(
            SimpleNamespace(seq_lens_cpu=torch.tensor([2]), batch_size=1)
        )


def test_dvr_linear_state_is_optional_for_plain_transformers():
    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle._state_adapter = None
    lifecycle.state_backup = object()

    lifecycle.backup_boundary_state(SimpleNamespace())

    assert not lifecycle.has_state_adapter
    assert lifecycle.state_backup is None

    assert (
        lifecycle.rollback_after_verify(
            batch=SimpleNamespace(
                spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
                enable_overlap=False,
                decoding_reqs=None,
            ),
            ctx=None,
            accept_lens=torch.empty(0, dtype=torch.int32),
        )
        is None
    )


def test_dvr_sync_rollback_keeps_authoritative_checkpoint_action():
    class Adapter:
        chunk_size = 64

        def commit_after_verify(self, **_kwargs):
            pass

    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle._state_adapter = Adapter()
    lifecycle.boundaries = {1: SimpleNamespace(rid="r0", track_idx=0, seq_len=64)}
    req = SimpleNamespace(rid="r0", req_pool_idx=1, mamba_last_track_seqlen=64)
    batch = SimpleNamespace(
        reqs=[req],
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
        enable_overlap=False,
        decoding_reqs=None,
    )
    ctx = SimpleNamespace(
        state_cache=object(),
        state_input_indices=torch.tensor([1]),
        live_indices=torch.tensor([2]),
        boundary_indices=torch.tensor([3]),
    )

    actions = lifecycle.rollback_after_verify(
        batch=batch,
        ctx=ctx,
        accept_lens=torch.tensor([1], dtype=torch.int32),
    )

    assert actions is not None
    assert actions.pending_checkpoints == [(0, 128)]


def test_dvr_self_draft_requires_graph_for_gdn_normal_decode():
    worker = object.__new__(DecodeVerifyRollbackWorkerV2)
    worker.cuda_graph_runner_for_draft_decode = None
    worker.linear_state = SimpleNamespace(has_state_adapter=True)

    with pytest.raises(RuntimeError, match="requires the dedicated CUDA graph"):
        worker._draft_decode_forward(
            SimpleNamespace(seq_lens_cpu=torch.tensor([3]), batch_size=1)
        )


def test_dvr_boundary_metadata_advances_from_published_checkpoint():
    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle._state_adapter = SimpleNamespace(chunk_size=64)
    lifecycle.boundaries = {
        0: SimpleNamespace(rid="r0", track_idx=1, seq_len=64, publish_pending=False)
    }
    lifecycle._ensure_boundary_state = lambda *_args, **_kwargs: None
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="r0", req_pool_idx=0, mamba_last_track_seqlen=128)],
        enable_overlap=True,
    )

    lifecycle.prepare_for_draft(batch)

    checkpoint = lifecycle.boundaries[0]
    assert (checkpoint.rid, checkpoint.track_idx, checkpoint.seq_len) == (
        "r0",
        1,
        128,
    )


def test_dvr_existing_boundary_does_not_resolve_gpu_lengths():
    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle._state_adapter = SimpleNamespace(chunk_size=64)
    lifecycle.boundaries = {
        0: SimpleNamespace(rid="r0", track_idx=1, seq_len=64, publish_pending=False)
    }
    lifecycle.batch_seq_lens_cpu = lambda _batch: pytest.fail(
        "steady decode must not copy seq_lens to the host"
    )
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="r0", req_pool_idx=0, mamba_last_track_seqlen=64)],
        enable_overlap=True,
    )

    lifecycle.prepare_for_draft(batch)


def test_dvr_self_draft_graph_does_not_publish_intermediate_war_event():
    assert not DVRDraftDecodeCudaGraphRunner.record_war_fastpath_event


def test_dvr_prefill_boundary_uses_request_local_track_slot():
    class Pool:
        mamba_pool = SimpleNamespace(copy_from=lambda *_args: None)

        @staticmethod
        def get_mamba_ping_pong_other_idx(track_idx):
            return 1 - track_idx

        @staticmethod
        def get_mamba_ping_pong_keep_idx(req):
            return 1 - req.mamba_next_track_idx

    class Adapter:
        chunk_size = 64

        @staticmethod
        def state_input_window():
            return SimpleNamespace(set_tail_lens=lambda **_kwargs: None)

        @staticmethod
        def zero_recurrent_state(**_kwargs):
            pass

    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle._state_adapter = Adapter()
    lifecycle.boundaries = {}
    lifecycle.state_backup = None
    req = SimpleNamespace(
        rid="r0",
        req_pool_idx=0,
        mamba_last_track_seqlen=None,
        mamba_next_track_idx=0,
        mamba_ping_pong_track_buffer=torch.tensor([10, 11]),
    )
    batch = SimpleNamespace(req_to_token_pool=Pool(), reqs=[req])
    ctx = SimpleNamespace(
        state_cache=object(),
        state_input_indices=torch.tensor([1]),
        live_indices=torch.tensor([10]),
    )

    lifecycle._ensure_boundary_state(
        batch,
        ctx=ctx,
        seq_lens_cpu=[64],
        prefill_prefix_lens=[64],
        publish_to_request=False,
    )
    checkpoint = lifecycle.boundaries[0]
    assert (checkpoint.rid, checkpoint.track_idx, checkpoint.seq_len) == (
        "r0",
        0,
        64,
    )
    assert checkpoint.publish_pending
    assert req.mamba_next_track_idx == 0

    ensure_boundary_state = lifecycle._ensure_boundary_state
    lifecycle._ensure_boundary_state = lambda *_args, **_kwargs: None
    lifecycle.prepare_for_draft(batch, seq_lens_cpu=[64])
    lifecycle._ensure_boundary_state = ensure_boundary_state
    assert req.mamba_next_track_idx == 1

    lifecycle.boundaries.clear()
    req.mamba_last_track_seqlen = None
    with pytest.raises(RuntimeError, match="did not publish"):
        lifecycle._ensure_boundary_state(
            batch,
            ctx=ctx,
            seq_lens_cpu=[128],
            publish_to_request=True,
        )


def test_dvr_boundary_backup_tracks_logical_slot_across_physical_rebind():
    class Adapter:
        def __init__(self):
            self.backup_indices = []
            self.draft_reuses_target_state = True

        def backup_recurrent_state(self, *, indices, include_temporal=True, **_kwargs):
            self.backup_indices.append((indices.tolist(), include_temporal))
            return len(self.backup_indices)

    lifecycle = object.__new__(DVRLinearStateLifecycle)
    checkpoint = SimpleNamespace(
        rid="r0", track_idx=1, seq_len=64, publish_pending=False
    )
    lifecycle.boundaries = {1: checkpoint}
    lifecycle.state_backup = None
    lifecycle.server_args = SimpleNamespace(disable_radix_cache=False)
    adapter = Adapter()
    lifecycle._state_adapter = adapter
    context = SimpleNamespace(
        state_cache=object(),
        state_input_indices=torch.tensor([1]),
        boundary_indices=torch.tensor([11]),
        live_indices=torch.tensor([12]),
    )
    rebound_context = SimpleNamespace(
        state_cache=object(),
        state_input_indices=torch.tensor([1]),
        boundary_indices=torch.tensor([21]),
        live_indices=torch.tensor([22]),
    )
    state_context_calls = []

    def state_context(batch, require_boundary):
        state_context_calls.append((batch, require_boundary))
        return rebound_context if len(state_context_calls) > 1 else context

    lifecycle.state_context = state_context
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="r0", req_pool_idx=1)],
        req_to_token_pool=SimpleNamespace(req_to_token=torch.empty(3, 0)),
        enable_overlap=True,
    )

    lifecycle.backup_boundary_state(batch)
    assert adapter.backup_indices == [([11], True), ([12], False)]
    assert len(state_context_calls) == 1

    # Radix may rebind the physical slot while the logical ping-pong owner is
    # unchanged. Preserve the authoritative verify snapshot before resolving
    # that mutable request mapping again.
    checkpoint.seq_len = 128
    lifecycle.backup_boundary_state(batch)
    assert adapter.backup_indices == [([11], True), ([12], False)]
    assert len(state_context_calls) == 1

    # Post-verify commit supplies the exact physical context used by target
    # verify, independent of the request mapping currently visible to radix.
    lifecycle.backup_boundary_state(batch, ctx=rebound_context)
    assert adapter.backup_indices == [
        ([11], True),
        ([12], False),
        ([21], True),
        ([22], False),
    ]
    assert len(state_context_calls) == 1

    checkpoint.track_idx = 0
    lifecycle.backup_boundary_state(batch)
    assert adapter.backup_indices == [
        ([11], True),
        ([12], False),
        ([21], True),
        ([22], False),
        ([21], True),
        ([22], False),
    ]
    assert len(state_context_calls) == 2

    # A separate EAGLE/MTP draft model cannot mutate the target live slot.
    adapter.draft_reuses_target_state = False
    lifecycle.state_backup = None
    lifecycle.backup_boundary_state(batch, ctx=context)
    assert adapter.backup_indices[-1] == ([11], True)
    assert len(adapter.backup_indices) == 7


def test_dvr_target_extend_invalidates_prior_boundary_snapshot():
    class Adapter:
        draft_reuses_target_state = False
        chunk_size = 64

        def __init__(self):
            self.backup_indices = []

        def backup_recurrent_state(self, *, indices, **_kwargs):
            self.backup_indices.append(indices.tolist())
            return object()

    adapter = Adapter()
    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle.server_args = SimpleNamespace(disable_radix_cache=False)
    lifecycle._state_adapter = adapter
    lifecycle.boundaries = {
        1: SimpleNamespace(
            rid="r0", track_idx=0, seq_len=64, publish_pending=False
        )
    }
    lifecycle.state_backup = None
    lifecycle._ensure_boundary_state = lambda *_args, **_kwargs: None
    contexts = iter(
        [
            SimpleNamespace(
                state_cache=object(),
                state_input_indices=torch.tensor([1]),
                boundary_indices=torch.tensor([11]),
                live_indices=torch.tensor([12]),
            ),
            SimpleNamespace(
                state_cache=object(),
                state_input_indices=torch.tensor([1]),
                boundary_indices=torch.tensor([21]),
                live_indices=torch.tensor([22]),
            ),
        ]
    )
    lifecycle.state_context = lambda *_args, **_kwargs: next(contexts)
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                rid="r0",
                req_pool_idx=1,
                mamba_last_track_seqlen=64,
            )
        ],
        req_to_token_pool=SimpleNamespace(req_to_token=torch.empty(3, 0)),
    )

    lifecycle.backup_boundary_state(batch)
    lifecycle.prepare_for_draft(
        batch,
        seq_lens_cpu=[128],
        prefill_prefix_lens=[64],
    )
    lifecycle.backup_boundary_state(batch)

    assert adapter.backup_indices == [[11], [21]]


def test_dvr_request_local_backup_survives_interleaved_batch_and_rebind():
    conv = torch.zeros(1, 6, 2)
    temporal = torch.zeros(1, 6, 2)
    conv[:, 0] = torch.tensor([1.0, 2.0])
    temporal[:, 0] = torch.tensor([3.0, 4.0])
    conv[:, 3] = torch.tensor([5.0, 6.0])
    conv[:, 1] = torch.tensor([11.0, 12.0])
    temporal[:, 1] = torch.tensor([13.0, 14.0])
    conv[:, 4] = torch.tensor([15.0, 16.0])
    state_cache = SimpleNamespace(conv=(conv,), temporal=temporal)
    adapter = DVRGDNStateAdapter(kernel_dispatcher=None, draft_reuses_target_state=True)

    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle.server_args = SimpleNamespace(disable_radix_cache=False)
    lifecycle._state_adapter = adapter
    lifecycle.boundaries = {
        1: SimpleNamespace(rid="r0", track_idx=0, seq_len=64),
        2: SimpleNamespace(rid="r1", track_idx=0, seq_len=64),
    }
    lifecycle.state_backup = None
    lifecycle._ensure_boundary_state = lambda *_args, **_kwargs: None

    req0 = SimpleNamespace(rid="r0", req_pool_idx=1)
    req1 = SimpleNamespace(rid="r1", req_pool_idx=2)
    pool = SimpleNamespace(req_to_token=torch.empty(3, 0))
    batch0 = SimpleNamespace(reqs=[req0], req_to_token_pool=pool)
    batch1 = SimpleNamespace(reqs=[req1], req_to_token_pool=pool)
    r0_contexts = iter(
        [
            SimpleNamespace(
                state_cache=state_cache,
                state_input_indices=torch.tensor([1]),
                boundary_indices=torch.tensor([0]),
                live_indices=torch.tensor([3]),
            ),
            SimpleNamespace(
                state_cache=state_cache,
                state_input_indices=torch.tensor([1]),
                boundary_indices=torch.tensor([2]),
                live_indices=torch.tensor([5]),
            ),
        ]
    )
    r1_context = SimpleNamespace(
        state_cache=state_cache,
        state_input_indices=torch.tensor([2]),
        boundary_indices=torch.tensor([1]),
        live_indices=torch.tensor([4]),
    )
    lifecycle.state_context = lambda batch, **_kwargs: (
        next(r0_contexts) if batch is batch0 else r1_context
    )

    lifecycle.backup_boundary_state(batch0)
    lifecycle.backup_boundary_state(batch1)
    conv[:, 0].zero_()
    temporal[:, 0].zero_()
    conv[:, 3].zero_()

    # The same logical owner now points at rebound physical slots. Draft
    # preparation must retain r0's old snapshot instead of reading those slots.
    lifecycle.backup_boundary_state(batch0)
    lifecycle.restore_for_verify(batch0)

    torch.testing.assert_close(conv[:, 2], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(temporal[:, 2], torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(conv[:, 5], torch.tensor([[5.0, 6.0]]))
    torch.testing.assert_close(temporal[:, 5], torch.tensor([[3.0, 4.0]]))


def test_dvr_cache_release_restores_committed_request_snapshot():
    conv = torch.zeros(1, 4, 2)
    temporal = torch.zeros(1, 4, 2)
    conv[:, 2] = torch.tensor([1.0, 2.0])
    temporal[:, 2] = torch.tensor([3.0, 4.0])
    conv[:, 3] = torch.tensor([5.0, 6.0])
    state_cache = SimpleNamespace(conv=(conv,), temporal=temporal)
    adapter = DVRGDNStateAdapter(kernel_dispatcher=None, draft_reuses_target_state=True)

    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle.server_args = SimpleNamespace(disable_radix_cache=False)
    lifecycle._state_adapter = adapter
    lifecycle.boundaries = {1: SimpleNamespace(rid="r0", track_idx=0, seq_len=64)}
    lifecycle.state_backup = None
    req = SimpleNamespace(
        rid="r0",
        req_pool_idx=1,
        mamba_next_track_idx=0,
        mamba_ping_pong_track_buffer=torch.tensor([2, 1]),
    )
    pool = SimpleNamespace(
        req_to_token=torch.empty(2, 0),
        get_speculative_mamba2_params_all_layers=lambda: state_cache,
        get_mamba_ping_pong_other_idx=lambda track_idx: 1 - track_idx,
    )
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=pool)
    ctx = SimpleNamespace(
        state_cache=state_cache,
        state_input_indices=torch.tensor([1]),
        boundary_indices=torch.tensor([2]),
        live_indices=torch.tensor([3]),
    )

    lifecycle.backup_boundary_state(batch, ctx=ctx)
    conv[:, 2].zero_()
    temporal[:, 2].zero_()
    lifecycle.restore_for_cache_release(req, pool)

    torch.testing.assert_close(conv[:, 2], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(temporal[:, 2], torch.tensor([[3.0, 4.0]]))
    assert req.mamba_next_track_idx == 1
    assert lifecycle.boundaries == {}
    assert lifecycle.state_backup.slot_owners == {}


@pytest.mark.parametrize(
    ("reuse_target", "overlap", "disable_radix", "expected"),
    [
        (True, False, False, [([11], True), ([12], False)]),
        (True, True, True, [([12], False)]),
        (False, False, False, [([11], True)]),
        (False, True, True, []),
    ],
)
def test_dvr_boundary_backup_covers_radix_or_self_draft_mutation(
    reuse_target, overlap, disable_radix, expected
):
    class Adapter:
        draft_reuses_target_state = reuse_target

        def __init__(self):
            self.backup_indices = []

        def backup_recurrent_state(self, *, indices, include_temporal=True, **_kwargs):
            self.backup_indices.append((indices.tolist(), include_temporal))
            return object()

    adapter = Adapter()
    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle.server_args = SimpleNamespace(disable_radix_cache=disable_radix)
    lifecycle._state_adapter = adapter
    lifecycle.boundaries = {1: SimpleNamespace(rid="r0", track_idx=0, seq_len=64)}
    lifecycle.state_backup = None
    lifecycle.state_context = lambda *_args, **_kwargs: SimpleNamespace(
        state_cache=object(),
        state_input_indices=torch.tensor([1]),
        boundary_indices=torch.tensor([11]),
        live_indices=torch.tensor([12]),
    )
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="r0", req_pool_idx=1)],
        req_to_token_pool=SimpleNamespace(req_to_token=torch.empty(3, 0)),
        enable_overlap=overlap,
    )

    lifecycle.backup_boundary_state(batch)

    assert adapter.backup_indices == expected
