from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.dvr_draft_cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
    _ensure_decode_custom_all_reduce_comm,
    _patch_draft_decode_backend_defaults,
    dvr_self_draft_graph_block_reason,
    iter_dvr_attention_backends,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dvr_worker import DecodeVerifyRollbackWorkerV2
from sglang.srt.speculative.dvr_core import DVRRollbackActions
from sglang.srt.speculative.dvr_state_flow import DVRLinearStateLifecycle
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


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
        pending_mamba_checkpoints=[
            (0, 128),
        ],
    )
    assert dvr_rollback_actions.commit_checkpoint_after_decode(
        req=req,
        batch=batch,
        req_index=0,
        tree_cache=tree_cache,
    )
    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 1

    dvr_rollback_actions.pending_mamba_checkpoints = [(1, 128)]
    assert dvr_rollback_actions.commit_checkpoint_after_decode(
        req=req,
        batch=batch,
        req_index=0,
        tree_cache=tree_cache,
    )
    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 1

    dvr_rollback_actions.pending_mamba_checkpoints = [(2, 192)]
    req.kv_committed_len = 127
    assert dvr_rollback_actions.commit_checkpoint_after_decode(
        req=req,
        batch=batch,
        req_index=0,
        tree_cache=tree_cache,
    )
    assert req.mamba_last_track_seqlen == 128
    req.kv_committed_len = 128

    dvr_rollback_actions.pending_mamba_checkpoints = None
    with pytest.raises(RuntimeError, match="missing Mamba checkpoint actions"):
        dvr_rollback_actions.commit_checkpoint_after_decode(
            req=req,
            batch=batch,
            req_index=0,
            tree_cache=tree_cache,
        )

    dvr_rollback_actions.pending_mamba_checkpoints = [(2, 128)]
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


def test_dvr_draft_restores_backend_decode_defaults(monkeypatch):
    monkeypatch.delenv("SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", raising=False)
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

    def patch_attr(obj, name, value):
        if hasattr(obj, name):
            setattr(obj, name, value)

    FlashAttentionBackend = type("FlashAttentionBackend", (), {})
    fa3 = FlashAttentionBackend()
    fa3.fa_impl_ver = 3
    fa3.num_splits = 1
    _patch_draft_decode_backend_defaults(fa3, model_runner, patch_attr)
    assert fa3.num_splits == 0

    FlashInferAttnBackend = type("FlashInferAttnBackend", (), {})
    flashinfer = FlashInferAttnBackend()
    flashinfer.decode_split_tile_size = 2048
    flashinfer.disable_cuda_graph_kv_split = True
    flashinfer.decode_use_tensor_cores = True
    monkeypatch.setenv("SGLANG_FLASHINFER_USE_TENSOR_CORE", "false")
    _patch_draft_decode_backend_defaults(flashinfer, model_runner, patch_attr)
    assert flashinfer.decode_split_tile_size is None
    assert not flashinfer.disable_cuda_graph_kv_split
    assert not flashinfer.decode_use_tensor_cores

    TritonAttnBackend = type("TritonAttnBackend", (), {})
    triton_backend = TritonAttnBackend()
    triton_backend.max_context_len = 8192
    triton_backend.max_kv_splits = 32
    triton_backend.split_tile_size = 256
    triton_backend.static_kv_splits = False
    triton_backend.cuda_graph_attn_logits = torch.zeros(4, 2, 32, 8)
    triton_backend.cuda_graph_attn_lse = torch.zeros(4, 2, 32)
    triton_backend.cuda_graph_swa_attn_logits = None
    triton_backend.cuda_graph_num_kv_splits = torch.full(
        (4,), 32, dtype=torch.int32
    )
    _patch_draft_decode_backend_defaults(triton_backend, model_runner, patch_attr)
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


def test_dvr_self_draft_graph_runner_rejects_short_boundary():
    graph_runner = object.__new__(DVRDraftDecodeCudaGraphRunner)
    graph_runner.runner = SimpleNamespace(can_run_graph=lambda forward_batch: True)

    short_batch = SimpleNamespace(seq_lens_cpu=torch.tensor([2]), batch_size=1)
    assert dvr_self_draft_graph_block_reason(short_batch) is not None
    assert not graph_runner.can_run(short_batch)
    assert graph_runner.can_run(
        SimpleNamespace(
            dvr_disable_draft_cuda_graph=True,
            seq_lens_cpu=torch.tensor([4]),
            batch_size=1,
        )
    )
    assert graph_runner.can_run(
        SimpleNamespace(seq_lens_cpu=torch.tensor([3]), batch_size=1)
    )


def test_dvr_self_draft_rejects_short_boundary_without_eager_fallback():
    worker = object.__new__(DecodeVerifyRollbackWorkerV2)
    worker.cuda_graph_runner_for_draft_decode = None

    with pytest.raises(RuntimeError, match="no eager fallback"):
        worker._draft_decode_forward(
            SimpleNamespace(seq_lens_cpu=torch.tensor([2]), batch_size=1)
        )


def test_dvr_self_draft_requires_graph_for_gdn_normal_decode():
    worker = object.__new__(DecodeVerifyRollbackWorkerV2)
    worker.cuda_graph_runner_for_draft_decode = None
    worker.model_runner = SimpleNamespace(hybrid_gdn_config=object())

    with pytest.raises(RuntimeError, match="requires the dedicated CUDA graph"):
        worker._draft_decode_forward(
            SimpleNamespace(seq_lens_cpu=torch.tensor([3]), batch_size=1)
        )


def test_dvr_boundary_metadata_advances_from_next_scheduler_length():
    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle.boundary_seqlen = {"r0": 64}
    lifecycle.boundary_track_idx = {"r0": 1}
    lifecycle.ensure_boundary_state = lambda batch, seq_lens_cpu: []
    batch = SimpleNamespace(reqs=[SimpleNamespace(rid="r0")])

    lifecycle.prepare_for_draft(batch, seq_lens_cpu=[128])

    assert lifecycle.boundary_seqlen == {"r0": 128}
    assert lifecycle.boundary_track_idx == {"r0": 1}


def test_dvr_boundary_backup_tracks_logical_slot_across_physical_rebind():
    class Adapter:
        def __init__(self):
            self.backup_indices = []

        def backup_recurrent_state(self, *, indices, **_kwargs):
            self.backup_indices.append(indices.tolist())
            return len(self.backup_indices)

    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle.boundary_seqlen = {"r0": 64}
    lifecycle.boundary_track_idx = {"r0": 1}
    lifecycle.boundary_backup = None
    lifecycle.boundary_backup_keys = None
    lifecycle.live_backup = None
    adapter = Adapter()
    context = SimpleNamespace(
        state_adapter=adapter,
        state_cache=object(),
        boundary_indices=torch.tensor([11]),
        live_indices=torch.tensor([12]),
    )
    rebound_context = SimpleNamespace(
        state_adapter=adapter,
        state_cache=object(),
        boundary_indices=torch.tensor([21]),
        live_indices=torch.tensor([22]),
    )
    state_context_calls = []

    def state_context(batch, require_boundary):
        state_context_calls.append((batch, require_boundary))
        return rebound_context if len(state_context_calls) > 1 else context

    lifecycle.state_context = state_context
    batch = SimpleNamespace(reqs=[SimpleNamespace(rid="r0")])

    lifecycle.backup_boundary_state(batch)
    assert adapter.backup_indices == [[11], [12]]
    assert len(state_context_calls) == 1

    # Radix may rebind the physical slot while the logical ping-pong owner is
    # unchanged. Preserve the authoritative verify snapshot before resolving
    # that mutable request mapping again.
    lifecycle.boundary_seqlen["r0"] = 128
    lifecycle.backup_boundary_state(batch, preserve_existing=True)
    assert adapter.backup_indices == [[11], [12]]
    assert len(state_context_calls) == 1

    # Post-verify commit supplies the exact physical context used by target
    # verify, independent of the request mapping currently visible to radix.
    lifecycle.backup_boundary_state(
        batch, preserve_existing=False, ctx=rebound_context
    )
    assert adapter.backup_indices == [[11], [12], [21], [22]]
    assert len(state_context_calls) == 1

    lifecycle.boundary_track_idx["r0"] = 0
    lifecycle.backup_boundary_state(batch, preserve_existing=True)
    assert adapter.backup_indices == [[11], [12], [21], [22], [21], [22]]
    assert len(state_context_calls) == 2
