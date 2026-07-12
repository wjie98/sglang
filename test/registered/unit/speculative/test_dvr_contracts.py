from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.dvr_draft_cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
    _ensure_decode_custom_all_reduce_comm,
    _patch_draft_decode_backend_defaults,
    dvr_self_draft_graph_block_reason,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dvr_worker import DecodeVerifyRollbackWorkerV2
from sglang.srt.speculative.dvr_core import (
    DVRRollbackActions,
    append_dvr_batch_output_tokens,
    compact_dvr_accepted_input_tokens_and_cache_locs,
    compact_dvr_output_rows,
    request_dvr_output_prefix_token_ids,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class _MockReq:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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
    with pytest.raises(RuntimeError, match="precedes output materialization"):
        dvr_rollback_actions.commit_checkpoint_after_decode(
            req=req,
            batch=batch,
            req_index=0,
            tree_cache=tree_cache,
        )

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


def test_dvr_replay_prefix_records_only_visible_output_tokens():
    req = _MockReq(
        rid="r0",
        origin_input_ids=[101, 102, 103],
        output_ids=[],
    )
    batch = SimpleNamespace(reqs=[req])
    journal = {}

    # Target EXTEND publishes the first client-visible token before overlap
    # scheduling has necessarily materialized it into Req.output_ids.
    append_dvr_batch_output_tokens(
        journal,
        batch,
        [[748]],
    )
    assert request_dvr_output_prefix_token_ids(
        journal,
        req,
        4,
        error_prefix="DVR EAGLE output replay prefix",
    ) == [101, 102, 103, 748]
    with pytest.raises(RuntimeError, match="not yet owned"):
        request_dvr_output_prefix_token_ids(
            journal,
            req,
            5,
            error_prefix="DVR EAGLE output replay prefix",
        )

    # A rejected EAGLE draft must append the target-predicted output tokens,
    # not the rejected draft candidates.
    req.output_ids = [748]
    append_dvr_batch_output_tokens(
        journal,
        batch,
        [[749, 750]],
        base_seq_lens_cpu=[4],
        error_prefix="DVR EAGLE output replay prefix",
    )

    assert request_dvr_output_prefix_token_ids(
        journal,
        req,
        6,
        error_prefix="DVR EAGLE final logprob",
    ) == [101, 102, 103, 748, 749, 750]

    req.output_ids = [748, 999]
    with pytest.raises(RuntimeError, match="diverged"):
        request_dvr_output_prefix_token_ids(
            journal,
            req,
            5,
            error_prefix="DVR EAGLE output replay prefix",
        )


def test_dvr_output_replay_prefix_tracks_self_draft_visible_output():
    req = _MockReq(
        rid="r0",
        origin_input_ids=[101, 102],
        output_ids=[201],
    )
    batch = SimpleNamespace(reqs=[req])
    journal = {}

    assert request_dvr_output_prefix_token_ids(
        journal,
        req,
        3,
        error_prefix="DVR spec-v2",
    ) == [101, 102, 201]

    append_dvr_batch_output_tokens(
        journal,
        batch,
        [[202, 203]],
    )

    assert request_dvr_output_prefix_token_ids(
        journal,
        req,
        5,
        error_prefix="DVR spec-v2",
    ) == [101, 102, 201, 202, 203]


def test_dvr_output_replay_prefix_is_req_lifecycle_scoped():
    old_req = _MockReq(
        rid="reused",
        origin_input_ids=[101, 102],
        output_ids=[],
    )
    new_req = _MockReq(
        rid="reused",
        origin_input_ids=[101, 102],
        output_ids=[301],
    )
    journal = {}

    append_dvr_batch_output_tokens(
        journal, SimpleNamespace(reqs=[old_req]), [[201, 202]]
    )
    append_dvr_batch_output_tokens(journal, SimpleNamespace(reqs=[new_req]), [[]])

    # rid is client/protocol state and can be reused.  The pending output
    # journal must be scoped to the live Req object, otherwise a new request can
    # inherit stale overlap output tokens from a completed request.
    assert request_dvr_output_prefix_token_ids(
        journal,
        new_req,
        3,
        error_prefix="DVR spec-v2",
    ) == [101, 102, 301]


def test_dvr_eagle_compacts_accepted_output_rows():
    accept_tokens = torch.tensor(
        [
            [10, 11, 99, 99],
            [20, 21, 22, 99],
        ],
        dtype=torch.int32,
    )
    accept_lens = torch.tensor([2, 3], dtype=torch.int32)

    accept_lens_cpu, token_ids_per_req = compact_dvr_output_rows(
        output_tokens=accept_tokens,
        accept_lens=accept_lens,
        tokens_per_req=4,
    )
    assert accept_lens_cpu == [2, 3]
    assert token_ids_per_req == [[10, 11], [20, 21, 22]]


def test_dvr_eagle_state_replay_uses_verify_input_path():
    batch = SimpleNamespace(
        out_cache_loc=torch.tensor(
            [100, 101, 102, 103, 200, 201, 202, 203], dtype=torch.int64
        )
    )
    verify_input_tokens = torch.tensor(
        [271, 2523, 0, 0, 248068, 271, 0, 0], dtype=torch.int32
    )
    accept_lens = torch.tensor([2, 3], dtype=torch.int32)

    tokens, cache_locs = compact_dvr_accepted_input_tokens_and_cache_locs(
        batch=batch,
        verify_input_tokens=verify_input_tokens,
        accept_lens=accept_lens,
        num_draft_tokens=4,
    )

    # EAGLE output tokens are sampled from verifier logits and are one row ahead
    # of the target sequence.  DVR state replay must advance the target-owned
    # verify-input path instead of replaying the client-visible bonus tokens.
    assert tokens.tolist() == [271, 2523, 248068, 271, 0]
    assert cache_locs.tolist() == [100, 101, 200, 201, 202]
