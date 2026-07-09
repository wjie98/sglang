from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.dvr_draft_cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
)
from sglang.srt.speculative.dvr_core import (
    _final_output_len_if_repair_needed,
)
from sglang.srt.speculative.dvr_worker import _DVRSelfDraftCore
from sglang.srt.speculative.dvr_info import (
    DVRDeferredActions,
    DVRDeferredOutput,
    DVRFinalLogprobRepair,
    DVRMambaCheckpoint,
    DVRPendingOutputPrefix,
    allow_dvr_non_streaming_logprob_output,
    compact_dvr_accepted_tokens_and_cache_locs,
    compact_dvr_output_rows,
    defer_dvr_non_streaming_logprob_output,
    should_hold_dvr_non_streaming_logprob_output,
    try_claim_dvr_final_logprob_repair,
)
from sglang.srt.speculative.dvr_scheduler import (
    _commit_pending_mamba_checkpoint_from_result,
    apply_dvr_deferred_output_from_result,
    maybe_filter_running_batch_with_dvr_state,
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


def test_dvr_published_seq_len_filter_hook():
    req = SimpleNamespace(
        origin_input_ids=[1, 2, 3],
        finished=lambda: False,
        sampling_params=SimpleNamespace(max_new_tokens=4),
    )
    batch = SimpleNamespace(
        enable_overlap=True,
        seq_lens_cpu=torch.tensor([6]),
        seq_lens=None,
        reqs=[req],
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
        filter_batch=lambda keep_indices: setattr(
            batch, "filtered_keep_indices", keep_indices
        ),
    )
    future_map = SimpleNamespace(
        resolve_seq_lens_cpu=lambda resolved_batch: setattr(
            resolved_batch, "seq_lens_resolved", True
        )
    )

    assert maybe_filter_running_batch_with_dvr_state(
        batch=batch,
        future_map=future_map,
        enable_overlap=True,
    )
    assert batch.seq_lens_resolved
    assert batch.filtered_keep_indices == []

    batch.seq_lens_cpu = torch.tensor([5])
    assert maybe_filter_running_batch_with_dvr_state(
        batch=batch,
        future_map=future_map,
        enable_overlap=True,
    )
    assert batch.filtered_keep_indices == [0]

    batch.spec_algorithm = SpeculativeAlgorithm.EAGLE
    assert not maybe_filter_running_batch_with_dvr_state(
        batch=batch,
        future_map=future_map,
        enable_overlap=True,
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
    batch = SimpleNamespace(req_to_token_pool=Pool())
    tree_cache = SimpleNamespace(page_size=64)

    result = SimpleNamespace(
        dvr_aux=DVRDeferredActions(
            pending_mamba_checkpoints=[
                DVRMambaCheckpoint(track_idx=0, seqlen=128),
            ],
        )
    )
    _commit_pending_mamba_checkpoint_from_result(
        req=req,
        batch=batch,
        result=result,
        req_index=0,
        tree_cache=tree_cache,
    )
    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 1

    result.dvr_aux.pending_mamba_checkpoints = [
        DVRMambaCheckpoint(track_idx=1, seqlen=128)
    ]
    _commit_pending_mamba_checkpoint_from_result(
        req=req,
        batch=batch,
        result=result,
        req_index=0,
        tree_cache=tree_cache,
    )
    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 1

    result.dvr_aux.pending_mamba_checkpoints = [
        DVRMambaCheckpoint(track_idx=2, seqlen=192)
    ]
    _commit_pending_mamba_checkpoint_from_result(
        req=req,
        batch=batch,
        result=result,
        req_index=0,
        tree_cache=tree_cache,
    )
    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 1


def test_dvr_self_draft_graph_runner_only_skips_known_short_boundary():
    graph_runner = object.__new__(DVRDraftDecodeCudaGraphRunner)
    graph_runner.runner = SimpleNamespace(can_run_graph=lambda forward_batch: True)

    assert not graph_runner.can_run(
        SimpleNamespace(seq_lens_cpu=torch.tensor([2]), batch_size=1)
    )
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


def test_dvr_self_draft_requires_graph_for_gdn_normal_decode():
    worker = object.__new__(_DVRSelfDraftCore)
    worker.cuda_graph_runner_for_draft_decode = None
    worker.model_runner = SimpleNamespace(hybrid_gdn_config=object())

    with pytest.raises(RuntimeError, match="requires the dedicated CUDA graph"):
        worker._draft_decode_forward(
            SimpleNamespace(seq_lens_cpu=torch.tensor([3]), batch_size=1)
        )


def test_dvr_request_flags_defer_non_streaming_logprob():
    req = SimpleNamespace(
        return_logprob=True,
        stream=False,
        output_ids=[1, 2],
    )

    assert not should_hold_dvr_non_streaming_logprob_output(
        req=req,
        return_logprob=True,
    )

    defer_dvr_non_streaming_logprob_output(req)
    assert should_hold_dvr_non_streaming_logprob_output(
        req=req,
        return_logprob=True,
    )


def test_dvr_request_flags_do_not_defer_streaming_or_non_logprob_output():
    streaming_req = SimpleNamespace(
        return_logprob=True,
        stream=True,
        output_ids=[1, 2],
    )
    defer_dvr_non_streaming_logprob_output(streaming_req)
    assert not should_hold_dvr_non_streaming_logprob_output(
        req=streaming_req,
        return_logprob=True,
    )

    no_logprob_req = SimpleNamespace(
        return_logprob=False,
        stream=False,
        output_ids=[1, 2],
    )
    defer_dvr_non_streaming_logprob_output(no_logprob_req)
    assert not should_hold_dvr_non_streaming_logprob_output(
        req=no_logprob_req,
        return_logprob=True,
    )
    assert not should_hold_dvr_non_streaming_logprob_output(
        req=no_logprob_req,
        return_logprob=False,
    )


def test_dvr_request_flags_final_logprob_repair_claim_is_once_only():
    req = SimpleNamespace(
        return_logprob=True,
        stream=False,
        output_ids=[1, 2],
    )

    defer_dvr_non_streaming_logprob_output(req)
    assert try_claim_dvr_final_logprob_repair(req)
    assert not try_claim_dvr_final_logprob_repair(req)
    assert should_hold_dvr_non_streaming_logprob_output(
        req=req,
        return_logprob=True,
        require_final_repair=True,
    )

    allow_dvr_non_streaming_logprob_output(req)
    assert not should_hold_dvr_non_streaming_logprob_output(
        req=req,
        return_logprob=True,
        require_final_repair=True,
    )
    assert not try_claim_dvr_final_logprob_repair(req)


def test_dvr_final_logprob_repair_applies_after_materialization():
    req = SimpleNamespace(
        rid="r0",
        return_logprob=True,
        output_ids=[10, 11, 12],
        logprob=SimpleNamespace(
            output_token_logprobs_val=[-1.0],
            output_token_logprobs_idx=[10],
        ),
    )
    result = SimpleNamespace(
        dvr_aux=DVRDeferredActions(
            output=DVRDeferredOutput(
                final_logprob_repairs=[
                    DVRFinalLogprobRepair(
                        output_ids=[10, 11, 12],
                        output_logprobs=[-0.1, -0.2, -0.3],
                    )
                ]
            )
        )
    )

    apply_dvr_deferred_output_from_result(
        SimpleNamespace(
            reqs=[req],
            spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
        ),
        result,
    )

    assert req.logprob.output_token_logprobs_val == [-0.1, -0.2, -0.3]
    assert req.logprob.output_token_logprobs_idx == [10, 11, 12]


def test_dvr_final_logprob_repair_rejects_mismatched_output_ids():
    req = SimpleNamespace(
        rid="r0",
        return_logprob=True,
        output_ids=[10, 99],
        logprob=SimpleNamespace(
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
        ),
    )
    result = SimpleNamespace(
        dvr_aux=DVRDeferredActions(
            output=DVRDeferredOutput(
                final_logprob_repairs=[
                    DVRFinalLogprobRepair(
                        output_ids=[10, 11],
                        output_logprobs=[-0.1, -0.2],
                    )
                ]
            )
        )
    )

    with pytest.raises(RuntimeError, match="materialized output ids"):
        apply_dvr_deferred_output_from_result(
            SimpleNamespace(
                reqs=[req],
                spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
            ),
            result,
        )


def test_dvr_final_logprob_repair_rejects_mismatched_lengths():
    req = SimpleNamespace(
        rid="r0",
        return_logprob=True,
        output_ids=[10, 11],
        logprob=SimpleNamespace(
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
        ),
    )
    result = SimpleNamespace(
        dvr_aux=DVRDeferredActions(
            output=DVRDeferredOutput(
                final_logprob_repairs=[
                    DVRFinalLogprobRepair(
                        output_ids=[10, 11],
                        output_logprobs=[-0.1],
                    )
                ]
            )
        )
    )

    with pytest.raises(RuntimeError, match="inconsistent ids/logprobs length"):
        apply_dvr_deferred_output_from_result(
            SimpleNamespace(
                reqs=[req],
                spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
            ),
            result,
        )


def test_dvr_replay_prefix_records_only_visible_output_tokens():
    req = _MockReq(
        rid="r0",
        origin_input_ids=[101, 102, 103],
        output_ids=[],
    )
    batch = SimpleNamespace(reqs=[req])
    prefix = DVRPendingOutputPrefix()

    # Target EXTEND publishes the first client-visible token before overlap
    # scheduling has necessarily materialized it into Req.output_ids.
    prefix.append_batch_output_tokens(
        batch,
        [[748]],
    )
    assert prefix.request_output_prefix_token_ids(
        req,
        4,
        error_prefix="DVR EAGLE output replay prefix",
    ) == [101, 102, 103, 748]
    with pytest.raises(RuntimeError, match="not yet owned"):
        prefix.request_output_prefix_token_ids(
            req,
            5,
            error_prefix="DVR EAGLE output replay prefix",
        )

    # A rejected EAGLE draft must append the target-predicted output tokens,
    # not the rejected draft candidates.
    req.output_ids = [748]
    prefix.append_batch_output_tokens(
        batch,
        [[749, 750]],
        base_seq_lens_cpu=[4],
        error_prefix="DVR EAGLE output replay prefix",
    )

    assert prefix.request_output_prefix_token_ids(
        req,
        6,
        error_prefix="DVR EAGLE final logprob",
    ) == [101, 102, 103, 748, 749, 750]

    req.output_ids = [748, 999]
    with pytest.raises(RuntimeError, match="diverged"):
        prefix.request_output_prefix_token_ids(
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
    prefix = DVRPendingOutputPrefix()

    assert prefix.request_output_prefix_token_ids(
        req,
        3,
        error_prefix="DVR spec-v2",
    ) == [101, 102, 201]

    prefix.append_batch_output_tokens(
        batch,
        [[202, 203]],
    )

    assert prefix.request_output_prefix_token_ids(
        req,
        5,
        error_prefix="DVR spec-v2",
    ) == [101, 102, 201, 202, 203]


def test_dvr_output_journal_builds_final_logprob_repair():
    req = _MockReq(
        rid="r0",
        origin_input_ids=[101, 102],
        output_ids=[201],
        logprob=SimpleNamespace(output_token_logprobs_val=[-0.1]),
    )
    batch = SimpleNamespace(reqs=[req])
    prefix = DVRPendingOutputPrefix()

    prefix.append_batch_output_tokens(
        batch,
        [[202, 203]],
        token_logprobs_per_req=[[-0.2, -0.3]],
        base_seq_lens_cpu=[3],
        error_prefix="DVR spec-v2",
    )

    repair = prefix.final_logprob_repair(
        req,
        3,
        error_prefix="DVR spec-v2 final logprob",
    )
    assert repair.output_ids == [201, 202, 203]
    assert repair.output_logprobs == [-0.1, -0.2, -0.3]


def test_dvr_output_journal_requires_exact_logprobs():
    req = _MockReq(
        rid="r0",
        origin_input_ids=[101, 102],
        output_ids=[201],
        logprob=SimpleNamespace(output_token_logprobs_val=[]),
    )
    batch = SimpleNamespace(reqs=[req])
    prefix = DVRPendingOutputPrefix()

    prefix.append_batch_output_tokens(
        batch,
        [[202]],
        base_seq_lens_cpu=[3],
        error_prefix="DVR spec-v2",
    )

    with pytest.raises(RuntimeError, match="missing verify logprobs"):
        prefix.final_logprob_repair(
            req,
            2,
            error_prefix="DVR spec-v2 final logprob",
        )


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
    prefix = DVRPendingOutputPrefix()

    prefix.append_batch_output_tokens(SimpleNamespace(reqs=[old_req]), [[201, 202]])
    prefix.prune_to_batch(SimpleNamespace(reqs=[new_req]))

    # rid is client/protocol state and can be reused.  The pending output
    # journal must be scoped to the live Req object, otherwise a new request can
    # inherit stale overlap output tokens from a completed request.
    assert prefix.request_output_prefix_token_ids(
        new_req,
        3,
        error_prefix="DVR spec-v2",
    ) == [101, 102, 301]


def test_dvr_final_logprob_overlap_bonus_can_finish_request():
    req = SimpleNamespace(
        return_logprob=True,
        stream=False,
        origin_input_ids=[1, 2, 3, 4],
        sampling_params=SimpleNamespace(max_new_tokens=17),
    )

    # Spec-v2 overlap can have model-side seq_len one token behind the
    # scheduler-visible output because the verifier preclaims the bonus slot.
    # If that bonus slot exactly reaches max_new_tokens, final logprob repair
    # must replay the completed output instead of waiting for another step.
    assert (
        _final_output_len_if_repair_needed(
            req=req,
            req_i=0,
            seq_len=len(req.origin_input_ids),
            accept_len=16,
            observed_output_len=0,
            compact_output_token_ids_per_req=None,
        )
        == 17
    )

    assert (
        _final_output_len_if_repair_needed(
            req=req,
            req_i=0,
            seq_len=len(req.origin_input_ids) + 16,
            accept_len=0,
            observed_output_len=0,
            compact_output_token_ids_per_req=None,
        )
        is None
    )

    assert (
        _final_output_len_if_repair_needed(
            req=req,
            req_i=0,
            seq_len=len(req.origin_input_ids),
            accept_len=0,
            observed_output_len=17,
            compact_output_token_ids_per_req=None,
        )
        == 17
    )


def test_dvr_eagle_compacts_accepted_output_rows():
    accept_tokens = torch.tensor(
        [
            [10, 11, 99, 99],
            [20, 21, 22, 99],
        ],
        dtype=torch.int32,
    )
    accept_lens = torch.tensor([2, 3], dtype=torch.int32)

    _, accept_lens_cpu, token_ids_per_req = compact_dvr_output_rows(
        batch=SimpleNamespace(seq_lens=None),
        output_tokens=accept_tokens,
        accept_lens=accept_lens,
        tokens_per_req=4,
    )
    assert accept_lens_cpu == [2, 3]
    assert token_ids_per_req == [[10, 11], [20, 21, 22]]


def test_dvr_eagle_replay_tokens_follow_spec_v2_output_order():
    batch = SimpleNamespace(
        out_cache_loc=torch.tensor(
            [100, 101, 102, 103, 200, 201, 202, 203], dtype=torch.int64
        )
    )
    predict = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23], dtype=torch.int32)
    accept_index = torch.tensor([[0, 2, -1], [4, 7, 6]], dtype=torch.int32)
    accept_lens = torch.tensor([2, 3], dtype=torch.int32)

    tokens, cache_locs = compact_dvr_accepted_tokens_and_cache_locs(
        batch=batch,
        predict=predict,
        accept_index=accept_index,
        accept_lens=accept_lens,
        num_draft_tokens=4,
    )

    # Spec-v2 output processing emits compact per-request predict slices.
    # Tree accept_index still owns the KV/cache rows for the accepted path.
    assert tokens.tolist() == [10, 11, 20, 21, 22]
    assert cache_locs.tolist() == [100, 102, 200, 203, 202]
