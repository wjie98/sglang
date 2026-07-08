from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.dvr_draft_cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
)
from sglang.srt.mem_cache.dvr_mamba_radix_cache_policy import (
    get_req_mamba_radix_insert_snapshot,
    get_unfinished_insert_state,
    mark_req_skip_mamba_radix_finished_insert,
    set_req_mamba_radix_insert_snapshot,
    should_insert_finished_req,
)
from sglang.srt.speculative.dvr_draft_decode_context import draft_decode_performance_context
from sglang.srt.speculative.dvr_worker import DecodeVerifyRollbackWorker
from sglang.srt.speculative.dvr_logprob_repair import (
    _final_output_len_if_repair_needed,
    _try_live_cache_locs_for_final_replay,
)
from sglang.srt.arg_groups.speculative_hook import (
    speculative_uses_draft_decode_custom_all_reduce,
)
from sglang.srt.speculative.dvr_scheduler_utils import (
    DVRFinalLogprobRepair,
    DVRMambaCheckpoint,
    DVRReplayPrefixTracker,
    DVRSpecResultAux,
    apply_dvr_final_logprob_repairs_from_result,
    compact_output_token_rows,
    commit_pending_mamba_checkpoint_from_result,
)
from sglang.srt.speculative.dvr_output_policy import (
    allow_req_non_streaming_logprob_output,
    defer_req_non_streaming_logprob_output,
    should_hold_non_streaming_logprob_output,
    try_claim_req_final_logprob_repair,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_policy import get_spec_algorithm_policy


def _policy(algorithm):
    return get_spec_algorithm_policy(algorithm)


def test_dvr_spec_v2_policy():
    self_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
    eagle_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE

    assert _policy(self_draft).uses_spec_v2(enable_overlap=True)
    assert not _policy(self_draft).uses_spec_v2(enable_overlap=False)
    # DVR-EAGLE is implemented on EAGLE's v2 schema only. Disabling overlap
    # selects synchronous v2, not a separate EAGLE-v1 worker.
    assert _policy(eagle_draft).uses_spec_v2(enable_overlap=True)
    assert _policy(eagle_draft).uses_spec_v2(enable_overlap=False)


def test_dvr_target_verify_capture_hidden_policy():
    self_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
    eagle_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE
    standalone = SpeculativeAlgorithm.STANDALONE

    assert (
        _policy(self_draft).target_verify_capture_hidden_mode(CaptureHiddenMode.FULL)
        == CaptureHiddenMode.NULL
    )
    assert (
        _policy(eagle_draft).target_verify_capture_hidden_mode(CaptureHiddenMode.FULL)
        == CaptureHiddenMode.FULL
    )
    assert (
        _policy(standalone).target_verify_capture_hidden_mode(CaptureHiddenMode.FULL)
        == CaptureHiddenMode.FULL
    )
    assert (
        _policy(standalone).target_verify_capture_hidden_mode(
            CaptureHiddenMode.FULL,
            null_for_standalone=True,
        )
        == CaptureHiddenMode.NULL
    )


def test_dvr_mamba_radix_snapshot_policy():
    assert (
        _policy(
            SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
        ).needs_mamba_radix_snapshot_for_spec_v2()
    )
    assert (
        _policy(
            SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE
        ).needs_mamba_radix_snapshot_for_spec_v2()
    )
    assert not _policy(SpeculativeAlgorithm.EAGLE).needs_mamba_radix_snapshot_for_spec_v2()
    assert not _policy(SpeculativeAlgorithm.NONE).needs_mamba_radix_snapshot_for_spec_v2()


def test_dvr_published_seq_len_filter_policy():
    dvr_policy = _policy(SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK)
    eagle_policy = _policy(SpeculativeAlgorithm.EAGLE)
    req = SimpleNamespace(
        origin_input_ids=[1, 2, 3],
        sampling_params=SimpleNamespace(max_new_tokens=4),
    )
    batch = SimpleNamespace(
        is_spec_v2=True,
        seq_lens_cpu=torch.tensor([6]),
        seq_lens=None,
        reqs=[req],
    )

    assert dvr_policy.requires_seq_lens_cpu_before_filter(
        batch=batch,
        enable_overlap=True,
    )
    assert not dvr_policy.requires_seq_lens_cpu_before_filter(
        batch=batch,
        enable_overlap=False,
    )
    assert not eagle_policy.requires_seq_lens_cpu_before_filter(
        batch=batch,
        enable_overlap=True,
    )

    assert dvr_policy.is_finished_by_published_seq_len(
        batch=batch,
        req_index=0,
    )
    batch.seq_lens_cpu = torch.tensor([5])
    assert not dvr_policy.is_finished_by_published_seq_len(
        batch=batch,
        req_index=0,
    )
    assert not eagle_policy.is_finished_by_published_seq_len(
        batch=batch,
        req_index=0,
    )


def test_mamba_radix_request_policy_helpers():
    req = SimpleNamespace()

    assert should_insert_finished_req(req, default_is_insert=True)
    assert not should_insert_finished_req(req, default_is_insert=False)
    assert get_req_mamba_radix_insert_snapshot(req) is None

    mark_req_skip_mamba_radix_finished_insert(req)
    assert not should_insert_finished_req(req, default_is_insert=True)

    set_req_mamba_radix_insert_snapshot(req, indices="idx", seqlen=64)
    snapshot = get_req_mamba_radix_insert_snapshot(req)
    assert snapshot.indices == "idx"
    assert snapshot.seqlen == 64


def test_mamba_radix_unfinished_insert_plan_prefers_snapshot():
    req = SimpleNamespace(mamba_last_track_seqlen=32)

    cache_len, snapshot_indices = get_unfinished_insert_state(
        req,
        enable_mamba_extra_buffer=True,
        token_count=96,
    )
    assert cache_len == 32
    assert snapshot_indices is None

    set_req_mamba_radix_insert_snapshot(req, indices="snapshot_idx", seqlen=64)
    cache_len, snapshot_indices = get_unfinished_insert_state(
        req,
        enable_mamba_extra_buffer=True,
        token_count=96,
    )
    assert cache_len == 64
    assert snapshot_indices == "snapshot_idx"

    cache_len, snapshot_indices = get_unfinished_insert_state(
        req,
        enable_mamba_extra_buffer=False,
        token_count=96,
    )
    assert cache_len == 96
    assert snapshot_indices is None


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
        spec_aux=DVRSpecResultAux(
            pending_mamba_checkpoints=[
                DVRMambaCheckpoint(track_idx=0, seqlen=128),
            ],
        )
    )
    commit_pending_mamba_checkpoint_from_result(
        req=req,
        batch=batch,
        result=result,
        req_index=0,
        tree_cache=tree_cache,
    )
    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 1

    result.spec_aux.pending_mamba_checkpoints = [
        DVRMambaCheckpoint(track_idx=1, seqlen=128)
    ]
    commit_pending_mamba_checkpoint_from_result(
        req=req,
        batch=batch,
        result=result,
        req_index=0,
        tree_cache=tree_cache,
    )
    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 1

    result.spec_aux.pending_mamba_checkpoints = [
        DVRMambaCheckpoint(track_idx=2, seqlen=192)
    ]
    commit_pending_mamba_checkpoint_from_result(
        req=req,
        batch=batch,
        result=result,
        req_index=0,
        tree_cache=tree_cache,
    )
    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 1


def test_dvr_linear_state_extension_policy():
    runner_without_gdn = SimpleNamespace(hybrid_gdn_config=None)
    runner_with_gdn = SimpleNamespace(hybrid_gdn_config=object())

    assert (
        _policy(SpeculativeAlgorithm.EAGLE).linear_speculative_state_extension_factory(
            runner_with_gdn
        )
        is None
    )
    assert (
        _policy(
            SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
        ).linear_speculative_state_extension_factory(
            runner_without_gdn
        )
        is None
    )
    assert (
        _policy(
            SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
        ).linear_speculative_state_extension_factory(
            runner_with_gdn
        )
        is not None
    )


def test_dvr_self_draft_reuses_target_kv_pool():
    assert _policy(
        SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
    ).uses_target_kv_pool_for_draft()
    assert (
        not _policy(
            SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE
        ).uses_target_kv_pool_for_draft()
    )
    assert not _policy(SpeculativeAlgorithm.EAGLE).uses_target_kv_pool_for_draft()


def test_dvr_draft_decode_custom_all_reduce_policy_hook():
    assert speculative_uses_draft_decode_custom_all_reduce("DECODE_VERIFY_ROLLBACK")
    assert speculative_uses_draft_decode_custom_all_reduce(
        "DECODE_VERIFY_ROLLBACK_EAGLE"
    )
    assert not speculative_uses_draft_decode_custom_all_reduce("EAGLE")
    assert not speculative_uses_draft_decode_custom_all_reduce(None)


def test_draft_extend_selected_logits_policy_is_capability_gated():
    model = SimpleNamespace(supports_draft_extend_selected_logits=True)
    model_without_capability = SimpleNamespace()

    dvr_eagle = _policy(SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE)
    assert dvr_eagle.uses_draft_extend_selected_logits(
        topk=1,
        model=model,
        is_v2=True,
        requires_gathered_buffer=False,
    )
    assert not dvr_eagle.uses_draft_extend_selected_logits(
        topk=2,
        model=model,
        is_v2=True,
        requires_gathered_buffer=False,
    )
    assert not dvr_eagle.uses_draft_extend_selected_logits(
        topk=1,
        model=model_without_capability,
        is_v2=True,
        requires_gathered_buffer=False,
    )
    assert not dvr_eagle.uses_draft_extend_selected_logits(
        topk=1,
        model=model,
        is_v2=True,
        requires_gathered_buffer=True,
    )

    # Keep the selected-logits optimization scoped to DVR-EAGLE so ordinary
    # EAGLE/MTP behavior stays exactly on the upstream draft-extend path.
    assert not _policy(SpeculativeAlgorithm.EAGLE).uses_draft_extend_selected_logits(
        topk=1,
        model=model,
        is_v2=True,
        requires_gathered_buffer=False,
    )
    assert not _policy(SpeculativeAlgorithm.NONE).uses_draft_extend_selected_logits(
        topk=1,
        model=model,
        is_v2=True,
        requires_gathered_buffer=False,
    )


def test_draft_decode_context_is_noop_for_regular_eagle():
    runner = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.EAGLE)
    with draft_decode_performance_context(runner) as ctx:
        assert ctx is None


def test_dvr_self_draft_graph_runner_only_skips_known_short_boundary():
    graph_runner = object.__new__(DVRDraftDecodeCudaGraphRunner)
    graph_runner.runner = SimpleNamespace(can_run=lambda forward_batch: True)

    assert not graph_runner.can_run(
        SimpleNamespace(
            dvr_disable_draft_cuda_graph=True,
            seq_lens_cpu=torch.tensor([4]),
            batch_size=1,
        )
    )
    assert not graph_runner.can_run(
        SimpleNamespace(seq_lens_cpu=torch.tensor([2]), batch_size=1)
    )
    assert graph_runner.can_run(
        SimpleNamespace(seq_lens_cpu=torch.tensor([3]), batch_size=1)
    )


def test_dvr_self_draft_requires_graph_for_gdn_normal_decode():
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.cuda_graph_runner_for_draft_decode = None
    worker.model_runner = SimpleNamespace(hybrid_gdn_config=object())

    with pytest.raises(RuntimeError, match="requires the dedicated CUDA graph"):
        worker._draft_decode_forward(
            SimpleNamespace(seq_lens_cpu=torch.tensor([3]), batch_size=1)
        )


def test_output_policy_defer_non_streaming_logprob():
    req = SimpleNamespace(
        return_logprob=True,
        stream=False,
        output_ids=[1, 2],
    )

    assert not should_hold_non_streaming_logprob_output(
        req=req,
        return_logprob=True,
    )

    defer_req_non_streaming_logprob_output(req)
    assert should_hold_non_streaming_logprob_output(
        req=req,
        return_logprob=True,
    )


def test_output_policy_does_not_defer_streaming_or_non_logprob_output():
    streaming_req = SimpleNamespace(
        return_logprob=True,
        stream=True,
        output_ids=[1, 2],
    )
    defer_req_non_streaming_logprob_output(streaming_req)
    assert not should_hold_non_streaming_logprob_output(
        req=streaming_req,
        return_logprob=True,
    )

    no_logprob_req = SimpleNamespace(
        return_logprob=False,
        stream=False,
        output_ids=[1, 2],
    )
    defer_req_non_streaming_logprob_output(no_logprob_req)
    assert not should_hold_non_streaming_logprob_output(
        req=no_logprob_req,
        return_logprob=True,
    )
    assert not should_hold_non_streaming_logprob_output(
        req=no_logprob_req,
        return_logprob=False,
    )


def test_output_policy_final_logprob_repair_claim_is_once_only():
    req = SimpleNamespace(
        return_logprob=True,
        stream=False,
        output_ids=[1, 2],
    )

    defer_req_non_streaming_logprob_output(req)
    assert try_claim_req_final_logprob_repair(req)
    assert not try_claim_req_final_logprob_repair(req)
    assert should_hold_non_streaming_logprob_output(
        req=req,
        return_logprob=True,
        require_final_repair=True,
    )

    allow_req_non_streaming_logprob_output(req)
    assert not should_hold_non_streaming_logprob_output(
        req=req,
        return_logprob=True,
        require_final_repair=True,
    )
    assert not try_claim_req_final_logprob_repair(req)


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
        spec_aux=DVRSpecResultAux(
            final_logprob_repairs=[
                DVRFinalLogprobRepair(
                    output_ids=[10, 11, 12],
                    output_logprobs=[-0.1, -0.2, -0.3],
                )
            ]
        )
    )

    apply_dvr_final_logprob_repairs_from_result(SimpleNamespace(reqs=[req]), result)

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
        spec_aux=DVRSpecResultAux(
            final_logprob_repairs=[
                DVRFinalLogprobRepair(
                    output_ids=[10, 11],
                    output_logprobs=[-0.1, -0.2],
                )
            ]
        )
    )

    with pytest.raises(RuntimeError, match="materialized output ids"):
        apply_dvr_final_logprob_repairs_from_result(SimpleNamespace(reqs=[req]), result)


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
        spec_aux=DVRSpecResultAux(
            final_logprob_repairs=[
                DVRFinalLogprobRepair(
                    output_ids=[10, 11],
                    output_logprobs=[-0.1],
                )
            ]
        )
    )

    with pytest.raises(RuntimeError, match="inconsistent ids/logprobs length"):
        apply_dvr_final_logprob_repairs_from_result(SimpleNamespace(reqs=[req]), result)


def test_dvr_eagle_replay_prefix_splits_verifier_and_output_streams():
    req = SimpleNamespace(
        rid="r0",
        origin_input_ids=[101, 102, 103],
        output_ids=[],
    )
    batch = SimpleNamespace(reqs=[req])
    verifier_prefix = DVRReplayPrefixTracker()
    output_prefix = DVRReplayPrefixTracker()

    # Target EXTEND publishes the first client-visible token before overlap
    # scheduling has necessarily materialized it into Req.output_ids.
    output_prefix.append_batch_output_tokens(
        batch,
        [[748]],
    )
    assert output_prefix.request_output_prefix_token_ids(
        req,
        4,
        error_prefix="DVR EAGLE output replay prefix",
    ) == [101, 102, 103, 748]
    with pytest.raises(RuntimeError, match="cannot reconstruct"):
        verifier_prefix.request_verifier_prefix_token_ids(
            req,
            4,
            error_prefix="DVR EAGLE verifier replay prefix",
        )

    verifier_prefix.append_batch_verifier_tokens(
        batch,
        [[900, 901]],
    )
    output_prefix.append_batch_output_tokens(
        batch,
        [[749, 750]],
    )
    req.output_ids = [748]

    assert verifier_prefix.request_verifier_prefix_token_ids(
        req,
        5,
        error_prefix="DVR EAGLE verifier replay prefix",
    ) == [101, 102, 103, 900, 901]
    assert output_prefix.request_output_prefix_token_ids(
        req,
        6,
        error_prefix="DVR EAGLE final logprob",
    ) == [101, 102, 103, 748, 749, 750]


def test_dvr_output_replay_prefix_tracks_self_draft_visible_output():
    req = SimpleNamespace(
        rid="r0",
        origin_input_ids=[101, 102],
        output_ids=[201],
    )
    batch = SimpleNamespace(reqs=[req])
    prefix = DVRReplayPrefixTracker()

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
            allow_preclaimed_final_token=True,
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
            allow_preclaimed_final_token=True,
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
            allow_preclaimed_final_token=True,
        )
        == 17
    )


def test_dvr_final_logprob_replay_reuses_only_complete_live_mapping():
    req = SimpleNamespace(req_pool_idx=0)
    batch = SimpleNamespace(
        reqs=[req],
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.tensor([[3, 4, 5, 6]], dtype=torch.int32)
        ),
    )

    assert _try_live_cache_locs_for_final_replay(batch, 3).tolist() == [3, 4, 5]

    batch.req_to_token_pool.req_to_token[0, 1] = 0
    assert _try_live_cache_locs_for_final_replay(batch, 3) is None


def test_dvr_eagle_compacts_accepted_output_rows():
    accept_tokens = torch.tensor(
        [
            [10, 11, 99, 99],
            [20, 21, 22, 99],
        ],
        dtype=torch.int32,
    )
    accept_lens = torch.tensor([2, 3], dtype=torch.int32)

    assert compact_output_token_rows(
        accept_tokens,
        accept_lens,
    ) == [[10, 11], [20, 21, 22]]
    assert (
        compact_output_token_rows(
            None,
            accept_lens,
        )
        is None
    )
