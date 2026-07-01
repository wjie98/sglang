from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.draft_decode_context import draft_decode_performance_context
from sglang.srt.speculative.dvr_logprob_repair import (
    _DVRFinalLogprobReplayPlan,
    _can_reuse_live_cache_locs_for_final_replay,
    _final_output_len_if_repair_needed,
    _is_kv_allocation_failure,
)
from sglang.srt.speculative.dvr_output_replay import compact_output_token_rows
from sglang.srt.speculative.dvr_scheduler_utils import (
    DVRFinalLogprobRepair,
    DVRReplayPrefixTracker,
    DVRSpecResultAux,
    apply_dvr_final_logprob_repairs_from_result,
)
from sglang.srt.speculative.output_policy import (
    allow_req_non_streaming_logprob_output,
    defer_req_non_streaming_logprob_output,
    should_defer_finished_non_streaming_logprob_output,
    should_emit_non_streaming_output_chunk,
    try_expect_req_final_logprob_repair,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def test_dvr_spec_v2_policy():
    self_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
    eagle_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE

    assert self_draft.uses_spec_v2(enable_overlap=True)
    assert not self_draft.uses_spec_v2(enable_overlap=False)
    assert eagle_draft.uses_spec_v2(enable_overlap=True)
    assert eagle_draft.uses_spec_v2(enable_overlap=False)


def test_dvr_target_verify_capture_hidden_policy():
    self_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
    eagle_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE
    standalone = SpeculativeAlgorithm.STANDALONE

    assert (
        self_draft.target_verify_capture_hidden_mode(CaptureHiddenMode.FULL)
        == CaptureHiddenMode.NULL
    )
    assert (
        eagle_draft.target_verify_capture_hidden_mode(CaptureHiddenMode.FULL)
        == CaptureHiddenMode.FULL
    )
    assert (
        standalone.target_verify_capture_hidden_mode(CaptureHiddenMode.FULL)
        == CaptureHiddenMode.FULL
    )
    assert (
        standalone.target_verify_capture_hidden_mode(
            CaptureHiddenMode.FULL,
            null_for_standalone=True,
        )
        == CaptureHiddenMode.NULL
    )


def test_dvr_mamba_radix_snapshot_policy():
    assert (
        SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK.needs_mamba_radix_snapshot_for_spec_v2()
    )
    assert (
        SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE.needs_mamba_radix_snapshot_for_spec_v2()
    )
    assert not SpeculativeAlgorithm.EAGLE.needs_mamba_radix_snapshot_for_spec_v2()
    assert not SpeculativeAlgorithm.NONE.needs_mamba_radix_snapshot_for_spec_v2()


def test_dvr_linear_state_extension_policy():
    runner_without_gdn = SimpleNamespace(hybrid_gdn_config=None)
    runner_with_gdn = SimpleNamespace(hybrid_gdn_config=object())

    assert (
        SpeculativeAlgorithm.EAGLE.linear_speculative_state_extension_factory(
            runner_with_gdn
        )
        is None
    )
    assert (
        SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK.linear_speculative_state_extension_factory(
            runner_without_gdn
        )
        is None
    )
    assert (
        SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK.linear_speculative_state_extension_factory(
            runner_with_gdn
        )
        is not None
    )


def test_dvr_self_draft_reuses_target_kv_pool():
    assert SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK.uses_target_kv_pool_for_draft()
    assert (
        not SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE.uses_target_kv_pool_for_draft()
    )
    assert not SpeculativeAlgorithm.EAGLE.uses_target_kv_pool_for_draft()


def test_spec_accept_rate_proposal_width_policy():
    assert (
        SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK.proposed_draft_tokens_per_verify(
            speculative_num_steps=15,
            speculative_num_draft_tokens=16,
        )
        == 15
    )
    assert (
        SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE.proposed_draft_tokens_per_verify(
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
        )
        == 3
    )
    assert (
        SpeculativeAlgorithm.EAGLE.proposed_draft_tokens_per_verify(
            speculative_num_steps=3,
            speculative_num_draft_tokens=8,
        )
        == 3
    )
    assert (
        SpeculativeAlgorithm.DFLASH.proposed_draft_tokens_per_verify(
            speculative_num_steps=3,
            speculative_num_draft_tokens=8,
        )
        == 7
    )


def test_draft_decode_context_is_noop_for_regular_eagle():
    runner = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.EAGLE)
    with draft_decode_performance_context(runner) as ctx:
        assert ctx is None


def test_output_policy_defer_non_streaming_logprob():
    req = SimpleNamespace(
        return_logprob=True,
        stream=False,
        output_ids=[1, 2],
    )

    assert should_emit_non_streaming_output_chunk(
        req=req,
        return_logprob=True,
        force_stream_interval=2,
    )

    defer_req_non_streaming_logprob_output(req)
    assert not should_emit_non_streaming_output_chunk(
        req=req,
        return_logprob=True,
        force_stream_interval=2,
    )


def test_output_policy_final_logprob_repair_claim_is_once_only():
    req = SimpleNamespace(
        return_logprob=True,
        stream=False,
        output_ids=[1, 2],
    )

    defer_req_non_streaming_logprob_output(req)
    assert try_expect_req_final_logprob_repair(req)
    assert not try_expect_req_final_logprob_repair(req)
    assert should_defer_finished_non_streaming_logprob_output(
        req=req,
        return_logprob=True,
    )

    allow_req_non_streaming_logprob_output(req)
    assert not should_defer_finished_non_streaming_logprob_output(
        req=req,
        return_logprob=True,
    )
    assert not try_expect_req_final_logprob_repair(req)


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
        initialize_from_req_output=True,
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

    verifier_prefix.append_output_tokens(
        req,
        [900, 901],
        initialize_from_req_output=False,
    )
    output_prefix.append_output_tokens(
        req,
        [749, 750],
        initialize_from_req_output=True,
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


def test_dvr_self_draft_replay_prefix_tracks_client_visible_output():
    req = SimpleNamespace(
        rid="r0",
        origin_input_ids=[101, 102],
        output_ids=[201],
    )
    batch = SimpleNamespace(reqs=[req])
    prefix = DVRReplayPrefixTracker()

    assert prefix.request_self_draft_prefix_token_ids(
        req,
        3,
        error_prefix="DVR spec-v2",
    ) == [101, 102, 201]

    prefix.append_self_draft_output_tokens(batch, [[202, 203]])

    assert prefix.request_self_draft_prefix_token_ids(
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


def test_dvr_final_logprob_replay_reuses_live_slots_only_for_all_final_rows():
    all_final = _DVRFinalLogprobReplayPlan(
        req_indices=[0, 1],
        input_ids=[],
        logprob_token_ids=[],
        extend_lens_cpu=[8, 9],
        final_seq_lens_cpu=[8, 9],
        final_score_specs=[(0, object(), 2, [10, 11]), (1, object(), 1, [20])],
    )
    single_final = _DVRFinalLogprobReplayPlan(
        req_indices=[0],
        input_ids=[],
        logprob_token_ids=[],
        extend_lens_cpu=[8],
        final_seq_lens_cpu=[8],
        final_score_specs=[(0, object(), 2, [10, 11])],
    )
    malformed_partial_final = _DVRFinalLogprobReplayPlan(
        req_indices=[0, 1],
        input_ids=[],
        logprob_token_ids=[],
        extend_lens_cpu=[8, 9],
        final_seq_lens_cpu=[8, 9],
        final_score_specs=[(0, object(), 2, [10, 11])],
    )

    assert _can_reuse_live_cache_locs_for_final_replay(all_final)
    assert _can_reuse_live_cache_locs_for_final_replay(single_final)
    assert not _can_reuse_live_cache_locs_for_final_replay(malformed_partial_final)


def test_dvr_final_logprob_replay_identifies_allocator_oom():
    assert _is_kv_allocation_failure(
        RuntimeError(
            "Prefill out of memory. Try to lower your batch size.\n"
            "Try to allocate 725 tokens."
        )
    )
    assert not _is_kv_allocation_failure(RuntimeError("CUDA out of memory"))


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
