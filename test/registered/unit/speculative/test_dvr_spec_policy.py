from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.draft_decode_context import draft_decode_performance_context
from sglang.srt.speculative.dvr_scheduler_utils import (
    DVRFinalLogprobRepair,
    DVRSpecResultAux,
    apply_dvr_final_logprob_repairs_from_result,
)
from sglang.srt.speculative.output_policy import (
    defer_req_non_streaming_logprob_output,
    should_emit_non_streaming_output_chunk,
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
