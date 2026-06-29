from types import SimpleNamespace

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.draft_decode_context import draft_decode_performance_context
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
