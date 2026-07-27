"""Unit tests for Spec V2 grammar truncation in _resolve_spec_v2_tokens.

The grammar-constrained spec path stops accepting at the grammar-terminating
token, so the over-drafted suffix is never committed to KV nor emitted.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components import (
    batch_result_processor as result_processor_module,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeGrammar:
    """Grammar stub that terminates after `terminate_after` accepted tokens."""

    def __init__(self, terminate_after: int):
        self.accepted = []
        self.finished = False
        self._terminate_after = terminate_after

    def accept_token(self, token_id: int):
        self.accepted.append(token_id)

    def is_terminated(self) -> bool:
        return len(self.accepted) >= self._terminate_after


class _FakeSpecAlgorithm:
    def is_none(self) -> bool:
        return False

    def is_dflash(self) -> bool:
        return False


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.spec_algorithm = _FakeSpecAlgorithm()


def _make_processor() -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_metrics=False),
        model_config=SimpleNamespace(think_end_id=None),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(),
        draft_worker=None,
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        abort_request=lambda *a, **k: None,
    )


def _make_req(terminate_after: int) -> Req:
    sp = SamplingParams(max_new_tokens=256, temperature=0)
    sp.normalize(None)
    req = Req(
        rid="r0",
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sp,
    )
    req.grammar = _FakeGrammar(terminate_after=terminate_after)
    req.kv_committed_len = 0
    return req


def _make_result(num_draft_tokens, accept_lens, flat_tokens):
    return SimpleNamespace(
        next_token_ids=torch.tensor(flat_tokens, dtype=torch.long),
        accept_lens=torch.tensor(accept_lens, dtype=torch.long),
        speculative_num_draft_tokens=num_draft_tokens,
        num_correct_drafts=None,
        num_correct_drafts_per_req_cpu=None,
    )


class TestSpecV2GrammarTruncation(CustomTestCase):
    def test_prefill_release_prepares_worker_state_before_free(self):
        events = []
        proc = SimpleNamespace(
            is_generation=True,
            tree_cache=None,
            model_worker=SimpleNamespace(
                prepare_for_kv_cache_release=lambda _req: events.append("prepare")
            ),
            move_logprobs_to_cpu=lambda **_kwargs: None,
            _validate_pp_skip_output_comm=lambda *_args: None,
            _maybe_update_reasoning_tokens=lambda *_args: None,
            _maybe_collect_routed_experts=lambda *_args: None,
            _maybe_collect_indexer_topk=lambda *_args: None,
            _maybe_collect_customized_info=lambda *_args: None,
            output_streamer=SimpleNamespace(stream_output=lambda *_args: None),
            metrics_reporter=SimpleNamespace(
                report_prefill_stats=lambda **_kwargs: None
            ),
        )
        finished = [False]
        req = SimpleNamespace(
            output_ids=[],
            inflight_middle_chunks=0,
            is_retracted=False,
            return_hidden_states=False,
            return_sampling_mask=False,
            grammar=None,
            finished=lambda: finished[0],
            update_finish_state=lambda: finished.__setitem__(0, True),
            time_stats=SimpleNamespace(
                set_prefill_finished_time=lambda: None,
                set_completion_time=lambda: None,
            ),
        )
        batch = SimpleNamespace(
            reqs=[req],
            decoding_reqs=[],
            return_logprob=False,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = SimpleNamespace(
            copy_done=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            logits_output=SimpleNamespace(customized_info=None, hidden_states=None),
            next_token_ids=torch.tensor([7]),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            can_run_cuda_graph=False,
        )
        with patch.object(
            result_processor_module,
            "release_kv_cache",
            side_effect=lambda *_args, **_kwargs: events.append("release"),
        ):
            SchedulerBatchResultProcessor.process_batch_result_prefill(
                proc, batch, result
            )

        self.assertEqual(events, ["prepare", "release"])

    def test_resolve_truncates_after_grammar_completion(self):
        req = _make_req(terminate_after=2)
        proc = _make_processor()
        # stride=4, accept_len=3 -> proposed [101, 102, 103]; grammar finishes at 102.
        result = _make_result(4, [3], [101, 102, 103, 0])

        predict_tokens = proc._resolve_spec_v2_tokens(result, _FakeBatch([req]))

        self.assertEqual(predict_tokens, [[101, 102]])
        # No pre-claim: commit the full retained run (no -1 refund).
        self.assertEqual(req.kv_committed_len, 2)

    def test_resolve_keeps_all_when_grammar_not_terminated(self):
        req = _make_req(terminate_after=99)
        proc = _make_processor()
        result = _make_result(4, [3], [201, 202, 203, 0])

        predict_tokens = proc._resolve_spec_v2_tokens(result, _FakeBatch([req]))

        self.assertEqual(predict_tokens, [[201, 202, 203]])
        self.assertEqual(req.kv_committed_len, 3)


if __name__ == "__main__":
    unittest.main()
