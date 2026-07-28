from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import sglang.srt.speculative.dvr_sampling as dvr_sampling_module
import sglang.srt.speculative.dvr_worker as dvr_worker_module
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dvr_worker import (
    DecodeVerifyRollbackWorker,
)
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    create_dummy_verify_input,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _sampling_info(top_ks, top_ps, min_ps):
    return SimpleNamespace(
        top_ks=torch.tensor(top_ks),
        top_ps=torch.tensor(top_ps),
        min_ps=torch.tensor(min_ps),
        temperatures=torch.ones((len(top_ks), 1)),
        need_top_k_sampling=any(value != 0 for value in top_ks),
        need_top_p_sampling=any(value != 1.0 for value in top_ps),
        need_min_p_sampling=any(value != 0.0 for value in min_ps),
        is_all_greedy=False,
        apply_logits_bias=lambda _logits: None,
    )


def test_dvr_algorithm_contracts():
    self_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
    eagle_draft = SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE

    assert self_draft.is_dvr_self_draft() and not self_draft.is_dvr_eagle()
    assert eagle_draft.is_dvr_eagle() and not eagle_draft.is_eagle()
    assert not self_draft.need_topk()
    assert eagle_draft.need_topk()
    assert self_draft.has_draft_kv()
    assert eagle_draft.has_draft_kv()


def test_dvr_sampling_probs_match_target_distribution():
    proposal = dvr_sampling_module.dvr_sampling_probs(
        torch.tensor([[0.6, 0.3, 0.1]]),
        _sampling_info([3], [1.0], [0.5]),
    )
    torch.testing.assert_close(proposal, torch.tensor([[2 / 3, 1 / 3, 0.0]]))

    joint = dvr_sampling_module.dvr_sampling_probs(
        torch.tensor([[0.6, 0.25, 0.15]]),
        _sampling_info([2], [0.7], [0.0]),
    )
    torch.testing.assert_close(joint, torch.tensor([[12 / 17, 5 / 17, 0.0]]))


def test_dvr_sampling_probs_handle_mixed_and_repeated_rows():
    probs = torch.tensor([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])
    filtered = dvr_sampling_module.dvr_sampling_probs(
        probs,
        _sampling_info([1, 3], [1.0, 1.0], [0.0, 0.0]),
    )
    torch.testing.assert_close(filtered[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(filtered[1], probs[1])

    repeated = dvr_sampling_module.dvr_sampling_probs(
        torch.tensor([[0.6, 0.4]] * 4),
        _sampling_info([1, 2], [1.0, 1.0], [0.0, 0.0]),
        repeat=2,
    )
    torch.testing.assert_close(repeated[:2], torch.tensor([[1.0, 0.0]] * 2))
    torch.testing.assert_close(repeated[2:], torch.tensor([[0.6, 0.4]] * 2))


@pytest.mark.parametrize(
    ("top_ks", "top_ps", "min_ps"),
    [
        ([4, 2], [1.0, 1.0], [0.0, 0.0]),
        ([4, 4], [0.75, 0.55], [0.0, 0.0]),
        ([3, 2], [0.8, 0.7], [0.15, 0.35]),
    ],
)
def test_pytorch_dvr_sampling_probs_match_sampler(monkeypatch, top_ks, top_ps, min_ps):
    probs = torch.tensor([[0.40, 0.30, 0.20, 0.10], [0.50, 0.25, 0.15, 0.10]])
    sampling_info = _sampling_info(top_ks, top_ps, min_ps)
    captured = None

    def capture_multinomial(filtered, **_kwargs):
        nonlocal captured
        captured = filtered.clone()
        return torch.zeros((filtered.shape[0], 1), dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", capture_multinomial)
    top_k_top_p_min_p_sampling_from_probs_torch(
        probs,
        sampling_info.top_ks,
        sampling_info.top_ps,
        sampling_info.min_ps,
        sampling_info.need_min_p_sampling,
        sampling_seed=None,
        positions=torch.arange(probs.shape[0]),
    )

    sorted_indices = probs.argsort(dim=-1, descending=True)
    expected = torch.zeros_like(probs).scatter_(-1, sorted_indices, captured)
    expected /= expected.sum(dim=-1, keepdim=True)
    actual = dvr_sampling_module.dvr_sampling_probs(probs, sampling_info)
    torch.testing.assert_close(actual, expected)


def test_dvr_draft_sample_returns_the_distribution_it_samples(monkeypatch):
    sampling_info = _sampling_info([2], [0.8], [0.0])
    sampling_info.temperatures.fill_(2.0)
    sampling_info.sampling_seed = torch.tensor([2026])
    sampling_info.apply_logits_bias = lambda logits: logits.add_(
        torch.tensor([[0.0, 0.5, -0.5]])
    )
    sampled = None

    def sample(proposal, seeds, positions):
        nonlocal sampled
        assert seeds.tolist() == [2026]
        assert positions.tolist() == [7]
        sampled = proposal
        return torch.tensor([1])

    monkeypatch.setattr(dvr_sampling_module, "dvr_sample_from_probs", sample)
    logits = torch.tensor([[2.0, 1.0, 0.0]])

    token_ids, proposal = dvr_sampling_module.dvr_draft_sample(
        logits, sampling_info, torch.tensor([7])
    )

    expected_probs = torch.softmax(torch.tensor([[2.0, 1.5, -0.5]]) / 2.0, dim=-1)
    expected = dvr_sampling_module.dvr_sampling_probs(expected_probs, sampling_info)
    torch.testing.assert_close(proposal, expected)
    assert sampled is proposal
    assert token_ids.tolist() == [1]


def test_dvr_draft_sample_greedy_applies_logits_bias(monkeypatch):
    sampling_info = _sampling_info([1], [1.0], [0.0])
    sampling_info.is_all_greedy = True
    sampling_info.apply_logits_bias = lambda logits: logits.add_(
        torch.tensor([[0.0, 2.0]])
    )
    monkeypatch.setattr(
        dvr_sampling_module,
        "dvr_sample_from_probs",
        lambda *_args, **_kwargs: pytest.fail("greedy draft must not sample"),
    )

    token_ids, proposal = dvr_sampling_module.dvr_draft_sample(
        torch.tensor([[1.0, 0.0]]), sampling_info, torch.tensor([3])
    )

    assert token_ids.tolist() == [1]
    assert proposal is None


@pytest.mark.parametrize("uses_eagle_draft", [False, True])
def test_short_prefix_uses_one_root_verify_sentinel(uses_eagle_draft):
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.uses_eagle_draft = uses_eagle_draft
    worker.draft_backend = SimpleNamespace(
        target_capture_hidden_mode=(
            dvr_worker_module.CaptureHiddenMode.FULL
            if uses_eagle_draft
            else dvr_worker_module.CaptureHiddenMode.NULL
        )
    )
    worker.device = "cpu"
    worker.num_draft_tokens = 4
    worker.chain_retrieve_index = torch.arange(8).view(2, 4)
    worker.chain_retrieve_sibling = torch.full((2, 4), -1)
    worker.chain_position_offsets = torch.arange(4)
    batch = SimpleNamespace(
        spec_info=dvr_worker_module.EagleDraftInput(bonus_tokens=torch.tensor([6, 7])),
        seq_lens=torch.tensor([1, 65]),
        seq_lens_cpu=torch.tensor([1, 65]),
        seq_lens_sum=66,
    )

    verify_input = worker.build_root_only_verify_input(batch)

    assert verify_input.draft_token.tolist() == [6] * 4 + [7] * 4
    assert verify_input.spec_steps == 0
    assert verify_input.retrieve_next_token.eq(-1).all()
    assert verify_input.positions.tolist() == [1, 2, 3, 4, 65, 66, 67, 68]


def test_short_prefix_sentinel_marks_only_new_prefill_requests():
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.uses_eagle_draft = False
    worker.pending_seed_rows = set()
    worker.draft_backend = SimpleNamespace(
        target_capture_hidden_mode=dvr_worker_module.CaptureHiddenMode.NULL,
        finish_prefill=lambda _batch, _result: "next-draft",
    )
    worker.state_lifecycle = SimpleNamespace(
        prepare_target_extend=lambda _batch: None,
        finish_target_extend=lambda _batch: None,
        prepare_for_cache_release=lambda _req: None,
    )
    worker.target_model_worker = SimpleNamespace(
        forward_batch_generation=lambda _batch: SimpleNamespace()
    )
    new_req = SimpleNamespace(rid="new", req_pool_idx=1, origin_input_ids=[7])
    running_req = SimpleNamespace(rid="running", req_pool_idx=2, origin_input_ids=[8])
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        is_extend_in_batch=True,
        reqs=[new_req, running_req],
        decoding_reqs=[running_req],
        seq_lens=torch.tensor([1, 1]),
    )

    result = worker.forward_batch_generation(batch)

    assert worker.pending_seed_rows == {1}
    assert result.new_seq_lens is batch.seq_lens
    assert result.next_draft_input == "next-draft"
    worker.prepare_for_kv_cache_release(new_req)
    assert not worker.pending_seed_rows


@pytest.mark.parametrize(
    ("additive_penalties", "scaling_penalties", "orchestrator_required"),
    [
        (torch.zeros(1, 2), None, False),
        (None, torch.ones(1, 2), False),
        (None, None, True),
    ],
)
def test_dvr_rejects_dynamic_token_penalties(
    additive_penalties, scaling_penalties, orchestrator_required
):
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.uses_eagle_draft = False
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
        sampling_info=SimpleNamespace(
            acc_additive_penalties=additive_penalties,
            acc_scaling_penalties=scaling_penalties,
            penalizer_orchestrator=SimpleNamespace(
                is_required=orchestrator_required
            ),
        ),
    )

    with pytest.raises(ValueError, match="dynamic token penalties"):
        worker.forward_batch_generation(batch)


def test_draft_backends_finalize_the_common_verify_result():
    bonus_tokens = torch.tensor([3, 5])
    result = SimpleNamespace(
        next_draft_input=dvr_worker_module.EagleDraftInput(bonus_tokens=bonus_tokens)
    )
    owner = SimpleNamespace()
    dvr_worker_module.SelfDraftBackend(owner).finish_verify(None, result)
    torch.testing.assert_close(result.next_draft_input.bonus_tokens, bonus_tokens)
    torch.testing.assert_close(
        result.next_draft_input.topk_index, bonus_tokens[:, None]
    )

    calls = []
    eagle_worker = SimpleNamespace(
        _draft_extend_for_decode=lambda batch, output: calls.append((batch, output))
    )
    backend = dvr_worker_module.EagleDraftBackend(owner, eagle_worker)
    backend.context = nullcontext
    batch = object()
    backend.finish_verify(batch, result)
    assert calls == [(batch, result)]


@pytest.mark.parametrize("uses_eagle_draft", [False, True])
def test_weight_update_recaptures_the_complete_dvr_draft_graph_set(
    uses_eagle_draft,
):
    calls = []
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.uses_eagle_draft = uses_eagle_draft
    worker.draft_graph_buffers = {"stale": object()}
    worker.init_cuda_graphs = lambda: calls.append(("recapture",))
    worker.draft_backend = SimpleNamespace(graph_runner=object())

    def update_draft_weights(model_path, load_format, **kwargs):
        calls.append(("load", model_path, load_format, kwargs))
        return True, "loaded"

    worker.draft_model_worker = SimpleNamespace(
        draft_runner=SimpleNamespace(
            update_weights_from_disk=update_draft_weights,
        )
    )
    request = SimpleNamespace(
        model_path="updated-model",
        load_format="auto",
        recapture_cuda_graph=True,
    )

    success, _ = worker.update_weights_from_disk(request)

    assert success
    if uses_eagle_draft:
        assert calls == [
            (
                "load",
                "updated-model",
                "auto",
                {"recapture_cuda_graph": False},
            ),
            ("recapture",),
        ]
        assert worker.draft_backend.graph_runner is not None
    else:
        assert calls == [("recapture",)]
        assert worker.draft_backend.graph_runner is None
    assert not worker.draft_graph_buffers


def test_weight_update_keeps_dvr_graphs_when_recapture_is_disabled():
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.uses_eagle_draft = False
    worker.draft_graph_buffers = {"current": object()}
    graph_runner = object()
    worker.draft_backend = SimpleNamespace(graph_runner=graph_runner)
    worker.init_cuda_graphs = lambda: pytest.fail("unexpected graph recapture")
    request = SimpleNamespace(
        model_path="updated-model",
        load_format="auto",
        recapture_cuda_graph=False,
    )

    success, _ = worker.update_weights_from_disk(request)

    assert success
    assert worker.draft_backend.graph_runner is graph_runner
    assert "current" in worker.draft_graph_buffers


def test_cache_release_waits_for_pending_dvr_rollback(monkeypatch):
    calls = []
    read_done = object()
    state_done = object()
    stream = SimpleNamespace(wait_event=lambda value: calls.append(("wait", value)))
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda _device: SimpleNamespace(current_stream=lambda: stream),
    )
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.device = "cuda"
    worker.uses_eagle_draft = False
    worker.pending_seed_rows = {3}
    worker.target_model_worker = SimpleNamespace(
        model_runner=SimpleNamespace(war_fastpath_read_done_event=read_done)
    )
    worker.rollback_done_event = state_done
    worker.state_lifecycle = SimpleNamespace(
        prepare_for_cache_release=lambda req: calls.append(("release", req.rid))
    )
    req = SimpleNamespace(rid="done", req_pool_idx=3)

    worker.prepare_for_kv_cache_release(req)

    assert calls == [
        ("wait", read_done),
        ("wait", state_done),
        ("release", "done"),
    ]
    assert not worker.pending_seed_rows


def test_self_verify_fences_state_commit_before_overlap_publish(monkeypatch):
    calls = []

    class Event:
        def record(self):
            calls.append("record_commit")

    class FillBonusTokens:
        def __getitem__(self, _grid):
            def run(tokens, _accept_lens, output, _width):
                calls.append("fill_bonus")
                output.copy_(tokens.reshape(-1)[: output.shape[0]])

            return run

    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda _device: SimpleNamespace(
            Event=Event, current_stream=lambda: SimpleNamespace()
        ),
    )
    monkeypatch.setattr(
        dvr_worker_module,
        "eagle_prepare_for_verify",
        lambda *_args, **_kwargs: (SimpleNamespace(), False),
    )
    monkeypatch.setattr(dvr_worker_module, "fill_bonus_tokens", FillBonusTokens())

    runner = SimpleNamespace(war_fastpath_read_done_event=None)
    logits_output = SimpleNamespace(
        next_token_logits=torch.tensor([[1.0, 0.0]]),
        hidden_states=None,
    )
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.uses_eagle_draft = False
    worker.device = "cpu"
    worker.num_draft_tokens = 2
    worker.verify_plan_stream = None
    worker.req_to_token_pool = object()
    worker.target_model_worker = SimpleNamespace(
        model_runner=runner,
        forward_batch_generation=lambda **_kwargs: SimpleNamespace(
            logits_output=logits_output,
            routed_experts_output=None,
            indexer_topk_output=None,
        ),
    )
    worker.state_lifecycle = SimpleNamespace(
        rollback=lambda **_kwargs: calls.append("commit")
    )
    worker.sample_verified_tokens = lambda *_args: (
        torch.tensor([7, 8], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[0]], dtype=torch.int32),
    )
    batch = SimpleNamespace(
        seq_lens=torch.tensor([64], dtype=torch.int32),
        seq_lens_cpu_cache=None,
        forward_mode=ForwardMode.DECODE,
        return_logprob=False,
    )
    spec_info = SimpleNamespace(
        is_verify_input=lambda: True,
        draft_token_num=2,
        num_tokens_per_req=None,
        custom_mask=object(),
        seq_lens_cpu=torch.tensor([64], dtype=torch.int32),
        spec_steps=1,
    )

    worker.verify(
        batch,
        spec_info,
        rollback_plan=object(),
        on_publish=lambda _seq_lens: calls.append("publish"),
    )

    assert calls == ["fill_bonus", "commit", "record_commit", "publish"]
    assert runner.war_fastpath_read_done_event is worker.rollback_done_event


def test_self_draft_copies_each_graph_proposal_before_next_replay(monkeypatch):
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.num_draft_tokens = 3
    worker.num_draft_steps = 2
    backend = dvr_worker_module.SelfDraftBackend(worker)
    backend.proposal_prob_buffer = torch.empty((1, 2, 3))

    sampling_info = _sampling_info([3], [1.0], [0.0])
    sampling_info.sampling_seed = torch.tensor([2026])
    sampled_tokens = iter((torch.tensor([0]), torch.tensor([2])))

    def sample(_proposal, seeds, positions):
        assert seeds.tolist() == [2026]
        assert positions.shape == (1,)
        return next(sampled_tokens)

    worker.model_runner = SimpleNamespace(
        maybe_update_ngram_token_table=lambda _token_ids, _batch: None,
    )
    static_logits = torch.empty((1, 3))
    per_step = iter(
        (
            torch.tensor([[0.6, 0.3, 0.1]]).log(),
            torch.tensor([[0.1, 0.2, 0.7]]).log(),
        )
    )

    def draft_forward(_batch):
        static_logits.copy_(next(per_step))
        return LogitsProcessorOutput(next_token_logits=static_logits)

    backend.decode_forward = draft_forward
    monkeypatch.setattr(dvr_sampling_module, "dvr_sample_from_probs", sample)
    forward_batch = SimpleNamespace(
        spec_info=dvr_worker_module.EagleDraftInput(bonus_tokens=torch.tensor([1])),
        out_cache_loc=torch.arange(3),
        batch_size=1,
        seq_lens=torch.tensor([10]),
        seq_lens_cpu=torch.tensor([10]),
        seq_lens_sum=10,
        positions=torch.tensor([10]),
        sampling_info=sampling_info,
    )

    _, proposals = backend.draft_tokens(forward_batch)

    torch.testing.assert_close(proposals[0, 0], torch.tensor([0.6, 0.3, 0.1]))
    torch.testing.assert_close(proposals[0, 1], torch.tensor([0.1, 0.2, 0.7]))


def test_self_draft_restores_batch_output_flags(monkeypatch):
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.num_draft_tokens = 2
    worker.topk = 1
    worker.model_runner = object()
    worker.chain_position_offsets = torch.arange(2)
    worker.chain_retrieve_index = torch.arange(2).view(1, 2)
    worker.chain_retrieve_next = torch.tensor([[1, -1]])
    worker.chain_retrieve_sibling = torch.full((1, 2), -1)
    backend = dvr_worker_module.SelfDraftBackend(worker)
    backend.draft_tokens = lambda _forward_batch: (_ for _ in ()).throw(
        RuntimeError("draft failed")
    )
    monkeypatch.setattr(
        dvr_worker_module.ForwardBatch,
        "init_new",
        lambda _batch, _runner: object(),
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_info=dvr_worker_module.EagleDraftInput(bonus_tokens=torch.tensor([1])),
        seq_lens=torch.tensor([4]),
        req_pool_indices=torch.tensor([0]),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(8).view(1, 8)),
        return_logprob=True,
        return_hidden_states=True,
    )

    with pytest.raises(RuntimeError, match="draft failed"):
        backend.propose(batch)

    assert batch.return_logprob is True
    assert batch.return_hidden_states is True


def test_self_draft_positions_do_not_alias_sequence_lengths(monkeypatch):
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.num_draft_tokens = 2
    worker.topk = 1
    worker.model_runner = object()
    worker.chain_position_offsets = torch.arange(2)
    worker.chain_retrieve_index = torch.arange(2).view(1, 2)
    worker.chain_retrieve_next = torch.tensor([[1, -1]])
    worker.chain_retrieve_sibling = torch.full((1, 2), -1)

    def init_new(batch, _runner):
        assert batch.spec_info.positions.data_ptr() != batch.seq_lens.data_ptr()
        batch.spec_info.positions.add_(1)
        torch.testing.assert_close(batch.seq_lens, torch.tensor([4]))
        raise RuntimeError("stop after initialization")

    monkeypatch.setattr(dvr_worker_module.ForwardBatch, "init_new", init_new)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_info=dvr_worker_module.EagleDraftInput(bonus_tokens=torch.tensor([1])),
        seq_lens=torch.tensor([4]),
        req_pool_indices=torch.tensor([0]),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(8).view(1, 8)),
        return_logprob=False,
        return_hidden_states=False,
    )

    with pytest.raises(RuntimeError, match="stop after initialization"):
        dvr_worker_module.SelfDraftBackend(worker).propose(batch)


@pytest.mark.parametrize(
    ("algorithm", "expected_hidden"),
    [
        (SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK, "NULL"),
        (SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE, "FULL"),
    ],
)
def test_dvr_dummy_verify_input_matches_draft_backend(algorithm, expected_hidden):
    spec_info = create_dummy_verify_input(
        spec_algorithm=algorithm,
        server_args=SimpleNamespace(
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
        ),
        custom_mask=torch.ones(1, dtype=torch.bool),
        num_tokens_per_bs=4,
        is_draft_worker=False,
    )

    assert spec_info is not None
    assert spec_info.capture_hidden_mode.name == expected_hidden
