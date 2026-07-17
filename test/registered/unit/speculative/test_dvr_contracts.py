import weakref
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.speculative.dvr_worker as dvr_worker_module
import sglang.srt.speculative.eagle_utils as eagle_utils_module
import sglang.srt.speculative.spec_utils as spec_utils_module
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.managers.overlap_utils import decide_needs_cpu_seq_lens
from sglang.srt.managers.utils import GenerationBatchResult
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
    _DVRBoundaryCheckpoint,
)
from sglang.srt.speculative.dvr_worker import (
    DecodeVerifyRollbackWorkerV2,
)
from sglang.srt.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    SPEC_ACCEPT_HASH_STREAM,
    SPEC_DRAFT_SAMPLE_HASH_STREAM,
    SPEC_FINAL_SAMPLE_HASH_STREAM,
    fast_sample,
    renorm_draft_probs,
    renorm_sampling_probs,
)


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
        shape=SimpleNamespace(
            conv=((2, 4),),
            temporal=(2, 2, 2),
            state_size=2,
            conv_dim=6,
            intermediate_size=2,
            num_heads=2,
        ),
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

    assert runner._calculate_mamba_ratio() == 2
    assert server_args.max_mamba_cache_size == 16


def test_overlap_batch_results_default_to_no_result_process_fence():
    assert GenerationBatchResult().result_process_ready_event is None


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
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=True,
    )
    proposal = renorm_sampling_probs(torch.tensor([[0.6, 0.3, 0.1]]), sampling_info)

    torch.testing.assert_close(proposal, torch.tensor([[2 / 3, 1 / 3, 0.0]]))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_dvr_sampling_filters_use_joint_top_k_top_p_semantics(device, monkeypatch):
    monkeypatch.setattr(
        spec_utils_module,
        "get_global_server_args",
        lambda: SimpleNamespace(sampling_backend="pytorch"),
    )
    sampling_info = SimpleNamespace(
        top_ks=torch.tensor([2], device=device),
        top_ps=torch.tensor([0.7], device=device),
        min_ps=torch.tensor([0.0], device=device),
        need_top_k_sampling=True,
        need_top_p_sampling=True,
        need_min_p_sampling=False,
    )

    proposal = renorm_sampling_probs(
        torch.tensor([[0.6, 0.25, 0.15]], device=device), sampling_info
    )

    # Joint filtering evaluates top-p against the original distribution, so
    # the first two tokens survive. Top-k-first would renormalize to 0.706/0.294
    # before top-p and incorrectly collapse this distribution to top-1.
    torch.testing.assert_close(
        proposal, torch.tensor([[12 / 17, 5 / 17, 0.0]], device=device)
    )


def test_dvr_sampling_filters_preserve_mixed_greedy_rows():
    probs = torch.tensor([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])
    sampling_info = SimpleNamespace(
        top_ks=torch.tensor([1, 3]),
        top_ps=torch.ones(2),
        min_ps=torch.zeros(2),
        need_top_k_sampling=True,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
    )

    filtered = renorm_sampling_probs(probs, sampling_info)

    torch.testing.assert_close(filtered[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(filtered[1], probs[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dvr_sampling_filters_match_flashinfer_min_p_order(monkeypatch):
    monkeypatch.setattr(
        spec_utils_module,
        "get_global_server_args",
        lambda: SimpleNamespace(sampling_backend="flashinfer"),
    )
    sampling_info = SimpleNamespace(
        top_ks=torch.tensor([2], device="cuda"),
        top_ps=torch.tensor([0.7], device="cuda"),
        min_ps=torch.tensor([0.0], device="cuda"),
        need_top_k_sampling=True,
        need_top_p_sampling=True,
        need_min_p_sampling=True,
    )

    proposal = renorm_sampling_probs(
        torch.tensor([[0.6, 0.25, 0.15]], device="cuda"), sampling_info
    )

    # FlashInfer's min-p sampler first renormalizes top-k, so the first token
    # alone crosses top-p=0.7. DVR must publish that exact proposal as q.
    torch.testing.assert_close(
        proposal, torch.tensor([[1.0, 0.0, 0.0]], device="cuda")
    )


def test_seeded_draft_sampling_excludes_the_one_endpoint(monkeypatch):
    from sglang.srt.layers.utils import hash as hash_module

    def max_uint32_hash(seed, positions, streams):
        return torch.full(
            (seed.shape[0], streams.shape[0]),
            torch.iinfo(torch.uint32).max,
            dtype=torch.uint32,
            device=seed.device,
        )

    monkeypatch.setattr(hash_module, "murmur_hash32", max_uint32_hash)
    _, sample_index = fast_sample(
        torch.tensor([[1.0, 0.0]]),
        sampling_seed=torch.tensor([7]),
        positions=torch.tensor([65]),
    )

    assert sample_index.item() == 0


def test_dvr_sampling_filters_skip_identity_work():
    probs = torch.tensor([[0.6, 0.3, 0.1]])
    sampling_info = SimpleNamespace(
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
    )

    assert renorm_sampling_probs(probs, sampling_info) is probs


def test_speculative_hash_streams_do_not_overlap_sampler_token_ids(monkeypatch):
    from sglang.srt.layers.utils import hash as hash_module

    captured = {}

    def capture_stream(seed, positions, streams):
        captured["streams"] = streams.clone()
        return torch.zeros(
            (seed.shape[0], streams.shape[0]),
            dtype=torch.uint32,
            device=seed.device,
        )

    monkeypatch.setattr(hash_module, "murmur_hash32", capture_stream)
    fast_sample(
        torch.tensor([[0.5, 0.5]]),
        sampling_seed=torch.tensor([7]),
        positions=torch.tensor([65]),
    )

    streams = {
        SPEC_DRAFT_SAMPLE_HASH_STREAM,
        SPEC_ACCEPT_HASH_STREAM,
        SPEC_FINAL_SAMPLE_HASH_STREAM,
    }
    assert len(streams) == 3
    assert min(streams) > torch.iinfo(torch.int32).max
    assert captured["streams"].item() == SPEC_DRAFT_SAMPLE_HASH_STREAM


def test_eagle_draft_proposal_matches_greedy_and_stochastic_rows():
    logits = torch.tensor([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    sampling_info = SimpleNamespace(
        temperatures=torch.tensor([[0.5], [0.5]]),
        top_ks=torch.tensor([1, 3]),
    )

    proposal = renorm_draft_probs(logits, sampling_info, True)

    torch.testing.assert_close(proposal[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(
        proposal[1],
        torch.softmax(logits[1] / sampling_info.temperatures[1], dim=-1),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_draft_proposal_is_batch_order_independent():
    device = "cuda"
    probs = torch.tensor(
        [
            [0.10, 0.20, 0.70],
            [0.60, 0.25, 0.15],
            [0.30, 0.40, 0.30],
        ],
        device=device,
    )
    seeds = torch.tensor([7, 11, 7], dtype=torch.int64, device=device)
    positions = torch.tensor([65, 129, 66], dtype=torch.int64, device=device)
    permutation = torch.tensor([2, 0, 1], device=device)

    sample_p, sample_index = fast_sample(
        probs, sampling_seed=seeds, positions=positions
    )
    permuted_p, permuted_index = fast_sample(
        probs[permutation],
        sampling_seed=seeds[permutation],
        positions=positions[permutation],
    )
    inverse = torch.argsort(permutation)

    torch.testing.assert_close(permuted_p[inverse], sample_p)
    torch.testing.assert_close(permuted_index[inverse], sample_index)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejection_sampling_selects_final_coin_at_accepted_position():
    candidates = torch.zeros((1, 3), dtype=torch.long, device="cuda")
    retrieve_index = torch.arange(3, dtype=torch.long, device="cuda").view(1, 3)
    target_probs = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [0.75, 0.25]]], device="cuda"
    )
    draft_probs = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], device="cuda")
    predict = torch.zeros(3, dtype=torch.int32, device="cuda")
    accept_index = torch.full((1, 3), -1, dtype=torch.int32, device="cuda")
    accept_lens = torch.empty(1, dtype=torch.int32, device="cuda")

    chain_speculative_sampling_triton(
        predicts=predict,
        accept_index=accept_index,
        accept_token_num=accept_lens,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=torch.zeros((1, 3), device="cuda"),
        uniform_samples_for_final_sampling=torch.tensor(
            [[0.1, 0.1, 0.9]], device="cuda"
        ),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )

    assert accept_lens.item() == 2
    assert predict[2].item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejection_sampling_zero_residual_falls_back_to_target():
    candidates = torch.zeros((1, 2), dtype=torch.long, device="cuda")
    retrieve_index = torch.arange(2, dtype=torch.long, device="cuda").view(1, 2)
    target_probs = torch.tensor([[[0.25, 0.75], [1.0, 0.0]]], device="cuda")
    draft_probs = torch.tensor([[[0.25, 0.75]]], device="cuda")
    predict = torch.zeros(2, dtype=torch.int32, device="cuda")
    accept_index = torch.full((1, 2), -1, dtype=torch.int32, device="cuda")
    accept_lens = torch.empty(1, dtype=torch.int32, device="cuda")

    # coin=1 forces the numerically degenerate p == q rejection that normal
    # [0, 1) random generation excludes. The fallback must sample p, not the
    # vocabulary endpoint by accident.
    chain_speculative_sampling_triton(
        predicts=predict,
        accept_index=accept_index,
        accept_token_num=accept_lens,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=torch.ones((1, 2), device="cuda"),
        uniform_samples_for_final_sampling=torch.tensor([[0.1, 0.1]], device="cuda"),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )

    assert accept_lens.item() == 0
    assert predict[0].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejection_sampling_draws_from_positive_residual():
    candidates = torch.tensor([[0, 0]], dtype=torch.long, device="cuda")
    retrieve_index = torch.arange(2, dtype=torch.long, device="cuda").view(1, 2)
    target_probs = torch.tensor([[[0.2, 0.8], [1.0, 0.0]]], device="cuda")
    draft_probs = torch.tensor([[[0.8, 0.2]]], device="cuda")
    predict = torch.zeros(2, dtype=torch.int32, device="cuda")
    accept_index = torch.full((1, 2), -1, dtype=torch.int32, device="cuda")
    accept_lens = torch.empty(1, dtype=torch.int32, device="cuda")

    chain_speculative_sampling_triton(
        predicts=predict,
        accept_index=accept_index,
        accept_token_num=accept_lens,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=torch.tensor([[0.9, 0.0]], device="cuda"),
        uniform_samples_for_final_sampling=torch.zeros((1, 2), device="cuda"),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )

    assert accept_lens.item() == 0
    assert predict[0].item() == 1


def _reference_chain_rejection(
    candidates, retrieve_index, target_probs, draft_probs, coins, final_coins
):
    """Small readable oracle for the chain Triton kernel."""
    batch_size, num_slots = candidates.shape
    predicts = torch.zeros(batch_size * num_slots, dtype=torch.int32)
    accept_index = torch.full((batch_size, num_slots), -1, dtype=torch.int32)
    accept_lens = torch.zeros(batch_size, dtype=torch.int32)

    for batch_idx in range(batch_size):
        current_row = 0
        last_index = int(retrieve_index[batch_idx, 0])
        accept_index[batch_idx, 0] = last_index
        for step in range(1, num_slots):
            token = int(candidates[batch_idx, step])
            p = target_probs[batch_idx, current_row, token]
            q = draft_probs[batch_idx, current_row, token]
            if coins[batch_idx, step - 1] * q >= p:
                break
            predicts[last_index] = token
            current_row = step
            accept_lens[batch_idx] += 1
            last_index = int(retrieve_index[batch_idx, step])
            accept_index[batch_idx, current_row] = last_index

        distribution = target_probs[batch_idx, current_row]
        if current_row != num_slots - 1:
            residual = (distribution - draft_probs[batch_idx, current_row]).clamp_min(0)
            if residual.sum() > 0:
                distribution = residual
        threshold = final_coins[batch_idx, current_row] * distribution.sum()
        matches = torch.nonzero(distribution.cumsum(0) > threshold)
        predicts[last_index] = int(matches[0]) if matches.numel() else len(distribution) - 1

    return predicts, accept_index, accept_lens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejection_sampling_matches_reference_across_rejection_boundaries():
    batch_size, num_slots, vocab_size = 3, 5, 7
    generator = torch.Generator().manual_seed(1234)
    draft_probs = torch.rand(
        batch_size, num_slots - 1, vocab_size, generator=generator
    )
    draft_probs /= draft_probs.sum(dim=-1, keepdim=True)
    target_probs = torch.cat(
        [
            draft_probs.clone(),
            torch.rand(batch_size, 1, vocab_size, generator=generator),
        ],
        dim=1,
    )
    target_probs[:, -1] /= target_probs[:, -1].sum(dim=-1, keepdim=True)

    candidates = torch.zeros((batch_size, num_slots), dtype=torch.long)
    for step in range(1, num_slots):
        candidates[:, step] = torch.multinomial(
            draft_probs[:, step - 1], 1, generator=generator
        ).squeeze(1)

    # Force rejection at the first edge, the third edge, and never. Moving
    # probability mass preserves normalized p while producing a real residual.
    for batch_idx, reject_step in ((0, 1), (1, 3)):
        row = reject_step - 1
        token = int(candidates[batch_idx, reject_step])
        replacement = (token + 1) % vocab_size
        removed = target_probs[batch_idx, row, token] * 0.9
        target_probs[batch_idx, row, token] -= removed
        target_probs[batch_idx, row, replacement] += removed

    coins = torch.full((batch_size, num_slots), 0.5)
    final_coins = torch.tensor(
        [[0.1, 0.3, 0.5, 0.7, 0.9]] * batch_size, dtype=torch.float32
    )
    retrieve_index = torch.arange(batch_size * num_slots).reshape(
        batch_size, num_slots
    )
    expected = _reference_chain_rejection(
        candidates,
        retrieve_index,
        target_probs,
        draft_probs,
        coins,
        final_coins,
    )

    predicts = torch.zeros(batch_size * num_slots, dtype=torch.int32, device="cuda")
    accept_index = torch.full(
        (batch_size, num_slots), -1, dtype=torch.int32, device="cuda"
    )
    accept_lens = torch.empty(batch_size, dtype=torch.int32, device="cuda")
    chain_speculative_sampling_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_lens,
        candidates=candidates.cuda(),
        retrive_index=retrieve_index.cuda(),
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=coins.cuda(),
        uniform_samples_for_final_sampling=final_coins.cuda(),
        target_probs=target_probs.cuda(),
        draft_probs=draft_probs.cuda(),
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )

    assert accept_lens.cpu().tolist() == [0, 2, num_slots - 1]
    for actual, reference in zip(
        (predicts.cpu(), accept_index.cpu(), accept_lens.cpu()), expected
    ):
        torch.testing.assert_close(actual, reference)


def test_rejection_sampling_rejects_malformed_proposal_shape():
    with pytest.raises(ValueError, match="draft_probs has shape"):
        chain_speculative_sampling_triton(
            predicts=torch.empty(3, dtype=torch.int32),
            accept_index=torch.empty((1, 3), dtype=torch.int32),
            accept_token_num=torch.empty(1, dtype=torch.int32),
            candidates=torch.empty((1, 3), dtype=torch.long),
            retrive_index=torch.empty((1, 3), dtype=torch.long),
            retrive_next_token=None,
            retrive_next_sibling=None,
            uniform_samples=torch.empty((1, 3)),
            uniform_samples_for_final_sampling=torch.empty((1, 3)),
            target_probs=torch.empty((1, 3, 5)),
            draft_probs=torch.empty((1, 3, 5)),
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
        )


def test_target_only_eagle_uses_one_final_coin_per_request(monkeypatch):
    import sgl_kernel
    import sglang.srt.distributed as distributed_module
    import sglang.srt.layers.dp_attention as dp_attention_module
    import sglang.srt.server_args as server_args_module

    captured = {}

    def fake_target_sampling(**kwargs):
        captured["final_coin_shape"] = tuple(
            kwargs["uniform_samples_for_final_sampling"].shape
        )
        kwargs["predicts"].zero_()
        kwargs["accept_index"].zero_()
        kwargs["accept_token_num"].zero_()

    monkeypatch.setattr(sgl_kernel, "top_k_renorm_prob", lambda probs, _: probs)
    monkeypatch.setattr(sgl_kernel, "top_p_renorm_prob", lambda probs, _: probs)
    monkeypatch.setattr(
        sgl_kernel, "tree_speculative_sampling_target_only", fake_target_sampling
    )
    monkeypatch.setattr(
        distributed_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(dp_attention_module, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(
        server_args_module,
        "get_global_server_args",
        lambda: SimpleNamespace(
            speculative_use_rejection_sampling=False,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        ),
    )

    batch_size, draft_tokens, vocab_size = 2, 3, 5
    sampling_info = SimpleNamespace(
        is_all_greedy=False,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        temperatures=torch.ones((batch_size, 1)),
        top_ks=torch.full((batch_size,), vocab_size, dtype=torch.int32),
        top_ps=torch.ones(batch_size),
        sampling_seed=None,
    )
    verify_input = SimpleNamespace(
        draft_token=torch.zeros(batch_size * draft_tokens, dtype=torch.long),
        draft_token_num=draft_tokens,
        max_tree_depth=draft_tokens,
        retrieve_index=torch.arange(batch_size * draft_tokens).reshape(
            batch_size, draft_tokens
        ),
        retrieve_next_token=torch.zeros(
            (batch_size, draft_tokens), dtype=torch.long
        ),
        retrieve_next_sibling=torch.zeros(
            (batch_size, draft_tokens), dtype=torch.long
        ),
        draft_probs=None,
        spec_steps=draft_tokens - 1,
        positions=torch.arange(batch_size * draft_tokens),
    )
    batch = SimpleNamespace(
        device="cpu",
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.ones(batch_size, dtype=torch.int32),
        sampling_info=sampling_info,
    )

    eagle_utils_module.eagle_sample(
        verify_input,
        batch,
        SimpleNamespace(
            next_token_logits=torch.zeros(
                (batch_size * draft_tokens, vocab_size), dtype=torch.float32
            )
        ),
    )

    assert captured["final_coin_shape"] == (batch_size,)


def test_rejection_sentinel_uses_full_filters_and_seeded_coin(monkeypatch):
    import sgl_kernel
    import sglang.srt.distributed as distributed_module
    import sglang.srt.layers.dp_attention as dp_attention_module
    import sglang.srt.server_args as server_args_module
    from sglang.srt.layers.utils import hash as hash_module

    captured = {}

    def fake_target_sampling(**kwargs):
        captured["target_probs"] = kwargs["target_probs"].clone()
        captured["final_coins"] = kwargs["uniform_samples_for_final_sampling"].clone()
        kwargs["predicts"].zero_()
        kwargs["accept_index"].zero_()
        kwargs["accept_token_num"].zero_()

    def half_hash(seed, positions, streams):
        captured["streams"] = streams.clone()
        return torch.full(
            (seed.shape[0], streams.shape[0]),
            0x80000000,
            dtype=torch.uint32,
            device=seed.device,
        )

    monkeypatch.setattr(
        sgl_kernel, "tree_speculative_sampling_target_only", fake_target_sampling
    )
    monkeypatch.setattr(
        distributed_module, "get_tp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(dp_attention_module, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(hash_module, "murmur_hash32", half_hash)
    monkeypatch.setattr(
        server_args_module,
        "get_global_server_args",
        lambda: SimpleNamespace(
            sampling_backend="pytorch",
            speculative_use_rejection_sampling=True,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        ),
    )

    sampling_info = SimpleNamespace(
        is_all_greedy=False,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        temperatures=torch.ones((1, 1)),
        top_ks=torch.tensor([3], dtype=torch.int32),
        top_ps=torch.ones(1),
        min_ps=torch.tensor([0.75]),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=True,
        sampling_seed=torch.tensor([7], dtype=torch.int64),
    )
    verify_input = SimpleNamespace(
        draft_token=torch.zeros(1, dtype=torch.long),
        draft_token_num=1,
        max_tree_depth=1,
        retrieve_index=torch.zeros((1, 1), dtype=torch.long),
        retrieve_next_token=torch.zeros((1, 1), dtype=torch.long),
        retrieve_next_sibling=torch.zeros((1, 1), dtype=torch.long),
        draft_probs=None,
        spec_steps=0,
        positions=torch.tensor([64], dtype=torch.long),
    )
    batch = SimpleNamespace(
        device="cpu",
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.ones(1, dtype=torch.int32),
        sampling_info=sampling_info,
    )

    eagle_utils_module.eagle_sample(
        verify_input,
        batch,
        SimpleNamespace(next_token_logits=torch.log(torch.tensor([[0.6, 0.3, 0.1]]))),
    )

    torch.testing.assert_close(
        captured["target_probs"], torch.tensor([[[1.0, 0.0, 0.0]]])
    )
    torch.testing.assert_close(captured["final_coins"], torch.tensor([0.5]))
    assert captured["streams"].tolist() == [
        SPEC_ACCEPT_HASH_STREAM,
        SPEC_FINAL_SAMPLE_HASH_STREAM,
    ]


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


def test_dvr_rollback_action_publishes_owned_checkpoint():
    calls = []

    class LinearState:
        @staticmethod
        def publish_checkpoint(req):
            calls.append(req)

    req = SimpleNamespace(rid="r0")
    actions = DVRRollbackActions(linear_state=LinearState())

    assert actions.commit_checkpoint_after_decode(req=req)
    assert calls == [req]


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

    fa4 = SimpleNamespace(fa_impl_ver=4)
    assert (
        FlashAttentionBackend.get_nondeterministic_decode_overrides(fa4, model_runner)
        is None
    )

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


class _Req(SimpleNamespace):
    """Weak-referenceable request stub for DVR lifecycle tests."""


def _dvr_checkpoint(req, *, seq_len=64):
    return _DVRBoundaryCheckpoint(
        request_ref=weakref.ref(req),
        retraction_count=getattr(req, "retraction_count", 0),
        seq_len=seq_len,
    )


def _dvr_lifecycle(adapter, *, disable_radix_cache=False):
    lifecycle = object.__new__(DVRLinearStateLifecycle)
    lifecycle.server_args = SimpleNamespace(disable_radix_cache=disable_radix_cache)
    lifecycle.model_runner = None
    lifecycle._state_adapter = adapter
    lifecycle.boundaries = {}
    lifecycle.draft_state_backup = None
    lifecycle.physical_boundary_lens = torch.zeros(8, dtype=torch.int64)
    lifecycle.boundary_track_indices = torch.zeros(8, dtype=torch.int64)
    return lifecycle


def test_dvr_linear_state_is_optional_for_plain_transformers():
    lifecycle = _dvr_lifecycle(None)

    assert lifecycle.prepare_for_draft(SimpleNamespace()) is None
    assert (
        lifecycle.rollback_after_verify(
            batch=SimpleNamespace(),
            ctx=None,
            accept_lens=torch.empty(0, dtype=torch.int32),
        )
        is None
    )


def test_dvr_linear_state_requires_matching_cache_and_verify_chunks():
    lifecycle = DVRLinearStateLifecycle(
        server_args=SimpleNamespace(
            mamba_track_interval=64,
            mamba_cache_chunk_size=128,
        ),
        model_runner=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="mamba_cache_chunk_size"):
        lifecycle.bind_state_adapter(SimpleNamespace(chunk_size=64))


@pytest.mark.parametrize(("slot_count", "expected_track"), [(1, 0), (2, 1)])
def test_dvr_sync_rollback_uses_one_checkpoint_action(slot_count, expected_track):
    calls = []

    class Adapter:
        chunk_size = 64

        @staticmethod
        def commit_after_verify(**kwargs):
            calls.append(kwargs)
            return torch.tensor([True])

    lifecycle = _dvr_lifecycle(Adapter())
    ctx = SimpleNamespace(
        state_cache=object(),
        state_input_indices=torch.tensor([1]),
        live_indices=torch.tensor([2]),
        boundary_indices=torch.tensor([3]),
        previous_boundary_indices=torch.tensor([4]),
        boundary_track_indices=torch.tensor([0]),
    )
    lifecycle.physical_boundary_lens[1] = 64
    actions = lifecycle.rollback_after_verify(
        batch=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                get_mamba_ping_pong_other_idx=lambda idx: (
                    1 - idx if slot_count == 2 else idx
                )
            )
        ),
        ctx=ctx,
        accept_lens=torch.tensor([5], dtype=torch.int32),
    )

    assert isinstance(actions, DVRRollbackActions)
    assert calls[0]["accepted_token_counts"].tolist() == [5]
    assert calls[0]["previous_boundary_indices"].tolist() == [4]
    assert lifecycle.physical_boundary_lens[1].item() == 128
    assert lifecycle.boundary_track_indices[1].item() == expected_track


def test_dvr_self_draft_requires_graph_for_gdn_normal_decode():
    worker = object.__new__(DecodeVerifyRollbackWorkerV2)
    worker.cuda_graph_runner_for_draft_decode = None
    worker.linear_state = SimpleNamespace(has_state_adapter=True)

    with pytest.raises(RuntimeError, match="requires the dedicated CUDA graph"):
        worker._draft_decode_forward(
            SimpleNamespace(seq_lens_cpu=torch.tensor([3]), batch_size=1)
        )


def test_dvr_publish_checkpoint_updates_only_visible_boundary():
    lifecycle = _dvr_lifecycle(SimpleNamespace(chunk_size=64))
    req = _Req(
        rid="r0",
        req_pool_idx=1,
        retraction_count=0,
        origin_input_ids=[1] * 65,
        output_ids_through_stop=[2] * 64,
        kv_committed_len=128,
        mamba_last_track_seqlen=64,
        mamba_next_track_idx=0,
    )
    lifecycle.boundaries[1] = _dvr_checkpoint(req)

    lifecycle.publish_checkpoint(req)

    assert lifecycle.boundaries[1].seq_len == 128
    assert req.mamba_last_track_seqlen == 64
    assert req.mamba_next_track_idx == 0


@pytest.mark.parametrize(
    ("track_slots", "physical_track", "expected_boundary", "expected_previous"),
    [((10, 11), 1, 11, 10), ((10,), 0, 10, 10)],
)
def test_dvr_state_context_follows_request_checkpoint_owner(
    track_slots, physical_track, expected_boundary, expected_previous
):
    class Adapter:
        chunk_size = 64
        draft_reuses_target_state = False

        @staticmethod
        def get_live_indices(*, batch):
            return torch.tensor([20])

        @staticmethod
        def get_state_input_indices(*, batch, device):
            return torch.tensor([1], device=device)

        @staticmethod
        def get_state_cache(*, batch):
            return object()

    lifecycle = _dvr_lifecycle(
        Adapter(), disable_radix_cache=len(track_slots) == 1
    )
    req = _Req(
        rid="r0",
        req_pool_idx=1,
        retraction_count=0,
        mamba_ping_pong_track_buffer=torch.tensor(track_slots),
    )
    lifecycle.boundaries[1] = _dvr_checkpoint(req)
    lifecycle.boundary_track_indices[1] = physical_track
    batch = SimpleNamespace(
        reqs=[req],
        req_pool_indices=torch.tensor([1]),
        req_to_token_pool=SimpleNamespace(
            get_mamba_ping_pong_slots=lambda indices: torch.tensor(
                [[0] * len(track_slots), list(track_slots)]
            )[indices],
            get_mamba_ping_pong_other_idx=lambda idx: (
                1 - idx if len(track_slots) == 2 else idx
            )
        ),
        batch_size=lambda: 1,
    )

    ctx = lifecycle.state_context(batch, require_boundary=True)

    assert ctx.boundary_indices.tolist() == [expected_boundary]
    assert ctx.previous_boundary_indices.tolist() == [expected_previous]
    assert ctx.boundary_track_indices.tolist() == [physical_track]


def test_dvr_checkpoint_owner_rejects_slot_aba_and_retraction():
    lifecycle = _dvr_lifecycle(SimpleNamespace(chunk_size=64))
    first = _Req(rid="same", req_pool_idx=1, retraction_count=0)
    lifecycle.boundaries[1] = _dvr_checkpoint(first)

    replacement = _Req(rid="same", req_pool_idx=1, retraction_count=0)
    assert lifecycle._checkpoint(replacement) is None
    assert lifecycle.boundaries == {}

    lifecycle.boundaries[1] = _dvr_checkpoint(first)
    first.retraction_count += 1
    assert lifecycle._checkpoint(first) is None
    assert lifecycle.boundaries == {}


def test_dvr_checkpoint_does_not_retain_finished_request():
    req = _Req(rid="r0", req_pool_idx=1, retraction_count=0)
    checkpoint = _dvr_checkpoint(req)
    request_ref = checkpoint.request_ref

    del req

    assert request_ref() is None


def test_dvr_existing_boundary_does_not_resolve_gpu_lengths():
    adapter = SimpleNamespace(chunk_size=64, draft_reuses_target_state=False)
    lifecycle = _dvr_lifecycle(adapter)
    req = _Req(rid="r0", req_pool_idx=1, retraction_count=0)
    lifecycle.boundaries[1] = _dvr_checkpoint(req)
    lifecycle.batch_seq_lens_cpu = lambda _batch: pytest.fail(
        "steady decode must not copy seq_lens to the host"
    )
    lifecycle._ensure_boundary_state = lambda *_args, **_kwargs: None
    lifecycle.state_context = lambda _batch, require_boundary=False: object()

    assert lifecycle.prepare_for_draft(SimpleNamespace(reqs=[req])) is not None


def _boundary_fixture(
    *,
    seq_len,
    last_track,
    prefix_len,
    disable_radix_cache=False,
    track_slots=(10, 11),
):
    copies = []
    zeroed = []
    tail_updates = []

    class Pool:
        req_to_token = torch.empty(3, 0)
        mamba_pool = SimpleNamespace(
            copy_from=lambda source, destination: copies.append(
                (source.tolist(), destination.tolist())
            )
        )

        @staticmethod
        def get_mamba_ping_pong_other_idx(idx):
            return 1 - idx if len(track_slots) == 2 else idx

        @staticmethod
        def get_mamba_ping_pong_keep_idx(req):
            return Pool.get_mamba_ping_pong_other_idx(req.mamba_next_track_idx)

    class Adapter:
        chunk_size = 64
        draft_reuses_target_state = False

        @staticmethod
        def get_live_indices(*, batch):
            return torch.tensor([20])

        @staticmethod
        def get_state_input_indices(*, batch, device):
            return torch.tensor([1], device=device)

        @staticmethod
        def get_state_cache(*, batch):
            return object()

        @staticmethod
        def zero_recurrent_state(*, state_cache, indices):
            zeroed.extend(indices.tolist())

        @staticmethod
        def state_input_window():
            return SimpleNamespace(
                set_tail_lens=lambda **kwargs: tail_updates.append(
                    (kwargs["indices"].tolist(), kwargs["value"].tolist())
                )
            )

    req = _Req(
        rid="r0",
        req_pool_idx=1,
        retraction_count=0,
        mamba_last_track_seqlen=last_track,
        mamba_next_track_idx=0,
        mamba_ping_pong_track_buffer=torch.tensor(track_slots),
    )
    batch = SimpleNamespace(
        reqs=[req],
        req_pool_indices=torch.tensor([1]),
        req_to_token_pool=Pool(),
        batch_size=lambda: 1,
    )
    lifecycle = _dvr_lifecycle(Adapter(), disable_radix_cache=disable_radix_cache)
    lifecycle._ensure_boundary_state(
        batch,
        seq_lens_cpu=[seq_len],
        prefill_prefix_lens=[prefix_len],
    )
    return lifecycle, req, copies, zeroed, tail_updates


def test_dvr_aligned_live_state_is_the_first_boundary_source():
    lifecycle, req, copies, zeroed, tails = _boundary_fixture(
        seq_len=64, last_track=64, prefix_len=0
    )

    assert copies == [([20], [10])]
    assert zeroed == []
    assert tails == [([1], [0])]
    assert lifecycle.boundary_track_indices[1].item() == 0


def test_dvr_unaligned_prefill_uses_ping_pong_keep_state():
    lifecycle, req, copies, zeroed, tails = _boundary_fixture(
        seq_len=65, last_track=64, prefix_len=0
    )

    assert copies == [([11], [10])]
    assert zeroed == []
    assert tails == [([1], [1])]
    assert lifecycle.boundaries[1].seq_len == 64


def test_dvr_unaligned_prefill_keeps_the_single_request_local_slot():
    lifecycle, req, copies, zeroed, tails = _boundary_fixture(
        seq_len=65,
        last_track=64,
        prefix_len=0,
        disable_radix_cache=True,
        track_slots=(10,),
    )

    assert copies == []
    assert zeroed == []
    assert tails == [([1], [1])]
    assert lifecycle.boundaries[1].seq_len == 64


def test_dvr_radix_prefix_boundary_is_captured_by_standard_extend():
    lifecycle, req, copies, zeroed, tails = _boundary_fixture(
        seq_len=65, last_track=None, prefix_len=64
    )

    assert copies == []
    assert zeroed == []
    assert tails == [([1], [1])]
    assert lifecycle.boundaries[1].seq_len == 64


def test_dvr_without_radix_rejects_a_prefix_only_boundary_source():
    with pytest.raises(RuntimeError, match="target EXTEND"):
        _boundary_fixture(
            seq_len=65,
            last_track=None,
            prefix_len=64,
            disable_radix_cache=True,
            track_slots=(10,),
        )


def test_dvr_zero_boundary_uses_zero_state():
    lifecycle, req, copies, zeroed, tails = _boundary_fixture(
        seq_len=1, last_track=None, prefix_len=0
    )

    assert copies == []
    assert zeroed == [10]
    assert tails == [([1], [1])]
    assert lifecycle.boundaries[1].seq_len == 0


def test_dvr_missing_boundary_fails_instead_of_using_a_slow_private_replay():
    with pytest.raises(RuntimeError, match="target EXTEND"):
        _boundary_fixture(seq_len=65, last_track=None, prefix_len=0)


def test_dvr_target_extend_reselects_a_new_boundary():
    adapter = SimpleNamespace(chunk_size=64, draft_reuses_target_state=False)
    lifecycle = _dvr_lifecycle(adapter)
    req = _Req(rid="r0", req_pool_idx=1, retraction_count=0)
    lifecycle.boundaries[1] = _dvr_checkpoint(req, seq_len=64)
    calls = []
    lifecycle._ensure_boundary_state = lambda *args, **kwargs: calls.append(kwargs)
    lifecycle.state_context = lambda _batch, require_boundary=False: object()

    lifecycle.prepare_for_draft(
        SimpleNamespace(reqs=[req]),
        seq_lens_cpu=[128],
        prefill_prefix_lens=[0],
    )

    assert lifecycle.boundaries == {}
    assert calls[0]["missing"] == [0]


def test_dvr_self_draft_backup_is_request_pool_indexed():
    calls = []

    class Adapter:
        chunk_size = 64
        draft_reuses_target_state = True

        @staticmethod
        def backup_draft_state(**kwargs):
            calls.append(kwargs)
            return kwargs["out"]

    lifecycle = _dvr_lifecycle(Adapter())
    lifecycle.draft_state_backup = object()
    req = _Req(rid="r1", req_pool_idx=2, retraction_count=0)
    lifecycle.boundaries[2] = _dvr_checkpoint(req)
    lifecycle._ensure_boundary_state = lambda *_args, **_kwargs: None
    lifecycle.state_context = lambda _batch, require_boundary=False: SimpleNamespace(
        state_cache=object(),
        live_indices=torch.tensor([7]),
        state_input_indices=torch.tensor([2]),
    )
    batch = SimpleNamespace(
        reqs=[req],
        req_to_token_pool=SimpleNamespace(req_to_token=torch.empty(4, 0)),
    )

    lifecycle.prepare_for_draft(batch)

    assert calls[0]["indices"].tolist() == [7]
    assert calls[0]["backup_indices"].tolist() == [2]


@pytest.mark.parametrize(
    ("physical_len", "physical_track", "expected_next_track"),
    [
        (128, 0, 1),
        (128, 1, 0),
        (192, 0, 0),
        (192, 1, 1),
    ],
)
def test_dvr_finished_request_publishes_exact_ping_pong_boundary(
    physical_len, physical_track, expected_next_track
):
    lifecycle = _dvr_lifecycle(SimpleNamespace(chunk_size=64))
    syncs = []
    pool = SimpleNamespace(get_mamba_ping_pong_other_idx=lambda idx: 1 - idx)
    lifecycle.model_runner = SimpleNamespace(
        forward_stream=SimpleNamespace(synchronize=lambda: syncs.append(True)),
        req_to_token_pool=pool,
    )
    req = _Req(
        rid="r0",
        req_pool_idx=1,
        retraction_count=0,
        skip_radix_cache_insert=False,
        mamba_last_track_seqlen=64,
        mamba_next_track_idx=1,
        cache_commit_len=128,
    )
    lifecycle.boundaries[1] = _dvr_checkpoint(req, seq_len=128)
    lifecycle.physical_boundary_lens[1] = physical_len
    lifecycle.boundary_track_indices[1] = physical_track

    lifecycle.prepare_for_cache_release(req)

    assert not req.skip_radix_cache_insert
    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == expected_next_track
    assert syncs == [True]
    assert lifecycle.boundaries == {}


def test_dvr_finished_request_respects_the_host_visible_commit_boundary():
    lifecycle = _dvr_lifecycle(SimpleNamespace(chunk_size=64))
    pool = SimpleNamespace(get_mamba_ping_pong_other_idx=lambda idx: 1 - idx)
    lifecycle.model_runner = SimpleNamespace(
        forward_stream=SimpleNamespace(synchronize=lambda: None),
        req_to_token_pool=pool,
    )
    req = _Req(
        rid="r0",
        req_pool_idx=1,
        retraction_count=0,
        skip_radix_cache_insert=False,
        mamba_last_track_seqlen=0,
        mamba_next_track_idx=0,
        cache_commit_len=64,
    )
    lifecycle.boundaries[1] = _dvr_checkpoint(req, seq_len=128)
    lifecycle.physical_boundary_lens[1] = 128
    lifecycle.boundary_track_indices[1] = 1

    lifecycle.prepare_for_cache_release(req)

    assert not req.skip_radix_cache_insert
    assert req.mamba_last_track_seqlen == 64
    assert req.mamba_next_track_idx == 1


def test_dvr_finished_request_skips_insert_when_exact_boundary_is_gone():
    lifecycle = _dvr_lifecycle(SimpleNamespace(chunk_size=64))
    pool = SimpleNamespace(get_mamba_ping_pong_other_idx=lambda idx: 1 - idx)
    lifecycle.model_runner = SimpleNamespace(
        forward_stream=SimpleNamespace(synchronize=lambda: None),
        req_to_token_pool=pool,
    )
    req = _Req(
        rid="r0",
        req_pool_idx=1,
        retraction_count=0,
        skip_radix_cache_insert=False,
        cache_commit_len=128,
    )
    lifecycle.boundaries[1] = _dvr_checkpoint(req, seq_len=128)
    lifecycle.physical_boundary_lens[1] = 256

    lifecycle.prepare_for_cache_release(req)

    assert req.skip_radix_cache_insert
    assert lifecycle.boundaries == {}


def test_dvr_without_radix_drops_request_checkpoint_without_publication():
    lifecycle = _dvr_lifecycle(SimpleNamespace(chunk_size=64), disable_radix_cache=True)
    lifecycle.model_runner = SimpleNamespace(
        forward_stream=SimpleNamespace(
            synchronize=lambda: pytest.fail("Radix-off release must not synchronize")
        )
    )
    req = _Req(
        rid="r0",
        req_pool_idx=1,
        retraction_count=0,
        skip_radix_cache_insert=False,
    )
    lifecycle.boundaries[1] = _dvr_checkpoint(req)

    lifecycle.prepare_for_cache_release(req)

    assert lifecycle.boundaries == {}


def test_dvr_self_draft_graph_does_not_publish_intermediate_war_event():
    assert not DVRDraftDecodeCudaGraphRunner.record_war_fastpath_event
