from types import SimpleNamespace

import pytest
import torch
import sglang.srt.layers.attention.dvr.gdn_backend as dvr_gdn_module
from sglang.srt.configs.mamba_utils import Mamba2StateShape
from sglang.srt.layers.attention.dvr.gdn_kernels import (
    dvr_chunk_gated_delta_rule,
)
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.dvr.gdn_backend import (
    DVRGDNStateAdapter,
    dvr_gdn_intermediate_bytes_per_request,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def create_gdn_adapter(
    *, num_layers, num_slots, num_draft_tokens, state_shape, dtype, device
):
    conv_dim, conv_window = state_shape.conv[0]
    state_cache = SimpleNamespace(
        conv=(
            torch.empty(
                num_layers,
                num_slots,
                conv_dim,
                conv_window,
                dtype=dtype,
                device=device,
            ),
        ),
        temporal=torch.empty(
            num_layers,
            num_slots,
            *state_shape.temporal,
            dtype=torch.float32,
            device=device,
        ),
    )
    req_to_token_pool = SimpleNamespace(
        mamba_pool=SimpleNamespace(mamba_cache=state_cache),
        req_to_token=torch.empty(num_slots, 1),
    )
    adapter = DVRGDNStateAdapter.for_gdn(
        None,
        model_runner=SimpleNamespace(
            mambaish_config=SimpleNamespace(
                mamba2_cache_params=SimpleNamespace(
                    shape=state_shape,
                    dtype=SimpleNamespace(conv=dtype),
                )
            ),
            req_to_token_pool=req_to_token_pool,
            server_args=SimpleNamespace(speculative_num_draft_tokens=num_draft_tokens),
            spec_algorithm=SimpleNamespace(is_dvr_self_draft=lambda: False),
            device=device,
        ),
    )
    return adapter


def create_test_adapter(
    *,
    transition_windows,
    state_cache=None,
    kernel_dispatcher=None,
    enable_self_draft_state=False,
):
    if state_cache is None:
        layers, slots = transition_windows[0].shape[:2]
        physical_slots = max(slots, 16)
        state_cache = SimpleNamespace(
            conv=(torch.empty(layers, physical_slots, 1, 1),),
            temporal=torch.empty(layers, physical_slots, 1),
        )
    layers, slots = transition_windows[0].shape[:2]
    draft_tokens = max(transition_windows[0].shape[2] - FLA_CHUNK_SIZE, 1)
    draft_state = torch.empty(
        layers,
        slots,
        1,
        *state_cache.temporal.shape[2:],
        dtype=state_cache.temporal.dtype,
        device=state_cache.temporal.device,
    )
    intermediate_conv_window = torch.empty(
        layers,
        slots,
        draft_tokens,
        *state_cache.conv[0].shape[2:],
        dtype=state_cache.conv[0].dtype,
        device=state_cache.conv[0].device,
    )
    return DVRGDNStateAdapter(
        kernel_dispatcher,
        state_cache=state_cache,
        draft_state=draft_state,
        intermediate_conv_window=intermediate_conv_window,
        transition_windows=transition_windows,
        enable_self_draft_state=enable_self_draft_state,
    )


def prepare_prefill(adapter, *, prefix_lens, extend_lens, request_rows):
    device = adapter.transition_windows[0].device
    prefix_lens = torch.tensor(prefix_lens, dtype=torch.long, device=device)
    extend_lens = torch.tensor(extend_lens, dtype=torch.long, device=device)
    extend_start_loc = torch.cat((extend_lens.new_zeros(1), extend_lens.cumsum(0)[:-1]))
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.tensor(request_rows, dtype=torch.long, device=device),
        extend_prefix_lens=prefix_lens,
        extend_seq_lens=extend_lens,
        extend_start_loc=extend_start_loc,
    )
    adapter.prepare_forward(
        forward_batch=forward_batch,
        forward_metadata=SimpleNamespace(
            mamba_cache_indices=torch.tensor(
                request_rows, dtype=torch.long, device=device
            )
        ),
    )
    return forward_batch


def test_dvr_chunk_boundary_outputs_are_all_or_none():
    with pytest.raises(ValueError, match="must be provided together"):
        dvr_chunk_gated_delta_rule(
            torch.empty(1, 1, 1, 1, dtype=torch.bfloat16),
            torch.empty(1, 1, 1, 1, dtype=torch.bfloat16),
            torch.empty(1, 1, 1, 1, dtype=torch.bfloat16),
            torch.empty(1, 1, 1),
            torch.empty(1, 1, 1),
            initial_state=torch.empty(1, 1, 1, 1),
            initial_state_indices=torch.zeros(1, dtype=torch.int32),
            boundary_state=torch.empty(1),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dvr_chunk_preserves_source_and_exports_fp32_boundary():
    torch.manual_seed(7)
    q = torch.randn(1, 80, 1, 16, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, 80, 1, dtype=torch.float32, device="cuda")
    )
    beta = torch.sigmoid(torch.randn(1, 80, 1, dtype=torch.float32, device="cuda"))
    source = torch.randn(2, 1, 16, 16, device="cuda")
    original = source.clone()
    boundary = torch.zeros_like(source)

    output, _, chunk_states = dvr_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=source,
        initial_state_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        boundary_state=boundary,
        boundary_state_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
        boundary_state_steps=torch.tensor([1], dtype=torch.int32, device="cuda"),
    )

    assert output.shape[1] == 80
    assert chunk_states.dtype == q.dtype
    assert torch.equal(source, original)

    reference = original.clone()
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule

    chunk_gated_delta_rule(
        q[:, :64],
        k[:, :64],
        v[:, :64],
        g[:, :64],
        beta[:, :64],
        initial_state=reference,
        initial_state_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        use_qk_l2norm_in_kernel=True,
    )
    assert torch.equal(boundary[0], reference[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dvr_chunk_exports_boundary_at_aligned_sequence_end():
    torch.manual_seed(11)
    q = torch.randn(1, 64, 1, 16, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, 64, 1, dtype=torch.float32, device="cuda")
    )
    beta = torch.sigmoid(torch.randn(1, 64, 1, dtype=torch.float32, device="cuda"))
    source = torch.randn(2, 1, 16, 16, device="cuda")
    boundary = torch.zeros_like(source)

    dvr_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=source,
        initial_state_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        boundary_state=boundary,
        boundary_state_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
        boundary_state_steps=torch.tensor([1], dtype=torch.int32, device="cuda"),
    )

    reference = source.clone()
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule

    chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=reference,
        initial_state_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        use_qk_l2norm_in_kernel=True,
    )
    assert torch.equal(boundary[0], reference[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gdn_state_inputs_support_distinct_key_and_value_heads():
    state_shape = Mamba2StateShape.create(
        tp_world_size=2,
        intermediate_size=32 * 128,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
    )

    adapter = create_gdn_adapter(
        num_layers=1,
        num_slots=3,
        num_draft_tokens=16,
        state_shape=state_shape,
        dtype=torch.float32,
        device="cuda",
    )

    k_cache, v_cache, g_cache, beta_cache = adapter.transition_windows
    assert k_cache.shape == (1, 3, FLA_CHUNK_SIZE + 16, 8, 128)
    assert v_cache.shape == (1, 3, FLA_CHUNK_SIZE + 16, 16, 128)
    assert g_cache.shape == (1, 3, FLA_CHUNK_SIZE + 16, 16)
    assert beta_cache.shape == (1, 3, FLA_CHUNK_SIZE + 16, 16)

    k = torch.randn(2, 8, 128, device="cuda")
    v = torch.randn(2, 16, 128, device="cuda")
    g = torch.randn(2, 16, device="cuda")
    beta = torch.randn(2, 16, device="cuda")
    forward_batch = prepare_prefill(
        adapter,
        prefix_lens=[FLA_CHUNK_SIZE],
        extend_lens=[2],
        request_rows=[1],
    )
    adapter.cache_prefill_transitions(
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g,
        beta=beta,
        layer_idx=0,
        forward_batch=forward_batch,
    )

    k_cache, v_cache, g_cache, beta_cache = (
        state_input[0] for state_input in adapter.transition_windows
    )
    assert torch.equal(k_cache[1, :2], k)
    assert torch.equal(v_cache[1, :2], v)
    assert torch.equal(g_cache[1, :2], g)
    assert torch.equal(beta_cache[1, :2], beta)


def test_gdn_extend_requires_forward_metadata():
    adapter = create_test_adapter(
        transition_windows=tuple(torch.zeros(1, 2, 66, 1) for _ in range(4)),
    )

    with pytest.raises(RuntimeError, match="metadata was not prepared"):
        adapter.cache_prefill_transitions(
            k=torch.zeros(1, 1, 1),
            v=torch.zeros(1, 1, 1),
            g=torch.zeros(1, 1),
            beta=torch.zeros(1, 1),
            layer_idx=0,
            forward_batch=SimpleNamespace(
                req_pool_indices=None,
                extend_prefix_lens=None,
                extend_seq_lens=None,
                extend_start_loc=None,
            ),
        )


def test_gdn_memory_estimate_matches_allocation():
    state_shape = Mamba2StateShape.create(
        tp_world_size=4,
        intermediate_size=32 * 128,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
    )
    num_layers = 3
    num_slots = 5
    num_draft_tokens = 16
    adapter = create_gdn_adapter(
        num_layers=num_layers,
        num_slots=num_slots,
        num_draft_tokens=num_draft_tokens,
        state_shape=state_shape,
        dtype=torch.bfloat16,
        device="cpu",
    )
    params = SimpleNamespace(
        layers=tuple(range(num_layers)),
        shape=state_shape,
        dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
    )

    allocated_per_request = sum(
        t.numel() * t.element_size() for t in adapter.transition_windows
    )
    allocated_per_request += (
        adapter.verify_conv.numel() * adapter.verify_conv.element_size()
    )
    allocated_per_request += (
        adapter.draft_state.numel() * adapter.draft_state.element_size()
    )
    allocated_per_request += adapter.intermediate_conv_window.untyped_storage().nbytes()
    allocated_per_request //= num_slots
    checkpoint_lanes = 2
    allocated_per_request += (1 + checkpoint_lanes) * torch.int64.itemsize
    assert allocated_per_request == dvr_gdn_intermediate_bytes_per_request(
        params,
        num_draft_tokens,
        checkpoint_lanes=checkpoint_lanes,
        dedup_conv_window=True,
    )


def test_dvr_gdn_adapter_maps_request_and_state_slots():
    state_shape = Mamba2StateShape.create(
        tp_world_size=2,
        intermediate_size=32 * 128,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
    )
    all_layers_state_cache = SimpleNamespace(
        conv=(torch.zeros(1, 3, 4096, 3),),
        temporal=torch.zeros(1, 3, 16, 128, 128),
    )
    req_to_token_pool = SimpleNamespace(
        mamba_pool=SimpleNamespace(mamba_cache=all_layers_state_cache),
        req_to_token=torch.empty(3, 1),
        get_mamba_indices=lambda req_pool_indices: torch.zeros_like(req_pool_indices),
    )
    batch = SimpleNamespace(
        batch_size=lambda: 1,
        req_to_token_pool=req_to_token_pool,
        req_pool_indices=torch.tensor([2]),
        reqs=[SimpleNamespace(mamba_ping_pong_track_buffer=torch.tensor([0]))],
    )
    adapter = DVRGDNStateAdapter.for_gdn(
        None,
        model_runner=SimpleNamespace(
            mambaish_config=SimpleNamespace(
                mamba2_cache_params=SimpleNamespace(
                    shape=state_shape,
                    dtype=SimpleNamespace(conv=torch.float32),
                )
            ),
            req_to_token_pool=req_to_token_pool,
            server_args=SimpleNamespace(speculative_num_draft_tokens=4),
            spec_algorithm=SimpleNamespace(is_dvr_self_draft=lambda: False),
            device="cpu",
        ),
    )

    request_rows, accepted_conv_slots = adapter.resolve_request_slots(batch=batch)

    assert adapter.state_cache is all_layers_state_cache
    assert request_rows.tolist() == [2]
    assert accepted_conv_slots.tolist() == [0]
    assert adapter.transition_windows[0].shape[2] == FLA_CHUNK_SIZE + 4
    assert adapter.transition_windows[0][0].shape == (
        3,
        FLA_CHUNK_SIZE + 4,
        8,
        128,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gdn_extend_tail_cache_uses_target_request_slots():
    state_shape = Mamba2StateShape.create(
        tp_world_size=2,
        intermediate_size=32 * 128,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
    )
    adapter = create_gdn_adapter(
        num_layers=1,
        num_slots=3,
        num_draft_tokens=16,
        state_shape=state_shape,
        dtype=torch.float32,
        device="cuda",
    )
    for tensor in adapter.transition_windows:
        tensor[:, 1] = 1

    k = torch.randn(1, 2, 8, 128, device="cuda")
    v = torch.randn(1, 2, 16, 128, device="cuda")
    g = torch.randn(2, 16, device="cuda")
    beta = torch.randn(2, 16, device="cuda")

    forward_batch = prepare_prefill(
        adapter,
        prefix_lens=[FLA_CHUNK_SIZE],
        extend_lens=[2],
        request_rows=[1],
    )
    adapter.cache_prefill_transitions(
        k=k,
        v=v,
        g=g,
        beta=beta,
        layer_idx=0,
        forward_batch=forward_batch,
    )
    k_cache, v_cache, g_cache, beta_cache = (
        state_input[0] for state_input in adapter.transition_windows
    )
    assert torch.equal(k_cache[1, :2], k.reshape(2, 8, 128))
    assert torch.equal(v_cache[1, :2], v.reshape(2, 16, 128))
    assert torch.equal(g_cache[1, :2], g)
    assert torch.equal(beta_cache[1, :2], beta)
    # Rows after accepted_tail_lens are outside every consumed causal prefix. Keeping
    # them untouched avoids clearing the full k/v/g/beta window per verify.
    for tensor in (k_cache, v_cache, g_cache, beta_cache):
        assert torch.all(tensor[1, 2:] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gdn_prefill_transition_scatter_handles_mixed_chunk_boundaries():
    device = "cuda"
    adapter = create_test_adapter(
        transition_windows=tuple(
            torch.full((1, 6, 80, 2), -1.0, device=device) for _ in range(4)
        ),
    )
    prefix_lens = [62, 63, 126, 5, 0]
    extend_lens = [1, 3, 2, 130, 0]
    request_rows = [1, 2, 3, 4, 0]
    forward_batch = prepare_prefill(
        adapter,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        request_rows=request_rows,
    )

    total_tokens = sum(extend_lens)
    values = [
        torch.arange(total_tokens * 2, dtype=torch.float32, device=device).view(
            total_tokens, 2
        )
        + offset
        for offset in (0, 1000, 2000, 3000)
    ]
    adapter.cache_prefill_transitions(
        k=values[0].view(1, total_tokens, 1, 2),
        v=values[1].view(1, total_tokens, 1, 2),
        g=values[2],
        beta=values[3],
        layer_idx=0,
        forward_batch=forward_batch,
    )

    starts = [0, 1, 4, 6, 136]
    expected_ranges = (
        (1, 62, 1, starts[0]),
        (2, 0, 2, starts[1] + 1),
        (3, 0, 0, starts[2]),
        (4, 0, 7, starts[3] + 123),
        (0, 0, 0, starts[4]),
    )
    for cache, value in zip(adapter.transition_windows, values, strict=True):
        cache = cache[0]
        for slot, destination_start, count, source_start in expected_ranges:
            if count:
                assert torch.equal(
                    cache[slot, destination_start : destination_start + count],
                    value[source_start : source_start + count],
                )
            untouched = torch.ones(cache.shape[1], dtype=torch.bool, device=device)
            untouched[destination_start : destination_start + count] = False
            assert torch.all(cache[slot, untouched] == -1)


def test_gdn_state_input_window_rejects_invalid_tail():
    with pytest.raises(RuntimeError, match="invalid linear-state tail"):
        dvr_gdn_module._compact_gdn_transition_windows(
            (torch.zeros(1, 1, 80, 1),),
            indices=torch.tensor([0]),
            crosses_chunk_boundary=torch.tensor([True]),
            chunk_size=64,
            accepted_tail_lens=torch.tensor([17]),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gdn_state_input_window_compacts_only_valid_crossing_tail():
    values = torch.arange(2 * 80 * 2, dtype=torch.float32, device="cuda").view(
        1, 2, 80, 2
    )
    original = values.clone()
    dvr_gdn_module._compact_gdn_transition_windows(
        (values,),
        indices=torch.tensor([0, 1], device="cuda"),
        crosses_chunk_boundary=torch.tensor([False, True], device="cuda"),
        chunk_size=64,
        # A non-crossing lane may retain nearly a full chunk; only the compacted
        # crossing lane is constrained by the 16-row post-boundary capacity.
        accepted_tail_lens=torch.tensor([63, 3], device="cuda"),
    )

    torch.testing.assert_close(values[:, 0], original[:, 0])
    torch.testing.assert_close(values[:, 1, :3], original[:, 1, 64:67])
    torch.testing.assert_close(values[:, 1, 3:], original[:, 1, 3:])


def test_gdn_verify_uses_mamba_padding_sentinel():
    adapter = create_test_adapter(
        transition_windows=(torch.zeros(1, 3, 66, 1),),
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.long),
        req_pool_indices=torch.tensor([2, 0]),
        seq_lens=torch.tensor([63, 0]),
        spec_info=SimpleNamespace(draft_token_num=2),
    )

    adapter.prepare_target_verify(
        forward_batch=forward_batch,
        cache_indices=torch.tensor([7, PAD_SLOT_ID]),
    )
    (
        checkpoint_slots,
        request_rows,
        verify_conv_slots,
        accepted_tail_lens,
        valid_mask,
        boundary_state_steps,
    ) = adapter.verify_metadata

    assert torch.equal(checkpoint_slots, torch.tensor([7, 0]))
    assert torch.equal(request_rows, torch.tensor([2, 0]))
    assert torch.equal(verify_conv_slots, torch.tensor([7, 0]))
    assert torch.equal(accepted_tail_lens, torch.tensor([63, 0]))
    assert torch.equal(valid_mask, torch.tensor([True, False]))
    assert torch.equal(boundary_state_steps, torch.tensor([1, -1]))
    conv_state, conv_slots, intermediate_rows = adapter.verify_conv_state(0)
    assert conv_state.data_ptr() == adapter.verify_conv[0].data_ptr()
    assert torch.equal(conv_slots, torch.tensor([2, 0]))
    assert torch.equal(intermediate_rows, torch.tensor([2, 0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gdn_verify_exports_boundary_into_workspace(monkeypatch):
    state_shape = Mamba2StateShape.create(
        tp_world_size=2,
        intermediate_size=32 * 128,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
    )
    adapter = create_gdn_adapter(
        num_layers=1,
        num_slots=3,
        num_draft_tokens=2,
        state_shape=state_shape,
        dtype=torch.float32,
        device="cuda",
    )
    calls = []

    def extend(**kwargs):
        calls.append(kwargs)
        kwargs["boundary_state"][kwargs["boundary_state_indices"]] = 7
        return (
            torch.zeros(1, 66, 16, 128, device="cuda"),
            None,
            torch.empty(0, device="cuda"),
        )

    state_cache = SimpleNamespace(
        conv=(torch.zeros(1, 3, 1, 1, device="cuda"),),
        temporal=torch.zeros(1, 3, 16, 128, 128, device="cuda"),
    )
    adapter.state_cache = state_cache
    adapter.verify_metadata = (
        torch.tensor([0], device="cuda"),
        torch.tensor([1], device="cuda"),
        torch.tensor([0], device="cuda"),
        torch.tensor([2], device="cuda"),
        torch.tensor([True], device="cuda"),
        torch.tensor([-1], device="cuda"),
    )

    query = torch.ones(1, 2, 8, 128, device="cuda")
    monkeypatch.setattr(dvr_gdn_module, "dvr_chunk_gated_delta_rule", extend)
    output = adapter.forward_target_verify(
        query=query,
        key=torch.zeros(1, 2, 8, 128, device="cuda"),
        value=torch.zeros(1, 2, 16, 128, device="cuda"),
        g=torch.zeros(1, 2, 16, device="cuda"),
        beta=torch.zeros(1, 2, 16, device="cuda"),
        layer_idx=0,
    )
    assert output.shape == (1, 2, 16, 128)
    assert torch.equal(
        calls[0]["initial_state_indices"], torch.tensor([0], device="cuda")
    )
    assert torch.equal(
        calls[0]["boundary_state_indices"], torch.tensor([1], device="cuda")
    )
    assert torch.equal(
        calls[0]["boundary_state_steps"], torch.tensor([-1], device="cuda")
    )
    assert torch.count_nonzero(adapter.draft_state[:, 1]) > 0
    assert torch.count_nonzero(calls[0]["q"][:, :2]) == 0
    assert torch.equal(calls[0]["q"][:, 2:4], query)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gdn_verify_window_pack_matches_indexing_reference():
    torch.manual_seed(3)
    slots, window, draft_tokens = 4, 9, 3
    cache0 = torch.randn(slots, window, 2, 4, device="cuda")
    cache1 = torch.randn_like(cache0)
    original0 = cache0.clone()
    original1 = cache1.clone()
    candidate0 = torch.randn(3, draft_tokens, 2, 4, device="cuda")
    candidate1 = torch.randn_like(candidate0)
    indices = torch.tensor([1, 3, 0], device="cuda")
    accepted_tail_lens = torch.tensor([0, 5, 0], device="cuda")
    valid = torch.tensor([True, True, False], device="cuda")

    output0, output1 = dvr_gdn_module._pack_verify_window_pair(
        cache0,
        candidate0,
        cache1=cache1,
        candidate1=candidate1,
        request_rows=indices,
        accepted_tail_lens=accepted_tail_lens,
        valid_mask=valid,
    )

    expected0 = original0.clone()
    expected1 = original1.clone()
    for req in range(2):
        slot = int(indices[req])
        tail = int(accepted_tail_lens[req])
        expected0[slot, tail : tail + draft_tokens] = candidate0[req]
        expected1[slot, tail : tail + draft_tokens] = candidate1[req]
    expected_output0 = expected0[indices]
    expected_output1 = expected1[indices]
    expected_output0[2] = original0[0]
    expected_output1[2] = original1[0]

    torch.testing.assert_close(cache0, expected0)
    torch.testing.assert_close(cache1, expected1)
    torch.testing.assert_close(output0, expected_output0)
    torch.testing.assert_close(output1, expected_output1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gdn_verify_window_pack_can_keep_query_candidate_only():
    cache = torch.randn(2, 9, 2, 4, device="cuda")
    original = cache.clone()
    query = torch.randn(1, 3, 2, 4, device="cuda")
    key = torch.randn_like(query)

    packed_query, packed_key = dvr_gdn_module._pack_verify_window_pair(
        cache,
        query,
        cache1=cache,
        candidate1=key,
        request_rows=torch.tensor([1], device="cuda"),
        accepted_tail_lens=torch.tensor([4], device="cuda"),
        valid_mask=torch.tensor([True], device="cuda"),
        read_cache0=False,
        persist_cache0=False,
    )

    assert torch.count_nonzero(packed_query[:, :4]) == 0
    torch.testing.assert_close(packed_query[:, 4:7], query)
    torch.testing.assert_close(cache[0], original[0])
    torch.testing.assert_close(cache[1, 4:7], key[0])
    torch.testing.assert_close(packed_key, cache[torch.tensor([1], device="cuda")])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gdn_verify_output_gather_matches_logical_rows():
    source = torch.arange(2 * 9 * 3 * 4, device="cuda", dtype=torch.float32).view(
        1, 2 * 9, 3, 4
    )
    accepted_tail_lens = torch.tensor([1, 5], device="cuda")

    actual = dvr_gdn_module._gather_verify_output(
        source, accepted_tail_lens=accepted_tail_lens, draft_tokens=3
    )

    source_by_req = source.view(2, 9, 3, 4)
    expected = torch.cat((source_by_req[0, 1:4], source_by_req[1, 5:8])).view(
        1, 6, 3, 4
    )
    torch.testing.assert_close(actual, expected)


def test_gdn_self_draft_state_is_request_owned_and_keeps_target_unchanged(
    monkeypatch,
):
    conv = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    temporal = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
    state_cache = SimpleNamespace(conv=(conv,), temporal=temporal)
    adapter = create_test_adapter(
        state_cache=state_cache,
        transition_windows=(torch.empty(1, 4, 1),) * 4,
        enable_self_draft_state=True,
    )

    def rebuild(*_args, **kwargs):
        kwargs["draft_state"][:, kwargs["request_rows"], 0] = kwargs["boundary_state"][
            :, kwargs["boundary_slots"]
        ]

    monkeypatch.setattr(dvr_gdn_module, "_rebuild_gdn_self_draft_state", rebuild)
    adapter.initialize_self_draft_state(
        accepted_conv_slots=torch.tensor([0, 2]),
        request_rows=torch.tensor([1, 3]),
        accepted_tail_lens=torch.tensor([0, 0]),
    )
    original_conv = conv.clone()
    original_temporal = temporal.clone()
    layer_cache = SimpleNamespace(temporal=temporal[0])
    forward_batch = SimpleNamespace(
        req_pool_indices=torch.tensor([1, 3]), forward_mode=ForwardMode.DECODE
    )
    assert (
        adapter.decode_state(
            layer_cache=layer_cache,
            forward_batch=SimpleNamespace(
                req_pool_indices=torch.tensor([1, 3]),
                forward_mode=ForwardMode.EXTEND,
            ),
            layer_idx=0,
        )
        is None
    )

    draft_conv, draft_temporal, indices = adapter.decode_state(
        layer_cache=layer_cache, forward_batch=forward_batch, layer_idx=0
    )
    draft_conv[indices] = -1
    draft_temporal[indices] = -2

    torch.testing.assert_close(conv, original_conv)
    torch.testing.assert_close(temporal, original_temporal)
    assert torch.all(adapter.self_draft_conv[:, [1, 3]] == -1)
    assert torch.all(adapter.draft_state[:, [1, 3], 0] == -2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gdn_self_draft_rebuild_reads_boundary_and_writes_workspace():
    torch.manual_seed(1)
    layers, slots, tokens, dim = 2, 5, 80, 16
    k = torch.randn(layers, slots, tokens, 1, dim, device="cuda")
    v = torch.randn_like(k)
    g = torch.randn(layers, slots, tokens, 1, device="cuda") * 0.01
    beta = torch.sigmoid(torch.randn_like(g))
    temporal = torch.randn(layers, 7, 1, dim, dim, device="cuda") * 0.01
    original = temporal.clone()
    workspace = torch.zeros(layers, slots, 1, 1, dim, dim, device="cuda")
    request_rows = torch.tensor([1, 4], device="cuda")
    boundaries = torch.tensor([2, 6], device="cuda")
    token_counts = torch.tensor([0, 5], device="cuda")

    expected = torch.empty(layers, 2, 1, dim, dim, device="cuda")
    for layer in range(layers):
        for row, (request, boundary, count) in enumerate(
            zip(request_rows, boundaries, token_counts, strict=True)
        ):
            state = temporal[layer, boundary].clone()
            for step in range(int(count)):
                key = k[layer, request, step, 0].float()
                key /= torch.sqrt(torch.sum(key * key) + 1e-6)
                value = v[layer, request, step, 0].float()
                state *= torch.exp(g[layer, request, step, 0].float())
                value = (value - torch.sum(state[0] * key.unsqueeze(0), dim=1)) * beta[
                    layer, request, step, 0
                ].float()
                state += value[:, None] * key[None, :]
            expected[layer, row] = state

    dvr_gdn_module._rebuild_gdn_self_draft_state(
        (k, v, g, beta),
        boundary_state=temporal,
        draft_state=workspace,
        request_rows=request_rows,
        boundary_slots=boundaries,
        token_count=token_counts,
    )

    torch.testing.assert_close(temporal, original)
    torch.testing.assert_close(
        workspace[:, request_rows, 0], expected, rtol=2e-4, atol=2e-4
    )


def test_gdn_commit_alternates_boundary_slots(monkeypatch):
    temporal_scatters = []
    conv_scatters = []
    monkeypatch.setattr(
        dvr_gdn_module,
        "dvr_scatter_state",
        lambda _dst, _src, **kwargs: temporal_scatters.append(
            (
                kwargs["destination_rows"].tolist(),
                kwargs["source_steps"].tolist(),
                kwargs["source_rows"].tolist(),
            )
        ),
    )
    monkeypatch.setattr(
        dvr_gdn_module,
        "dvr_scatter_conv_window",
        lambda _dst, _src, **kwargs: conv_scatters.append(
            (
                kwargs["destination_rows"].tolist(),
                kwargs["source_steps"].tolist(),
                kwargs["source_rows"].tolist(),
            )
        ),
    )
    compacted_tails = []
    monkeypatch.setattr(
        dvr_gdn_module,
        "_compact_gdn_transition_windows",
        lambda *_args, **kwargs: compacted_tails.append(
            kwargs["accepted_tail_lens"].tolist()
        ),
    )
    state_cache = SimpleNamespace(
        temporal=torch.empty(1, 9, 1),
        conv=(torch.empty(1, 9, 1, 1),),
    )
    adapter = create_test_adapter(
        state_cache=state_cache,
        transition_windows=(torch.empty(1, 3, 80, 1),) * 4,
    )

    crossed = adapter.commit_accepted_state(
        request_rows=torch.tensor([1, 2]),
        accepted_conv_slots=torch.tensor([3, 4]),
        publish_boundary_slots=torch.tensor([7, 8]),
        tail_lens_before=torch.tensor([63, 1]),
        accepted_token_counts=torch.tensor([2, 2]),
    )

    assert crossed.tolist() == [True, False]
    assert temporal_scatters == [
        ([3, 4], [0, -1], [1, 2]),
        ([7, 8], [0, -1], [1, 2]),
    ]
    assert conv_scatters == [
        ([3, 4], [1, 1], [1, 2]),
        ([7, 8], [0, -1], [1, 2]),
    ]
    assert compacted_tails == [[1, 3]]
