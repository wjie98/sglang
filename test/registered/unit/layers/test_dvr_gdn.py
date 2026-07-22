from types import SimpleNamespace

import pytest
import torch
import sglang.srt.layers.attention.linear.dvr_gdn as dvr_gdn_module
from sglang.srt.configs.mamba_utils import Mamba2StateShape
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.linear.dvr_gdn import (
    DVRGDNStateAdapter,
    dvr_gdn_state_input_bytes_per_request,
)
from sglang.srt.layers.attention.linear.dvr_state import DVRStateInputCache
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID


def create_gdn_state_input_cache(
    *, num_layers, num_slots, num_draft_tokens, state_shape, dtype, device
):
    state_cache = SimpleNamespace(
        temporal=torch.empty(num_layers, 1, dtype=torch.float32),
        intermediate_ssm=torch.empty(num_layers, num_slots, 1, dtype=torch.float32),
    )
    req_to_token_pool = SimpleNamespace(
        get_speculative_mamba2_params_all_layers=lambda: state_cache
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
    return adapter.state_input_cache


def test_gdn_state_input_cache_supports_distinct_key_and_value_heads():
    state_shape = Mamba2StateShape.create(
        tp_world_size=2,
        intermediate_size=32 * 128,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
    )

    cache = create_gdn_state_input_cache(
        num_layers=1,
        num_slots=3,
        num_draft_tokens=16,
        state_shape=state_shape,
        dtype=torch.float32,
        device="cpu",
    )

    q_cache, k_cache, v_cache, g_cache, beta_cache = cache.tensors
    assert q_cache.shape == (1, 3, FLA_CHUNK_SIZE + 16, 8, 128)
    assert k_cache.shape == (1, 3, FLA_CHUNK_SIZE + 16, 8, 128)
    assert v_cache.shape == (1, 3, FLA_CHUNK_SIZE + 16, 16, 128)
    assert g_cache.shape == (1, 3, FLA_CHUNK_SIZE + 16, 16)
    assert beta_cache.shape == (1, 3, FLA_CHUNK_SIZE + 16, 16)

    q = torch.randn(2, 8, 128)
    k = torch.randn(2, 8, 128)
    v = torch.randn(2, 16, 128)
    g = torch.randn(2, 16)
    beta = torch.randn(2, 16)
    layer_cache = cache[0]
    layer_cache.write_extend_tail(
        values=(q, k, v, g, beta),
        indices=torch.tensor([1]),
        extend_prefix_lens_cpu=[FLA_CHUNK_SIZE],
        extend_seq_lens_cpu=[2],
        chunk_size=FLA_CHUNK_SIZE,
    )

    assert torch.equal(layer_cache.tail_lens[1], torch.tensor(2, dtype=torch.int32))
    q_cache, k_cache, v_cache, g_cache, beta_cache = layer_cache.tensors
    assert torch.equal(q_cache[1, :2], q)
    assert torch.equal(k_cache[1, :2], k)
    assert torch.equal(v_cache[1, :2], v)
    assert torch.equal(g_cache[1, :2], g)
    assert torch.equal(beta_cache[1, :2], beta)


def test_gdn_state_input_memory_estimate_matches_allocation():
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
    cache = create_gdn_state_input_cache(
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
        dtype=SimpleNamespace(conv=torch.bfloat16),
    )

    allocated = sum(t.numel() * t.element_size() for t in cache.tensors)
    allocated += cache.tail_lens.numel() * cache.tail_lens.element_size()
    assert allocated // num_slots == dvr_gdn_state_input_bytes_per_request(
        params, num_draft_tokens
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
        temporal=torch.zeros(1, 3, 16, 128, 128),
        # DVR stores only the exported chunk-boundary recurrent state here;
        # the state-input window still needs all four draft positions.
        intermediate_ssm=torch.zeros(1, 3, 1, 16, 128, 128),
    )
    req_to_token_pool = SimpleNamespace(
        mamba_pool=SimpleNamespace(mamba_cache=all_layers_state_cache),
        get_mamba_indices=lambda req_pool_indices: torch.zeros_like(req_pool_indices),
        get_speculative_mamba2_params_all_layers=lambda: all_layers_state_cache,
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

    returned_state_cache, state_input_indices, live_indices = adapter.batch_state(
        batch=batch
    )

    assert returned_state_cache is all_layers_state_cache
    assert state_input_indices.tolist() == [2]
    assert live_indices.tolist() == [0]
    window = adapter.state_input_window(layer_idx=0)
    assert window.capacity == FLA_CHUNK_SIZE + 4
    assert window.tensors[0].shape == (3, FLA_CHUNK_SIZE + 4, 8, 128)


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
    cache = create_gdn_state_input_cache(
        num_layers=1,
        num_slots=3,
        num_draft_tokens=16,
        state_shape=state_shape,
        dtype=torch.float32,
        device="cpu",
    )
    for tensor in cache.tensors:
        tensor[:, 1] = 1
    adapter = DVRGDNStateAdapter(kernel_dispatcher=None, state_input_cache=cache)

    q = torch.randn(1, 2, 8, 128)
    k = torch.randn(1, 2, 8, 128)
    v = torch.randn(1, 2, 16, 128)
    g = torch.randn(2, 16)
    beta = torch.randn(2, 16)

    target_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[FLA_CHUNK_SIZE],
        extend_seq_lens_cpu=[2],
        req_pool_indices=torch.tensor([1]),
    )
    adapter.cache_extend_tail(
        forward_batch=target_batch,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        layer_idx=0,
    )
    layer_cache = cache[0]
    assert torch.equal(layer_cache.tail_lens[1], torch.tensor(2, dtype=torch.int32))
    q_cache, k_cache, v_cache, g_cache, beta_cache = layer_cache.tensors
    assert torch.equal(q_cache[1, :2], q.reshape(2, 8, 128))
    assert torch.equal(k_cache[1, :2], k.reshape(2, 8, 128))
    assert torch.equal(v_cache[1, :2], v.reshape(2, 16, 128))
    assert torch.equal(g_cache[1, :2], g)
    assert torch.equal(beta_cache[1, :2], beta)
    # Rows after tail_lens are outside every consumed causal prefix. Keeping
    # them untouched avoids clearing the full q/k/v/g/beta window per verify.
    for tensor in layer_cache.tensors:
        assert torch.all(tensor[1, 2:] == 1)


def test_gdn_state_input_window_shifts_only_crossed_requests():
    values = torch.arange(2 * 80, dtype=torch.float32).view(2, 80, 1)
    original = values.clone()
    cache = DVRStateInputCache(
        tensors=(values,),
        tail_lens=torch.zeros(2, dtype=torch.int32),
        has_layer_dim=False,
    )

    cache.shift_after_boundary(
        indices=torch.tensor([0, 1]),
        crosses_chunk_boundary=torch.tensor([False, True]),
        chunk_size=64,
    )

    torch.testing.assert_close(values[0], original[0])
    torch.testing.assert_close(values[1, :16], original[1, 64:80])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gdn_state_input_window_compacts_only_valid_crossing_tail():
    values = torch.arange(2 * 80 * 2, dtype=torch.float32, device="cuda").view(
        1, 2, 80, 2
    )
    original = values.clone()
    cache = DVRStateInputCache(
        tensors=(values,),
        tail_lens=torch.zeros(2, dtype=torch.int32, device="cuda"),
    )

    cache.shift_after_boundary(
        indices=torch.tensor([0, 1], device="cuda"),
        crosses_chunk_boundary=torch.tensor([False, True], device="cuda"),
        chunk_size=64,
        tail_lens=torch.tensor([8, 3], device="cuda"),
    )

    torch.testing.assert_close(values[:, 0], original[:, 0])
    torch.testing.assert_close(values[:, 1, :3], original[:, 1, 64:67])
    torch.testing.assert_close(values[:, 1, 3:], original[:, 1, 3:])


def test_gdn_verify_uses_mamba_padding_sentinel():
    adapter = DVRGDNStateAdapter(
        kernel_dispatcher=None,
        verify_boundary_indices=torch.tensor([0, 11, 12]),
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.long),
        req_pool_indices=torch.tensor([2, 0]),
        spec_info=SimpleNamespace(draft_token_num=2),
    )

    boundary_indices, state_input_indices, valid_mask = adapter.target_verify_indices(
        forward_batch=forward_batch,
        cache_indices=torch.tensor([7, PAD_SLOT_ID]),
    )

    assert torch.equal(boundary_indices, torch.tensor([12, 0]))
    assert torch.equal(state_input_indices, torch.tensor([2, 0]))
    assert torch.equal(valid_mask, torch.tensor([True, False]))


def test_gdn_verify_rejects_backend_without_boundary_states():
    state_shape = Mamba2StateShape.create(
        tp_world_size=2,
        intermediate_size=32 * 128,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
    )
    cache = create_gdn_state_input_cache(
        num_layers=1,
        num_slots=3,
        num_draft_tokens=2,
        state_shape=state_shape,
        dtype=torch.float32,
        device="cpu",
    )
    calls = []

    def extend(**kwargs):
        calls.append(kwargs)
        return torch.empty(0), None, None

    dispatcher = SimpleNamespace(extend=extend)
    adapter = DVRGDNStateAdapter(
        kernel_dispatcher=dispatcher,
        state_input_cache=cache,
    )
    state_cache = SimpleNamespace(
        temporal=torch.zeros(3, 16, 128, 128),
        intermediate_ssm=torch.zeros(3, 1, 16, 128, 128),
    )

    with pytest.raises(RuntimeError, match="exact chunk-boundary states"):
        adapter.forward_target_verify(
            query=torch.zeros(1, 2, 8, 128),
            key=torch.zeros(1, 2, 8, 128),
            value=torch.zeros(1, 2, 16, 128),
            g=torch.zeros(1, 2, 16),
            beta=torch.zeros(1, 2, 16),
            state_cache=state_cache,
            boundary_indices=torch.tensor([0]),
            state_input_indices=torch.tensor([1]),
            valid_mask=torch.tensor([True]),
            layer_idx=0,
        )
    assert calls[0]["inplace_update"] is False
    assert torch.equal(calls[0]["cache_indices"], torch.tensor([0]))


def test_gdn_self_draft_state_is_request_owned_and_keeps_target_unchanged():
    conv = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    temporal = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
    workspace = torch.zeros(1, 4, 1, 6)
    state_cache = SimpleNamespace(
        conv=(conv,), temporal=temporal, intermediate_ssm=workspace
    )
    adapter = DVRGDNStateAdapter(kernel_dispatcher=None, draft_reuses_target_state=True)
    adapter.allocate_self_draft_state(state_cache=state_cache, num_slots=4)
    adapter.initialize_self_draft_state(
        state_cache=state_cache,
        live_indices=torch.tensor([0, 2]),
        state_input_indices=torch.tensor([1, 3]),
    )
    original_conv = conv.clone()
    original_temporal = temporal.clone()
    layer_cache = SimpleNamespace(temporal=temporal[0], intermediate_ssm=workspace[0])
    forward_batch = SimpleNamespace(req_pool_indices=torch.tensor([1, 3]))

    with adapter.self_draft_decode():
        draft_conv, draft_temporal, indices = adapter.decode_state(
            layer_cache=layer_cache, forward_batch=forward_batch, layer_idx=0
        )
        draft_conv[indices] = -1
        draft_temporal[indices] = -2

    torch.testing.assert_close(conv, original_conv)
    torch.testing.assert_close(temporal, original_temporal)
    assert torch.all(adapter.draft_conv_state[0][:, [1, 3]] == -1)
    assert torch.all(workspace[:, [1, 3], 0] == -2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gdn_self_draft_rebuild_reads_boundary_and_writes_workspace():
    torch.manual_seed(1)
    layers, slots, tokens, dim = 2, 5, 80, 16
    q = torch.randn(layers, slots, tokens, 1, dim, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn(layers, slots, tokens, 1, device="cuda") * 0.01
    beta = torch.sigmoid(torch.randn_like(g))
    window = DVRStateInputCache(
        tensors=(q, k, v, g, beta),
        tail_lens=torch.zeros(slots, dtype=torch.int32, device="cuda"),
    )
    temporal = torch.randn(layers, 7, 1, dim, dim, device="cuda") * 0.01
    original = temporal.clone()
    workspace = torch.zeros(layers, slots, 1, 1, dim, dim, device="cuda")
    state_cache = SimpleNamespace(
        temporal=temporal,
        intermediate_ssm=workspace,
    )
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
                value = (
                    value - torch.sum(state[0] * key.unsqueeze(0), dim=1)
                ) * beta[layer, request, step, 0].float()
                state += value[:, None] * key[None, :]
            expected[layer, row] = state

    dvr_gdn_module._rebuild_gdn_self_draft_state(
        window,
        state_cache=state_cache,
        state_input_indices=request_rows,
        boundary_indices=boundaries,
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
        "fused_mamba_state_scatter_with_mask",
        lambda _dst, _src, indices, steps, **kwargs: temporal_scatters.append(
            (
                indices.tolist(),
                steps.tolist(),
                kwargs.get("src_indices_raw").tolist(),
            )
        ),
    )
    monkeypatch.setattr(
        dvr_gdn_module,
        "fused_conv_window_scatter_with_mask",
        lambda _dst, _src, indices, steps: conv_scatters.append(
            (indices.tolist(), steps.tolist())
        ),
    )
    tail_updates = []
    window = SimpleNamespace(
        get_tail_lens=lambda **_kwargs: torch.tensor([63, 1]),
        shift_after_boundary=lambda **_kwargs: None,
        set_tail_lens=lambda **kwargs: tail_updates.append(kwargs["value"].tolist()),
    )
    adapter = DVRGDNStateAdapter(kernel_dispatcher=None, state_input_cache=window)
    state_cache = SimpleNamespace(
        temporal=torch.empty(1),
        intermediate_ssm=torch.empty(1),
        conv=(torch.empty(1),),
        intermediate_conv_window=(torch.empty(1),),
    )

    crossed = adapter.commit_after_verify(
        state_cache=state_cache,
        state_input_indices=torch.tensor([1, 2]),
        live_indices=torch.tensor([3, 4]),
        boundary_indices=torch.tensor([5, 6]),
        next_boundary_indices=torch.tensor([7, 8]),
        accepted_token_counts=torch.tensor([2, 2]),
    )

    assert crossed.tolist() == [True, False]
    assert temporal_scatters == [([7, 8], [0, -1], [1, 2])]
    assert conv_scatters == [([3, 4], [1, 1]), ([7, 8], [0, -1])]
    assert tail_updates == [[1, 3]]
