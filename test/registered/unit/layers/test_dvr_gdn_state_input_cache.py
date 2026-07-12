from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.mamba_utils import Mamba2StateShape
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.linear.dvr_gdn import (
    DVRGDNStateAdapter,
    create_gdn_state_input_cache,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fla_boundary_state_preserves_fp32_initial_state_dtype():
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule

    q = torch.randn(1, 64, 1, 16, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, 64, 1, dtype=torch.float32, device="cuda")
    )
    beta = torch.sigmoid(
        torch.randn(1, 64, 1, dtype=torch.float32, device="cuda")
    )
    initial_state = torch.zeros(
        1, 1, 16, 16, dtype=torch.float32, device="cuda"
    )
    initial_state_indices = torch.zeros(1, dtype=torch.int32, device="cuda")

    output, _, boundary_states = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    assert output.dtype == q.dtype
    assert boundary_states.dtype == initial_state.dtype


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
    )

    assert torch.equal(layer_cache.tail_lens[1], torch.tensor(2, dtype=torch.int32))
    q_cache, k_cache, v_cache, g_cache, beta_cache = layer_cache.tensors
    assert torch.equal(q_cache[1, :2], q)
    assert torch.equal(k_cache[1, :2], k)
    assert torch.equal(v_cache[1, :2], v)
    assert torch.equal(g_cache[1, :2], g)
    assert torch.equal(beta_cache[1, :2], beta)


def test_gdn_state_input_cache_is_its_own_window():
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
    )[0]

    assert cache.capacity == FLA_CHUNK_SIZE + 16


def test_dvr_gdn_adapter_owns_state_input_cache():
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
        intermediate_ssm=torch.zeros(1, 3, 4, 16, 128, 128),
    )
    req_to_token_pool = SimpleNamespace(
        mamba_pool=SimpleNamespace(mamba_cache=all_layers_state_cache),
        mamba_map={7: 0},
        get_mamba_indices=lambda req_pool_indices: req_pool_indices,
        get_speculative_mamba2_params_all_layers=lambda: all_layers_state_cache,
    )
    batch = SimpleNamespace(
        batch_size=lambda: 1,
        req_to_token_pool=req_to_token_pool,
        reqs=[SimpleNamespace(mamba_ping_pong_track_buffer=torch.tensor([0]))],
    )
    adapter = DVRGDNStateAdapter(
        kernel_dispatcher=None,
        state_shape=state_shape,
        conv_dtype=torch.float32,
        device="cpu",
    )

    returned_state_cache = adapter.get_state_cache(batch=batch)

    assert returned_state_cache is all_layers_state_cache
    assert adapter.state_input_cache is not None
    window = adapter.state_input_window(layer_idx=0)
    assert window.capacity == FLA_CHUNK_SIZE + 4
    assert window.tensors[0].shape == (4, FLA_CHUNK_SIZE + 4, 8, 128)

    # The adapter owns the side cache without mutating upstream pool/cache objects.
    assert not hasattr(req_to_token_pool, "_dvr_linear_state_input_cache")
    assert not hasattr(
        all_layers_state_cache,
        "linear_state_input_cache",
    )


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
    adapter = DVRGDNStateAdapter(kernel_dispatcher=None, state_input_cache=cache)

    q = torch.randn(1, 2, 8, 128)
    k = torch.randn(1, 2, 8, 128)
    v = torch.randn(1, 2, 16, 128)
    g = torch.randn(2, 16)
    beta = torch.randn(2, 16)

    target_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[FLA_CHUNK_SIZE],
        extend_seq_lens_cpu=[2],
        req_pool_indices=torch.tensor([0]),
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


def test_gdn_extend_preserves_cached_prefix_boundary_in_request_slot():
    conv = torch.arange(12, dtype=torch.float32).view(4, 3)
    temporal = torch.arange(20, dtype=torch.float32).view(4, 5)
    state_cache = SimpleNamespace(conv=(conv,), temporal=temporal)
    batch = SimpleNamespace(
        extend_prefix_lens=torch.tensor([64, 64]),
        seq_lens=torch.tensor([70, 128]),
        mamba_track_indices=torch.tensor([2, 3]),
        mamba_track_mask=torch.tensor([False, True]),
    )
    adapter = DVRGDNStateAdapter(kernel_dispatcher=None)

    adapter.capture_extend_prefix_boundary(
        forward_batch=batch,
        state_cache=state_cache,
        cache_indices=torch.tensor([0, 1]),
    )

    assert torch.equal(conv[2], conv[0])
    assert torch.equal(temporal[2], temporal[0])
    # The normal prefill tracker owns requests whose extend reaches a new chunk.
    assert not torch.equal(conv[3], conv[1])
    assert not torch.equal(temporal[3], temporal[1])
