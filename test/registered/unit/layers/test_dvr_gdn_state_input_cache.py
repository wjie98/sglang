from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import Mamba2StateShape
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.linear.dvr_gdn_state import (
    DVRGDNStateInputCache,
    DVRGDNStateInputs,
)
from sglang.srt.layers.attention.linear.dvr_state import DVRStateInputWindow
from sglang.srt.layers.attention.linear.dvr_state_adapter import (
    DVRGatedStateAdapter,
)


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

    cache = DVRGDNStateInputCache.create(
        num_layers=1,
        num_slots=3,
        num_draft_tokens=16,
        state_shape=state_shape,
        dtype=torch.float32,
        device="cpu",
    )

    cache_inputs = cache.state_inputs()
    assert cache_inputs.q.shape == (1, 3, FLA_CHUNK_SIZE + 16, 8, 128)
    assert cache_inputs.k.shape == (1, 3, FLA_CHUNK_SIZE + 16, 8, 128)
    assert cache_inputs.v.shape == (1, 3, FLA_CHUNK_SIZE + 16, 16, 128)
    assert cache_inputs.g.shape == (1, 3, FLA_CHUNK_SIZE + 16, 16)
    assert cache_inputs.beta.shape == (1, 3, FLA_CHUNK_SIZE + 16, 16)

    q = torch.randn(2, 8, 128)
    k = torch.randn(2, 8, 128)
    v = torch.randn(2, 16, 128)
    g = torch.randn(2, 16)
    beta = torch.randn(2, 16)
    state_inputs = DVRGDNStateInputs.from_tensors((q, k, v, g, beta))

    layer_cache = cache[0]
    state_inputs.write_extend_tail(
        layer_cache.window(),
        indices=torch.tensor([1]),
        extend_prefix_lens_cpu=[FLA_CHUNK_SIZE],
        extend_seq_lens_cpu=[2],
    )

    assert torch.equal(layer_cache.tail_lens[1], torch.tensor(2, dtype=torch.int32))
    layer_inputs = layer_cache.state_inputs()
    assert torch.equal(layer_inputs.q[1, :2], q)
    assert torch.equal(layer_inputs.k[1, :2], k)
    assert torch.equal(layer_inputs.v[1, :2], v)
    assert torch.equal(layer_inputs.g[1, :2], g)
    assert torch.equal(layer_inputs.beta[1, :2], beta)


def test_gdn_state_input_cache_uses_generic_linear_state_field():
    state_shape = Mamba2StateShape.create(
        tp_world_size=2,
        intermediate_size=32 * 128,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
    )
    cache = DVRGDNStateInputCache.create(
        num_layers=1,
        num_slots=3,
        num_draft_tokens=16,
        state_shape=state_shape,
        dtype=torch.float32,
        device="cpu",
    )[0]

    state_cache = SimpleNamespace(linear_state_input_cache=cache)
    window = DVRStateInputWindow.from_cache(state_cache)

    assert window.enabled
    assert window.capacity == FLA_CHUNK_SIZE + 16


def test_dvr_gdn_adapter_lazily_adds_state_input_cache_view():
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
    adapter = DVRGatedStateAdapter(
        ops=None,
        state_shape=state_shape,
        conv_dtype=torch.float32,
        device="cpu",
    )

    layer_state_cache = SimpleNamespace(
        intermediate_ssm=all_layers_state_cache.intermediate_ssm[0],
    )
    wrapped = adapter.get_layer_state_cache(
        req_to_token_pool=batch.req_to_token_pool,
        state_cache=layer_state_cache,
        layer_id=7,
    )

    assert wrapped is layer_state_cache
    assert wrapped.linear_state_input_cache is not None
    assert req_to_token_pool._dvr_linear_state_input_cache is not None
    window = DVRStateInputWindow.from_cache(wrapped)
    assert window.enabled
    assert window.capacity == FLA_CHUNK_SIZE + 4
    assert window.tensors()[0].shape == (4, FLA_CHUNK_SIZE + 4, 8, 128)


def test_gdn_extend_tail_cache_skips_draft_workers():
    state_shape = Mamba2StateShape.create(
        tp_world_size=2,
        intermediate_size=32 * 128,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
    )
    cache = DVRGDNStateInputCache.create(
        num_layers=1,
        num_slots=3,
        num_draft_tokens=16,
        state_shape=state_shape,
        dtype=torch.float32,
        device="cpu",
    )[0]
    state_cache = SimpleNamespace(linear_state_input_cache=cache)
    draft_adapter = DVRGatedStateAdapter(ops=None, is_draft_worker=True)
    target_adapter = DVRGatedStateAdapter(ops=None, is_draft_worker=False)

    q = torch.randn(1, 2, 8, 128)
    k = torch.randn(1, 2, 8, 128)
    v = torch.randn(1, 2, 16, 128)
    g = torch.randn(2, 16)
    beta = torch.randn(2, 16)

    draft_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[FLA_CHUNK_SIZE],
        extend_seq_lens_cpu=[2],
        req_pool_indices=torch.tensor([0]),
    )
    state_inputs = DVRGDNStateInputs.from_extend_forward(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
    )

    draft_adapter.cache_extend_tail(
        forward_batch=draft_batch,
        state_cache=state_cache,
        state_inputs=state_inputs,
    )
    assert torch.equal(cache.tail_lens[1], torch.tensor(0, dtype=torch.int32))

    target_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[FLA_CHUNK_SIZE],
        extend_seq_lens_cpu=[2],
        req_pool_indices=torch.tensor([0]),
    )
    target_adapter.cache_extend_tail(
        forward_batch=target_batch,
        state_cache=state_cache,
        state_inputs=state_inputs,
    )
    assert torch.equal(cache.tail_lens[1], torch.tensor(2, dtype=torch.int32))
    layer_inputs = cache.state_inputs()
    assert torch.equal(layer_inputs.q[1, :2], q.reshape(2, 8, 128))
    assert torch.equal(layer_inputs.k[1, :2], k.reshape(2, 8, 128))
    assert torch.equal(layer_inputs.v[1, :2], v.reshape(2, 16, 128))
    assert torch.equal(layer_inputs.g[1, :2], g)
    assert torch.equal(layer_inputs.beta[1, :2], beta)
