import torch

from sglang.srt.configs.mamba_utils import Mamba2StateShape
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.linear.dvr_gdn_state import (
    DVRGDNStateInputCache,
    DVRGDNStateInputs,
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
