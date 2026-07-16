from types import SimpleNamespace

import pytest
import torch
from sglang.srt.configs.mamba_utils import Mamba2StateShape
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.linear.dvr_gdn import (
    DVRGDNStateAdapter,
)
from sglang.srt.layers.attention.linear.dvr_state import DVRRecurrentStateBackup


def create_gdn_state_input_cache(
    *, num_layers, num_slots, num_draft_tokens, state_shape, dtype, device
):
    state_cache = SimpleNamespace(
        temporal=torch.empty(num_layers, 1, dtype=torch.float32),
        intermediate_ssm=torch.empty(
            num_layers, num_slots, 1, dtype=torch.float32
        ),
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
            server_args=SimpleNamespace(
                speculative_num_draft_tokens=num_draft_tokens
            ),
            spec_algorithm=SimpleNamespace(is_dvr_self_draft=lambda: False),
            device=device,
        ),
    )
    return adapter.state_input_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_fla_boundary_state_preserves_initial_state_dtype(state_dtype):
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
    initial_state = torch.zeros(1, 1, 16, 16, dtype=state_dtype, device="cuda")
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
        chunk_size=FLA_CHUNK_SIZE,
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
        # DVR stores only the exported chunk-boundary recurrent state here;
        # the state-input window still needs all four draft positions.
        intermediate_ssm=torch.zeros(1, 3, 1, 16, 128, 128),
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

    returned_state_cache = adapter.get_state_cache(batch=batch)

    assert returned_state_cache is all_layers_state_cache
    assert adapter.state_input_cache is not None
    window = adapter.state_input_window(layer_idx=0)
    assert window.capacity == FLA_CHUNK_SIZE + 4
    assert window.tensors[0].shape == (3, FLA_CHUNK_SIZE + 4, 8, 128)

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


def test_gdn_verify_uses_request_pool_dummy_row_directly():
    adapter = DVRGDNStateAdapter(kernel_dispatcher=None)
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.long),
        req_pool_indices=torch.tensor([2, 0]),
        spec_info=SimpleNamespace(draft_token_num=2),
    )

    dvr_indices, state_input_indices, valid_mask = adapter.target_verify_indices(
        forward_batch=forward_batch,
        cache_indices=torch.tensor([7, 9]),
    )

    assert torch.equal(dvr_indices, torch.tensor([7, 0]))
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
    dispatcher = SimpleNamespace(
        extend=lambda **kwargs: (torch.empty(0), None, None)
    )
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
            dvr_indices=torch.tensor([0]),
            state_input_indices=torch.tensor([1]),
            valid_mask=torch.tensor([True]),
            intermediate_state_indices=torch.tensor([0]),
            layer_idx=0,
        )


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


def test_gdn_self_draft_restores_from_stable_boundary_without_snapshot():
    conv = torch.zeros(1, 3, 2)
    temporal = torch.zeros(1, 3, 2)
    temporal[:, 1] = torch.tensor([3.0, 4.0])
    state_cache = SimpleNamespace(conv=(conv,), temporal=temporal)
    live_backup = DVRRecurrentStateBackup(
        conv=(torch.tensor([[[5.0, 6.0]]]),), temporal=None
    )
    adapter = DVRGDNStateAdapter(
        kernel_dispatcher=None, draft_reuses_target_state=True
    )

    adapter.prepare_recurrent_state_for_verify(
        state_cache=state_cache,
        live_indices=torch.tensor([2]),
        boundary_indices=torch.tensor([1]),
        boundary_backup=None,
        live_backup=live_backup,
    )

    assert torch.equal(temporal[:, 2], temporal[:, 1])
    assert torch.equal(conv[:, [2]], live_backup.conv[0])


def test_gdn_backup_uses_request_pool_rows():
    conv = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    temporal = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
    state_cache = SimpleNamespace(conv=(conv,), temporal=temporal)
    adapter = DVRGDNStateAdapter(kernel_dispatcher=None)

    backup = adapter.backup_recurrent_state(
        state_cache=state_cache,
        indices=torch.tensor([0, 2]),
        backup_indices=torch.tensor([1, 3]),
        backup_size=4,
    )
    torch.testing.assert_close(backup.conv[0][:, 1], conv[:, 0])
    torch.testing.assert_close(backup.conv[0][:, 3], conv[:, 2])
    torch.testing.assert_close(backup.temporal[:, 1], temporal[:, 0])
    torch.testing.assert_close(backup.temporal[:, 3], temporal[:, 2])

    conv[:, 1].fill_(99)
    updated = adapter.backup_recurrent_state(
        state_cache=state_cache,
        indices=torch.tensor([1]),
        backup_indices=torch.tensor([2]),
        backup_size=4,
        out=backup,
    )
    assert updated is backup
    torch.testing.assert_close(backup.conv[0][:, 2], conv[:, 1])
