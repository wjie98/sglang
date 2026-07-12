"""GDN-specific DVR state-input cache allocation and operators."""

from typing import Optional, Tuple, Union

import torch

from sglang.srt.configs.mamba_utils import Mamba2StateShape
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.fla.op import exp
from sglang.srt.layers.attention.linear.dvr_state import DVRStateInputCache
from sglang.srt.utils import is_cpu

__all__ = ["create_gdn_state_input_cache"]

if not is_cpu():
    import triton
    import triton.language as tl


def _infer_gdn_state_input_shapes(
    state_shape: Mamba2StateShape,
) -> Tuple[int, int, int, int]:
    """Infer local q/k and v cache dimensions from existing Mamba2 metadata.

    GDN's recurrent state is shaped by value heads, but the state-input cache
    stores q/k using key groups. Mamba2StateShape does not store n_groups
    directly, so keep this DVR-local by inverting the conv_dim formula used by
    Mamba2StateShape.create():
      conv_dim = intermediate_size + 2 * padded_n_groups * state_size.
    """
    local_value_heads, value_head_dim, key_head_dim = state_shape.temporal
    key_group_width = 2 * state_shape.state_size
    assert key_group_width > 0
    assert state_shape.conv_dim >= state_shape.intermediate_size
    assert (
        state_shape.conv_dim - state_shape.intermediate_size
    ) % key_group_width == 0

    padded_num_key_groups = (
        state_shape.conv_dim - state_shape.intermediate_size
    ) // key_group_width
    assert local_value_heads > 0
    assert state_shape.num_heads % local_value_heads == 0
    tp_world_size = state_shape.num_heads // local_value_heads
    assert padded_num_key_groups % tp_world_size == 0
    local_key_heads = padded_num_key_groups // tp_world_size
    return local_key_heads, key_head_dim, local_value_heads, value_head_dim


if not is_cpu():

    @triton.jit
    def _dvr_gdn_final_state_kernel(
        k,
        v,
        g,
        beta,
        initial_state,
        final_state,
        token_count,
        T: tl.constexpr,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        i_v, i_nh = tl.program_id(0), tl.program_id(1)
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        state_offset = i_n * HV * V * K + i_hv * V * K
        p_h0 = initial_state + state_offset + o_v[:, None] * K + o_k[None, :]
        state = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        steps = tl.load(token_count + i_n).to(tl.int64)
        p_k = k + ((i_n * T * H + i_h) * K + o_k)
        p_v = v + ((i_n * T * HV + i_hv) * V + o_v)
        p_g = g + (i_n * T * HV + i_hv)
        p_beta = beta + (i_n * T * HV + i_hv)

        for step in range(0, T):
            if step < steps:
                key = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
                value = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
                key /= tl.sqrt(tl.sum(key * key) + 1e-6)
                state *= exp(tl.load(p_g).to(tl.float32))
                value -= tl.sum(state * key[None, :], 1)
                value *= tl.load(p_beta).to(tl.float32)
                state += value[:, None] * key[None, :]

            p_k += H * K
            p_v += HV * V
            p_g += HV
            p_beta += HV

        p_ht = final_state + state_offset + o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, state.to(p_ht.dtype.element_ty), mask=mask_h)


def _zero_gdn_rows_after_token_count(
    tensor: torch.Tensor, token_count: torch.Tensor
) -> torch.Tensor:
    token_count = token_count.to(device=tensor.device, dtype=torch.long).clamp(
        min=0, max=tensor.shape[1]
    )
    rows = torch.arange(tensor.shape[1], device=tensor.device).unsqueeze(0)
    keep = rows < token_count.unsqueeze(1)
    view_shape = keep.shape + (1,) * (tensor.dim() - 2)
    return torch.where(keep.view(view_shape), tensor, torch.zeros_like(tensor))


def _rebuild_gdn_state_from_qkvg_beta_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    token_count: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor:
    """Rebuild a GDN state with the same chunkwise kernel used by prefill.

    The fast DVR recurrent kernel is sufficient for the all-accepted hot path,
    but partial acceptance is a correctness boundary: the next draft starts from
    a shortened suffix.  Reusing FLA's chunkwise path there keeps the rebuilt
    live state bit-aligned with deterministic prefill instead of maintaining a
    second copy of GDN's chunk-local gate math.
    """

    if token_count is None:
        token_count = q.shape[1]
    if isinstance(token_count, int):
        token_count = torch.full(
            (q.shape[0],), token_count, dtype=torch.long, device=q.device
        )

    q = _zero_gdn_rows_after_token_count(q, token_count)
    k = _zero_gdn_rows_after_token_count(k, token_count)
    v = _zero_gdn_rows_after_token_count(v, token_count)
    g = _zero_gdn_rows_after_token_count(g, token_count)
    beta = _zero_gdn_rows_after_token_count(beta, token_count)

    # FLA returns boundary states in `h`; it does not update `initial_state`
    # in-place.  Use the final exported state so chunkwise rebuild actually
    # advances DVR's live recurrent slot after partial/cross-boundary commits.
    initial_state = initial_state.clone()
    cache_indices = torch.arange(
        q.shape[0], dtype=torch.int32, device=q.device
    )
    _, _, h = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=cache_indices,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    return h[:, -1]


def _rebuild_gdn_state_for_self_draft(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    token_count: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor:
    """Rebuild the performance-only recurrent state consumed by self draft.

    Target verify restores its deterministic boundary checkpoint separately.
    Self draft needs only the accepted suffix's final state, so avoid FLA's
    general recurrent path, which also materializes every token output.
    """

    if is_cpu():
        raise NotImplementedError("DVR GDN self-draft state rebuild is GPU-only.")
    if token_count is None:
        token_count = q.shape[1]
    if isinstance(token_count, int):
        token_count = torch.full(
            (q.shape[0],), token_count, dtype=torch.long, device=q.device
        )

    _, T, H, K = q.shape
    batch_size, _, num_value_heads, value_dim = v.shape
    token_count = token_count.to(device=k.device, dtype=torch.long).clamp(
        min=0, max=T
    )
    final_state = torch.empty_like(initial_state)
    block_k = triton.next_power_of_2(K)
    block_v = min(triton.next_power_of_2(value_dim), 8)
    _dvr_gdn_final_state_kernel[
        (triton.cdiv(value_dim, block_v), batch_size * num_value_heads)
    ](
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        final_state=final_state,
        token_count=token_count,
        T=T,
        H=H,
        HV=num_value_heads,
        K=K,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        num_warps=1,
        num_stages=3,
    )
    return final_state


def create_gdn_state_input_cache(
    *,
    num_layers: int,
    num_slots: int,
    num_draft_tokens: int,
    state_shape: Mamba2StateShape,
    dtype: torch.dtype,
    device: str,
) -> DVRStateInputCache:
    """Allocate the q/k/v/g/beta rolling window used by GDN DVR."""

    local_key_heads, key_dim, local_value_heads, value_dim = (
        _infer_gdn_state_input_shapes(state_shape)
    )
    window_len = FLA_CHUNK_SIZE + num_draft_tokens
    q = torch.zeros(
        num_layers,
        num_slots,
        window_len,
        local_key_heads,
        key_dim,
        dtype=dtype,
        device=device,
    )
    v = torch.zeros(
        num_layers,
        num_slots,
        window_len,
        local_value_heads,
        value_dim,
        dtype=dtype,
        device=device,
    )
    gate = torch.zeros(
        num_layers,
        num_slots,
        window_len,
        local_value_heads,
        dtype=torch.float32,
        device=device,
    )
    return DVRStateInputCache(
        tensors=(q, torch.zeros_like(q), v, gate, torch.zeros_like(gate)),
        tail_lens=torch.zeros(
            num_layers,
            num_slots,
            dtype=torch.int32,
            device=device,
        ),
    )
