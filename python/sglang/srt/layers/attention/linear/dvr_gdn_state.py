"""GDN-specific DVR state-input cache allocation and operators."""

from typing import Optional, Tuple, Union

import torch

from sglang.srt.configs.mamba_utils import Mamba2StateShape
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.linear.dvr_state import DVRStateInputCache

__all__ = ["create_gdn_state_input_cache"]


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
