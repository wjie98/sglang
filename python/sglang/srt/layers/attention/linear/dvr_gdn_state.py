"""GDN-specific DVR state-input cache allocation and operators."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch

from sglang.srt.configs.mamba_utils import Mamba2StateShape
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.fla.op import exp
from sglang.srt.layers.attention.linear.dvr_state import (
    DVRStateInputCache,
    DVRStateInputs,
    DVRStateOps,
)
from sglang.srt.mem_cache.linear_state_extension import (
    LinearSpeculativeStateExtension,
    LinearSpeculativeStateExtensionConfig,
)
from sglang.srt.utils import is_cpu

__all__ = [
    "DVRGDNStateInputs",
    "DVRGDNStateInputCache",
    "DVRGDNStateOps",
    "create_dvr_gdn_speculative_state_extension",
]

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
    def _dvr_gdn_recurrent_state_kernel(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        final_state,
        token_count,
        scale,
        T: tl.constexpr,
        B: tl.constexpr,
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

        p_h0 = (
            initial_state
            + i_n * HV * V * K
            + i_hv * V * K
            + o_v[:, None] * K
            + o_k[None, :]
        )
        b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        steps = tl.load(token_count + i_n).to(tl.int64)
        p_q = q + ((i_n * T * H + i_h) * K + o_k)
        p_k = k + ((i_n * T * H + i_h) * K + o_k)
        p_v = v + ((i_n * T * HV + i_hv) * V + o_v)
        p_g = g + (i_n * T * HV + i_hv)
        p_beta = beta + (i_n * T * HV + i_hv)

        for step in range(0, T):
            if step < steps:
                b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
                b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
                b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
                b_q = b_q * scale
                b_g = tl.load(p_g).to(tl.float32)
                b_h *= exp(b_g)
                b_v -= tl.sum(b_h * b_k[None, :], 1)
                b_beta = tl.load(p_beta).to(tl.float32)
                b_v *= b_beta
                b_h += b_v[:, None] * b_k[None, :]

            p_q += H * K
            p_k += H * K
            p_v += HV * V
            p_g += HV
            p_beta += HV

        p_ht = (
            final_state
            + i_n * HV * V * K
            + i_hv * V * K
            + o_v[:, None] * K
            + o_k[None, :]
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def _rebuild_gdn_state_from_qkvg_beta_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    token_count: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor:
    if is_cpu():
        raise NotImplementedError("DVR GDN recurrent post-verify is GPU-only.")
    if token_count is None:
        token_count = q.shape[1]
    if isinstance(token_count, int):
        token_count = torch.full(
            (q.shape[0],), token_count, dtype=torch.long, device=q.device
        )
    token_count = token_count.to(device=q.device, dtype=torch.long)

    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 8)

    final_state = torch.empty_like(initial_state)
    grid = (triton.cdiv(V, BV), B * HV)
    _dvr_gdn_recurrent_state_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        final_state=final_state,
        token_count=token_count,
        scale=K**-0.5,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        num_warps=1,
        num_stages=3,
    )
    return final_state


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


@dataclass(frozen=True)
class DVRGDNStateInputs(DVRStateInputs):
    """Interpret a generic DVR state-input tuple as GDN q/k/v/g/beta tensors."""

    @classmethod
    def from_tensors(cls, tensors: Tuple[torch.Tensor, ...]) -> "DVRGDNStateInputs":
        tensors = tuple(tensors)
        assert len(tensors) == 5
        return cls(values=tensors)

    @property
    def q(self) -> torch.Tensor:
        return self.values[0]

    @property
    def k(self) -> torch.Tensor:
        return self.values[1]

    @property
    def v(self) -> torch.Tensor:
        return self.values[2]

    @property
    def g(self) -> torch.Tensor:
        return self.values[3]

    @property
    def beta(self) -> torch.Tensor:
        return self.values[4]

    @classmethod
    def from_draft_rows(
        cls,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        batch_size: int,
        draft_token_num: int,
        num_q_heads: int,
        head_q_dim: int,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
    ) -> "DVRGDNStateInputs":
        return cls.from_tensors(
            (
                q.reshape(batch_size, draft_token_num, num_q_heads, head_q_dim),
                k.reshape(batch_size, draft_token_num, num_k_heads, head_k_dim),
                v.reshape(batch_size, draft_token_num, num_v_heads, head_v_dim),
                g.reshape(batch_size, draft_token_num, num_v_heads),
                beta.reshape(batch_size, draft_token_num, num_v_heads),
            )
        )

    @classmethod
    def from_extend_forward(
        cls,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> "DVRGDNStateInputs":
        return cls.from_tensors(
            (
                q.reshape(q.shape[1], q.shape[2], q.shape[3]),
                k.reshape(k.shape[1], k.shape[2], k.shape[3]),
                v.reshape(v.shape[1], v.shape[2], v.shape[3]),
                g.reshape(-1, g.shape[-1]),
                beta.reshape(-1, beta.shape[-1]),
            )
        )


@dataclass(frozen=True)
class DVRGDNStateInputCache(DVRStateInputCache):
    """q/k/v/g/beta scratch cache plus rolling-window lengths for GDN DVR."""

    @classmethod
    def create(
        cls,
        *,
        num_layers: int,
        num_slots: int,
        num_draft_tokens: int,
        state_shape: Mamba2StateShape,
        dtype: torch.dtype,
        device: str,
    ) -> "DVRGDNStateInputCache":
        (
            local_key_heads,
            key_head_dim,
            local_value_heads,
            value_head_dim,
        ) = _infer_gdn_state_input_shapes(state_shape)
        window_len = FLA_CHUNK_SIZE + num_draft_tokens
        q_input = torch.zeros(
            size=(
                num_layers,
                num_slots,
                window_len,
                local_key_heads,
                key_head_dim,
            ),
            dtype=dtype,
            device=device,
        )
        state_inputs = [
            q_input,
            torch.zeros_like(q_input),
            torch.zeros(
                size=(
                    num_layers,
                    num_slots,
                    window_len,
                    local_value_heads,
                    value_head_dim,
                ),
                dtype=dtype,
                device=device,
            ),
            torch.zeros(
                size=(num_layers, num_slots, window_len, local_value_heads),
                dtype=torch.float32,
                device=device,
            ),
        ]
        state_inputs.append(torch.zeros_like(state_inputs[-1]))
        tail_lens = torch.zeros(
            size=(num_layers, num_slots),
            dtype=torch.int32,
            device=device,
        )
        return cls(tensors=tuple(state_inputs), tail_lens=tail_lens)

    def state_inputs(self) -> DVRGDNStateInputs:
        return DVRGDNStateInputs.from_tensors(self.tensors)


def create_dvr_gdn_speculative_state_extension(
    config: LinearSpeculativeStateExtensionConfig,
) -> LinearSpeculativeStateExtension:
    """Create DVR-GDN's extra state-input cache for target-verify replay."""

    return LinearSpeculativeStateExtension(
        # DVR verify commits only the first chunk-boundary recurrent state.
        # Draft-token q/k/v/g/beta rows are kept in the side cache below so the
        # target replay can rebuild prefill-equivalent recurrent states.
        intermediate_ssm_tokens=1,
        intermediate_conv_tokens=config.num_draft_tokens,
        state_input_cache=DVRGDNStateInputCache.create(
            num_layers=config.num_layers,
            # Slot 0 is the padded-row dummy; real DVR rows use req_pool_idx + 1
            # to preserve the v5 hot-path indexing convention.
            num_slots=config.spec_state_size + 2,
            num_draft_tokens=config.num_draft_tokens,
            state_shape=config.state_shape,
            dtype=config.conv_dtype,
            device=config.device,
        ),
        log_label="linear_state_input_cache",
    )


@dataclass
class DVRGDNStateOps(DVRStateOps):
    """GDN concrete operator bundle for DVR state replay."""

    chunkwise_scan_fn: Optional[Callable] = None
    rebuild_recurrent_state_fn: Optional[Callable] = None
    verify_conv_fn: Optional[Callable] = None
    state_scatter_fn: Optional[Callable] = None

    @classmethod
    def create(cls, kernel_dispatcher) -> "DVRGDNStateOps":
        from sglang.srt.layers.attention.mamba.causal_conv1d import (
            causal_conv1d_fn,
        )
        from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
            fused_mamba_state_scatter_with_mask,
        )

        return cls(
            chunk_size=FLA_CHUNK_SIZE,
            chunkwise_scan_fn=kernel_dispatcher.extend,
            rebuild_recurrent_state_fn=_rebuild_gdn_state_from_qkvg_beta_triton,
            verify_conv_fn=causal_conv1d_fn,
            state_scatter_fn=fused_mamba_state_scatter_with_mask,
        )

    def scan_chunkwise(
        self,
        *,
        state_inputs: DVRStateInputs,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple:
        assert self.chunkwise_scan_fn is not None
        state_inputs = DVRGDNStateInputs.from_tensors(state_inputs.tensors())
        return self.chunkwise_scan_fn(
            q=state_inputs.q,
            k=state_inputs.k,
            v=state_inputs.v,
            g=state_inputs.g,
            beta=state_inputs.beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def rebuild_recurrent_state(
        self,
        state_inputs: DVRStateInputs,
        *,
        initial_state: torch.Tensor,
        token_count: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        assert self.rebuild_recurrent_state_fn is not None
        state_inputs = DVRGDNStateInputs.from_tensors(state_inputs.tensors())
        return self.rebuild_recurrent_state_fn(
            state_inputs.q,
            state_inputs.k,
            state_inputs.v,
            state_inputs.g,
            state_inputs.beta,
            initial_state=initial_state,
            token_count=token_count,
        )

    def rebuild_recurrent_state_chunkwise(
        self,
        state_inputs: DVRStateInputs,
        *,
        initial_state: torch.Tensor,
        token_count: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        state_inputs = DVRGDNStateInputs.from_tensors(state_inputs.tensors())
        return _rebuild_gdn_state_from_qkvg_beta_chunkwise(
            state_inputs.q,
            state_inputs.k,
            state_inputs.v,
            state_inputs.g,
            state_inputs.beta,
            initial_state=initial_state,
            token_count=token_count,
        )

    def run_verify_conv(self, *args, **kwargs):
        assert self.verify_conv_fn is not None
        return self.verify_conv_fn(*args, **kwargs)

    def scatter_state(self, *args, **kwargs):
        assert self.state_scatter_fn is not None
        return self.state_scatter_fn(*args, **kwargs)
