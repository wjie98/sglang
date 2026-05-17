from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.fla.op import exp
from sglang.srt.utils import is_cpu

if not is_cpu():
    import triton
    import triton.language as tl

    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
        fused_recurrent_gated_delta_rule_update,
    )


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


def rebuild_gdn_state_from_qkvg_beta_triton(
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
    if triton.cdiv(K, BK) != 1:
        raise ValueError(f"DVR GDN recurrent rebuild only supports NK=1, K={K}.")

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


def rebuild_gdn_state_from_qkvg_beta(
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
        _, final_state = fused_recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        return final_state.to(initial_state.dtype, copy=False)

    if isinstance(token_count, int):
        token_count = torch.full(
            (q.shape[0],), token_count, dtype=torch.long, device=q.device
        )
    token_count = token_count.to(device=q.device, dtype=torch.long)
    max_count = q.shape[1]
    row_mask = (
        torch.arange(max_count, device=q.device, dtype=torch.long).unsqueeze(0)
        < token_count.unsqueeze(1)
    )
    row_idx, col_idx = row_mask.nonzero(as_tuple=True)
    q_packed = q[row_idx, col_idx].unsqueeze(0).contiguous()
    k_packed = k[row_idx, col_idx].unsqueeze(0).contiguous()
    v_packed = v[row_idx, col_idx].unsqueeze(0).contiguous()
    g_packed = g[row_idx, col_idx].unsqueeze(0).contiguous()
    beta_packed = beta[row_idx, col_idx].unsqueeze(0).contiguous()
    cu_seqlens = torch.empty(
        token_count.shape[0] + 1, dtype=torch.int32, device=q.device
    )
    cu_seqlens[0] = 0
    cu_seqlens[1:] = torch.cumsum(token_count.to(torch.int32), dim=0)
    state_indices = torch.arange(
        token_count.shape[0], dtype=torch.int32, device=q.device
    )
    final_state = initial_state.clone()
    fused_recurrent_gated_delta_rule_update(
        q=q_packed,
        k=k_packed,
        v=v_packed,
        g=g_packed,
        beta=beta_packed,
        initial_state_source=final_state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=False,
        disable_output_calculation=True,
        intermediate_state_indices=state_indices,
    )
    return final_state.to(initial_state.dtype, copy=False)


@dataclass
class DVRStateOps:
    """Concrete operators used by DVR for a known linear-state layer family."""

    chunk_scan: Optional[Callable] = None
    recurrent_state: Optional[Callable] = None
    verify_conv: Optional[Callable] = None
    state_scatter: Optional[Callable] = None
    chunk_size: int = FLA_CHUNK_SIZE

    @classmethod
    def for_gdn(
        cls,
        kernel_dispatcher,
    ) -> "DVRStateOps":
        from sglang.srt.layers.attention.mamba.causal_conv1d import (
            causal_conv1d_fn,
        )
        from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
            fused_mamba_state_scatter_with_mask,
        )

        return cls(
            chunk_scan=kernel_dispatcher.extend,
            recurrent_state=rebuild_gdn_state_from_qkvg_beta_triton,
            verify_conv=causal_conv1d_fn,
            state_scatter=fused_mamba_state_scatter_with_mask,
            chunk_size=FLA_CHUNK_SIZE,
        )

    def scan_chunkwise(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple:
        assert self.chunk_scan is not None
        return self.chunk_scan(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def rebuild_recurrent_state(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor,
        token_count: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        assert self.recurrent_state is not None
        return self.recurrent_state(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            token_count=token_count,
        )

    def run_verify_conv(self, *args, **kwargs):
        assert self.verify_conv is not None
        return self.verify_conv(*args, **kwargs)

    def scatter_state(self, *args, **kwargs):
        assert self.state_scatter is not None
        return self.state_scatter(*args, **kwargs)
