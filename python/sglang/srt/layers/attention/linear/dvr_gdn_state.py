"""GDN-specific DVR state inputs and operators."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.fla.op import exp
from sglang.srt.layers.attention.linear.dvr_state import (
    DVRStateInputCache,
    DVRStateInputWindow,
    DVRStateOps,
)
from sglang.srt.utils import is_cpu

__all__ = [
    "DVRGDNStateInputCache",
    "DVRGDNStateInputWindow",
    "DVRGDNStateOps",
]

if not is_cpu():
    import triton
    import triton.language as tl


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


@dataclass(frozen=True)
class DVRGDNStateInputWindow(DVRStateInputWindow):
    """GDN rolling input window storing q/k/v/g/beta rows."""

    @property
    def q(self) -> torch.Tensor:
        return self.tensors()[0]

    @property
    def k(self) -> torch.Tensor:
        return self.tensors()[1]

    @property
    def v(self) -> torch.Tensor:
        return self.tensors()[2]

    @property
    def g(self) -> torch.Tensor:
        return self.tensors()[3]

    @property
    def beta(self) -> torch.Tensor:
        return self.tensors()[4]

    def write_tail(
        self,
        *,
        dst: torch.Tensor,
        cols: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ):
        self.q[dst, cols] = q
        self.k[dst, cols] = k
        self.v[dst, cols] = v
        self.g[dst, cols] = g
        self.beta[dst, cols] = beta

    def write_draft_rows(
        self,
        *,
        indices: torch.Tensor,
        col_start: torch.Tensor,
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
    ):
        cols = (
            torch.arange(draft_token_num, dtype=torch.long, device=indices.device)
            .unsqueeze(0)
            .add(col_start.to(torch.long).unsqueeze(1))
        )
        rows = indices.unsqueeze(1).expand(-1, draft_token_num)
        self.q[rows, cols] = q.reshape(
            batch_size, draft_token_num, num_q_heads, head_q_dim
        )
        self.k[rows, cols] = k.reshape(
            batch_size, draft_token_num, num_k_heads, head_k_dim
        )
        self.v[rows, cols] = v.reshape(
            batch_size, draft_token_num, num_v_heads, head_v_dim
        )
        self.g[rows, cols] = g.reshape(batch_size, draft_token_num, num_v_heads)
        self.beta[rows, cols] = beta.reshape(batch_size, draft_token_num, num_v_heads)

    def write_extend_tail(
        self,
        *,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        extend_prefix_lens_cpu,
        extend_seq_lens_cpu,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = FLA_CHUNK_SIZE,
    ):
        src_base = 0
        for req_i, (prefix_len, extend_len) in enumerate(
            zip(extend_prefix_lens_cpu, extend_seq_lens_cpu, strict=True)
        ):
            seq_len = prefix_len + extend_len
            boundary = (seq_len // chunk_size) * chunk_size
            num_verified_tokens = seq_len - boundary
            dst = cache_indices[req_i].to(torch.long)
            self.set_tail_lens(indices=dst, value=num_verified_tokens)
            if num_verified_tokens == 0:
                src_base += extend_len
                continue

            write_start = max(prefix_len, boundary)
            write_end = seq_len
            if write_start >= write_end:
                src_base += extend_len
                continue

            src_start = src_base + (write_start - prefix_len)
            src_end = src_start + (write_end - write_start)
            cols = torch.arange(
                write_start - boundary,
                write_end - boundary,
                dtype=torch.long,
                device=cache_indices.device,
            )
            self.write_tail(
                dst=dst,
                cols=cols,
                q=q[src_start:src_end],
                k=k[src_start:src_end],
                v=v[src_start:src_end],
                g=g[src_start:src_end],
                beta=beta[src_start:src_end],
            )
            src_base += extend_len


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
        temporal_state_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: str,
    ) -> "DVRGDNStateInputCache":
        window_len = FLA_CHUNK_SIZE + num_draft_tokens
        q_input = torch.zeros(
            size=(
                num_layers,
                num_slots,
                window_len,
                temporal_state_shape[0],
                temporal_state_shape[1],
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
                    temporal_state_shape[0],
                    temporal_state_shape[2],
                ),
                dtype=dtype,
                device=device,
            ),
            torch.zeros(
                size=(num_layers, num_slots, window_len, temporal_state_shape[0]),
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

    def window(self) -> DVRGDNStateInputWindow:
        return DVRGDNStateInputWindow(inputs=self.tensors, pos=self.tail_lens)


@dataclass
class DVRGDNStateOps(DVRStateOps):
    """GDN concrete operator bundle for DVR state replay."""

    chunk_scan: Optional[Callable] = None
    recurrent_state: Optional[Callable] = None
    verify_conv: Optional[Callable] = None
    state_scatter: Optional[Callable] = None

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
            chunk_scan=kernel_dispatcher.extend,
            recurrent_state=_rebuild_gdn_state_from_qkvg_beta_triton,
            verify_conv=causal_conv1d_fn,
            state_scatter=fused_mamba_state_scatter_with_mask,
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
