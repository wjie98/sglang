from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE


@dataclass
class MambaDVRFlaOps:
    """FLA operators DVR needs from a Mamba-like linear attention backend."""

    chunk_scan: Optional[Callable] = None
    recurrent_state: Optional[Callable] = None
    chunk_size: int = FLA_CHUNK_SIZE

    def set_ops(self, *, chunk_scan: Callable, recurrent_state: Callable):
        self.chunk_scan = chunk_scan
        self.recurrent_state = recurrent_state

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


@dataclass(frozen=True)
class MambaDVRQKVGBetaCache:
    q: Optional[torch.Tensor]
    k: Optional[torch.Tensor]
    v: Optional[torch.Tensor]
    g: Optional[torch.Tensor]
    beta: Optional[torch.Tensor]
    pos: Optional[torch.Tensor]

    @classmethod
    def from_mamba_cache(cls, mamba_cache):
        return cls(
            q=getattr(mamba_cache, "dvr_q_state_cache", None),
            k=getattr(mamba_cache, "dvr_k_state_cache", None),
            v=getattr(mamba_cache, "dvr_v_state_cache", None),
            g=getattr(mamba_cache, "dvr_g_state_cache", None),
            beta=getattr(mamba_cache, "dvr_beta_state_cache", None),
            pos=getattr(mamba_cache, "dvr_qkvg_beta_pos", None),
        )

    @property
    def enabled(self) -> bool:
        return self.q is not None

    def tensors(self) -> Tuple[torch.Tensor, ...]:
        assert self.enabled
        assert self.q is not None
        assert self.k is not None
        assert self.v is not None
        assert self.g is not None
        assert self.beta is not None
        return self.q, self.k, self.v, self.g, self.beta

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
        q_cache, k_cache, v_cache, g_cache, beta_cache = self.tensors()
        q_cache[dst, cols] = q
        k_cache[dst, cols] = k
        v_cache[dst, cols] = v
        g_cache[dst, cols] = g
        beta_cache[dst, cols] = beta

    def write_verify_window(
        self,
        *,
        indices: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        batch_size: int,
        verify_window_size: int,
        num_q_heads: int,
        head_q_dim: int,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
    ):
        cols = torch.arange(
            verify_window_size, dtype=torch.long, device=indices.device
        ).unsqueeze(0)
        rows = indices.unsqueeze(1).expand(-1, verify_window_size)
        q_cache, k_cache, v_cache, g_cache, beta_cache = self.tensors()
        q_cache[rows, cols] = q.reshape(
            batch_size, verify_window_size, num_q_heads, head_q_dim
        )
        k_cache[rows, cols] = k.reshape(
            batch_size, verify_window_size, num_k_heads, head_k_dim
        )
        v_cache[rows, cols] = v.reshape(
            batch_size, verify_window_size, num_v_heads, head_v_dim
        )
        g_cache[rows, cols] = g.reshape(batch_size, verify_window_size, num_v_heads)
        beta_cache[rows, cols] = beta.reshape(
            batch_size, verify_window_size, num_v_heads
        )

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
        q_cache, k_cache, v_cache, g_cache, beta_cache = self.tensors()
        q_cache[rows, cols] = q.reshape(
            batch_size, draft_token_num, num_q_heads, head_q_dim
        )
        k_cache[rows, cols] = k.reshape(
            batch_size, draft_token_num, num_k_heads, head_k_dim
        )
        v_cache[rows, cols] = v.reshape(
            batch_size, draft_token_num, num_v_heads, head_v_dim
        )
        g_cache[rows, cols] = g.reshape(batch_size, draft_token_num, num_v_heads)
        beta_cache[rows, cols] = beta.reshape(
            batch_size, draft_token_num, num_v_heads
        )

    def read_window(self, *, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        q_cache, k_cache, v_cache, g_cache, beta_cache = self.tensors()
        return (
            q_cache[indices],
            k_cache[indices],
            v_cache[indices],
            g_cache[indices],
            beta_cache[indices],
        )

    def shift_suffix(self, *, slots: torch.Tensor, start: int, length: int):
        if length <= 0 or slots.numel() == 0:
            return
        has_layer_dim = self.pos is not None and self.pos.dim() == 2
        for cache in self.tensors():
            if has_layer_dim:
                cache[:, slots, :length] = cache[:, slots, start : start + length].clone()
            else:
                cache[slots, :length] = cache[slots, start : start + length].clone()


def write_dvr_chunk_boundary_state(
    *,
    h: Optional[torch.Tensor],
    intermediate_state_cache: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    batch_size: int,
    verify_window_size: int,
    chunk_size: int = FLA_CHUNK_SIZE,
):
    if h is None or h.shape[1] <= 1:
        return

    chunks_per_req = (verify_window_size - 1) // chunk_size + 1
    if h.shape[0] == batch_size:
        # Equal-length graph-friendly path: GDN DVR passes q/k/v/g/beta as
        # [batch, chunk+draft, ...], so the first chunk boundary is h[:, 1].
        boundary_state = h[:, 1]
    else:
        # Variable-length path packs all request chunks into h[0, total_chunks].
        # DVR's physical window is chunk + draft, so each request contributes
        # two chunks and the first boundary is at offsets 1, 3, 5, ...
        boundary_h_indices = (
            torch.arange(batch_size, dtype=torch.long, device=h.device)
            * chunks_per_req
            + 1
        )
        boundary_state = h.squeeze(0)[boundary_h_indices]
    # DVR only commits the first chunk-boundary state. Newer DVR cache layouts
    # store that state in a one-slot boundary buffer; the fallback keeps
    # compatibility with the older per-token speculative state cache.
    boundary_slot = 0 if intermediate_state_cache.shape[1] == 1 else chunk_size - 1
    intermediate_state_cache[
        intermediate_state_indices[:batch_size].to(torch.long),
        boundary_slot,
    ] = boundary_state.to(intermediate_state_cache.dtype)


def build_dvr_conv_windows(
    *,
    initial_conv_windows: torch.Tensor,
    mixed_qkv_reshaped: torch.Tensor,
    verify_window_size: int,
) -> torch.Tensor:
    conv_source = torch.cat([initial_conv_windows, mixed_qkv_reshaped], dim=2)
    state_len = initial_conv_windows.shape[-1]
    return conv_source.unfold(
        dimension=2, size=state_len, step=1
    )[:, :, 1 : verify_window_size + 1].transpose(1, 2)
