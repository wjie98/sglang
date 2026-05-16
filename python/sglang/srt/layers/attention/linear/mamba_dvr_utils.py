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


def write_dvr_conv_windows(
    *,
    intermediate_conv_window_cache: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    initial_conv_windows: torch.Tensor,
    mixed_qkv_reshaped: torch.Tensor,
    verify_window_size: int,
):
    conv_windows = build_dvr_conv_windows(
        initial_conv_windows=initial_conv_windows,
        mixed_qkv_reshaped=mixed_qkv_reshaped,
        verify_window_size=verify_window_size,
    )
    rows = (
        intermediate_state_indices[: initial_conv_windows.shape[0]]
        .to(torch.long)
        .unsqueeze(1)
        .expand(-1, verify_window_size)
    )
    cols = torch.arange(
        verify_window_size,
        dtype=torch.long,
        device=intermediate_state_indices.device,
    ).unsqueeze(0)
    intermediate_conv_window_cache[rows, cols] = conv_windows


def select_dvr_draft_suffix(
    core_attn_out: torch.Tensor,
    *,
    tail_lens: torch.Tensor,
    batch_size: int,
    verify_window_size: int,
    draft_token_num: int,
    num_v_heads: int,
    head_v_dim: int,
) -> torch.Tensor:
    core_attn_out = core_attn_out.view(
        batch_size, verify_window_size, num_v_heads, head_v_dim
    )
    rows = (
        torch.arange(batch_size, dtype=torch.long, device=core_attn_out.device)
        .unsqueeze(1)
        .expand(-1, draft_token_num)
    )
    cols = (
        torch.arange(draft_token_num, dtype=torch.long, device=core_attn_out.device)
        .unsqueeze(0)
        .add(tail_lens.unsqueeze(1))
    )
    return core_attn_out[rows, cols].reshape(
        1, batch_size * draft_token_num, num_v_heads, head_v_dim
    ).contiguous()


def rebuild_mamba_dvr_live_state_grouped(
    *,
    fla_ops: MambaDVRFlaOps,
    qkvg_beta_cache: MambaDVRQKVGBetaCache,
    temporal_state: torch.Tensor,
    live_indices: torch.Tensor,
    boundary_indices: torch.Tensor,
    req_indices: torch.Tensor,
    token_count: torch.Tensor,
):
    """Rebuild DVR live recurrent state from chunk-boundary checkpoints.

    The live state is consumed by the next self-draft decode, so it should use
    the recurrent semantics of normal decode. Grouping flattens
    layers x requests and calls the recurrent FLA kernel once per distinct
    accepted length instead of once per layer/request.
    """

    if req_indices.numel() == 0:
        return

    state_live_indices = live_indices[req_indices]
    state_boundary_indices = boundary_indices[req_indices]
    q_cache, k_cache, v_cache, g_cache, beta_cache = qkvg_beta_cache.tensors()
    num_layers = temporal_state.shape[0]
    num_reqs = state_live_indices.numel()
    token_count = token_count.to(device=temporal_state.device, dtype=torch.long)

    q = q_cache[:, state_live_indices]
    k = k_cache[:, state_live_indices]
    v = v_cache[:, state_live_indices]
    g = g_cache[:, state_live_indices]
    beta = beta_cache[:, state_live_indices]
    initial_state = temporal_state[:, state_boundary_indices]

    flat_shape = (num_layers * num_reqs,)
    q = q.reshape(*flat_shape, *q.shape[2:])
    k = k.reshape(*flat_shape, *k.shape[2:])
    v = v.reshape(*flat_shape, *v.shape[2:])
    g = g.reshape(*flat_shape, *g.shape[2:])
    beta = beta.reshape(*flat_shape, *beta.shape[2:])
    initial_state = initial_state.reshape(*flat_shape, *initial_state.shape[2:])
    flat_token_count = (
        token_count.unsqueeze(0).expand(num_layers, -1).reshape(-1).contiguous()
    )

    rebuilt_state = initial_state.clone()
    for count in torch.unique(flat_token_count).tolist():
        count = int(count)
        if count == 0:
            continue
        group = (flat_token_count == count).nonzero(as_tuple=True)[0]
        rebuilt_state[group] = fla_ops.rebuild_recurrent_state(
            q[group, :count],
            k[group, :count],
            v[group, :count],
            g[group, :count],
            beta[group, :count],
            initial_state=initial_state[group],
            token_count=count,
        )

    temporal_state[:, state_live_indices] = rebuilt_state.reshape(
        num_layers, num_reqs, *rebuilt_state.shape[1:]
    )
