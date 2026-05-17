from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE


@dataclass(frozen=True)
class DVRRecurrentStateBackup:
    conv: Tuple[torch.Tensor, ...]
    temporal: torch.Tensor
    indices: torch.Tensor


@dataclass(frozen=True)
class DVRStateInputCache:
    """q/k/v/g/beta scratch cache plus rolling-window lengths for DVR verify."""

    tensors: Tuple[torch.Tensor, ...]
    tail_lens: torch.Tensor

    def __getitem__(self, layer: int) -> "DVRStateInputCache":
        return DVRStateInputCache(
            tensors=tuple(tensor[layer] for tensor in self.tensors),
            tail_lens=self.tail_lens[layer],
        )

    def reset(self, indices: torch.Tensor):
        self.tail_lens[:, indices] = 0

    def mem_usage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors) + (
            self.tail_lens.numel() * self.tail_lens.element_size()
        )


@dataclass(frozen=True)
class DVRStateInputWindow:
    q: Optional[torch.Tensor]
    k: Optional[torch.Tensor]
    v: Optional[torch.Tensor]
    g: Optional[torch.Tensor]
    beta: Optional[torch.Tensor]
    pos: Optional[torch.Tensor]

    @classmethod
    def from_cache(cls, state_cache):
        cache = getattr(state_cache, "dvr_state_inputs", None)
        if cache is None:
            return cls(q=None, k=None, v=None, g=None, beta=None, pos=None)
        q, k, v, g, beta = cache.tensors
        return cls(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            pos=cache.tail_lens,
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

    @property
    def capacity(self) -> int:
        assert self.enabled
        assert self.q is not None
        # Full-cache layout is [layers, slots, window, ...]. Layer-local cache
        # layout is [slots, window, ...].
        return self.q.shape[2] if self.q.dim() >= 5 else self.q.shape[1]

    def tail_lens(self, *, indices: torch.Tensor) -> torch.Tensor:
        assert self.pos is not None
        indices = indices.to(device=self.pos.device, dtype=torch.long)
        if self.pos.dim() == 2:
            return self.pos[0, indices]
        return self.pos[indices]

    def set_tail_lens(self, *, indices: torch.Tensor, value: Union[int, torch.Tensor]):
        assert self.pos is not None
        indices = indices.to(device=self.pos.device, dtype=torch.long)
        value = torch.as_tensor(value, device=self.pos.device, dtype=self.pos.dtype)
        if self.pos.dim() == 2:
            self.pos[:, indices] = value
        else:
            self.pos[indices] = value

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

    def shift_after_boundary(
        self,
        *,
        live_indices: torch.Tensor,
        crossing: torch.Tensor,
        chunk_size: int = FLA_CHUNK_SIZE,
    ):
        tail_capacity = self.capacity - chunk_size
        if tail_capacity <= 0:
            return

        has_layer_dim = self.pos is not None and self.pos.dim() == 2
        mask = crossing.to(torch.bool)
        for cache in self.tensors():
            if has_layer_dim:
                dst = cache[:, live_indices, :tail_capacity]
                src = cache[:, live_indices, chunk_size : chunk_size + tail_capacity]
                mask_shape = (1, -1) + (1,) * (dst.dim() - 2)
                cache[:, live_indices, :tail_capacity] = torch.where(
                    mask.view(mask_shape), src, dst
                )
            else:
                dst = cache[live_indices, :tail_capacity]
                src = cache[live_indices, chunk_size : chunk_size + tail_capacity]
                mask_shape = (-1,) + (1,) * (dst.dim() - 1)
                cache[live_indices, :tail_capacity] = torch.where(
                    mask.view(mask_shape), src, dst
                )

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


def allocate_dvr_state_input_cache(
    *,
    num_layers: int,
    num_slots: int,
    num_draft_tokens: int,
    temporal_state_shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: str,
) -> DVRStateInputCache:
    """Allocate q/k/v/g/beta scratch inputs used by DVR linear-state adapters."""

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
    return DVRStateInputCache(tensors=tuple(state_inputs), tail_lens=tail_lens)
