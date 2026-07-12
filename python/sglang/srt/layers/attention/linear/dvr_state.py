"""Model-independent rolling state-input storage for DVR linear attention."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE


@dataclass(frozen=True)
class DVRRecurrentStateBackup:
    conv: Tuple[torch.Tensor, ...]
    temporal: torch.Tensor


@dataclass(frozen=True)
class DVRStateInputCache:
    """Rolling input cache shared by DVR linear-state adapters."""

    tensors: Tuple[torch.Tensor, ...]
    tail_lens: torch.Tensor

    def __getitem__(self, layer: int) -> "DVRStateInputCache":
        return type(self)(
            tensors=tuple(tensor[layer] for tensor in self.tensors),
            tail_lens=self.tail_lens[layer],
        )

    @property
    def capacity(self) -> int:
        if self.tail_lens.dim() == 2:
            return self.tensors[0].shape[2]
        return self.tensors[0].shape[1]

    def get_tail_lens(self, *, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(device=self.tail_lens.device, dtype=torch.long)
        if self.tail_lens.dim() == 2:
            return self.tail_lens[0, indices]
        return self.tail_lens[indices]

    def set_tail_lens(
        self, *, indices: torch.Tensor, value: Union[int, torch.Tensor]
    ) -> None:
        indices = indices.to(device=self.tail_lens.device, dtype=torch.long)
        value = torch.as_tensor(
            value, device=self.tail_lens.device, dtype=self.tail_lens.dtype
        )
        if self.tail_lens.dim() == 2:
            self.tail_lens[:, indices] = value
        else:
            self.tail_lens[indices] = value

    def read(self, *, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(tensor[indices] for tensor in self.tensors)

    def write_rows(
        self,
        *,
        indices: torch.Tensor,
        cols: torch.Tensor,
        values: Tuple[torch.Tensor, ...],
    ) -> None:
        for cache, value in zip(self.tensors, values, strict=True):
            cache[indices, cols] = value

    def write_extend_tail(
        self,
        *,
        values: Tuple[torch.Tensor, ...],
        indices: torch.Tensor,
        extend_prefix_lens_cpu,
        extend_seq_lens_cpu,
        chunk_size: int = FLA_CHUNK_SIZE,
    ) -> None:
        src_base = 0
        for req_i, (prefix_len, extend_len) in enumerate(
            zip(extend_prefix_lens_cpu, extend_seq_lens_cpu, strict=True)
        ):
            seq_len = prefix_len + extend_len
            boundary = (seq_len // chunk_size) * chunk_size
            tail_len = seq_len - boundary
            dst = indices[req_i].to(torch.long)
            self.set_tail_lens(indices=dst, value=tail_len)
            write_start = max(prefix_len, boundary)
            if write_start < seq_len:
                src_start = src_base + write_start - prefix_len
                src_end = src_start + seq_len - write_start
                self.write_rows(
                    indices=dst,
                    cols=torch.arange(
                        write_start - boundary,
                        seq_len - boundary,
                        dtype=torch.long,
                        device=indices.device,
                    ),
                    values=tuple(value[src_start:src_end] for value in values),
                )
            src_base += extend_len

    def zero_after_lens(
        self, *, indices: torch.Tensor, keep_lens: torch.Tensor
    ) -> None:
        indices = indices.to(device=self.tensors[0].device, dtype=torch.long)
        keep_lens = keep_lens.to(device=indices.device, dtype=torch.long)
        cols = torch.arange(self.capacity, dtype=torch.long, device=indices.device)
        stale = cols.unsqueeze(0) >= keep_lens.unsqueeze(1)

        for cache in self.tensors:
            if self.tail_lens.dim() == 2:
                rows = cache[:, indices]
                mask_shape = (1,) + stale.shape + (1,) * (rows.dim() - 3)
                cache[:, indices] = torch.where(
                    stale.view(mask_shape), torch.zeros_like(rows), rows
                )
            else:
                rows = cache[indices]
                mask_shape = stale.shape + (1,) * (rows.dim() - 2)
                cache[indices] = torch.where(
                    stale.view(mask_shape), torch.zeros_like(rows), rows
                )

    def shift_after_boundary(
        self,
        *,
        indices: torch.Tensor,
        crosses_chunk_boundary: torch.Tensor,
        chunk_size: int = FLA_CHUNK_SIZE,
    ) -> None:
        tail_capacity = self.capacity - chunk_size
        if tail_capacity <= 0:
            return

        mask = crosses_chunk_boundary.to(torch.bool)
        for cache in self.tensors:
            if self.tail_lens.dim() == 2:
                dst = cache[:, indices, :tail_capacity]
                src = cache[:, indices, chunk_size : chunk_size + tail_capacity]
                mask_shape = (1, -1) + (1,) * (dst.dim() - 2)
                cache[:, indices, :tail_capacity] = torch.where(
                    mask.view(mask_shape), src, dst
                )
            else:
                dst = cache[indices, :tail_capacity]
                src = cache[indices, chunk_size : chunk_size + tail_capacity]
                mask_shape = (-1,) + (1,) * (dst.dim() - 1)
                cache[indices, :tail_capacity] = torch.where(
                    mask.view(mask_shape), src, dst
                )


def run_dvr_chunkwise_verify(
    *,
    state_ops: Any,
    state_input_cache: DVRStateInputCache,
    draft_state_inputs: Tuple[torch.Tensor, ...],
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    state_input_indices: torch.Tensor,
    intermediate_state_cache: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    chunk_size: int = FLA_CHUNK_SIZE,
    tail_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    indices = cache_indices[:batch_size].to(torch.long)
    input_indices = state_input_indices[:batch_size].to(torch.long)
    if tail_lens is None:
        tail_lens = state_input_cache.get_tail_lens(indices=input_indices).to(torch.long)
    else:
        tail_lens = tail_lens[:batch_size].to(device=indices.device, dtype=torch.long)
    tail_lens = tail_lens.clamp(min=0, max=chunk_size)
    state_input_cache.write_rows(
        indices=input_indices.unsqueeze(1).expand(-1, draft_token_num),
        cols=(
            torch.arange(
                draft_token_num, dtype=torch.long, device=input_indices.device
            ).unsqueeze(0)
            + tail_lens.unsqueeze(1)
        ),
        values=draft_state_inputs,
    )
    state_input_cache.zero_after_lens(
        indices=input_indices, keep_lens=tail_lens + draft_token_num
    )
    verify_window_size = chunk_size + draft_token_num
    window_inputs = state_input_cache.read(indices=input_indices)
    core_attn_out, _, h = state_ops.scan_chunkwise(
        state_inputs=window_inputs,
        ssm_states=ssm_states,
        cache_indices=indices,
        query_start_loc=None,
    )
    if h is not None and h.shape[1] > 1:
        intermediate_state_cache[
            intermediate_state_indices[:batch_size].to(torch.long), 0
        ] = h[:batch_size, 1].to(intermediate_state_cache.dtype)

    value_shape = core_attn_out.shape[-2:]
    core_attn_out = core_attn_out.view(batch_size, verify_window_size, *value_shape)
    rows = torch.arange(
        batch_size, dtype=torch.long, device=core_attn_out.device
    ).unsqueeze(1)
    cols = torch.arange(
        draft_token_num, dtype=torch.long, device=core_attn_out.device
    ).unsqueeze(0) + tail_lens.unsqueeze(1)
    return core_attn_out[
        rows.expand(-1, draft_token_num), cols
    ].reshape(1, batch_size * draft_token_num, *value_shape).contiguous()


def rebuild_dvr_live_state_grouped(
    *,
    state_input_cache: DVRStateInputCache,
    temporal_state: torch.Tensor,
    state_input_indices: torch.Tensor,
    live_indices: torch.Tensor,
    boundary_indices: torch.Tensor,
    req_indices: torch.Tensor,
    token_count: torch.Tensor,
    rebuild_fn,
) -> None:
    if req_indices.numel() == 0:
        return

    selected_input_indices = state_input_indices[req_indices]
    state_live_indices = live_indices[req_indices]
    state_boundary_indices = boundary_indices[req_indices]
    num_layers = temporal_state.shape[0]
    num_reqs = state_live_indices.numel()
    token_count = token_count.to(device=temporal_state.device, dtype=torch.long)
    initial_state = temporal_state[:, state_boundary_indices]

    flat_dim = num_layers * num_reqs
    window_inputs = tuple(
        tensor[:, selected_input_indices].reshape(flat_dim, *tensor.shape[2:])
        for tensor in state_input_cache.tensors
    )
    initial_state = initial_state.reshape(flat_dim, *initial_state.shape[2:])
    flat_token_count = (
        token_count.unsqueeze(0).expand(num_layers, -1).reshape(-1).contiguous()
    )
    rebuilt_state = rebuild_fn(
        window_inputs,
        initial_state=initial_state,
        token_count=flat_token_count,
    )
    temporal_state[:, state_live_indices] = rebuilt_state.reshape(
        num_layers, num_reqs, *rebuilt_state.shape[1:]
    )
