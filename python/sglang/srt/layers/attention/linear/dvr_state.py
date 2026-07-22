"""Model-independent rolling state-input storage for DVR linear attention."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from sglang.srt.utils import is_cuda

if is_cuda():
    import triton
    import triton.language as tl


if is_cuda():

    @triton.jit
    def _dvr_compact_state_window_kernel(
        cache,
        indices,
        crosses_boundary,
        tail_lens,
        layer_stride,
        slot_stride,
        token_stride,
        S: tl.constexpr,
        T: tl.constexpr,
        E: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        i_req = tl.program_id(0)
        i_layer = tl.program_id(1).to(tl.int64)
        i_block = tl.program_id(2).to(tl.int64)
        if not tl.load(crosses_boundary + i_req):
            return

        slot = tl.load(indices + i_req).to(tl.int64)
        count = tl.load(tail_lens + i_req).to(tl.int64) * E
        offsets = i_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < count
        token = offsets // E
        element = offsets % E
        base = i_layer * layer_stride + slot * slot_stride
        values = tl.load(
            cache + base + (CHUNK_SIZE + token) * token_stride + element,
            mask=mask,
        )
        tl.store(cache + base + token * token_stride + element, values, mask=mask)


@dataclass(frozen=True)
class DVRStateInputCache:
    """Rolling input cache shared by DVR linear-state adapters."""

    tensors: Tuple[torch.Tensor, ...]
    tail_lens: torch.Tensor
    has_layer_dim: bool = True

    def __getitem__(self, layer: int) -> "DVRStateInputCache":
        return type(self)(
            tensors=tuple(tensor[layer] for tensor in self.tensors),
            tail_lens=self.tail_lens,
            has_layer_dim=False,
        )

    @property
    def capacity(self) -> int:
        return self.tensors[0].shape[2 if self.has_layer_dim else 1]

    def get_tail_lens(self, *, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(device=self.tail_lens.device, dtype=torch.long)
        return self.tail_lens[indices]

    def set_tail_lens(
        self, *, indices: torch.Tensor, value: Union[int, torch.Tensor]
    ) -> None:
        indices = indices.to(device=self.tail_lens.device, dtype=torch.long)
        value = torch.as_tensor(
            value, device=self.tail_lens.device, dtype=self.tail_lens.dtype
        )
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
        chunk_size: int,
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

    def shift_after_boundary(
        self,
        *,
        indices: torch.Tensor,
        crosses_chunk_boundary: torch.Tensor,
        chunk_size: int,
        tail_lens: Optional[torch.Tensor] = None,
    ) -> None:
        tail_capacity = self.capacity - chunk_size
        if tail_capacity <= 0:
            return

        mask = crosses_chunk_boundary.to(torch.bool)
        if tail_lens is None:
            tail_lens = torch.full_like(mask, tail_capacity, dtype=torch.long)
        else:
            tail_lens = tail_lens.to(device=mask.device, dtype=torch.long).clamp(
                min=0, max=tail_capacity
            )

        if self.tensors[0].is_cuda:
            indices = indices.to(device=mask.device, dtype=torch.long).contiguous()
            mask = mask.contiguous()
            tail_lens = tail_lens.contiguous()
            for cache in self.tensors:
                layered = cache if self.has_layer_dim else cache.unsqueeze(0)
                if not layered.is_contiguous():
                    raise RuntimeError("DVR state-input windows must be contiguous.")
                elements_per_token = layered[0, 0, 0].numel()
                grid = (
                    indices.numel(),
                    layered.shape[0],
                    triton.cdiv(tail_capacity * elements_per_token, 256),
                )
                _dvr_compact_state_window_kernel[grid](
                    layered,
                    indices,
                    mask,
                    tail_lens,
                    layered.stride(0),
                    layered.stride(1),
                    layered.stride(2),
                    S=layered.shape[1],
                    T=layered.shape[2],
                    E=elements_per_token,
                    CHUNK_SIZE=chunk_size,
                    BLOCK_SIZE=256,
                )
            return

        for cache in self.tensors:
            if self.has_layer_dim:
                dst = cache[:, indices, :tail_capacity]
                src = cache[:, indices, chunk_size : chunk_size + tail_capacity]
                mask_shape = (1, -1) + (1,) * (dst.dim() - 2)
                len_shape = (1, -1, 1) + (1,) * (dst.dim() - 3)
                cols = torch.arange(tail_capacity, device=cache.device).view(
                    (1, 1, -1) + (1,) * (dst.dim() - 3)
                )
                copy_mask = mask.view(mask_shape) & (cols < tail_lens.view(len_shape))
                cache[:, indices, :tail_capacity] = torch.where(copy_mask, src, dst)
            else:
                dst = cache[indices, :tail_capacity]
                src = cache[indices, chunk_size : chunk_size + tail_capacity]
                mask_shape = (-1,) + (1,) * (dst.dim() - 1)
                len_shape = (-1, 1) + (1,) * (dst.dim() - 2)
                cols = torch.arange(tail_capacity, device=cache.device).view(
                    (1, -1) + (1,) * (dst.dim() - 2)
                )
                copy_mask = mask.view(mask_shape) & (cols < tail_lens.view(len_shape))
                cache[indices, :tail_capacity] = torch.where(copy_mask, src, dst)
