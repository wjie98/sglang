"""Model-independent rolling state-input storage for DVR linear attention."""

from dataclasses import dataclass
from typing import Tuple, Union

import torch


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
    ) -> None:
        tail_capacity = self.capacity - chunk_size
        if tail_capacity <= 0:
            return

        mask = crosses_chunk_boundary.to(torch.bool)
        for cache in self.tensors:
            if self.has_layer_dim:
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
