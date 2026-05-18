"""Common DVR linear-state cache and operator interfaces.

This module intentionally knows only the DVR lifecycle:

- a rolling input window stores the unclosed chunk tail plus new draft tokens;
- an ops object provides chunkwise verify, recurrent rebuild, conv, and scatter;
- model-family details are supplied by small subclasses such as GDN.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE

__all__ = [
    "DVRRecurrentStateBackup",
    "DVRStateInputCache",
    "DVRStateInputWindow",
    "DVRStateOps",
]


@dataclass(frozen=True)
class DVRRecurrentStateBackup:
    conv: Tuple[torch.Tensor, ...]
    temporal: torch.Tensor
    indices: torch.Tensor


@dataclass(frozen=True)
class DVRStateInputWindow:
    """Base rolling-window view over DVR state inputs.

    Subclasses attach model-family names and writers. The base implementation
    keeps the common slot/length mechanics so future KDA/Mamba-style adapters
    do not need to reimplement shifting and tail-length bookkeeping.
    """

    inputs: Optional[Tuple[torch.Tensor, ...]]
    pos: Optional[torch.Tensor]

    @classmethod
    def from_cache(cls, state_cache) -> "DVRStateInputWindow":
        cache = getattr(state_cache, "dvr_state_inputs", None)
        if cache is None:
            return cls(inputs=None, pos=None)
        return cache.window()

    @property
    def enabled(self) -> bool:
        return self.inputs is not None

    def tensors(self) -> Tuple[torch.Tensor, ...]:
        assert self.inputs is not None
        return self.inputs

    @property
    def capacity(self) -> int:
        tensors = self.tensors()
        # Full-cache layout is [layers, slots, window, ...]. Layer-local cache
        # layout is [slots, window, ...].
        return tensors[0].shape[2] if tensors[0].dim() >= 5 else tensors[0].shape[1]

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

    def read_window(self, *, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(tensor[indices] for tensor in self.tensors())

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


@dataclass(frozen=True)
class DVRStateInputCache:
    """Base DVR rolling state-input cache.

    `memory_pool` owns allocation and reset. Adapters consume `window()` views
    and do not need to know whether a concrete cache stores q/k/v/g/beta or a
    different model-family input set.
    """

    tensors: Tuple[torch.Tensor, ...]
    tail_lens: torch.Tensor

    @classmethod
    def for_gdn(
        cls,
        *,
        num_layers: int,
        num_slots: int,
        num_draft_tokens: int,
        temporal_state_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: str,
    ) -> "DVRStateInputCache":
        from sglang.srt.layers.attention.linear.dvr_gdn_state import (
            DVRGDNStateInputCache,
        )

        return DVRGDNStateInputCache.create(
            num_layers=num_layers,
            num_slots=num_slots,
            num_draft_tokens=num_draft_tokens,
            temporal_state_shape=temporal_state_shape,
            dtype=dtype,
            device=device,
        )

    def __getitem__(self, layer: int) -> "DVRStateInputCache":
        return type(self)(
            tensors=tuple(tensor[layer] for tensor in self.tensors),
            tail_lens=self.tail_lens[layer],
        )

    def reset(self, indices: torch.Tensor):
        self.tail_lens[:, indices] = 0

    def mem_usage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors) + (
            self.tail_lens.numel() * self.tail_lens.element_size()
        )

    def window(self) -> DVRStateInputWindow:
        return DVRStateInputWindow(inputs=self.tensors, pos=self.tail_lens)


@dataclass
class DVRStateOps:
    """Base operator interface used by DVR state adapters.

    Concrete subclasses bind model-family kernels. The adapter depends on this
    interface rather than on FLA/GDN-specific function names.
    """

    chunk_size: int = FLA_CHUNK_SIZE

    @classmethod
    def for_gdn(cls, kernel_dispatcher) -> "DVRStateOps":
        from sglang.srt.layers.attention.linear.dvr_gdn_state import DVRGDNStateOps

        return DVRGDNStateOps.create(kernel_dispatcher)

    def scan_chunkwise(self, **kwargs) -> tuple:
        raise NotImplementedError

    def rebuild_recurrent_state(
        self,
        *args,
        initial_state: torch.Tensor,
        token_count: Optional[Union[int, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def run_verify_conv(self, *args, **kwargs):
        raise NotImplementedError

    def scatter_state(self, *args, **kwargs):
        raise NotImplementedError
