"""Common DVR linear-state cache and operator interfaces.

This module intentionally knows only the DVR lifecycle:

- a rolling input window stores the unclosed chunk tail plus new draft tokens;
- an ops object provides chunkwise verify, recurrent rebuild, conv, and scatter;
- model-family details are supplied by small concrete caches/ops such as GDN.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE

__all__ = [
    "DVRRecurrentStateBackup",
    "DVRStateInputs",
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
class DVRStateInputs:
    """Model-family state-input tensors stored inside a DVR rolling window."""

    values: Tuple[torch.Tensor, ...]

    @classmethod
    def from_tensors(cls, tensors: Tuple[torch.Tensor, ...]) -> "DVRStateInputs":
        return cls(values=tuple(tensors))

    def tensors(self) -> Tuple[torch.Tensor, ...]:
        return self.values

    def select(self, indices: torch.Tensor) -> "DVRStateInputs":
        return type(self).from_tensors(tuple(tensor[indices] for tensor in self.values))

    def select_all_layers(self, indices: torch.Tensor) -> "DVRStateInputs":
        return type(self).from_tensors(
            tuple(tensor[:, indices] for tensor in self.values)
        )

    def select_token_range(self, start: int, end: int) -> "DVRStateInputs":
        return type(self).from_tensors(
            tuple(tensor[start:end] for tensor in self.values)
        )

    def flatten_leading_dims(
        self, flat_dim: int, *, keep_from_dim: int
    ) -> "DVRStateInputs":
        return type(self).from_tensors(
            tuple(
                tensor.reshape(flat_dim, *tensor.shape[keep_from_dim:])
                for tensor in self.values
            )
        )

    def write_rows(
        self,
        state_window: "DVRStateInputWindow",
        *,
        indices: torch.Tensor,
        cols: torch.Tensor,
    ):
        state_window.write_rows(indices=indices, cols=cols, values=self)

    def write_draft_rows(
        self,
        state_window: "DVRStateInputWindow",
        *,
        indices: torch.Tensor,
        col_start: torch.Tensor,
        draft_token_num: int,
    ):
        cols = (
            torch.arange(draft_token_num, dtype=torch.long, device=indices.device)
            .unsqueeze(0)
            .add(col_start.to(torch.long).unsqueeze(1))
        )
        rows = indices.unsqueeze(1).expand(-1, draft_token_num)
        self.write_rows(state_window, indices=rows, cols=cols)

    def write_extend_tail(
        self,
        state_window: "DVRStateInputWindow",
        *,
        indices: torch.Tensor,
        extend_prefix_lens_cpu,
        extend_seq_lens_cpu,
        chunk_size: int = FLA_CHUNK_SIZE,
    ):
        src_base = 0
        for req_i, (prefix_len, extend_len) in enumerate(
            zip(extend_prefix_lens_cpu, extend_seq_lens_cpu, strict=True)
        ):
            seq_len = prefix_len + extend_len
            boundary = (seq_len // chunk_size) * chunk_size
            num_verified_tokens = seq_len - boundary
            dst = indices[req_i].to(torch.long)
            state_window.set_tail_lens(indices=dst, value=num_verified_tokens)
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
                device=indices.device,
            )
            self.select_token_range(src_start, src_end).write_rows(
                state_window, indices=dst, cols=cols
            )
            src_base += extend_len


@dataclass(frozen=True)
class DVRStateInputWindow:
    """Rolling-window view over DVR linear-state inputs.

    The window owns only slot, tail length, and shift mechanics. Model-family
    data layouts such as GDN q/k/v/g/beta live in small state-input wrappers.
    """

    inputs: Optional[DVRStateInputs]
    tail_lens_cache: Optional[torch.Tensor]

    @classmethod
    def from_cache(cls, state_cache) -> "DVRStateInputWindow":
        cache = getattr(state_cache, "linear_state_input_cache", None)
        if cache is None:
            return cls(inputs=None, tail_lens_cache=None)
        return cache.window()

    @property
    def enabled(self) -> bool:
        return self.inputs is not None

    def tensors(self) -> Tuple[torch.Tensor, ...]:
        assert self.inputs is not None
        return self.inputs.tensors()

    @property
    def capacity(self) -> int:
        tensors = self.tensors()
        # Full-cache layout is [layers, slots, window, ...]. Layer-local cache
        # layout is [slots, window, ...].
        return tensors[0].shape[2] if tensors[0].dim() >= 5 else tensors[0].shape[1]

    def tail_lens(self, *, indices: torch.Tensor) -> torch.Tensor:
        assert self.tail_lens_cache is not None
        indices = indices.to(device=self.tail_lens_cache.device, dtype=torch.long)
        if self.tail_lens_cache.dim() == 2:
            return self.tail_lens_cache[0, indices]
        return self.tail_lens_cache[indices]

    def set_tail_lens(self, *, indices: torch.Tensor, value: Union[int, torch.Tensor]):
        assert self.tail_lens_cache is not None
        indices = indices.to(device=self.tail_lens_cache.device, dtype=torch.long)
        value = torch.as_tensor(
            value,
            device=self.tail_lens_cache.device,
            dtype=self.tail_lens_cache.dtype,
        )
        if self.tail_lens_cache.dim() == 2:
            self.tail_lens_cache[:, indices] = value
        else:
            self.tail_lens_cache[indices] = value

    def read_window(self, *, indices: torch.Tensor) -> DVRStateInputs:
        assert self.inputs is not None
        return self.inputs.select(indices)

    def read_all_layers_window(self, *, indices: torch.Tensor) -> DVRStateInputs:
        assert self.inputs is not None
        return self.inputs.select_all_layers(indices)

    def write_rows(
        self,
        *,
        indices: torch.Tensor,
        cols: torch.Tensor,
        values: DVRStateInputs,
    ):
        for cache, value in zip(self.tensors(), values.tensors(), strict=True):
            cache[indices, cols] = value

    def backup_rows(self, *, indices: torch.Tensor) -> Optional[Tuple[torch.Tensor, ...]]:
        """Snapshot request rows that a diagnostic EXTEND replay may overwrite."""

        if self.inputs is None:
            return None
        indices = indices.to(device=self.tensors()[0].device, dtype=torch.long)
        has_layer_dim = (
            self.tail_lens_cache is not None and self.tail_lens_cache.dim() == 2
        )
        if has_layer_dim:
            return tuple(cache[:, indices].clone() for cache in self.tensors())
        return tuple(cache[indices].clone() for cache in self.tensors())

    def restore_rows(
        self, *, indices: torch.Tensor, backup: Optional[Tuple[torch.Tensor, ...]]
    ):
        if backup is None:
            return
        indices = indices.to(device=self.tensors()[0].device, dtype=torch.long)
        has_layer_dim = (
            self.tail_lens_cache is not None and self.tail_lens_cache.dim() == 2
        )
        for cache, saved in zip(self.tensors(), backup, strict=True):
            if has_layer_dim:
                cache[:, indices] = saved.to(cache.dtype, copy=False)
            else:
                cache[indices] = saved.to(cache.dtype, copy=False)

    def zero_after_lens(self, *, indices: torch.Tensor, keep_lens: torch.Tensor):
        """Clear stale rolling-window columns after each request's live suffix."""

        indices = indices.to(device=self.tensors()[0].device, dtype=torch.long)
        keep_lens = keep_lens.to(device=indices.device, dtype=torch.long)
        cols = torch.arange(self.capacity, dtype=torch.long, device=indices.device)
        stale = cols.unsqueeze(0) >= keep_lens.unsqueeze(1)

        has_layer_dim = (
            self.tail_lens_cache is not None and self.tail_lens_cache.dim() == 2
        )
        for cache in self.tensors():
            if has_layer_dim:
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
    ):
        tail_capacity = self.capacity - chunk_size
        if tail_capacity <= 0:
            return

        has_layer_dim = (
            self.tail_lens_cache is not None and self.tail_lens_cache.dim() == 2
        )
        mask = crosses_chunk_boundary.to(torch.bool)
        for cache in self.tensors():
            if has_layer_dim:
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


@dataclass(frozen=True)
class DVRStateInputCache:
    """Base DVR rolling state-input cache.

    `memory_pool` owns allocation. Adapters consume `window()` views and do not
    need to know whether a concrete cache stores q/k/v/g/beta or a different
    model-family input set.
    """

    tensors: Tuple[torch.Tensor, ...]
    tail_lens: torch.Tensor

    def __getitem__(self, layer: int) -> "DVRStateInputCache":
        return type(self)(
            tensors=tuple(tensor[layer] for tensor in self.tensors),
            tail_lens=self.tail_lens[layer],
        )

    def mem_usage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors) + (
            self.tail_lens.numel() * self.tail_lens.element_size()
        )

    def state_inputs(self) -> DVRStateInputs:
        return DVRStateInputs.from_tensors(self.tensors)

    def window(self) -> DVRStateInputWindow:
        return DVRStateInputWindow(
            inputs=self.state_inputs(), tail_lens_cache=self.tail_lens
        )


@dataclass
class DVRStateOps:
    """Base operator interface used by DVR state adapters.

    Concrete subclasses bind model-family kernels. The adapter depends on this
    interface rather than on FLA/GDN-specific function names.
    """

    chunk_size: int = FLA_CHUNK_SIZE

    def scan_chunkwise(self, *, state_inputs: DVRStateInputs, **kwargs) -> tuple:
        raise NotImplementedError

    def rebuild_recurrent_state(
        self,
        state_inputs: DVRStateInputs,
        *,
        initial_state: torch.Tensor,
        token_count: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def run_verify_conv(self, *args, **kwargs):
        raise NotImplementedError

    def scatter_state(self, *args, **kwargs):
        raise NotImplementedError
