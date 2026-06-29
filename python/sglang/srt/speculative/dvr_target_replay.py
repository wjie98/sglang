from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)


_TEMP_EXTEND_BATCH_FIELDS = (
    "forward_mode",
    "global_forward_mode",
    "input_ids",
    "input_embeds",
    "replace_embeds",
    "replace_positions",
    "out_cache_loc",
    "seq_lens",
    "seq_lens_cpu",
    "seq_lens_sum",
    "prefix_lens",
    "extend_lens",
    "extend_num_tokens",
    "extend_logprob_start_lens",
    "extend_input_logprob_token_ids",
    "global_num_tokens",
    "global_num_tokens_for_logprob",
    "is_extend_in_batch",
    "all_extend_in_batch",
    "spec_info",
    "capture_hidden_mode",
    "return_hidden_states",
    "return_hidden_states_before_norm",
    "return_logprob",
    "mamba_track_indices",
    "mamba_track_mask",
    "mamba_track_seqlens",
    "mamba_track_cache_seqlens",
    "mamba_cow_src_indices",
    "mamba_cow_dst_indices",
    "mamba_clear_indices",
    "multimodal_inputs",
)


@dataclass
class DVRTargetReplaySpec:
    input_ids: list[int]
    out_cache_locs: list[torch.Tensor]
    prefix_lens: list[int]
    extend_lens: list[int]
    final_seq_lens: list[int]
    extend_logprob_start_lens: list[int]
    extend_input_logprob_token_ids: Optional[list[int]] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    return_logprob: bool = False
    mamba_track_indices: Optional[torch.Tensor] = None
    mamba_track_mask: Optional[torch.Tensor] = None
    mamba_track_seqlens: Optional[torch.Tensor] = None
    mamba_clear_indices: Optional[torch.Tensor] = None
    multimodal_inputs: Optional[list[Any]] = None


@dataclass
class DVRTargetReplayContext:
    saved_fields: dict[str, Any]


@contextmanager
def target_extend_replay_batch(batch, spec: DVRTargetReplaySpec):
    """Run a target EXTEND replay without leaking mutations to the live batch."""

    saved_fields = {name: getattr(batch, name) for name in _TEMP_EXTEND_BATCH_FIELDS}
    device = batch.seq_lens.device
    try:
        batch.forward_mode = ForwardMode.EXTEND
        batch.global_forward_mode = None
        batch.input_ids = torch.tensor(spec.input_ids, dtype=torch.long, device=device)
        batch.input_embeds = None
        batch.replace_embeds = None
        batch.replace_positions = None
        batch.out_cache_loc = torch.cat(spec.out_cache_locs).to(device=device)
        batch.prefix_lens = [int(x) for x in spec.prefix_lens]
        batch.extend_lens = [int(x) for x in spec.extend_lens]
        batch.extend_num_tokens = len(spec.input_ids)
        batch.extend_logprob_start_lens = [
            int(x) for x in spec.extend_logprob_start_lens
        ]
        batch.extend_input_logprob_token_ids = (
            None
            if spec.extend_input_logprob_token_ids is None
            else torch.tensor(
                spec.extend_input_logprob_token_ids,
                dtype=torch.long,
                device=device,
            )
        )
        if saved_fields["global_num_tokens"] is not None:
            dp_world = len(saved_fields["global_num_tokens"])
            batch.global_num_tokens = [len(spec.input_ids)] * dp_world
            batch.global_num_tokens_for_logprob = [len(spec.input_ids)] * dp_world
        batch.seq_lens = torch.tensor(
            spec.final_seq_lens,
            dtype=torch.long,
            device=saved_fields["seq_lens"].device,
        )
        batch.seq_lens_cpu = torch.tensor(spec.final_seq_lens, dtype=torch.long)
        batch.seq_lens_sum = sum(spec.final_seq_lens)
        batch.is_extend_in_batch = True
        batch.all_extend_in_batch = True
        batch.spec_info = None
        batch.capture_hidden_mode = spec.capture_hidden_mode
        batch.return_hidden_states = False
        batch.return_hidden_states_before_norm = False
        batch.return_logprob = spec.return_logprob
        batch.mamba_track_indices = spec.mamba_track_indices
        batch.mamba_track_mask = spec.mamba_track_mask
        batch.mamba_track_seqlens = spec.mamba_track_seqlens
        batch.mamba_track_cache_seqlens = None
        batch.mamba_cow_src_indices = None
        batch.mamba_cow_dst_indices = None
        batch.mamba_clear_indices = spec.mamba_clear_indices
        batch.multimodal_inputs = spec.multimodal_inputs

        yield DVRTargetReplayContext(saved_fields=saved_fields)
    finally:
        for name, value in saved_fields.items():
            setattr(batch, name, value)
