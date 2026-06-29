from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata
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
    mamba_cow_src_indices: Optional[torch.Tensor] = None
    mamba_cow_dst_indices: Optional[torch.Tensor] = None
    mamba_clear_indices: Optional[torch.Tensor] = None
    multimodal_inputs: Optional[list[Any]] = None


@dataclass
class DVRTargetReplayContext:
    saved_fields: dict[str, Any]


@dataclass
class DVRLinearStateReplayContext:
    live_backup: Optional[Any]
    saved_tail_lens: Optional[torch.Tensor]
    saved_state_input_window: Any


@dataclass
class DVRSuffixDraftReplayPlan:
    """Plan a target EXTEND oracle over unclosed prefix tail plus draft rows."""

    base_seq_lens_cpu: list[int]
    boundary_lens: list[int]
    tail_lens_cpu: list[int]
    extend_lens_cpu: list[int]
    final_seq_lens_cpu: list[int]
    input_ids: list[int]
    out_cache_locs: list[torch.Tensor]
    draft_cache_locs: torch.Tensor
    draft_rows: torch.Tensor
    draft_offsets: torch.Tensor
    hidden_gather_indices: torch.Tensor


@contextmanager
def linear_state_replay_context(
    linear_state_ctx,
    *,
    clear_state_input_window: bool = False,
    restore_live_state: bool = True,
):
    """Save/restore linear-state side effects around a replay oracle.

    Suffix/full-prefix replay temporarily reuses the live request slots.  The
    oracle may clear state-input windows or mutate recurrent state through the
    normal EXTEND path; callers opt out of live-state restore only when the
    replay itself is intended to refresh committed live state.
    """

    state_adapter = linear_state_ctx.state_adapter
    state_cache = linear_state_ctx.state_cache
    saved_tail_lens = state_adapter.state_input_tail_lens(
        state_cache=state_cache,
        state_input_indices=linear_state_ctx.state_input_indices,
    )
    if saved_tail_lens is not None:
        saved_tail_lens = saved_tail_lens.clone()
    saved_state_input_window = state_adapter.backup_state_input_window(
        state_cache=state_cache,
        state_input_indices=linear_state_ctx.state_input_indices,
    )
    if clear_state_input_window and saved_tail_lens is not None:
        zero_tail_lens = torch.zeros_like(saved_tail_lens)
        state_adapter.set_state_input_tail_lens(
            state_cache=state_cache,
            state_input_indices=linear_state_ctx.state_input_indices,
            tail_lens=zero_tail_lens,
        )
        state_adapter.zero_state_input_after_lens(
            state_cache=state_cache,
            state_input_indices=linear_state_ctx.state_input_indices,
            keep_lens=zero_tail_lens,
        )

    live_backup = None
    if restore_live_state:
        live_backup = state_adapter.backup_recurrent_state(
            state_cache=state_cache,
            indices=linear_state_ctx.live_indices,
        )

    try:
        yield DVRLinearStateReplayContext(
            live_backup=live_backup,
            saved_tail_lens=saved_tail_lens,
            saved_state_input_window=saved_state_input_window,
        )
    finally:
        if live_backup is not None:
            state_adapter.restore_recurrent_state(
                state_cache=state_cache,
                backup=live_backup,
                indices=linear_state_ctx.live_indices,
            )
        if saved_tail_lens is not None:
            state_adapter.set_state_input_tail_lens(
                state_cache=state_cache,
                state_input_indices=linear_state_ctx.state_input_indices,
                tail_lens=saved_tail_lens,
            )
        state_adapter.restore_state_input_window(
            state_cache=state_cache,
            state_input_indices=linear_state_ctx.state_input_indices,
            backup=saved_state_input_window,
        )


def build_suffix_draft_replay_plan(
    *,
    batch,
    base_seq_lens_cpu: list[int],
    boundary_lens: list[int],
    draft_tokens: torch.Tensor,
    draft_cache_locs: torch.Tensor,
    request_token_ids_for_replay,
) -> Optional[DVRSuffixDraftReplayPlan]:
    """Build the common tail+draft replay shape for self-DVR and DVR-EAGLE."""

    bs = len(batch.seq_lens)
    draft_token_num = draft_tokens.numel() // max(bs, 1)
    draft_tokens = draft_tokens.reshape(bs, draft_token_num)
    draft_cache_locs = draft_cache_locs.reshape(bs, draft_token_num)
    draft_tokens_cpu = draft_tokens.detach().cpu().tolist()
    tail_lens_cpu = [
        int(seq_len) - int(boundary)
        for seq_len, boundary in zip(base_seq_lens_cpu, boundary_lens, strict=True)
    ]
    extend_lens_cpu = [tail + draft_token_num for tail in tail_lens_cpu]
    final_seq_lens_cpu = [
        int(boundary) + int(extend_len)
        for boundary, extend_len in zip(boundary_lens, extend_lens_cpu, strict=True)
    ]

    input_ids = []
    out_cache_locs = []
    for req_i, (req, seq_len, boundary, tail_len) in enumerate(
        zip(
            batch.reqs,
            base_seq_lens_cpu,
            boundary_lens,
            tail_lens_cpu,
            strict=True,
        )
    ):
        token_ids = request_token_ids_for_replay(req, int(seq_len))
        input_ids.extend(token_ids[int(boundary) : int(seq_len)])
        input_ids.extend(draft_tokens_cpu[req_i])
        if int(tail_len) > 0:
            out_cache_locs.append(
                batch.req_to_token_pool.req_to_token[
                    req.req_pool_idx, int(boundary) : int(seq_len)
                ].to(torch.long)
            )
        out_cache_locs.append(draft_cache_locs[req_i].to(torch.long))

    if not input_ids:
        return None

    draft_offsets = batch.seq_lens.to(torch.long).unsqueeze(1) + torch.arange(
        draft_token_num, dtype=torch.long, device=batch.seq_lens.device
    ).unsqueeze(0)
    draft_rows = batch.req_pool_indices.to(torch.long).unsqueeze(1).expand_as(
        draft_offsets
    )

    gather_indices = []
    offset = 0
    for tail_len, extend_len in zip(tail_lens_cpu, extend_lens_cpu, strict=True):
        gather_indices.extend(
            range(offset + int(tail_len), offset + int(tail_len) + draft_token_num)
        )
        offset += int(extend_len)
    hidden_gather_indices = torch.tensor(
        gather_indices, dtype=torch.long, device=batch.seq_lens.device
    )

    return DVRSuffixDraftReplayPlan(
        base_seq_lens_cpu=base_seq_lens_cpu,
        boundary_lens=boundary_lens,
        tail_lens_cpu=tail_lens_cpu,
        extend_lens_cpu=extend_lens_cpu,
        final_seq_lens_cpu=final_seq_lens_cpu,
        input_ids=input_ids,
        out_cache_locs=out_cache_locs,
        draft_cache_locs=draft_cache_locs,
        draft_rows=draft_rows,
        draft_offsets=draft_offsets,
        hidden_gather_indices=hidden_gather_indices,
    )


def draft_row_logits_from_replay_hidden_states(
    *,
    target_worker,
    forward_batch,
    hidden_states: torch.Tensor,
    hidden_gather_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits/hidden states for replayed draft-token rows only."""

    if hidden_states is None:
        raise RuntimeError("DVR target replay did not return hidden states.")
    gather_indices = hidden_gather_indices.to(
        device=hidden_states.device,
        dtype=torch.long,
    )
    draft_hidden_states = hidden_states[gather_indices]
    logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)
    logits_metadata.next_token_logits_buffer = None
    draft_logits = target_worker.model_runner.model.logits_processor._get_logits(
        draft_hidden_states,
        target_worker.model_runner.model.lm_head,
        logits_metadata,
    )
    return draft_logits, draft_hidden_states


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
        batch.mamba_cow_src_indices = spec.mamba_cow_src_indices
        batch.mamba_cow_dst_indices = spec.mamba_cow_dst_indices
        batch.mamba_clear_indices = spec.mamba_clear_indices
        batch.multimodal_inputs = (
            saved_fields["multimodal_inputs"]
            if spec.multimodal_inputs is None
            else spec.multimodal_inputs
        )

        yield DVRTargetReplayContext(saved_fields=saved_fields)
    finally:
        for name, value in saved_fields.items():
            setattr(batch, name, value)
