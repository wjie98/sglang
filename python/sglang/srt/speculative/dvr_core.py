from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.speculative.dvr_info import (
    DVRRollbackActions,
    compact_dvr_accepted_input_tokens_and_cache_locs,
    compact_dvr_output_rows,
)


@dataclass
class DVRVerifyOutput:
    """Client-visible DVR tokens produced by one target verify step."""

    replay_prefix: DVRRollbackActions
    tokens: torch.Tensor
    tokens_per_req: int
    base_seq_lens_cpu: list[int]
    error_prefix: str


def score_dvr_verify_outputs(
    *,
    batch: Any,
    output: DVRVerifyOutput,
    accept_lens: torch.Tensor,
) -> list[int]:
    """Record accepted DVR output tokens in scheduler materialization order."""

    accept_lens_cpu, token_ids_per_req = compact_dvr_output_rows(
        output_tokens=output.tokens,
        accept_lens=accept_lens,
        tokens_per_req=output.tokens_per_req,
    )
    output.replay_prefix.append_batch_output_tokens(
        batch,
        token_ids_per_req,
        base_seq_lens_cpu=output.base_seq_lens_cpu,
        error_prefix=f"{output.error_prefix} output prefix",
    )
    return accept_lens_cpu


def rollback_dvr_verify(
    *,
    batch: Any,
    linear_state: Any,
    linear_state_ctx: Any,
    accept_lens: torch.Tensor,
    accept_lens_cpu: Optional[list[int]],
    num_draft_tokens: int,
    output: Optional[DVRVerifyOutput] = None,
    accepted_input_tokens: Optional[torch.Tensor] = None,
    rollback_replay_kwargs: Optional[dict[str, Any]] = None,
    use_fast_self_draft_commit: bool = False,
) -> DVRRollbackActions:
    """Rollback target state and output metadata after a DVR verify step.

    Draft adapters stop at sampling.  Everything after that is rollback:
    scheduler-visible output prefix, accepted suffix state repair, and delayed
    checkpoint publication for spec-v2.
    """

    base_seq_lens_cpu = output.base_seq_lens_cpu if output is not None else None
    if output is not None and not batch.forward_mode.is_idle():
        accept_lens_cpu = score_dvr_verify_outputs(
            batch=batch,
            output=output,
            accept_lens=accept_lens,
        )
    elif accept_lens_cpu is None:
        accept_lens_cpu = accept_lens.detach().cpu().tolist()

    pending_track_indices = None
    pending_track_seqlens = None
    if linear_state_ctx is not None:
        accepted_suffix_replay = None
        if rollback_replay_kwargs is not None and torch.any(
            accept_lens < num_draft_tokens
        ).item():
            if accepted_input_tokens is None:
                raise RuntimeError(
                    "DVR rollback replay requires accepted verify-input tokens."
                )
            accepted_ids, accepted_cache_locs = (
                compact_dvr_accepted_input_tokens_and_cache_locs(
                    batch=batch,
                    verify_input_tokens=accepted_input_tokens,
                    accept_lens=accept_lens,
                    num_draft_tokens=num_draft_tokens,
                )
            )
            if accepted_ids.numel() > 0:
                accepted_suffix_replay = linear_state.rollback_live_state_with_accepted_suffix(
                    batch=batch,
                    accepted_token_counts_cpu=accept_lens_cpu,
                    accepted_ids=accepted_ids,
                    accepted_cache_locs=accepted_cache_locs,
                    **rollback_replay_kwargs,
                )

        pending_track_indices, pending_track_seqlens = linear_state.commit_after_verify(
            batch=batch,
            accepted_token_counts=accept_lens.to(torch.long),
            accepted_steps=(accept_lens - 1).to(torch.long),
            accepted_token_counts_cpu=accept_lens_cpu,
            ctx=linear_state_ctx,
            seq_lens_cpu=base_seq_lens_cpu or linear_state.batch_seq_lens_cpu(batch),
            live_state_already_replayed=(
                None
                if accepted_suffix_replay is None
                else accepted_suffix_replay.live_state_mask
            ),
            accepted_suffix_replay=accepted_suffix_replay,
            use_fast_self_draft_commit=use_fast_self_draft_commit,
        )

    rollback_actions = DVRRollbackActions()
    if pending_track_indices is not None or pending_track_seqlens is not None:
        if pending_track_indices is None:
            pending_track_indices = [
                None
            ] * (len(pending_track_seqlens) if pending_track_seqlens is not None else 0)
        if pending_track_seqlens is None:
            pending_track_seqlens = [None] * len(pending_track_indices)
        checkpoints = [
            (
                (int(track_idx), int(seqlen))
                if track_idx is not None and seqlen is not None
                else None
            )
            for track_idx, seqlen in zip(
                pending_track_indices, pending_track_seqlens, strict=True
            )
        ]
        rollback_actions.pending_mamba_checkpoints = checkpoints or None

    return rollback_actions
