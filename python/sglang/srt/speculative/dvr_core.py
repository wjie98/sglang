from __future__ import annotations

from typing import Any, Optional

import torch

from sglang.srt.speculative.dvr_info import (
    DVRDeferredActions,
    DVRFinalLogprobRepair,
    DVRPendingOutputPrefix,
    compact_dvr_accepted_tokens_and_cache_locs,
    compact_dvr_output_rows,
    defer_dvr_non_streaming_logprob_output,
    try_claim_dvr_final_logprob_repair,
)
from sglang.srt.speculative.dvr_state_flow import (
    replay_dvr_accepted_suffix_for_live_state,
)


def score_dvr_verify_outputs(
    *,
    batch: Any,
    replay_prefix: DVRPendingOutputPrefix,
    output_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    token_logprobs: Optional[torch.Tensor],
    tokens_per_req: int,
    base_seq_lens_cpu: list[int],
    error_prefix: str,
) -> tuple[list[int], Optional[list[Optional[DVRFinalLogprobRepair]]]]:
    """Record accepted DVR tokens and carry exact final non-streaming logprobs."""

    _, accept_lens_cpu, token_ids_per_req = compact_dvr_output_rows(
        batch=batch,
        output_tokens=output_tokens,
        accept_lens=accept_lens,
        tokens_per_req=tokens_per_req,
        base_seq_lens_cpu=base_seq_lens_cpu,
    )
    token_logprobs_per_req = None
    if token_logprobs is not None:
        logprob_rows = token_logprobs.detach().cpu().tolist()
        token_logprobs_per_req = [
            [float(value) for value in row[:accept_len]]
            for row, accept_len in zip(logprob_rows, accept_lens_cpu, strict=True)
        ]
    replay_prefix.append_batch_output_tokens(
        batch,
        token_ids_per_req,
        token_logprobs_per_req=token_logprobs_per_req,
        base_seq_lens_cpu=base_seq_lens_cpu,
        error_prefix=f"{error_prefix} output prefix",
    )
    if not batch.return_logprob:
        return accept_lens_cpu, None
    for req_i, req in enumerate(batch.reqs):
        if not (req.return_logprob and not req.stream):
            continue
        max_new_tokens = req.sampling_params.max_new_tokens
        if max_new_tokens is not None:
            prompt_len = len(req.origin_input_ids)
            prefix_output_len = max(0, int(base_seq_lens_cpu[req_i]) - prompt_len)
            if prefix_output_len >= int(max_new_tokens):
                continue
        defer_dvr_non_streaming_logprob_output(req)
    if batch.forward_mode.is_idle() or token_logprobs_per_req is None:
        return accept_lens_cpu, None

    repairs: list[Optional[DVRFinalLogprobRepair]] = [
        None for _ in range(len(batch.reqs))
    ]
    has_repair = False
    for req_i, (req, seq_len, accept_len) in enumerate(
        zip(batch.reqs, base_seq_lens_cpu, accept_lens_cpu, strict=True)
    ):
        observed_output_len = replay_prefix.observed_output_len(req)
        final_output_len = _final_output_len_if_repair_needed(
            req=req,
            req_i=req_i,
            seq_len=int(seq_len),
            accept_len=int(accept_len),
            observed_output_len=observed_output_len,
            compact_output_token_ids_per_req=token_ids_per_req,
        )
        if final_output_len is None:
            continue

        if not try_claim_dvr_final_logprob_repair(req):
            continue
        repairs[req_i] = replay_prefix.final_logprob_repair(
            req=req,
            output_len=final_output_len,
            error_prefix=f"{error_prefix} final logprob",
        )
        has_repair = True
    return accept_lens_cpu, repairs if has_repair else None


def _final_output_len_if_repair_needed(
    *,
    req: Any,
    req_i: int,
    seq_len: int,
    accept_len: int,
    observed_output_len: int,
    compact_output_token_ids_per_req: Optional[list[list[int]]],
) -> Optional[int]:
    """Return the final output length if this verify step finishes the request."""

    if not req.return_logprob or req.stream:
        return None

    max_new_tokens = req.sampling_params.max_new_tokens
    if max_new_tokens is None:
        return None

    prompt_len = len(req.origin_input_ids)
    prefix_output_len = max(0, seq_len - prompt_len)
    max_new_tokens = int(max_new_tokens)
    if observed_output_len >= max_new_tokens:
        # The DVR replay stream is advanced from compact, client-visible rows
        # before final repair. Treat it as authoritative around final overlap
        # steps where model-side seq_len may already be stale or preclaimed.
        return max_new_tokens
    if prefix_output_len >= max_new_tokens:
        return None

    length_remaining = max_new_tokens - prefix_output_len
    if length_remaining <= accept_len:
        return max_new_tokens if length_remaining > 0 else None

    if length_remaining == accept_len + 1:
        # Spec-v2 overlap preclaims one bonus slot. At the final step the
        # model-side seq_len can be one token behind the scheduler-visible
        # output while replay prefix already has the full token stream.
        return max_new_tokens if accept_len > 0 else None

    if compact_output_token_ids_per_req is None:
        return None

    if req.sampling_params.ignore_eos:
        return None
    stop_token_ids = req.sampling_params.stop_token_ids or set()
    eos_token_ids = req.eos_token_ids or set()
    tokenizer = getattr(req, "tokenizer", None)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    additional_stop_ids = (
        getattr(tokenizer, "additional_stop_token_ids", None) if tokenizer else None
    ) or set()
    for i, token_id in enumerate(compact_output_token_ids_per_req[req_i]):
        token_id = int(token_id)
        if (
            token_id in stop_token_ids
            or token_id in eos_token_ids
            or token_id == tokenizer_eos
            or token_id in additional_stop_ids
        ):
            return prefix_output_len + i + 1 if i < accept_len else None
        if token_id > req.vocab_size or token_id < 0:
            return prefix_output_len + i + 1 if i < accept_len else None
    return None


def finish_dvr_verify(
    *,
    batch: Any,
    linear_state: Any,
    linear_state_ctx: Any,
    accept_lens: torch.Tensor,
    accept_lens_cpu: Optional[list[int]],
    num_draft_tokens: int,
    replay_prefix: Optional[DVRPendingOutputPrefix] = None,
    output_tokens: Optional[torch.Tensor] = None,
    token_logprobs: Optional[torch.Tensor] = None,
    tokens_per_req: Optional[int] = None,
    base_seq_lens_cpu: Optional[list[int]] = None,
    error_prefix: str = "DVR",
    predict: Optional[torch.Tensor] = None,
    accept_index: Optional[torch.Tensor] = None,
    partial_suffix_replay_kwargs: Optional[dict[str, Any]] = None,
    use_fast_self_draft_commit: bool = False,
) -> DVRDeferredActions:
    """Record accepted output rows and commit target recurrent state.

    Draft adapters stop at sampling.  Everything after that is DVR core:
    scheduler-visible output prefix, optional exact logprob repair, accepted
    suffix state repair, and delayed checkpoint publication for spec-v2.
    """

    final_logprob_repairs = None
    if (
        replay_prefix is not None
        and output_tokens is not None
        and tokens_per_req is not None
        and base_seq_lens_cpu is not None
        and not batch.forward_mode.is_idle()
    ):
        accept_lens_cpu, final_logprob_repairs = score_dvr_verify_outputs(
            batch=batch,
            replay_prefix=replay_prefix,
            output_tokens=output_tokens,
            accept_lens=accept_lens,
            token_logprobs=token_logprobs,
            tokens_per_req=tokens_per_req,
            base_seq_lens_cpu=base_seq_lens_cpu,
            error_prefix=error_prefix,
        )
    elif accept_lens_cpu is None:
        accept_lens_cpu = accept_lens.detach().cpu().tolist()

    pending_track_indices = None
    pending_track_seqlens = None
    if linear_state_ctx is not None:
        live_state_already_replayed = None
        if partial_suffix_replay_kwargs is not None and torch.any(
            accept_lens < num_draft_tokens
        ).item():
            if predict is None or accept_index is None:
                raise RuntimeError("DVR partial suffix replay requires sampled rows.")
            accepted_ids, accepted_cache_locs = (
                compact_dvr_accepted_tokens_and_cache_locs(
                    batch=batch,
                    predict=predict,
                    accept_index=accept_index,
                    accept_lens=accept_lens,
                    num_draft_tokens=num_draft_tokens,
                )
            )
            if accepted_ids.numel() > 0:
                live_state_already_replayed = replay_dvr_accepted_suffix_for_live_state(
                    batch=batch,
                    accepted_token_counts_cpu=accept_lens_cpu,
                    accepted_ids=accepted_ids,
                    accepted_cache_locs=accepted_cache_locs,
                    **partial_suffix_replay_kwargs,
                )

        pending_track_indices, pending_track_seqlens = linear_state.commit_after_verify(
            batch=batch,
            accepted_token_counts=accept_lens.to(torch.long),
            accepted_steps=(accept_lens - 1).to(torch.long),
            accepted_token_counts_cpu=accept_lens_cpu,
            ctx=linear_state_ctx,
            seq_lens_cpu=base_seq_lens_cpu or linear_state.batch_seq_lens_cpu(batch),
            live_state_already_replayed=live_state_already_replayed,
            use_fast_self_draft_commit=use_fast_self_draft_commit,
        )

    deferred_actions = DVRDeferredActions()
    if (
        pending_track_indices is not None
        or pending_track_seqlens is not None
        or final_logprob_repairs is not None
    ):
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
        deferred_actions.pending_mamba_checkpoints = checkpoints or None
        deferred_actions.final_logprob_repairs = final_logprob_repairs

    return deferred_actions
