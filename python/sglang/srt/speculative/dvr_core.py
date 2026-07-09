from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sglang.srt.speculative.dvr_info import (
    DVRDeferredActions,
    DVRDeferredOutput,
    DVRFinalLogprobRepair,
    DVRMambaCheckpoint,
    DVRPendingOutputPrefix,
    compact_dvr_accepted_tokens_and_cache_locs,
    compact_dvr_output_rows,
    defer_dvr_non_streaming_logprob_output,
    try_claim_dvr_final_logprob_repair,
)
from sglang.srt.speculative.dvr_replay import (
    _linear_state_replay_context,
    build_dvr_private_extend_batch,
    replay_dvr_accepted_suffix_for_live_state,
)


@dataclass
class DVRDraftKVState:
    """Draft-model KV ownership seen by DVR core.

    Self draft shares the target model's scheduler-owned speculative KV window,
    so it does not need a separate accepted-suffix repair.  EAGLE/MTP keeps its
    own draft KV and asks DVR core to repair the target live recurrent state
    after partial acceptance.
    """

    owns_draft_kv_cache: bool

    @classmethod
    def self_draft(cls) -> "DVRDraftKVState":
        return cls(owns_draft_kv_cache=False)

    @classmethod
    def external_draft(cls) -> "DVRDraftKVState":
        return cls(owns_draft_kv_cache=True)

    @property
    def needs_accepted_suffix_repair(self) -> bool:
        return self.owns_draft_kv_cache


@dataclass
class DVRDraftResult:
    """Draft adapter output consumed by DVR target verify."""

    verify_input: Any
    kv_state: DVRDraftKVState

    @classmethod
    def self_draft(cls, verify_input: Any) -> "DVRDraftResult":
        return cls(verify_input=verify_input, kv_state=DVRDraftKVState.self_draft())

    @classmethod
    def external_draft(cls, verify_input: Any) -> "DVRDraftResult":
        return cls(
            verify_input=verify_input,
            kv_state=DVRDraftKVState.external_draft(),
        )


@dataclass
class DVRVerifyResult:
    """Core DVR post-verify result shared by self draft and EAGLE/MTP."""

    accept_lens_cpu: list[int]
    deferred_actions: Optional[DVRDeferredActions]


def score_dvr_verify_outputs(
    *,
    batch: Any,
    target_worker: Any,
    replay_prefix: DVRPendingOutputPrefix,
    linear_state_ctx: Any,
    output_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    token_logprobs: Optional[torch.Tensor],
    tokens_per_req: int,
    base_seq_lens_cpu: list[int],
    error_prefix: str,
    force_final_logprob_replay: bool = False,
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
        repairs[req_i] = _build_final_logprob_repair(
            batch=batch,
            target_worker=target_worker,
            linear_state_ctx=linear_state_ctx,
            replay_prefix=replay_prefix,
            req_i=req_i,
            req=req,
            final_output_len=final_output_len,
            error_prefix=f"{error_prefix} final logprob",
            force_final_logprob_replay=force_final_logprob_replay,
        )
        has_repair = True
    return accept_lens_cpu, repairs if has_repair else None


def _build_final_logprob_repair(
    *,
    batch: Any,
    target_worker: Any,
    linear_state_ctx: Any,
    replay_prefix: DVRPendingOutputPrefix,
    req_i: int,
    req: Any,
    final_output_len: int,
    error_prefix: str,
    force_final_logprob_replay: bool,
) -> DVRFinalLogprobRepair:
    if not (force_final_logprob_replay and linear_state_ctx is not None):
        return replay_prefix.final_logprob_repair(
            req,
            final_output_len,
            error_prefix=error_prefix,
        )

    # EAGLE/MTP uses compact suffix replay to pair verify logits with hidden
    # states for the next draft.  Final non-streaming logprobs are stricter:
    # return the same rows as a full target EXTEND over prompt + output.
    prompt_len = len(req.origin_input_ids)
    replay_seq_len = prompt_len + final_output_len
    replay_ids = replay_prefix.request_output_prefix_token_ids(
        req,
        replay_seq_len,
        error_prefix=error_prefix,
    )
    extend_len = len(replay_ids)
    replay_batch = build_dvr_private_extend_batch(
        batch,
        reqs=[req],
        input_ids=replay_ids,
        out_cache_locs=None,
        prefix_lens=[0],
        extend_lens=[extend_len],
        final_seq_lens=[extend_len],
        return_logprob=True,
        top_logprobs_nums=[0],
        token_ids_logprobs=[None],
        extend_logprob_start_lens=[0],
        extend_input_logprob_token_ids=replay_ids[1:] + [0],
        multimodal_inputs=[None],
        is_prefill_only=True,
        with_sampling_info=True,
    )
    replay_linear_state_ctx = _subset_linear_state_ctx(linear_state_ctx, [req_i])
    with _temporary_final_replay_cache_mapping(
        replay_batch,
        extend_len,
    ) as temp_cache_locs:
        device = replay_batch.seq_lens.device
        replay_batch.out_cache_loc = temp_cache_locs.to(
            device=device, dtype=torch.long
        )
        replay_batch.mamba_clear_indices = replay_linear_state_ctx.live_indices

        with (
            envs.SGLANG_EAGER_INPUT_NO_COPY.override(True),
            _linear_state_replay_context(
                replay_linear_state_ctx,
                clear_state_input_window=True,
                restore_live_state=True,
            ),
        ):
            replay_linear_state_ctx.state_adapter.zero_recurrent_state(
                state_cache=replay_linear_state_ctx.state_cache,
                indices=replay_linear_state_ctx.live_indices,
            )
            score_output = target_worker.forward_batch_generation(
                batch=replay_batch,
            )
            input_token_logprobs = score_output.logits_output.input_token_logprobs
            if input_token_logprobs is not None:
                input_token_logprobs = input_token_logprobs.detach().cpu()
        if input_token_logprobs is None:
            raise RuntimeError("DVR final logprob replay did not return logprobs.")

    output_logprob_start = prompt_len - 1
    final_output_ids = replay_ids[prompt_len : prompt_len + final_output_len]
    output_logprob_end = output_logprob_start + len(final_output_ids)
    return DVRFinalLogprobRepair(
        output_ids=final_output_ids,
        output_logprobs=input_token_logprobs[
            output_logprob_start:output_logprob_end
        ]
        .detach()
        .cpu()
        .tolist(),
    )


def _subset_linear_state_ctx(linear_state_ctx: Any, req_indices: list[int]) -> Any:
    index = torch.tensor(
        req_indices, dtype=torch.long, device=linear_state_ctx.live_indices.device
    )
    boundary_indices = getattr(linear_state_ctx, "boundary_indices", None)
    if boundary_indices is not None:
        boundary_indices = boundary_indices[index]
    return replace(
        linear_state_ctx,
        state_input_indices=linear_state_ctx.state_input_indices[index],
        live_indices=linear_state_ctx.live_indices[index],
        boundary_indices=boundary_indices,
    )


@contextmanager
def _temporary_final_replay_cache_mapping(
    batch: Any,
    extend_len: int,
):
    device = batch.seq_lens.device
    page_size = getattr(batch.tree_cache, "page_size", 1)
    if page_size == 1:
        temp_cache_locs = alloc_token_slots(batch.tree_cache, extend_len)
    else:
        prefix_lens_cpu = torch.zeros(1, dtype=torch.int64)
        seq_lens_cpu = torch.tensor([extend_len], dtype=torch.int64)
        temp_cache_locs = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=prefix_lens_cpu.to(device=device, non_blocking=True),
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens_cpu.to(device=device, non_blocking=True),
            seq_lens_cpu=seq_lens_cpu,
            last_loc=torch.full((1,), -1, dtype=torch.long, device=device),
            extend_num_tokens=extend_len,
        )
    write_offsets = torch.arange(int(extend_len), dtype=torch.long, device=device)
    write_rows = torch.full(
        (int(extend_len),),
        int(batch.reqs[0].req_pool_idx),
        dtype=torch.long,
        device=device,
    )
    saved_locs = batch.req_to_token_pool.req_to_token[
        write_rows, write_offsets
    ].clone()
    try:
        batch.req_to_token_pool.write(
            (write_rows, write_offsets), temp_cache_locs.to(torch.int32)
        )
        yield temp_cache_locs
    finally:
        batch.req_to_token_pool.write((write_rows, write_offsets), saved_locs)
        batch.token_to_kv_pool_allocator.free(temp_cache_locs)


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
    target_worker: Optional[Any] = None,
    output_tokens: Optional[torch.Tensor] = None,
    token_logprobs: Optional[torch.Tensor] = None,
    tokens_per_req: Optional[int] = None,
    base_seq_lens_cpu: Optional[list[int]] = None,
    error_prefix: str = "DVR",
    draft_kv_state: Optional[DVRDraftKVState] = None,
    predict: Optional[torch.Tensor] = None,
    accept_index: Optional[torch.Tensor] = None,
    partial_suffix_replay_kwargs: Optional[dict[str, Any]] = None,
    use_fast_self_draft_commit: bool = False,
) -> DVRVerifyResult:
    """Record accepted output rows and commit target recurrent state.

    Draft adapters stop at sampling.  Everything after that is DVR core:
    scheduler-visible output prefix, optional exact logprob repair, accepted
    suffix state repair, and delayed checkpoint publication for spec-v2.
    """

    final_logprob_repairs = None
    if (
        replay_prefix is not None
        and target_worker is not None
        and output_tokens is not None
        and tokens_per_req is not None
        and base_seq_lens_cpu is not None
        and not batch.forward_mode.is_idle()
    ):
        # External draft models seed the next draft from suffix-oracle hidden
        # states, but exact final non-streaming logprobs must still match a full
        # target EXTEND when prefix-cache/GDN chunk boundaries are involved.
        force_final_logprob_replay = (
            draft_kv_state is not None
            and draft_kv_state.needs_accepted_suffix_repair
            and linear_state_ctx is not None
        )
        accept_lens_cpu, final_logprob_repairs = score_dvr_verify_outputs(
            batch=batch,
            target_worker=target_worker,
            replay_prefix=replay_prefix,
            linear_state_ctx=linear_state_ctx,
            output_tokens=output_tokens,
            accept_lens=accept_lens,
            token_logprobs=token_logprobs,
            tokens_per_req=tokens_per_req,
            base_seq_lens_cpu=base_seq_lens_cpu,
            error_prefix=error_prefix,
            force_final_logprob_replay=force_final_logprob_replay,
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
            publish_boundary_checkpoint=False,
            return_pending_boundary=True,
        )

    deferred_actions = None
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
                DVRMambaCheckpoint(track_idx=track_idx, seqlen=seqlen)
                if track_idx is not None and seqlen is not None
                else None
            )
            for track_idx, seqlen in zip(
                pending_track_indices, pending_track_seqlens, strict=True
            )
        ]
        deferred_actions = DVRDeferredActions(
            pending_mamba_checkpoints=checkpoints or None,
            output=(
                DVRDeferredOutput(final_logprob_repairs=final_logprob_repairs)
                if final_logprob_repairs is not None
                else None
            ),
        )

    return DVRVerifyResult(
        accept_lens_cpu=accept_lens_cpu,
        deferred_actions=deferred_actions,
    )
