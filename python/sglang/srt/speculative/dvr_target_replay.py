from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)


@dataclass
class DVRPrivateExtendBatchSpec:
    """Inputs for a DVR-owned EXTEND batch detached from scheduler mutation.

    DVR uses private EXTEND batches for target verify oracles, boundary-state
    materialization, and final logprob scoring.  Keeping their ScheduleBatch
    construction in one place avoids three copies of upstream-sensitive field
    plumbing while leaving each replay path responsible for its own semantics.
    """

    reqs: list[Any]
    input_ids: list[int]
    out_cache_locs: Optional[list[torch.Tensor] | torch.Tensor]
    prefix_lens: list[int]
    extend_lens: list[int]
    final_seq_lens: list[int]
    return_logprob: bool = False
    top_logprobs_nums: Optional[list[int]] = None
    token_ids_logprobs: Optional[list[Any]] = None
    extend_logprob_start_lens: Optional[list[int]] = None
    extend_input_logprob_token_ids: Optional[list[int]] = None
    multimodal_inputs: Optional[list[Any]] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    is_extend_in_batch: bool = True
    all_extend_in_batch: bool = True
    is_prefill_only: Optional[bool] = None
    with_sampling_info: bool = False
    mamba_track_indices: Optional[torch.Tensor] = None
    mamba_track_mask: Optional[torch.Tensor] = None
    mamba_track_seqlens: Optional[torch.Tensor] = None
    mamba_cow_src_indices: Optional[torch.Tensor] = None
    mamba_cow_dst_indices: Optional[torch.Tensor] = None
    mamba_clear_indices: Optional[torch.Tensor] = None


@dataclass
class DVRSuffixReplayPlan:
    """Plan a target EXTEND replay over unclosed prefix tail plus appended rows.

    The appended rows are draft rows for verifier oracles and accepted rows for
    live-state repair.  Keeping one shape makes both paths share the same
    suffix replay plumbing while preserving their different callers.
    """

    base_seq_lens_cpu: list[int]
    boundary_lens: list[int]
    tail_lens_cpu: list[int]
    extend_lens_cpu: list[int]
    final_seq_lens_cpu: list[int]
    input_ids: list[int]
    out_cache_locs: list[torch.Tensor]
    append_cache_locs: torch.Tensor
    append_rows: Optional[torch.Tensor] = None
    append_offsets: Optional[torch.Tensor] = None
    hidden_gather_indices: Optional[torch.Tensor] = None


@dataclass
class DVRBoundaryReplayPlan:
    """Plan an EXTEND replay that materializes missing chunk-boundary state."""

    reqs: list[Any]
    input_ids: list[int]
    out_cache_locs: list[torch.Tensor]
    prefix_lens: list[int]
    extend_lens: list[int]
    final_seq_lens: list[int]
    boundary_indices: torch.Tensor


def _suffix_replay_lengths(
    *,
    base_seq_lens_cpu: list[int],
    boundary_lens: list[int],
    append_token_counts_cpu: list[int],
) -> tuple[list[int], list[int], list[int]]:
    tail_lens_cpu = [
        int(seq_len) - int(boundary)
        for seq_len, boundary in zip(base_seq_lens_cpu, boundary_lens, strict=True)
    ]
    extend_lens_cpu = [
        int(tail) + int(append_count)
        for tail, append_count in zip(
            tail_lens_cpu, append_token_counts_cpu, strict=True
        )
    ]
    final_seq_lens_cpu = [
        int(boundary) + int(extend_len)
        for boundary, extend_len in zip(boundary_lens, extend_lens_cpu, strict=True)
    ]
    return tail_lens_cpu, extend_lens_cpu, final_seq_lens_cpu


def _suffix_replay_inputs(
    *,
    batch,
    base_seq_lens_cpu: list[int],
    boundary_lens: list[int],
    tail_lens_cpu: list[int],
    append_tokens_cpu_by_req: list[list[int]],
    append_cache_locs_by_req: list[torch.Tensor],
    request_token_ids_for_replay,
) -> tuple[list[int], list[torch.Tensor]]:
    input_ids = []
    out_cache_locs = []
    for req, seq_len, boundary, tail_len, append_tokens, append_cache_locs in zip(
        batch.reqs,
        base_seq_lens_cpu,
        boundary_lens,
        tail_lens_cpu,
        append_tokens_cpu_by_req,
        append_cache_locs_by_req,
        strict=True,
    ):
        token_ids = request_token_ids_for_replay(req, int(seq_len))
        input_ids.extend(token_ids[int(boundary) : int(seq_len)])
        input_ids.extend(append_tokens)
        if int(tail_len) > 0:
            out_cache_locs.append(
                batch.req_to_token_pool.req_to_token[
                    req.req_pool_idx, int(boundary) : int(seq_len)
                ].to(torch.long)
            )
        if len(append_tokens) > 0:
            out_cache_locs.append(append_cache_locs.to(torch.long))
    return input_ids, out_cache_locs


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
        yield
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
) -> Optional[DVRSuffixReplayPlan]:
    """Build the common tail+draft replay shape for self-DVR and DVR-EAGLE."""

    bs = len(batch.seq_lens)
    draft_token_num = draft_tokens.numel() // max(bs, 1)
    draft_tokens = draft_tokens.reshape(bs, draft_token_num)
    draft_cache_locs = draft_cache_locs.reshape(bs, draft_token_num)
    draft_tokens_cpu = draft_tokens.detach().cpu().tolist()
    tail_lens_cpu, extend_lens_cpu, final_seq_lens_cpu = _suffix_replay_lengths(
        base_seq_lens_cpu=base_seq_lens_cpu,
        boundary_lens=boundary_lens,
        append_token_counts_cpu=[draft_token_num] * bs,
    )
    input_ids, out_cache_locs = _suffix_replay_inputs(
        batch=batch,
        base_seq_lens_cpu=base_seq_lens_cpu,
        boundary_lens=boundary_lens,
        tail_lens_cpu=tail_lens_cpu,
        append_tokens_cpu_by_req=draft_tokens_cpu,
        append_cache_locs_by_req=[draft_cache_locs[i] for i in range(bs)],
        request_token_ids_for_replay=request_token_ids_for_replay,
    )

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

    return DVRSuffixReplayPlan(
        base_seq_lens_cpu=base_seq_lens_cpu,
        boundary_lens=boundary_lens,
        tail_lens_cpu=tail_lens_cpu,
        extend_lens_cpu=extend_lens_cpu,
        final_seq_lens_cpu=final_seq_lens_cpu,
        input_ids=input_ids,
        out_cache_locs=out_cache_locs,
        append_cache_locs=draft_cache_locs,
        append_rows=draft_rows,
        append_offsets=draft_offsets,
        hidden_gather_indices=hidden_gather_indices,
    )


def build_suffix_draft_mrope_positions(
    replay_batch: ScheduleBatch,
    replay_plan: DVRSuffixReplayPlan,
) -> torch.Tensor:
    """Build flattened mrope positions matching suffix tail+draft input order."""

    draft_token_num = int(replay_plan.append_cache_locs.shape[1])
    device = replay_batch.seq_lens.device
    mrope_chunks = []
    mm_inputs = replay_batch.multimodal_inputs
    for req_i, (seq_len, boundary, tail_len) in enumerate(
        zip(
            replay_plan.base_seq_lens_cpu,
            replay_plan.boundary_lens,
            replay_plan.tail_lens_cpu,
            strict=True,
        )
    ):
        mm_input = (
            None if mm_inputs is None or req_i >= len(mm_inputs) else mm_inputs[req_i]
        )
        mm_positions = getattr(mm_input, "mrope_positions", None)
        chunk_parts = []
        if mm_positions is not None:
            tail_positions = mm_positions[:, int(boundary) : int(seq_len)].to(
                device=device, dtype=torch.long
            )
            if tail_positions.shape[1] > 0:
                chunk_parts.append(tail_positions)
        filled_tail = sum(part.shape[1] for part in chunk_parts)
        if filled_tail < int(tail_len):
            fallback_tail = (
                torch.arange(
                    int(boundary) + filled_tail,
                    int(seq_len),
                    dtype=torch.long,
                    device=device,
                )
                .unsqueeze(0)
                .repeat(3, 1)
            )
            chunk_parts.append(fallback_tail)
        draft_positions = (
            torch.arange(
                int(seq_len),
                int(seq_len) + draft_token_num,
                dtype=torch.long,
                device=device,
            )
            .unsqueeze(0)
            .repeat(3, 1)
        )
        chunk_parts.append(draft_positions)
        mrope_chunks.append(torch.cat(chunk_parts, dim=1))
    return torch.cat(mrope_chunks, dim=1)


def build_suffix_target_replay_batch(
    batch,
    replay_plan: DVRSuffixReplayPlan,
    *,
    capture_hidden_mode: CaptureHiddenMode,
    return_logprob: bool = False,
    mamba_cow_src_indices: Optional[torch.Tensor] = None,
    mamba_cow_dst_indices: Optional[torch.Tensor] = None,
    mamba_clear_indices: Optional[torch.Tensor] = None,
) -> ScheduleBatch:
    """Create the private target EXTEND batch for a suffix replay oracle."""

    return build_private_extend_batch(
        batch,
        DVRPrivateExtendBatchSpec(
            reqs=batch.reqs,
            input_ids=replay_plan.input_ids,
            out_cache_locs=replay_plan.out_cache_locs,
            prefix_lens=[int(x) for x in replay_plan.boundary_lens],
            extend_lens=[int(x) for x in replay_plan.extend_lens_cpu],
            final_seq_lens=replay_plan.final_seq_lens_cpu,
            return_logprob=return_logprob,
            extend_logprob_start_lens=[int(x) for x in replay_plan.extend_lens_cpu],
            capture_hidden_mode=capture_hidden_mode,
            is_prefill_only=batch.is_prefill_only,
            mamba_cow_src_indices=mamba_cow_src_indices,
            mamba_cow_dst_indices=mamba_cow_dst_indices,
            mamba_clear_indices=mamba_clear_indices,
        ),
    )


def build_accepted_suffix_replay_plan(
    *,
    batch,
    base_seq_lens_cpu: list[int],
    boundary_lens: list[int],
    accepted_tokens: torch.Tensor,
    accepted_cache_locs: torch.Tensor,
    accepted_token_counts_cpu: list[int],
    num_draft_tokens: int,
    request_token_ids_for_replay,
) -> Optional[DVRSuffixReplayPlan]:
    """Build the live-state repair replay shape for partially accepted chains."""

    if accepted_tokens is None or accepted_tokens.numel() == 0:
        return None

    total_accepted = sum(int(x) for x in accepted_token_counts_cpu)
    if (
        accepted_tokens.numel() != total_accepted
        or accepted_cache_locs.numel() != total_accepted
    ):
        return None

    if all(int(accepted) >= num_draft_tokens for accepted in accepted_token_counts_cpu):
        return None

    accepted_cache_locs = accepted_cache_locs.to(torch.long)
    tail_lens_cpu, extend_lens_cpu, final_seq_lens_cpu = _suffix_replay_lengths(
        base_seq_lens_cpu=base_seq_lens_cpu,
        boundary_lens=boundary_lens,
        append_token_counts_cpu=accepted_token_counts_cpu,
    )

    accepted_tokens_cpu = accepted_tokens.detach().cpu().tolist()
    accepted_tokens_by_req = []
    accepted_cache_locs_by_req = []
    accepted_rows = []
    accepted_offsets = []
    token_offset = 0
    for req_i, (seq_len, accepted_count) in enumerate(
        zip(
            base_seq_lens_cpu,
            accepted_token_counts_cpu,
            strict=True,
        )
    ):
        accepted_count = int(accepted_count)
        accepted_tokens_by_req.append(
            accepted_tokens_cpu[token_offset : token_offset + accepted_count]
        )
        accepted_cache_locs_by_req.append(
            accepted_cache_locs[token_offset : token_offset + accepted_count]
        )
        if accepted_count > 0:
            accepted_rows.append(
                batch.req_pool_indices[req_i].to(dtype=torch.long).repeat(
                    accepted_count
                )
            )
            accepted_offsets.append(
                torch.arange(
                    int(seq_len),
                    int(seq_len) + accepted_count,
                    dtype=torch.long,
                    device=batch.seq_lens.device,
                )
            )
        token_offset += accepted_count

    input_ids, out_cache_locs = _suffix_replay_inputs(
        batch=batch,
        base_seq_lens_cpu=base_seq_lens_cpu,
        boundary_lens=boundary_lens,
        tail_lens_cpu=tail_lens_cpu,
        append_tokens_cpu_by_req=accepted_tokens_by_req,
        append_cache_locs_by_req=accepted_cache_locs_by_req,
        request_token_ids_for_replay=request_token_ids_for_replay,
    )

    if not input_ids:
        return None

    return DVRSuffixReplayPlan(
        base_seq_lens_cpu=base_seq_lens_cpu,
        boundary_lens=boundary_lens,
        tail_lens_cpu=tail_lens_cpu,
        extend_lens_cpu=extend_lens_cpu,
        final_seq_lens_cpu=final_seq_lens_cpu,
        input_ids=input_ids,
        out_cache_locs=out_cache_locs,
        append_cache_locs=accepted_cache_locs,
        append_rows=torch.cat(accepted_rows) if accepted_rows else None,
        append_offsets=torch.cat(accepted_offsets) if accepted_offsets else None,
    )


def build_boundary_replay_plan(
    *,
    batch,
    tasks,
    state_adapter,
    request_token_ids_for_replay,
) -> Optional[DVRBoundaryReplayPlan]:
    """Build replay inputs for missing chunk-boundary checkpoints.

    Boundary replay is a real checkpoint materialization step, not a temporary
    verifier oracle.  It therefore runs on a narrow ScheduleBatch containing
    only the requests whose radix prefix did not already provide the requested
    chunk-aligned state.
    """

    reqs = [task.req for task in tasks]
    input_ids = []
    out_cache_locs = []
    prefix_lens = []
    extend_lens = []
    final_seq_lens = []

    for task in tasks:
        token_ids = request_token_ids_for_replay(task.req, task.boundary_seqlen)
        replay_token_ids = token_ids[task.source_seqlen : task.boundary_seqlen]
        input_ids.extend(replay_token_ids)
        out_cache_locs.append(
            batch.req_to_token_pool.req_to_token[
                task.req.req_pool_idx,
                task.source_seqlen : task.boundary_seqlen,
            ].to(torch.long)
        )
        prefix_lens.append(task.source_seqlen)
        extend_lens.append(task.boundary_seqlen - task.source_seqlen)
        final_seq_lens.append(task.boundary_seqlen)

    if not input_ids:
        return None

    boundary_indices = state_adapter.get_boundary_indices_for_reqs(
        reqs=reqs,
        track_indices=[task.boundary_track_idx for task in tasks],
        device=batch.device,
    )
    return DVRBoundaryReplayPlan(
        reqs=reqs,
        input_ids=input_ids,
        out_cache_locs=out_cache_locs,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        final_seq_lens=final_seq_lens,
        boundary_indices=boundary_indices,
    )


def build_boundary_replay_batch(batch, plan: DVRBoundaryReplayPlan) -> ScheduleBatch:
    """Create the narrow ScheduleBatch used for boundary checkpoint replay."""

    device = batch.device
    return build_private_extend_batch(
        batch,
        DVRPrivateExtendBatchSpec(
            reqs=plan.reqs,
            input_ids=plan.input_ids,
            out_cache_locs=plan.out_cache_locs,
            prefix_lens=plan.prefix_lens,
            extend_lens=plan.extend_lens,
            final_seq_lens=plan.final_seq_lens,
            extend_logprob_start_lens=plan.prefix_lens,
            is_extend_in_batch=False,
            all_extend_in_batch=False,
            is_prefill_only=True,
            mamba_track_indices=plan.boundary_indices,
            mamba_track_mask=torch.ones(
                len(plan.reqs), dtype=torch.bool, device=device
            ),
            mamba_track_seqlens=torch.tensor(
                plan.final_seq_lens, dtype=torch.int64, device=device
            ),
        ),
    )


def build_private_extend_batch(
    batch,
    spec: DVRPrivateExtendBatchSpec,
) -> ScheduleBatch:
    """Create a DVR-owned EXTEND batch with upstream ScheduleBatch plumbing."""

    device = batch.seq_lens.device
    global_num_tokens = None
    global_num_tokens_for_logprob = None
    if batch.global_num_tokens is not None:
        dp_world = len(batch.global_num_tokens)
        global_num_tokens = [len(spec.input_ids)] * dp_world
        global_num_tokens_for_logprob = [len(spec.input_ids)] * dp_world

    req_pool_indices = torch.tensor(
        [req.req_pool_idx for req in spec.reqs],
        dtype=torch.int64,
        device=device,
    )
    final_seq_lens = torch.tensor(spec.final_seq_lens, dtype=torch.int64, device=device)
    out_cache_loc = None
    if spec.out_cache_locs is not None:
        out_cache_loc = (
            torch.cat(spec.out_cache_locs)
            if isinstance(spec.out_cache_locs, list)
            else spec.out_cache_locs
        ).to(device=device)
    extend_logprob_start_lens = (
        spec.prefix_lens
        if spec.extend_logprob_start_lens is None
        else spec.extend_logprob_start_lens
    )
    multimodal_inputs = (
        [req.multimodal_inputs for req in spec.reqs]
        if spec.multimodal_inputs is None
        else spec.multimodal_inputs
    )

    replay_batch = ScheduleBatch(
        reqs=spec.reqs,
        req_to_token_pool=batch.req_to_token_pool,
        token_to_kv_pool_allocator=batch.token_to_kv_pool_allocator,
        tree_cache=batch.tree_cache,
        model_config=batch.model_config,
        enable_overlap=batch.enable_overlap,
        device=batch.device,
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.tensor(spec.input_ids, dtype=torch.int64, device=device),
        req_pool_indices=req_pool_indices,
        req_pool_indices_cpu=req_pool_indices.cpu(),
        seq_lens=final_seq_lens,
        out_cache_loc=out_cache_loc,
        seq_lens_cpu=final_seq_lens.cpu(),
        seq_lens_sum=sum(spec.final_seq_lens),
        return_logprob=spec.return_logprob,
        top_logprobs_nums=spec.top_logprobs_nums,
        token_ids_logprobs=spec.token_ids_logprobs,
        global_num_tokens=global_num_tokens,
        global_num_tokens_for_logprob=global_num_tokens_for_logprob,
        is_extend_in_batch=spec.is_extend_in_batch,
        all_extend_in_batch=spec.all_extend_in_batch,
        can_run_dp_cuda_graph=False,
        can_run_dp_breakable_cuda_graph=False,
        tbo_split_seq_index=None,
        global_forward_mode=None,
        extend_num_tokens=len(spec.input_ids),
        extend_lens=spec.extend_lens,
        prefix_lens=spec.prefix_lens,
        extend_logprob_start_lens=extend_logprob_start_lens,
        extend_input_logprob_token_ids=(
            None
            if spec.extend_input_logprob_token_ids is None
            else torch.tensor(
                spec.extend_input_logprob_token_ids,
                dtype=torch.int64,
                device=device,
            )
        ),
        multimodal_inputs=multimodal_inputs,
        encoder_cached=None,
        encoder_lens=None,
        encoder_lens_cpu=None,
        encoder_out_cache_loc=None,
        sampling_info=None,
        orig_seq_lens=final_seq_lens.to(dtype=torch.int32),
        input_embeds=None,
        ne_token_table=None,
        spec_algorithm=batch.spec_algorithm,
        spec_info=None,
        capture_hidden_mode=spec.capture_hidden_mode,
        hicache_consumer_index=-1,
        is_prefill_only=(
            batch.is_prefill_only
            if spec.is_prefill_only is None
            else spec.is_prefill_only
        ),
        dllm_config=batch.dllm_config,
        has_grammar=False,
        return_hidden_states=False,
        return_hidden_states_before_norm=False,
        mamba_track_indices=spec.mamba_track_indices,
        mamba_track_mask=spec.mamba_track_mask,
        mamba_track_seqlens=spec.mamba_track_seqlens,
        mamba_track_cache_seqlens=None,
        mamba_cow_src_indices=spec.mamba_cow_src_indices,
        mamba_cow_dst_indices=spec.mamba_cow_dst_indices,
        mamba_clear_indices=spec.mamba_clear_indices,
    )
    if spec.with_sampling_info:
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

        replay_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            replay_batch,
            batch.model_config.vocab_size,
        )
    return replay_batch


@contextmanager
def suffix_draft_replay_batch_context(
    *,
    batch,
    linear_state,
    linear_state_ctx,
    base_seq_lens_cpu: list[int],
    draft_tokens: torch.Tensor,
    draft_cache_locs: torch.Tensor,
    request_token_ids_for_replay,
    full_prefix_replay: bool = False,
    restore_boundary_state: bool = False,
    use_mamba_cow_from_boundary: bool = False,
):
    """Prepare a suffix+draft replay batch and protect live DVR state.

    Callers still own the actual forward path: self-DVR uses the normal target
    worker batch entrypoint, while DVR-EAGLE may need a prebuilt ForwardBatch to
    attach M-RoPE positions and to return hidden states for the next MTP step.
    """

    if batch.forward_mode.is_idle() or linear_state_ctx is None:
        yield None
        return

    live_indices = linear_state_ctx.live_indices
    boundary_indices = linear_state_ctx.boundary_indices
    if boundary_indices is None:
        yield None
        return

    boundary_lens = linear_state.boundary_lens_for_replay(batch, base_seq_lens_cpu)
    if full_prefix_replay:
        boundary_lens = [0 for _ in boundary_lens]
    replay_plan = build_suffix_draft_replay_plan(
        batch=batch,
        base_seq_lens_cpu=base_seq_lens_cpu,
        boundary_lens=boundary_lens,
        draft_tokens=draft_tokens,
        draft_cache_locs=draft_cache_locs,
        request_token_ids_for_replay=request_token_ids_for_replay,
    )
    if replay_plan is None:
        yield None
        return

    # Suffix replay is an oracle for verifier rows.  Checkpoint publication is
    # handled by the normal DVR commit path, not by this temporary EXTEND.
    linear_state.set_suffix_replay_boundary_track_mask(None)
    replay_batch = build_suffix_target_replay_batch(
        batch,
        replay_plan,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        mamba_cow_src_indices=(
            boundary_indices
            if use_mamba_cow_from_boundary and not full_prefix_replay
            else None
        ),
        mamba_cow_dst_indices=(
            live_indices if use_mamba_cow_from_boundary and not full_prefix_replay else None
        ),
        mamba_clear_indices=live_indices if full_prefix_replay else None,
    )

    with linear_state_replay_context(
        linear_state_ctx,
        clear_state_input_window=full_prefix_replay,
        restore_live_state=True,
    ):
        if restore_boundary_state and not full_prefix_replay:
            linear_state.restore_boundary_state_for_suffix_replay(linear_state_ctx)

        # Draft KV rows must be visible at absolute positions
        # base_seq_len..base_seq_len+draft during the oracle forward.
        assert replay_plan.append_rows is not None
        assert replay_plan.append_offsets is not None
        replay_batch.req_to_token_pool.write(
            (replay_plan.append_rows, replay_plan.append_offsets),
            replay_plan.append_cache_locs.to(torch.int32),
        )
        yield replay_batch, replay_plan


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


def run_suffix_draft_replay_oracle(
    *,
    target_worker,
    replay_batch: ScheduleBatch,
    replay_plan: DVRSuffixReplayPlan,
    use_forward_batch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a suffix+draft replay oracle and return only draft-row outputs."""

    model_runner = target_worker.model_runner
    if use_forward_batch:
        forward_batch = ForwardBatch.init_new(replay_batch, model_runner)
        if model_runner.model_is_mrope:
            forward_batch.mrope_positions = build_suffix_draft_mrope_positions(
                replay_batch, replay_plan
            )
        oracle_output = target_worker.forward_batch_generation(
            batch=None,
            forward_batch=forward_batch,
            is_verify=True,
        )
    else:
        oracle_output = target_worker.forward_batch_generation(
            batch=replay_batch,
            is_verify=True,
        )
        forward_batch = ForwardBatch.init_new(replay_batch, model_runner)

    assert replay_plan.hidden_gather_indices is not None
    return draft_row_logits_from_replay_hidden_states(
        target_worker=target_worker,
        forward_batch=forward_batch,
        hidden_states=oracle_output.logits_output.hidden_states,
        hidden_gather_indices=replay_plan.hidden_gather_indices,
    )
