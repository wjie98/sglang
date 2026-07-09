from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_components.output_policy import (
    defer_req_non_streaming_logprob_output,
    try_claim_req_final_logprob_repair,
)
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dvr_scheduler_utils import (
    DVRFinalLogprobRepair,
    DVRReplayPrefixTracker,
)


@dataclass
class _DVRPrivateExtendBatchSpec:
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
class _DVRSuffixReplayPlan:
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
class _DVRBoundaryReplayPlan:
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
def _linear_state_replay_context(
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


def _build_suffix_draft_replay_plan(
    *,
    batch,
    base_seq_lens_cpu: list[int],
    boundary_lens: list[int],
    draft_tokens: torch.Tensor,
    draft_cache_locs: torch.Tensor,
    request_token_ids_for_replay,
) -> Optional[_DVRSuffixReplayPlan]:
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

    return _DVRSuffixReplayPlan(
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


def _build_suffix_draft_mrope_positions(
    replay_batch: ScheduleBatch,
    replay_plan: _DVRSuffixReplayPlan,
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


def _build_suffix_target_replay_batch(
    batch,
    replay_plan: _DVRSuffixReplayPlan,
    *,
    capture_hidden_mode: CaptureHiddenMode,
    return_logprob: bool = False,
    mamba_track_indices: Optional[torch.Tensor] = None,
    mamba_track_mask: Optional[torch.Tensor] = None,
    mamba_track_seqlens: Optional[torch.Tensor] = None,
    mamba_cow_src_indices: Optional[torch.Tensor] = None,
    mamba_cow_dst_indices: Optional[torch.Tensor] = None,
    mamba_clear_indices: Optional[torch.Tensor] = None,
) -> ScheduleBatch:
    """Create the private target EXTEND batch for a suffix replay oracle."""

    return _build_private_extend_batch(
        batch,
        _DVRPrivateExtendBatchSpec(
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
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=mamba_track_seqlens,
            mamba_cow_src_indices=mamba_cow_src_indices,
            mamba_cow_dst_indices=mamba_cow_dst_indices,
            mamba_clear_indices=mamba_clear_indices,
        ),
    )


def _build_accepted_suffix_replay_plan(
    *,
    batch,
    base_seq_lens_cpu: list[int],
    boundary_lens: list[int],
    accepted_tokens: torch.Tensor,
    accepted_cache_locs: torch.Tensor,
    accepted_token_counts_cpu: list[int],
    num_draft_tokens: int,
    request_token_ids_for_replay,
) -> Optional[_DVRSuffixReplayPlan]:
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

    return _DVRSuffixReplayPlan(
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
) -> Optional[_DVRBoundaryReplayPlan]:
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
    return _DVRBoundaryReplayPlan(
        reqs=reqs,
        input_ids=input_ids,
        out_cache_locs=out_cache_locs,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        final_seq_lens=final_seq_lens,
        boundary_indices=boundary_indices,
    )


def build_boundary_replay_batch(batch, plan: _DVRBoundaryReplayPlan) -> ScheduleBatch:
    """Create the narrow ScheduleBatch used for boundary checkpoint replay."""

    device = batch.device
    return _build_private_extend_batch(
        batch,
        _DVRPrivateExtendBatchSpec(
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


def _build_private_extend_batch(
    batch,
    spec: _DVRPrivateExtendBatchSpec,
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
    track_replay_boundary_checkpoint: bool = False,
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
    replay_plan = _build_suffix_draft_replay_plan(
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

    # Suffix replay is an oracle for verifier rows.  When it crosses the next
    # chunk boundary, let the normal EXTEND tracking path materialize that
    # boundary checkpoint; DVR commit will decide after sampling whether the
    # checkpoint is actually accepted.
    boundary_track_mask = None
    if track_replay_boundary_checkpoint and not full_prefix_replay:
        append_count = int(replay_plan.append_cache_locs.shape[1])
        chunk_size = int(linear_state.server_args.mamba_track_interval)
        boundary_track_mask = torch.tensor(
            [
                int(tail_len) + append_count >= chunk_size
                for tail_len in replay_plan.tail_lens_cpu
            ],
            dtype=torch.bool,
            device=batch.seq_lens.device,
        )
        if not bool(boundary_track_mask.any().item()):
            boundary_track_mask = None
    linear_state.set_suffix_replay_boundary_track_mask(boundary_track_mask)
    replay_batch = _build_suffix_target_replay_batch(
        batch,
        replay_plan,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        mamba_track_indices=boundary_indices if boundary_track_mask is not None else None,
        mamba_track_mask=boundary_track_mask,
        mamba_track_seqlens=(
            torch.tensor(
                replay_plan.final_seq_lens_cpu,
                dtype=torch.int64,
                device=batch.seq_lens.device,
            )
            if boundary_track_mask is not None
            else None
        ),
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

    with _linear_state_replay_context(
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


def _draft_row_logits_from_replay_hidden_states(
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


def replay_accepted_suffix_for_live_state(
    *,
    batch: ScheduleBatch,
    target_worker,
    linear_state,
    linear_state_ctx,
    base_seq_lens_cpu: list[int],
    accepted_token_counts_cpu: list[int],
    accepted_ids: torch.Tensor,
    accepted_cache_locs: torch.Tensor,
    num_draft_tokens: int,
    request_token_ids_for_replay,
) -> Optional[torch.Tensor]:
    """Refresh live recurrent state after a partial verify acceptance.

    The TARGET_VERIFY hot path forwards every proposed draft token.  When a row
    is rejected, however, the committed suffix is shorter and may include a
    target-predicted token.  Replaying only "unclosed chunk tail + accepted
    tokens" updates the live recurrent slot to the exact committed sequence
    without replaying the full prefix.
    """

    if batch.forward_mode.is_idle() or linear_state_ctx is None:
        return None
    if accepted_ids is None or accepted_ids.numel() == 0:
        return None

    live_indices = linear_state_ctx.live_indices
    boundary_indices = linear_state_ctx.boundary_indices
    assert boundary_indices is not None

    boundary_lens = linear_state.boundary_lens_for_replay(batch, base_seq_lens_cpu)
    replay_plan = _build_accepted_suffix_replay_plan(
        batch=batch,
        base_seq_lens_cpu=base_seq_lens_cpu,
        boundary_lens=boundary_lens,
        accepted_tokens=accepted_ids,
        accepted_cache_locs=accepted_cache_locs,
        accepted_token_counts_cpu=accepted_token_counts_cpu,
        num_draft_tokens=num_draft_tokens,
        request_token_ids_for_replay=request_token_ids_for_replay,
    )
    if replay_plan is None:
        return None

    # Commit repair, not checkpoint publication: leave the live recurrent slot
    # updated by the replay so the normal commit path can copy it directly.
    linear_state.set_suffix_replay_boundary_track_mask(None)
    replay_batch = _build_suffix_target_replay_batch(
        batch,
        replay_plan,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        mamba_cow_src_indices=boundary_indices,
        mamba_cow_dst_indices=live_indices,
    )
    with _linear_state_replay_context(linear_state_ctx, restore_live_state=False):
        if replay_plan.append_rows is not None:
            assert replay_plan.append_offsets is not None
            replay_batch.req_to_token_pool.write(
                (replay_plan.append_rows, replay_plan.append_offsets),
                replay_plan.append_cache_locs.to(
                    device=replay_batch.seq_lens.device,
                    dtype=torch.int32,
                ),
            )
        target_worker.forward_batch_generation(batch=replay_batch, is_verify=True)
        return torch.ones(
            len(batch.reqs), dtype=torch.bool, device=live_indices.device
        )


def run_suffix_draft_replay_oracle(
    *,
    target_worker,
    replay_batch: ScheduleBatch,
    replay_plan: _DVRSuffixReplayPlan,
    use_forward_batch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a suffix+draft replay oracle and return only draft-row outputs."""

    model_runner = target_worker.model_runner
    if use_forward_batch:
        forward_batch = ForwardBatch.init_new(replay_batch, model_runner)
        if model_runner.model_is_mrope:
            forward_batch.mrope_positions = _build_suffix_draft_mrope_positions(
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
    return _draft_row_logits_from_replay_hidden_states(
        target_worker=target_worker,
        forward_batch=forward_batch,
        hidden_states=oracle_output.logits_output.hidden_states,
        hidden_gather_indices=replay_plan.hidden_gather_indices,
    )


def defer_dvr_non_streaming_logprob_output_until_finish(
    batch: ScheduleBatch,
    *,
    base_seq_lens_cpu: Optional[list[int]] = None,
) -> None:
    """Hold non-streaming DVR logprob chunks until final repair can overwrite them."""

    for req_i, req in enumerate(batch.reqs):
        if not (req.return_logprob and not req.stream):
            continue
        if base_seq_lens_cpu is not None:
            max_new_tokens = req.sampling_params.max_new_tokens
            if max_new_tokens is not None:
                prompt_len = len(req.origin_input_ids)
                prefix_output_len = max(
                    0,
                    int(base_seq_lens_cpu[req_i]) - prompt_len,
                )
                if prefix_output_len >= int(max_new_tokens):
                    continue
        defer_req_non_streaming_logprob_output(req)


def score_deferred_dvr_final_logprob_repairs(
    *,
    batch: ScheduleBatch,
    target_worker: Any,
    replay_prefix: DVRReplayPrefixTracker,
    linear_state_ctx: Any,
    base_seq_lens_cpu: list[int],
    accept_lens_cpu: list[int],
    compact_output_token_ids_per_req: Optional[list[list[int]]] = None,
    error_prefix: str,
    allow_preclaimed_final_token: bool = False,
) -> Optional[list[Optional[DVRFinalLogprobRepair]]]:
    """Defer non-streaming output and score exact final DVR logprob repairs."""

    if not batch.return_logprob:
        return None
    defer_dvr_non_streaming_logprob_output_until_finish(
        batch,
        base_seq_lens_cpu=base_seq_lens_cpu,
    )
    if batch.forward_mode.is_idle() or linear_state_ctx is None:
        return None

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
            compact_output_token_ids_per_req=compact_output_token_ids_per_req,
            allow_preclaimed_final_token=allow_preclaimed_final_token,
        )
        if final_output_len is None:
            continue

        prompt_len = len(req.origin_input_ids)
        replay_seq_len = prompt_len + final_output_len
        replay_ids = _final_replay_ids_for_req(
            req=req,
            req_i=req_i,
            replay_prefix=replay_prefix,
            base_seq_len=int(seq_len),
            replay_seq_len=replay_seq_len,
            compact_output_token_ids_per_req=compact_output_token_ids_per_req,
            error_prefix=error_prefix,
        )
        if not try_claim_req_final_logprob_repair(req):
            continue

        # Keep final repair request-by-request. Generation and verify stay
        # batched, while this full-prefix scoring oracle must match the
        # max_new_tokens=0 reference exactly at GDN chunk boundaries.
        input_token_logprobs = _run_final_logprob_replay(
            batch=batch,
            target_worker=target_worker,
            linear_state_ctx=linear_state_ctx,
            req_i=req_i,
            req=req,
            input_ids=replay_ids,
        )
        output_logprob_start = prompt_len - 1
        final_output_ids = replay_ids[prompt_len : prompt_len + final_output_len]
        output_logprob_end = output_logprob_start + len(final_output_ids)
        repairs[req_i] = DVRFinalLogprobRepair(
            output_ids=final_output_ids,
            output_logprobs=input_token_logprobs[
                output_logprob_start:output_logprob_end
            ]
            .detach()
            .cpu()
            .tolist(),
        )
        has_repair = True
    return repairs if has_repair else None


def _final_output_len_if_repair_needed(
    *,
    req: Any,
    req_i: int,
    seq_len: int,
    accept_len: int,
    observed_output_len: int,
    compact_output_token_ids_per_req: Optional[list[list[int]]],
    allow_preclaimed_final_token: bool,
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

    if allow_preclaimed_final_token and length_remaining == accept_len + 1:
        # Spec-v2 overlap preclaims one bonus slot. At the final step the
        # model-side seq_len can be one token behind the scheduler-visible
        # output while replay prefix already has the full token stream.
        return max_new_tokens if accept_len > 0 else None

    if compact_output_token_ids_per_req is None:
        return None

    stop_pos = _first_token_finish_pos(
        req,
        compact_output_token_ids_per_req[req_i],
    )
    if stop_pos is not None and stop_pos < accept_len:
        return prefix_output_len + stop_pos + 1
    return None


def _final_replay_ids_for_req(
    *,
    req: Any,
    req_i: int,
    replay_prefix: DVRReplayPrefixTracker,
    base_seq_len: int,
    replay_seq_len: int,
    compact_output_token_ids_per_req: Optional[list[list[int]]],
    error_prefix: str,
) -> list[int]:
    """Build a final replay row from the stable prefix plus current emissions."""

    token_ids = replay_prefix.try_request_output_prefix_token_ids(
        req,
        replay_seq_len,
    )
    if token_ids is not None and len(token_ids) >= replay_seq_len:
        return token_ids[:replay_seq_len]

    prompt_len = len(req.origin_input_ids)
    materialized_seq_len = prompt_len + len(req.output_ids)
    stable_seq_len = min(replay_seq_len, max(base_seq_len, materialized_seq_len))
    output_len = stable_seq_len - prompt_len
    # DVR-EAGLE's replay tracker stores the verifier prefix, which can differ
    # from already-materialized client output by one bonus token. Final response
    # repair must match Req.output_ids exactly, so prefer that stable prefix.
    if output_len > 0 and len(req.output_ids) >= output_len:
        base_ids = list(req.origin_input_ids) + list(req.output_ids[:output_len])
    else:
        base_ids = replay_prefix.request_output_prefix_token_ids(
            req,
            stable_seq_len,
            error_prefix=error_prefix,
        )[:stable_seq_len]

    current_needed = replay_seq_len - stable_seq_len
    if current_needed == 0:
        return base_ids
    if compact_output_token_ids_per_req is not None and req_i < len(
        compact_output_token_ids_per_req
    ):
        current_tokens = compact_output_token_ids_per_req[req_i]
        if len(current_tokens) >= current_needed:
            return base_ids + [int(x) for x in current_tokens[:current_needed]]

    token_ids = replay_prefix.request_output_prefix_token_ids(
        req,
        replay_seq_len,
        error_prefix=error_prefix,
    )
    return token_ids[:replay_seq_len]


def _first_token_finish_pos(req: Any, token_ids: list[int]) -> Optional[int]:
    """Return the first accepted-token index that would finish this request."""

    if req.sampling_params.ignore_eos:
        return None

    stop_token_ids = req.sampling_params.stop_token_ids or set()
    eos_token_ids = req.eos_token_ids or set()
    tokenizer = getattr(req, "tokenizer", None)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    additional_stop_ids = (
        getattr(tokenizer, "additional_stop_token_ids", None) if tokenizer else None
    ) or set()

    for i, token_id in enumerate(token_ids):
        token_id = int(token_id)
        if (
            token_id in stop_token_ids
            or token_id in eos_token_ids
            or token_id == tokenizer_eos
            or token_id in additional_stop_ids
        ):
            return i
        if token_id > req.vocab_size or token_id < 0:
            return i
    return None


def _run_final_logprob_replay(
    *,
    batch: ScheduleBatch,
    target_worker: Any,
    linear_state_ctx: Any,
    req_i: int,
    req: Any,
    input_ids: list[int],
) -> torch.Tensor:
    extend_len = len(input_ids)
    replay_batch = _build_private_extend_batch(
        batch,
        _DVRPrivateExtendBatchSpec(
            reqs=[req],
            input_ids=input_ids,
            out_cache_locs=None,
            prefix_lens=[0],
            extend_lens=[extend_len],
            final_seq_lens=[extend_len],
            return_logprob=True,
            top_logprobs_nums=[0],
            token_ids_logprobs=[None],
            extend_logprob_start_lens=[0],
            extend_input_logprob_token_ids=input_ids[1:] + [0],
            multimodal_inputs=[None],
            is_prefill_only=True,
            with_sampling_info=True,
        ),
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
            # Final logprob repair is a full-prefix scoring oracle. Unlike
            # suffix replay, it must start from an empty recurrent state.
            replay_linear_state_ctx.state_adapter.zero_recurrent_state(
                state_cache=replay_linear_state_ctx.state_cache,
                indices=replay_linear_state_ctx.live_indices,
            )
            score_output = target_worker.forward_batch_generation(
                batch=replay_batch,
            )
            input_token_logprobs = score_output.logits_output.input_token_logprobs
            if input_token_logprobs is not None:
                # Materialize scores before restoring req-to-token mappings and
                # freeing temporary KV slots.
                input_token_logprobs = input_token_logprobs.detach().cpu()
        if input_token_logprobs is None:
            raise RuntimeError("DVR final logprob replay did not return logprobs.")
        return input_token_logprobs


def _subset_linear_state_ctx(linear_state_ctx: Any, req_indices: list[int]) -> Any:
    """Restrict replay state backup/clear/restore to final-response rows."""

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
    batch: ScheduleBatch,
    extend_len: int,
):
    allocated_cache_locs = None
    temp_cache_locs = _try_live_cache_locs_for_final_replay(batch, extend_len)
    if temp_cache_locs is None:
        temp_cache_locs = _alloc_final_replay_cache_locs(batch, extend_len)
        allocated_cache_locs = temp_cache_locs
    write_rows, write_offsets = _final_replay_req_to_token_indices(batch, extend_len)
    saved_locs = batch.req_to_token_pool.req_to_token[
        write_rows, write_offsets
    ].clone()
    try:
        # This scoring oracle temporarily maps full replay rows so logprob
        # metadata is valid, then restores scheduler ownership before result
        # processing releases live request slots.
        batch.req_to_token_pool.write(
            (write_rows, write_offsets), temp_cache_locs.to(torch.int32)
        )
        yield temp_cache_locs
    finally:
        batch.req_to_token_pool.write((write_rows, write_offsets), saved_locs)
        if allocated_cache_locs is not None:
            batch.token_to_kv_pool_allocator.free(allocated_cache_locs)


def _try_live_cache_locs_for_final_replay(
    batch: ScheduleBatch,
    extend_len: int,
) -> Optional[torch.Tensor]:
    """Return existing final-response KV slots when the full prefix is mapped."""

    req = batch.reqs[0]
    cache_locs = batch.req_to_token_pool.req_to_token[
        req.req_pool_idx, : int(extend_len)
    ].to(torch.long)
    if cache_locs.numel() != int(extend_len):
        return None
    if torch.any(cache_locs <= 0):
        return None
    return cache_locs


def _alloc_final_replay_cache_locs(
    batch: ScheduleBatch,
    extend_len: int,
) -> torch.Tensor:
    """Allocate temporary KV slots for side-effect-free final logprob replay."""

    page_size = getattr(batch.tree_cache, "page_size", 1)
    if page_size == 1:
        return alloc_token_slots(batch.tree_cache, extend_len)

    device = batch.seq_lens.device
    prefix_lens_cpu = torch.zeros(1, dtype=torch.int64)
    seq_lens_cpu = torch.tensor([extend_len], dtype=torch.int64)
    prefix_lens = prefix_lens_cpu.to(device=device, non_blocking=True)
    seq_lens = seq_lens_cpu.to(device=device, non_blocking=True)
    last_loc = torch.full(
        (1,),
        -1,
        dtype=torch.long,
        device=device,
    )
    return alloc_paged_token_slots_extend(
        tree_cache=batch.tree_cache,
        prefix_lens=prefix_lens,
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        last_loc=last_loc,
        extend_num_tokens=extend_len,
    )


def _final_replay_req_to_token_indices(
    batch: ScheduleBatch,
    extend_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = batch.seq_lens.device
    req = batch.reqs[0]
    rows = torch.full(
        (int(extend_len),),
        int(req.req_pool_idx),
        dtype=torch.long,
        device=device,
    )
    offsets = torch.arange(int(extend_len), dtype=torch.long, device=device)
    return rows, offsets
