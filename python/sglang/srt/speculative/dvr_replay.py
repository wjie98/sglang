from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dvr_info import (
    DVRFinalLogprobRepair,
    DVRPendingOutputPrefix,
    compact_dvr_output_rows,
    defer_dvr_non_streaming_logprob_output,
    try_claim_dvr_final_logprob_repair,
)


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
    append_token_counts_cpu: list[int]
    input_ids: list[int]
    out_cache_locs: list[torch.Tensor]
    append_cache_locs: torch.Tensor
    append_rows: Optional[torch.Tensor] = None
    append_offsets: Optional[torch.Tensor] = None
    hidden_gather_indices: Optional[torch.Tensor] = None


def _build_suffix_replay_plan(
    *,
    batch,
    base_seq_lens_cpu: list[int],
    boundary_lens: list[int],
    append_tokens_cpu_by_req: list[list[int]],
    append_cache_locs_by_req: list[torch.Tensor],
    append_cache_locs: torch.Tensor,
    append_token_counts_cpu: list[int],
    request_token_ids_for_replay,
    hidden_gather_append_count: Optional[int] = None,
) -> Optional[_DVRSuffixReplayPlan]:
    tail_lens_cpu = []
    extend_lens_cpu = []
    final_seq_lens_cpu = []
    input_ids = []
    out_cache_locs = []
    append_rows = []
    append_offsets = []

    for req_i, (
        req,
        seq_len,
        boundary,
        append_tokens,
        append_locs,
        append_count,
    ) in enumerate(
        zip(
            batch.reqs,
            base_seq_lens_cpu,
            boundary_lens,
            append_tokens_cpu_by_req,
            append_cache_locs_by_req,
            append_token_counts_cpu,
            strict=True,
        )
    ):
        seq_len = int(seq_len)
        boundary = int(boundary)
        append_count = int(append_count)
        tail_len = seq_len - boundary
        extend_len = tail_len + append_count

        tail_lens_cpu.append(tail_len)
        extend_lens_cpu.append(extend_len)
        final_seq_lens_cpu.append(boundary + extend_len)

        token_ids = request_token_ids_for_replay(req, seq_len)
        input_ids.extend(token_ids[boundary:seq_len])
        input_ids.extend(append_tokens)
        if tail_len > 0:
            out_cache_locs.append(
                batch.req_to_token_pool.req_to_token[
                    req.req_pool_idx, boundary:seq_len
                ].to(torch.long)
            )
        if append_count > 0:
            out_cache_locs.append(append_locs.to(torch.long))
            append_rows.append(
                batch.req_pool_indices[req_i].to(dtype=torch.long).repeat(append_count)
            )
            append_offsets.append(
                torch.arange(
                    seq_len,
                    seq_len + append_count,
                    dtype=torch.long,
                    device=batch.seq_lens.device,
                )
            )
    if not input_ids:
        return None

    hidden_gather_indices = None
    if hidden_gather_append_count is not None:
        gather_indices = []
        offset = 0
        for tail_len, extend_len in zip(tail_lens_cpu, extend_lens_cpu, strict=True):
            gather_indices.extend(
                range(
                    offset + int(tail_len),
                    offset + int(tail_len) + hidden_gather_append_count,
                )
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
        append_token_counts_cpu=append_token_counts_cpu,
        input_ids=input_ids,
        out_cache_locs=out_cache_locs,
        append_cache_locs=append_cache_locs.to(torch.long).reshape(-1),
        append_rows=torch.cat(append_rows) if append_rows else None,
        append_offsets=torch.cat(append_offsets) if append_offsets else None,
        hidden_gather_indices=hidden_gather_indices,
    )


@contextmanager
def _linear_state_replay_context(
    linear_state_ctx,
    *,
    clear_state_input_window: bool = False,
    restore_live_state: bool = True,
):
    """Save/restore linear-state side effects around a replay oracle.

    Suffix replay temporarily reuses the live request slots.  The oracle may
    clear state-input windows or mutate recurrent state through the normal
    EXTEND path; callers opt out of live-state restore only when the replay
    itself is intended to refresh committed live state.
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


def _build_suffix_draft_mrope_positions(
    replay_batch: ScheduleBatch,
    replay_plan: _DVRSuffixReplayPlan,
) -> torch.Tensor:
    """Build flattened mrope positions matching suffix tail+draft input order."""

    draft_token_num = int(replay_plan.append_token_counts_cpu[0])
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
    )


def build_boundary_replay_batch(
    *,
    batch,
    tasks,
    boundary_indices: torch.Tensor,
    request_token_ids_for_replay,
) -> Optional[ScheduleBatch]:
    """Create the private batch for missing chunk-boundary checkpoints.

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

    device = batch.device
    return _build_private_extend_batch(
        batch,
        reqs=reqs,
        input_ids=input_ids,
        out_cache_locs=out_cache_locs,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        final_seq_lens=final_seq_lens,
        extend_logprob_start_lens=prefix_lens,
        is_extend_in_batch=False,
        all_extend_in_batch=False,
        is_prefill_only=True,
        mamba_track_indices=boundary_indices,
        mamba_track_mask=torch.ones(len(reqs), dtype=torch.bool, device=device),
        mamba_track_seqlens=torch.tensor(
            final_seq_lens, dtype=torch.int64, device=device
        ),
    )


def _build_private_extend_batch(
    batch,
    *,
    reqs: list[Any],
    input_ids: list[int],
    out_cache_locs: Optional[list[torch.Tensor] | torch.Tensor],
    prefix_lens: list[int],
    extend_lens: list[int],
    final_seq_lens: list[int],
    return_logprob: bool = False,
    top_logprobs_nums: Optional[list[int]] = None,
    token_ids_logprobs: Optional[list[Any]] = None,
    extend_logprob_start_lens: Optional[list[int]] = None,
    extend_input_logprob_token_ids: Optional[list[int]] = None,
    multimodal_inputs: Optional[list[Any]] = None,
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
    is_extend_in_batch: bool = True,
    all_extend_in_batch: bool = True,
    is_prefill_only: Optional[bool] = None,
    with_sampling_info: bool = False,
    mamba_track_indices: Optional[torch.Tensor] = None,
    mamba_track_mask: Optional[torch.Tensor] = None,
    mamba_track_seqlens: Optional[torch.Tensor] = None,
    mamba_cow_src_indices: Optional[torch.Tensor] = None,
    mamba_cow_dst_indices: Optional[torch.Tensor] = None,
    mamba_clear_indices: Optional[torch.Tensor] = None,
) -> ScheduleBatch:
    """Create a DVR-owned EXTEND batch with upstream ScheduleBatch plumbing."""

    device = batch.seq_lens.device
    global_num_tokens = None
    global_num_tokens_for_logprob = None
    if batch.global_num_tokens is not None:
        dp_world = len(batch.global_num_tokens)
        global_num_tokens = [len(input_ids)] * dp_world
        global_num_tokens_for_logprob = [len(input_ids)] * dp_world

    req_pool_indices = torch.tensor(
        [req.req_pool_idx for req in reqs],
        dtype=torch.int64,
        device=device,
    )
    final_seq_lens_tensor = torch.tensor(
        final_seq_lens, dtype=torch.int64, device=device
    )
    out_cache_loc = None
    if out_cache_locs is not None:
        out_cache_loc = (
            torch.cat(out_cache_locs)
            if isinstance(out_cache_locs, list)
            else out_cache_locs
        ).to(device=device)
    extend_logprob_start_lens = (
        prefix_lens
        if extend_logprob_start_lens is None
        else extend_logprob_start_lens
    )
    multimodal_inputs = (
        [req.multimodal_inputs for req in reqs]
        if multimodal_inputs is None
        else multimodal_inputs
    )

    replay_batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=batch.req_to_token_pool,
        token_to_kv_pool_allocator=batch.token_to_kv_pool_allocator,
        tree_cache=batch.tree_cache,
        model_config=batch.model_config,
        enable_overlap=batch.enable_overlap,
        spec_algorithm=batch.spec_algorithm,
        dllm_config=batch.dllm_config,
    )
    replay_batch.forward_mode = ForwardMode.EXTEND
    replay_batch.input_ids = torch.tensor(
        input_ids, dtype=torch.int64, device=device
    )
    replay_batch.req_pool_indices = req_pool_indices
    replay_batch.req_pool_indices_cpu = req_pool_indices.cpu()
    replay_batch.seq_lens = final_seq_lens_tensor
    replay_batch.seq_lens_cpu = final_seq_lens_tensor.cpu()
    replay_batch.seq_lens_sum = sum(final_seq_lens)
    replay_batch.out_cache_loc = out_cache_loc
    replay_batch.orig_seq_lens = final_seq_lens_tensor.to(dtype=torch.int32)
    replay_batch.extend_num_tokens = len(input_ids)
    replay_batch.extend_lens = extend_lens
    replay_batch.prefix_lens = prefix_lens
    replay_batch.extend_logprob_start_lens = extend_logprob_start_lens
    replay_batch.extend_input_logprob_token_ids = (
        None
        if extend_input_logprob_token_ids is None
        else torch.tensor(
            extend_input_logprob_token_ids,
            dtype=torch.int64,
            device=device,
        )
    )
    replay_batch.multimodal_inputs = multimodal_inputs
    replay_batch.return_logprob = return_logprob
    replay_batch.top_logprobs_nums = top_logprobs_nums
    replay_batch.token_ids_logprobs = token_ids_logprobs
    replay_batch.global_num_tokens = global_num_tokens
    replay_batch.global_num_tokens_for_logprob = global_num_tokens_for_logprob
    replay_batch.is_extend_in_batch = is_extend_in_batch
    replay_batch.all_extend_in_batch = all_extend_in_batch
    replay_batch.can_run_dp_cuda_graph = False
    replay_batch.can_run_dp_breakable_cuda_graph = False
    replay_batch.tbo_split_seq_index = None
    replay_batch.global_forward_mode = None
    replay_batch.encoder_cached = None
    replay_batch.encoder_lens = None
    replay_batch.encoder_lens_cpu = None
    replay_batch.encoder_out_cache_loc = None
    replay_batch.sampling_info = None
    replay_batch.input_embeds = None
    replay_batch.ne_token_table = None
    replay_batch.spec_info = None
    replay_batch.capture_hidden_mode = capture_hidden_mode
    replay_batch.hicache_consumer_index = -1
    replay_batch.is_prefill_only = (
        batch.is_prefill_only if is_prefill_only is None else is_prefill_only
    )
    replay_batch.has_grammar = False
    replay_batch.return_hidden_states = False
    replay_batch.return_hidden_states_before_norm = False
    replay_batch.mamba_track_indices = mamba_track_indices
    replay_batch.mamba_track_mask = mamba_track_mask
    replay_batch.mamba_track_seqlens = mamba_track_seqlens
    replay_batch.mamba_cow_src_indices = mamba_cow_src_indices
    replay_batch.mamba_cow_dst_indices = mamba_cow_dst_indices
    replay_batch.mamba_clear_indices = mamba_clear_indices

    if with_sampling_info:
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

        replay_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            replay_batch,
            batch.model_config.vocab_size,
        )
    return replay_batch


@contextmanager
def dvr_suffix_replay_context(
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
    bs = len(batch.seq_lens)
    draft_token_num = draft_tokens.numel() // max(bs, 1)
    draft_tokens = draft_tokens.reshape(bs, draft_token_num)
    draft_cache_locs = draft_cache_locs.reshape(bs, draft_token_num)
    replay_plan = _build_suffix_replay_plan(
        batch=batch,
        base_seq_lens_cpu=base_seq_lens_cpu,
        boundary_lens=boundary_lens,
        append_tokens_cpu_by_req=draft_tokens.detach().cpu().tolist(),
        append_cache_locs_by_req=[draft_cache_locs[i] for i in range(bs)],
        append_cache_locs=draft_cache_locs,
        append_token_counts_cpu=[draft_token_num] * bs,
        request_token_ids_for_replay=request_token_ids_for_replay,
        hidden_gather_append_count=draft_token_num,
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
        chunk_size = int(linear_state.server_args.mamba_track_interval)
        boundary_track_mask = torch.tensor(
            [
                int(tail_len) + int(append_count) >= chunk_size
                for tail_len, append_count in zip(
                    replay_plan.tail_lens_cpu,
                    replay_plan.append_token_counts_cpu,
                    strict=True,
                )
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


def replay_dvr_accepted_suffix_for_live_state(
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

    total_accepted = sum(int(x) for x in accepted_token_counts_cpu)
    if (
        accepted_ids.numel() != total_accepted
        or accepted_cache_locs.numel() != total_accepted
    ):
        return None
    if all(int(accepted) >= num_draft_tokens for accepted in accepted_token_counts_cpu):
        return None

    boundary_lens = linear_state.boundary_lens_for_replay(batch, base_seq_lens_cpu)
    accepted_cache_locs = accepted_cache_locs.to(torch.long)
    accepted_tokens_cpu = accepted_ids.detach().cpu().tolist()
    accepted_tokens_by_req = []
    accepted_cache_locs_by_req = []
    token_offset = 0
    for accepted_count in accepted_token_counts_cpu:
        accepted_count = int(accepted_count)
        accepted_tokens_by_req.append(
            accepted_tokens_cpu[token_offset : token_offset + accepted_count]
        )
        accepted_cache_locs_by_req.append(
            accepted_cache_locs[token_offset : token_offset + accepted_count]
        )
        token_offset += accepted_count
    replay_plan = _build_suffix_replay_plan(
        batch=batch,
        base_seq_lens_cpu=base_seq_lens_cpu,
        boundary_lens=boundary_lens,
        append_tokens_cpu_by_req=accepted_tokens_by_req,
        append_cache_locs_by_req=accepted_cache_locs_by_req,
        append_cache_locs=accepted_cache_locs,
        append_token_counts_cpu=accepted_token_counts_cpu,
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


def run_dvr_suffix_replay_oracle(
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
    hidden_states = oracle_output.logits_output.hidden_states
    if hidden_states is None:
        raise RuntimeError("DVR target replay did not return hidden states.")
    draft_hidden_states = hidden_states[
        replay_plan.hidden_gather_indices.to(
            device=hidden_states.device,
            dtype=torch.long,
        )
    ]
    logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)
    logits_metadata.next_token_logits_buffer = None
    draft_logits = target_worker.model_runner.model.logits_processor._get_logits(
        draft_hidden_states,
        target_worker.model_runner.model.lm_head,
        logits_metadata,
    )
    return draft_logits, draft_hidden_states


def score_dvr_verify_outputs(
    *,
    batch: ScheduleBatch,
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
    batch: ScheduleBatch,
    target_worker: Any,
    linear_state_ctx: Any,
    replay_prefix: DVRPendingOutputPrefix,
    req_i: int,
    req: Any,
    final_output_len: int,
    error_prefix: str,
    force_final_logprob_replay: bool,
) -> DVRFinalLogprobRepair:
    if not force_final_logprob_replay or linear_state_ctx is None:
        try:
            return replay_prefix.final_logprob_repair(
                req,
                final_output_len,
                error_prefix=error_prefix,
            )
        except RuntimeError as exc:
            message = str(exc)
            if (
                "missing verify logprobs" not in message
                and "is incomplete" not in message
            ):
                raise
            if linear_state_ctx is None:
                raise

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
    replay_batch = _build_private_extend_batch(
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
    batch: ScheduleBatch,
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

    stop_pos = _first_token_finish_pos(
        req,
        compact_output_token_ids_per_req[req_i],
    )
    if stop_pos is not None and stop_pos < accept_len:
        return prefix_output_len + stop_pos + 1
    return None


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
