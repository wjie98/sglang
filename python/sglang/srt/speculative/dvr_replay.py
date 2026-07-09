from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
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


def build_dvr_private_extend_batch(
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
    linear_state.suffix_replay_boundary_track_mask = (
        None if boundary_track_mask is None else boundary_track_mask.detach()
    )
    replay_batch = build_dvr_private_extend_batch(
        batch,
        reqs=batch.reqs,
        input_ids=replay_plan.input_ids,
        out_cache_locs=replay_plan.out_cache_locs,
        prefix_lens=[int(x) for x in replay_plan.boundary_lens],
        extend_lens=[int(x) for x in replay_plan.extend_lens_cpu],
        final_seq_lens=replay_plan.final_seq_lens_cpu,
        extend_logprob_start_lens=[int(x) for x in replay_plan.extend_lens_cpu],
        capture_hidden_mode=CaptureHiddenMode.FULL,
        is_prefill_only=batch.is_prefill_only,
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
            boundary_backup = linear_state.boundary_backup
            if boundary_backup is None:
                boundary_backup = linear_state_ctx.state_adapter.backup_recurrent_state(
                    state_cache=linear_state_ctx.state_cache,
                    indices=boundary_indices,
                )
            linear_state_ctx.state_adapter.restore_recurrent_state(
                state_cache=linear_state_ctx.state_cache,
                backup=boundary_backup,
                indices=live_indices,
            )

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
    linear_state.suffix_replay_boundary_track_mask = None
    replay_batch = build_dvr_private_extend_batch(
        batch,
        reqs=batch.reqs,
        input_ids=replay_plan.input_ids,
        out_cache_locs=replay_plan.out_cache_locs,
        prefix_lens=[int(x) for x in replay_plan.boundary_lens],
        extend_lens=[int(x) for x in replay_plan.extend_lens_cpu],
        final_seq_lens=replay_plan.final_seq_lens_cpu,
        extend_logprob_start_lens=[int(x) for x in replay_plan.extend_lens_cpu],
        capture_hidden_mode=CaptureHiddenMode.NULL,
        is_prefill_only=batch.is_prefill_only,
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
