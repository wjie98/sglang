from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dvr_scheduler_utils import (
    DVRFinalLogprobRepair,
    DVRReplayPrefixTracker,
)
from sglang.srt.speculative.dvr_target_replay import (
    DVRTargetReplaySpec,
    linear_state_replay_context,
    target_extend_replay_batch,
)
from sglang.srt.speculative.output_policy import (
    defer_req_non_streaming_logprob_output,
    try_expect_req_final_logprob_repair,
)


@dataclass
class _DVRFinalLogprobReplayPlan:
    """Full-prefill scoring plan for final non-streaming DVR output logprobs."""

    input_ids: list[int]
    logprob_token_ids: list[int]
    extend_lens_cpu: list[int]
    final_seq_lens_cpu: list[int]
    final_score_specs: list[tuple[int, Any, int, list[int]]]


def defer_dvr_non_streaming_logprob_output_until_finish(
    batch: ScheduleBatch,
    *,
    base_seq_lens_cpu: Optional[list[int]] = None,
) -> None:
    """Hold non-streaming DVR logprob chunks until final repair can overwrite them."""

    for req_i, req in enumerate(batch.reqs):
        if req.return_logprob and not req.stream:
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


def score_dvr_final_logprob_repairs(
    *,
    batch: ScheduleBatch,
    target_worker: Any,
    replay_prefix: DVRReplayPrefixTracker,
    linear_state_ctx: Any,
    base_seq_lens_cpu: list[int],
    accept_lens_cpu: list[int],
    accepted_token_ids_per_req: Optional[list[list[int]]] = None,
    error_prefix: str,
) -> Optional[list[Optional[DVRFinalLogprobRepair]]]:
    """Score final non-streaming DVR output logprobs with one full-prefill replay.

    Spec-v2 materializes accepted tokens after the worker returns.  During
    generation we may record fast-path logprobs from target verify rows, but those
    rows are not guaranteed to be bitwise identical to the KL oracle at every GDN
    chunk boundary.  Non-streaming responses can defer output, so repair only the
    final response with one full-prefix replay instead of replaying the full
    prefix after every verify step.
    """

    if batch.forward_mode.is_idle() or linear_state_ctx is None:
        return None

    replay_plan = _build_final_logprob_replay_plan(
        batch=batch,
        replay_prefix=replay_prefix,
        base_seq_lens_cpu=base_seq_lens_cpu,
        accept_lens_cpu=accept_lens_cpu,
        accepted_token_ids_per_req=accepted_token_ids_per_req,
        error_prefix=error_prefix,
    )
    if replay_plan is None:
        return None

    input_token_logprobs = _run_final_logprob_replay(
        batch=batch,
        target_worker=target_worker,
        linear_state_ctx=linear_state_ctx,
        replay_plan=replay_plan,
    )
    return _final_logprob_repairs_from_replay(
        batch=batch,
        replay_plan=replay_plan,
        input_token_logprobs=input_token_logprobs,
    )


def _build_final_logprob_replay_plan(
    *,
    batch: ScheduleBatch,
    replay_prefix: DVRReplayPrefixTracker,
    base_seq_lens_cpu: list[int],
    accept_lens_cpu: list[int],
    accepted_token_ids_per_req: Optional[list[list[int]]],
    error_prefix: str,
) -> Optional[_DVRFinalLogprobReplayPlan]:
    input_ids: list[int] = []
    logprob_token_ids: list[int] = []
    extend_lens_cpu: list[int] = []
    final_seq_lens_cpu: list[int] = []
    final_score_specs: list[tuple[int, Any, int, list[int]]] = []

    final_replay_specs: dict[int, int] = {}
    for req_i, (req, seq_len, accept_len) in enumerate(
        zip(batch.reqs, base_seq_lens_cpu, accept_lens_cpu, strict=True)
    ):
        if not req.return_logprob or req.stream:
            continue

        max_new_tokens = req.sampling_params.max_new_tokens
        if max_new_tokens is None:
            continue

        prompt_len = len(req.origin_input_ids)
        prefix_output_len = max(0, int(seq_len) - prompt_len)
        accept_len = int(accept_len)
        if prefix_output_len >= max_new_tokens:
            continue

        final_output_len: Optional[int] = None
        accepted_in_final_step: Optional[int] = None
        length_remaining = int(max_new_tokens) - prefix_output_len
        if length_remaining <= accept_len:
            # Req.update_finish_state checks the length cap before token-based
            # EOS/stop handling, so mirror that priority here.
            final_output_len = int(max_new_tokens)
            accepted_in_final_step = length_remaining
        elif length_remaining == accept_len + 1:
            # Spec-v2 overlap preclaims one bonus slot.  At the final step the
            # model-side seq_len can be one token behind the scheduler-visible
            # output, while the replay prefix already has the full token stream.
            final_output_len = int(max_new_tokens)
            accepted_in_final_step = accept_len
        elif accepted_token_ids_per_req is not None:
            stop_pos = _first_token_finish_pos(
                req,
                accepted_token_ids_per_req[req_i],
            )
            if stop_pos is not None and stop_pos < accept_len:
                final_output_len = prefix_output_len + stop_pos + 1
                accepted_in_final_step = stop_pos + 1

        if final_output_len is None or accepted_in_final_step is None:
            continue

        if accepted_in_final_step <= 0:
            continue
        if not try_expect_req_final_logprob_repair(req):
            continue

        final_replay_specs[req_i] = final_output_len

    if not final_replay_specs:
        return None

    # ForwardBatch currently derives req_pool_indices from the live ScheduleBatch.
    # Keep the replay sequence count aligned with that live batch, and only emit
    # repairs for requests that finish in this verify step.
    for req_i, (req, seq_len, accept_len) in enumerate(
        zip(batch.reqs, base_seq_lens_cpu, accept_lens_cpu, strict=True)
    ):
        prompt_len = len(req.origin_input_ids)
        final_spec = final_replay_specs.get(req_i)
        if final_spec is None:
            # Non-final rows are present only to keep replay metadata aligned with
            # the live batch. Their existing prefix is sufficient; current
            # accepted-token cache locs are owned by overlap bookkeeping and are
            # not needed for final logprob repair.
            replay_seq_len = int(seq_len)
        else:
            replay_seq_len = prompt_len + final_spec

        token_ids = replay_prefix.request_token_ids(
            req,
            replay_seq_len,
            initialize_from_req_output=False,
            include_full_untruncated_fill_ids=True,
            error_prefix=error_prefix,
        )
        replay_ids = token_ids[:replay_seq_len]
        replay_offset = len(input_ids)

        input_ids.extend(replay_ids)
        logprob_token_ids.extend(replay_ids[1:])
        logprob_token_ids.append(0)

        extend_len = len(replay_ids)
        extend_lens_cpu.append(extend_len)
        final_seq_lens_cpu.append(extend_len)
        if final_spec is not None:
            final_output_ids = replay_ids[prompt_len : prompt_len + final_spec]
            final_score_specs.append((req_i, req, replay_offset, final_output_ids))

    return _DVRFinalLogprobReplayPlan(
        input_ids=input_ids,
        logprob_token_ids=logprob_token_ids,
        extend_lens_cpu=extend_lens_cpu,
        final_seq_lens_cpu=final_seq_lens_cpu,
        final_score_specs=final_score_specs,
    )


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
    replay_plan: _DVRFinalLogprobReplayPlan,
) -> torch.Tensor:
    temp_cache_locs = _alloc_final_replay_cache_locs(batch, replay_plan)
    replay_spec = DVRTargetReplaySpec(
        input_ids=replay_plan.input_ids,
        out_cache_locs=[temp_cache_locs.to(torch.long)],
        prefix_lens=[0 for _ in replay_plan.extend_lens_cpu],
        extend_lens=[int(x) for x in replay_plan.extend_lens_cpu],
        final_seq_lens=replay_plan.final_seq_lens_cpu,
        extend_logprob_start_lens=[0 for _ in replay_plan.extend_lens_cpu],
        extend_input_logprob_token_ids=replay_plan.logprob_token_ids,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_logprob=True,
        mamba_clear_indices=linear_state_ctx.live_indices,
        multimodal_inputs=[None for _ in replay_plan.extend_lens_cpu],
    )

    with (
        envs.SGLANG_EAGER_INPUT_NO_COPY.override(True),
        linear_state_replay_context(
            linear_state_ctx,
            clear_state_input_window=True,
            restore_live_state=True,
        ),
        target_extend_replay_batch(batch, replay_spec),
    ):
        write_rows, write_offsets = _final_replay_req_to_token_indices(
            batch, replay_plan
        )
        saved_locs = batch.req_to_token_pool.req_to_token[
            write_rows, write_offsets
        ].clone()
        try:
            # The final replay is a scoring oracle.  It temporarily maps the
            # full replay rows so logprob metadata is valid, then restores the
            # scheduler-owned mapping before result processing releases slots.
            batch.req_to_token_pool.write(
                (write_rows, write_offsets), temp_cache_locs.to(torch.int32)
            )
            score_output = target_worker.forward_batch_generation(
                batch=batch,
                is_verify=True,
            )
            input_token_logprobs = score_output.logits_output.input_token_logprobs
            if input_token_logprobs is None:
                raise RuntimeError("DVR final logprob replay did not return logprobs.")
            return input_token_logprobs
        finally:
            batch.req_to_token_pool.write((write_rows, write_offsets), saved_locs)
            batch.token_to_kv_pool_allocator.free(temp_cache_locs)


def _alloc_final_replay_cache_locs(
    batch: ScheduleBatch,
    replay_plan: _DVRFinalLogprobReplayPlan,
) -> torch.Tensor:
    """Allocate temporary KV slots for side-effect-free final logprob replay."""

    num_tokens = len(replay_plan.input_ids)
    page_size = getattr(batch.tree_cache, "page_size", 1)
    if page_size == 1:
        return alloc_token_slots(batch.tree_cache, num_tokens)

    device = batch.seq_lens.device
    prefix_lens_cpu = torch.zeros(len(replay_plan.extend_lens_cpu), dtype=torch.int64)
    seq_lens_cpu = torch.tensor(replay_plan.final_seq_lens_cpu, dtype=torch.int64)
    prefix_lens = prefix_lens_cpu.to(device=device, non_blocking=True)
    seq_lens = seq_lens_cpu.to(device=device, non_blocking=True)
    last_loc = torch.full(
        (len(replay_plan.extend_lens_cpu),),
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
        extend_num_tokens=num_tokens,
    )


def _final_replay_req_to_token_indices(
    batch: ScheduleBatch,
    replay_plan: _DVRFinalLogprobReplayPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    offsets = []
    device = batch.seq_lens.device
    for req, extend_len in zip(
        batch.reqs, replay_plan.extend_lens_cpu, strict=True
    ):
        extend_len = int(extend_len)
        rows.append(
            torch.full(
                (extend_len,),
                int(req.req_pool_idx),
                dtype=torch.long,
                device=device,
            )
        )
        offsets.append(torch.arange(extend_len, dtype=torch.long, device=device))
    return torch.cat(rows), torch.cat(offsets)


def _final_logprob_repairs_from_replay(
    *,
    batch: ScheduleBatch,
    replay_plan: _DVRFinalLogprobReplayPlan,
    input_token_logprobs: torch.Tensor,
) -> list[Optional[DVRFinalLogprobRepair]]:
    final_logprob_repairs: list[Optional[DVRFinalLogprobRepair]] = [
        None for _ in range(len(batch.reqs))
    ]
    for req_i, req, replay_offset, final_output_ids in replay_plan.final_score_specs:
        prompt_len = len(req.origin_input_ids)
        output_logprob_start = replay_offset + prompt_len - 1
        output_logprob_end = output_logprob_start + len(final_output_ids)
        final_logprobs = input_token_logprobs[
            output_logprob_start:output_logprob_end
        ]
        final_logprob_repairs[req_i] = DVRFinalLogprobRepair(
            output_ids=final_output_ids,
            output_logprobs=final_logprobs.detach().cpu().tolist(),
        )
    return final_logprob_repairs
