from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sglang.srt.speculative.dvr_scheduler_utils import (
    DVRFinalLogprobRepair,
    DVRReplayPrefixTracker,
)
from sglang.srt.speculative.dvr_target_replay import (
    DVRPrivateExtendBatchSpec,
    build_private_extend_batch,
    linear_state_replay_context,
)
from sglang.srt.speculative.output_policy import (
    defer_req_non_streaming_logprob_output,
    try_expect_req_final_logprob_repair,
)


@dataclass
class _DVRFinalLogprobReplayPlan:
    """Full-prefill scoring plan for final non-streaming DVR output logprobs."""

    req_indices: list[int]
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
    """Hold non-streaming DVR logprob chunks until final repair can overwrite them.

    Streaming logprob chunks are intentionally excluded: once emitted, they
    cannot be repaired at the final response, so workers must use their per-step
    oracle path for streaming exact logprobs.
    """

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
    compact_output_token_ids_per_req: Optional[list[list[int]]] = None,
    error_prefix: str,
    allow_preclaimed_final_token: bool = False,
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
        compact_output_token_ids_per_req=compact_output_token_ids_per_req,
        error_prefix=error_prefix,
        allow_preclaimed_final_token=allow_preclaimed_final_token,
    )
    if replay_plan is None:
        return None

    if len(replay_plan.final_score_specs) > 1:
        return _score_final_logprob_repairs_row_by_row(
            batch=batch,
            target_worker=target_worker,
            linear_state_ctx=linear_state_ctx,
            replay_plan=replay_plan,
        )

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


def _score_final_logprob_repairs_row_by_row(
    *,
    batch: ScheduleBatch,
    target_worker: Any,
    linear_state_ctx: Any,
    replay_plan: _DVRFinalLogprobReplayPlan,
) -> list[Optional[DVRFinalLogprobRepair]]:
    """Score exact final logprobs with the same single-row oracle as clients.

    Public KL checks score each completed request with one max_new_tokens=0
    prefill.  GDN batched prefill can differ slightly from that single-row
    oracle for short tails, so split only final logprob repair while leaving
    generation and state commit on their normal batched paths.
    """

    repairs: list[Optional[DVRFinalLogprobRepair]] = [
        None for _ in range(len(batch.reqs))
    ]
    req_to_plan_row = {
        req_i: row for row, req_i in enumerate(replay_plan.req_indices)
    }
    for req_i, req, replay_offset, final_output_ids in replay_plan.final_score_specs:
        plan_row = req_to_plan_row[req_i]
        extend_len = int(replay_plan.extend_lens_cpu[plan_row])
        row_input_ids = replay_plan.input_ids[
            replay_offset : replay_offset + extend_len
        ]
        row_plan = _DVRFinalLogprobReplayPlan(
            req_indices=[req_i],
            input_ids=row_input_ids,
            logprob_token_ids=row_input_ids[1:] + [0],
            extend_lens_cpu=[extend_len],
            final_seq_lens_cpu=[extend_len],
            final_score_specs=[(req_i, req, 0, final_output_ids)],
        )
        input_token_logprobs = _run_final_logprob_replay(
            batch=batch,
            target_worker=target_worker,
            linear_state_ctx=linear_state_ctx,
            replay_plan=row_plan,
        )
        row_repairs = _final_logprob_repairs_from_replay(
            batch=batch,
            replay_plan=row_plan,
            input_token_logprobs=input_token_logprobs,
        )
        repairs[req_i] = row_repairs[req_i]
    return repairs


def defer_and_score_dvr_final_logprob_repairs(
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
    """Defer non-streaming chunks and score exact final DVR output logprobs."""

    defer_dvr_non_streaming_logprob_output_until_finish(
        batch,
        base_seq_lens_cpu=base_seq_lens_cpu,
    )
    return score_dvr_final_logprob_repairs(
        batch=batch,
        target_worker=target_worker,
        replay_prefix=replay_prefix,
        linear_state_ctx=linear_state_ctx,
        base_seq_lens_cpu=base_seq_lens_cpu,
        accept_lens_cpu=accept_lens_cpu,
        compact_output_token_ids_per_req=compact_output_token_ids_per_req,
        error_prefix=error_prefix,
        allow_preclaimed_final_token=allow_preclaimed_final_token,
    )


def _build_final_logprob_replay_plan(
    *,
    batch: ScheduleBatch,
    replay_prefix: DVRReplayPrefixTracker,
    base_seq_lens_cpu: list[int],
    accept_lens_cpu: list[int],
    compact_output_token_ids_per_req: Optional[list[list[int]]],
    error_prefix: str,
    allow_preclaimed_final_token: bool,
) -> Optional[_DVRFinalLogprobReplayPlan]:
    req_indices: list[int] = []
    input_ids: list[int] = []
    logprob_token_ids: list[int] = []
    extend_lens_cpu: list[int] = []
    final_seq_lens_cpu: list[int] = []
    final_score_specs: list[tuple[int, Any, int, list[int]]] = []

    final_replay_specs: dict[int, int] = {}
    for req_i, (req, seq_len, accept_len) in enumerate(
        zip(batch.reqs, base_seq_lens_cpu, accept_lens_cpu, strict=True)
    ):
        stream = replay_prefix.stream_for_req(req, initialize_from_req_output=True)
        observed_output_len = max(len(req.output_ids), len(stream))
        final_output_len = _final_output_len_if_repair_needed(
            req=req,
            req_i=req_i,
            seq_len=int(seq_len),
            accept_len=int(accept_len),
            observed_output_len=observed_output_len,
            compact_output_token_ids_per_req=compact_output_token_ids_per_req,
            allow_preclaimed_final_token=allow_preclaimed_final_token,
        )
        if final_output_len is not None:
            final_replay_specs[req_i] = final_output_len

    if not final_replay_specs:
        return None

    for req_i, (req, seq_len, accept_len) in enumerate(
        zip(batch.reqs, base_seq_lens_cpu, accept_lens_cpu, strict=True)
    ):
        prompt_len = len(req.origin_input_ids)
        final_spec = final_replay_specs.get(req_i)
        if final_spec is None:
            continue

        replay_seq_len = prompt_len + final_spec

        replay_ids = _final_replay_ids_for_req(
            req=req,
            req_i=req_i,
            replay_prefix=replay_prefix,
            base_seq_len=int(seq_len),
            replay_seq_len=replay_seq_len,
            compact_output_token_ids_per_req=compact_output_token_ids_per_req,
            error_prefix=error_prefix,
        )
        replay_offset = len(input_ids)

        input_ids.extend(replay_ids)
        logprob_token_ids.extend(replay_ids[1:])
        logprob_token_ids.append(0)

        extend_len = len(replay_ids)
        req_indices.append(req_i)
        extend_lens_cpu.append(extend_len)
        final_seq_lens_cpu.append(extend_len)
        if try_expect_req_final_logprob_repair(req):
            final_output_ids = replay_ids[prompt_len : prompt_len + final_spec]
            final_score_specs.append((req_i, req, replay_offset, final_output_ids))

    if not final_score_specs:
        return None

    return _DVRFinalLogprobReplayPlan(
        req_indices=req_indices,
        input_ids=input_ids,
        logprob_token_ids=logprob_token_ids,
        extend_lens_cpu=extend_lens_cpu,
        final_seq_lens_cpu=final_seq_lens_cpu,
        final_score_specs=final_score_specs,
    )


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
        # The DVR replay stream is advanced from compact, client-visible
        # accepted rows before final repair. In synchronous V2 the model-side
        # seq_len can be stale or already preclaimed around the final step, so
        # use the observed output stream as the authoritative finish signal.
        return max_new_tokens
    if prefix_output_len >= max_new_tokens:
        return None

    length_remaining = max_new_tokens - prefix_output_len
    if length_remaining <= accept_len:
        # Req.update_finish_state checks the length cap before token-based
        # EOS/stop handling, so mirror that priority here.
        return max_new_tokens if length_remaining > 0 else None

    if allow_preclaimed_final_token and length_remaining == accept_len + 1:
        # Spec-v2 overlap preclaims one bonus slot. At the final step the
        # model-side seq_len can be one token behind the scheduler-visible
        # output, while the replay prefix already has the full token stream.
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
    """Build a final replay row from the stable prefix plus current emissions.

    In overlap and DVR-EAGLE, Req.output_ids may still lag while final exact
    logprobs are scored.  The replay prefix can reconstruct the pre-verify
    prefix; the worker-provided accepted-token rows are the authoritative tokens
    emitted by this verify step.  These rows must already be compact: no padded
    verify tail tokens should be present.
    """

    if replay_seq_len <= base_seq_len:
        token_ids = replay_prefix.request_output_prefix_token_ids(
            req,
            replay_seq_len,
            error_prefix=error_prefix,
        )
        return token_ids[:replay_seq_len]

    token_ids = replay_prefix.try_request_output_prefix_token_ids(
        req,
        replay_seq_len,
    )
    if token_ids is not None and len(token_ids) >= replay_seq_len:
        return token_ids[:replay_seq_len]

    prompt_len = len(req.origin_input_ids)
    materialized_seq_len = prompt_len + len(req.output_ids)
    stable_seq_len = min(replay_seq_len, max(base_seq_len, materialized_seq_len))
    base_ids = _materialized_or_replay_prefix_ids(
        req=req,
        replay_prefix=replay_prefix,
        seq_len=stable_seq_len,
        error_prefix=error_prefix,
    )

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


def _materialized_or_replay_prefix_ids(
    *,
    req: Any,
    replay_prefix: DVRReplayPrefixTracker,
    seq_len: int,
    error_prefix: str,
) -> list[int]:
    prompt_len = len(req.origin_input_ids)
    output_len = seq_len - prompt_len
    # DVR-EAGLE's replay tracker stores the verifier prefix, which can differ
    # from already-materialized client output by one bonus token.  The final
    # response repair must match Req.output_ids exactly, so prefer that stable
    # materialized prefix when it covers the requested length.
    if output_len > 0 and len(req.output_ids) >= output_len:
        return list(req.origin_input_ids) + list(req.output_ids[:output_len])

    token_ids = replay_prefix.request_output_prefix_token_ids(
        req,
        seq_len,
        error_prefix=error_prefix,
    )
    return token_ids[:seq_len]


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
    reqs = [batch.reqs[i] for i in replay_plan.req_indices]
    replay_batch = build_private_extend_batch(
        batch,
        DVRPrivateExtendBatchSpec(
            reqs=reqs,
            input_ids=replay_plan.input_ids,
            out_cache_locs=None,
            prefix_lens=[0 for _ in replay_plan.extend_lens_cpu],
            extend_lens=replay_plan.extend_lens_cpu,
            final_seq_lens=replay_plan.final_seq_lens_cpu,
            return_logprob=True,
            top_logprobs_nums=[0 for _ in reqs],
            token_ids_logprobs=[None for _ in reqs],
            extend_logprob_start_lens=[0 for _ in replay_plan.extend_lens_cpu],
            extend_input_logprob_token_ids=replay_plan.logprob_token_ids,
            multimodal_inputs=[None for _ in reqs],
            is_prefill_only=True,
            with_sampling_info=True,
        ),
    )
    replay_linear_state_ctx = _subset_linear_state_ctx(
        linear_state_ctx, replay_plan.req_indices
    )
    with _temporary_final_replay_cache_mapping(
        replay_batch,
        replay_plan,
    ) as temp_cache_locs:
        device = replay_batch.seq_lens.device
        replay_batch.out_cache_loc = temp_cache_locs.to(
            device=device, dtype=torch.long
        )
        replay_batch.mamba_clear_indices = replay_linear_state_ctx.live_indices

        with (
            envs.SGLANG_EAGER_INPUT_NO_COPY.override(True),
            linear_state_replay_context(
                replay_linear_state_ctx,
                clear_state_input_window=True,
                restore_live_state=True,
            ),
        ):
            # This is an external scoring oracle equivalent to
            # max_new_tokens=0, not a target-verify pass.  Keep is_verify=False
            # so TP worker runs the normal prefill-only logprob path.
            score_output = target_worker.forward_batch_generation(
                batch=replay_batch,
            )
            input_token_logprobs = score_output.logits_output.input_token_logprobs
            if input_token_logprobs is not None:
                # Materialize replay scores before restoring req-to-token
                # mappings and freeing temporary KV slots. Some attention/GDN
                # kernels may still have outstanding reads from those slots
                # when the tensor object is returned to Python.
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
    replay_plan: _DVRFinalLogprobReplayPlan,
):
    allocated_cache_locs = None
    try:
        temp_cache_locs = _alloc_final_replay_cache_locs(batch, replay_plan)
        allocated_cache_locs = temp_cache_locs
    except RuntimeError as exc:
        if not _should_reuse_live_cache_locs_after_alloc_failure(exc, replay_plan):
            raise
        # This replay is a final-response scoring oracle and every row belongs
        # to a request that will finish in this verify step.  Reuse the live KV
        # slots instead of requiring a second full-prefix copy; this keeps exact
        # logprob repair viable under tight 80B token budgets.
        temp_cache_locs = _live_cache_locs_for_final_replay(batch, replay_plan)
    write_rows, write_offsets = _final_replay_req_to_token_indices(
        batch, replay_plan
    )
    saved_locs = batch.req_to_token_pool.req_to_token[
        write_rows, write_offsets
    ].clone()
    try:
        # The final replay is a scoring oracle. It temporarily maps full replay
        # rows so logprob metadata is valid, then restores scheduler ownership
        # before result processing releases the live request slots.
        batch.req_to_token_pool.write(
            (write_rows, write_offsets), temp_cache_locs.to(torch.int32)
        )
        yield temp_cache_locs
    finally:
        batch.req_to_token_pool.write((write_rows, write_offsets), saved_locs)
        if allocated_cache_locs is not None:
            batch.token_to_kv_pool_allocator.free(allocated_cache_locs)


def _should_reuse_live_cache_locs_after_alloc_failure(
    exc: RuntimeError,
    replay_plan: _DVRFinalLogprobReplayPlan,
) -> bool:
    """Return whether final scoring may fall back to live request KV slots."""

    message = str(exc).lower()
    return (
        "out of memory" in message
        and "try to allocate" in message
        and len(replay_plan.final_score_specs) == len(replay_plan.extend_lens_cpu)
    )


def _live_cache_locs_for_final_replay(
    batch: ScheduleBatch,
    replay_plan: _DVRFinalLogprobReplayPlan,
) -> torch.Tensor:
    """Return the existing per-request KV slots for final full-prefix replay."""

    cache_locs = []
    for req, extend_len in zip(
        batch.reqs, replay_plan.extend_lens_cpu, strict=True
    ):
        cache_locs.append(
            batch.req_to_token_pool.req_to_token[
                req.req_pool_idx, : int(extend_len)
            ].to(torch.long)
        )
    cache_locs = torch.cat(cache_locs)
    if torch.any(cache_locs <= 0):
        raise RuntimeError(
            "DVR final logprob repair cannot reuse live KV slots because the "
            "request mapping does not cover the full replay prefix."
        )
    return cache_locs


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
