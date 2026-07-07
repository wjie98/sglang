from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
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
    try_claim_req_final_logprob_repair,
)


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
    """Score final non-streaming DVR output logprobs with exact replay oracles.

    Spec-v2 materializes accepted tokens after the worker returns.  During
    generation we may record fast-path logprobs from target verify rows, but those
    rows are not guaranteed to be bitwise identical to the KL oracle at every GDN
    chunk boundary.  Non-streaming responses can defer output, so repair only
    the final response.  Keep the final replay request-by-request: the fixed DVR
    guards use this public single-request oracle as the exact reference, and it
    avoids batched GDN replay edge cases at request boundaries.
    """

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

        # The fixed KL guards catch batched final replay drifting from the
        # per-request max_new_tokens=0 oracle on short prompts.  Keep final
        # repair request-by-request; generation and verify stay batched.
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
    output_len = stable_seq_len - prompt_len
    # DVR-EAGLE's replay tracker stores the verifier prefix, which can differ
    # from already-materialized client output by one bonus token.  The final
    # response repair must match Req.output_ids exactly, so prefer that stable
    # materialized prefix when it covers the requested length.
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
    replay_batch = build_private_extend_batch(
        batch,
        DVRPrivateExtendBatchSpec(
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
