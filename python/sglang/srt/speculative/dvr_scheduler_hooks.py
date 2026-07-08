from __future__ import annotations

from typing import Any

from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.speculative.spec_policy import get_spec_algorithm_policy


def maybe_filter_running_batch_with_spec_state(
    *,
    batch: Any,
    future_map: Any,
    enable_overlap: bool,
    v1_spec_info_filtered: bool,
) -> bool:
    """Filter running batch with spec-owned logical-finish state if needed.

    DVR spec-v2 overlap can publish a row's new seq_len before accepted tokens
    are materialized into ``Req.output_ids``.  Keep that DVR-specific finish
    oracle out of ``ScheduleBatch.filter_batch`` so its default behavior stays
    aligned with upstream: filter only by ``Req.finished()`` unless callers pass
    explicit ``keep_indices``.
    """

    if not enable_overlap or not batch.is_spec_v2:
        return False
    if not get_spec_algorithm_policy(batch.spec_algorithm).is_dvr():
        return False

    future_map.resolve_seq_lens_cpu(batch)
    keep_indices = [
        i
        for i, req in enumerate(batch.reqs)
        if not req.finished()
        and not _dvr_is_finished_by_published_seq_len(batch=batch, req_index=i)
    ]
    batch.filter_batch(
        keep_indices=keep_indices,
        v1_spec_info_filtered=v1_spec_info_filtered,
    )
    return True


def _dvr_is_finished_by_published_seq_len(*, batch: Any, req_index: int) -> bool:
    req = batch.reqs[req_index]
    max_new_tokens = req.sampling_params.max_new_tokens
    if max_new_tokens is None:
        return False
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens <= 0:
        return False

    if batch.seq_lens_cpu is not None:
        seq_len = int(batch.seq_lens_cpu[req_index].item())
    elif batch.seq_lens is not None:
        seq_len = int(batch.seq_lens[req_index].item())
    else:
        return False

    # Decode seq_lens includes KV-visible generated tokens; the newest sampled
    # bonus token is materialized into Req.output_ids one result-processing step
    # later, hence the final visible token corresponds to max_new_tokens - 1.
    return seq_len - len(req.origin_input_ids) >= max_new_tokens - 1


def apply_spec_final_logprob_repairs_from_result(batch: Any, result: Any) -> None:
    """Apply spec-worker final-response logprob repairs, if any."""

    if not get_spec_algorithm_policy(batch.spec_algorithm).is_dvr():
        return

    from sglang.srt.speculative.dvr_scheduler_utils import (
        apply_dvr_final_logprob_repairs_from_result,
    )

    apply_dvr_final_logprob_repairs_from_result(batch, result)


def cache_unfinished_prefill_req_with_spec_state(
    *,
    req: Any,
    batch: Any,
    req_index: int,
    tree_cache: Any,
    enable_hisparse: bool,
    hisparse_coordinator: Any,
) -> None:
    """Cache an unfinished prefill request with any spec-owned state snapshot."""

    if get_spec_algorithm_policy(
        batch.spec_algorithm
    ).needs_mamba_radix_snapshot_for_spec_v2():
        from sglang.srt.speculative.dvr_scheduler_utils import (
            cache_unfinished_prefill_req_with_dvr_mamba_snapshot,
        )

        cache_unfinished_prefill_req_with_dvr_mamba_snapshot(
            req=req,
            batch=batch,
            req_index=req_index,
            tree_cache=tree_cache,
            enable_hisparse=enable_hisparse,
            hisparse_coordinator=hisparse_coordinator,
        )
        return

    should_cache_unfinished = not batch.decoding_reqs or req not in batch.decoding_reqs
    if not should_cache_unfinished:
        return

    maybe_cache_unfinished_req(req, tree_cache)
    if enable_hisparse:
        hisparse_coordinator.admit_request_into_staging(req)


def maybe_handle_spec_mamba_checkpoint_after_decode(
    *,
    req: Any,
    batch: Any,
    result: Any,
    req_index: int,
    tree_cache: Any,
) -> bool:
    """Handle spec-owned mamba checkpoint commit after decode, if any."""

    if not get_spec_algorithm_policy(batch.spec_algorithm).is_dvr():
        return False

    from sglang.srt.speculative.dvr_scheduler_utils import (
        maybe_handle_dvr_mamba_checkpoint_after_decode,
    )

    return maybe_handle_dvr_mamba_checkpoint_after_decode(
        req=req,
        batch=batch,
        result=result,
        req_index=req_index,
        tree_cache=tree_cache,
    )
