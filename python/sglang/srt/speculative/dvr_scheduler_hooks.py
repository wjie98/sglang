from __future__ import annotations

from typing import Any

from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.speculative.spec_policy import get_spec_algorithm_policy


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
