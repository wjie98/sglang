from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MambaRadixInsertSnapshot:
    indices: Any
    seqlen: int


@dataclass(frozen=True)
class MambaRadixUnfinishedInsertPlan:
    cache_len: Optional[int]
    snapshot_indices: Optional[Any] = None


@dataclass
class MambaRadixCachePolicy:
    """Request-local mamba radix cache policy.

    Speculative workers may need to hand a freshly materialized mamba
    checkpoint to radix-cache insertion, but radix cache should not depend on
    a concrete speculative algorithm.  Keep the request payload generic here
    and let algorithms set it from their own scheduler helpers.
    """

    skip_finished_insert: bool = False
    insert_snapshot: Optional[MambaRadixInsertSnapshot] = None


def _get_req_mamba_radix_cache_policy(
    req: Any, *, create: bool = False
) -> Optional[MambaRadixCachePolicy]:
    policy = getattr(req, "mamba_radix_cache_policy", None)
    if policy is None and create:
        policy = MambaRadixCachePolicy()
        setattr(req, "mamba_radix_cache_policy", policy)
    return policy


def mark_req_skip_mamba_radix_finished_insert(req: Any) -> None:
    _get_req_mamba_radix_cache_policy(
        req, create=True
    ).skip_finished_insert = True


def should_insert_finished_req(req: Any, *, default_is_insert: bool) -> bool:
    """Apply request-local cache policy to a finished-request insert decision."""

    policy = _get_req_mamba_radix_cache_policy(req)
    return bool(default_is_insert and not (policy and policy.skip_finished_insert))


def set_req_mamba_radix_insert_snapshot(
    req: Any, *, indices: Any, seqlen: int
) -> None:
    _get_req_mamba_radix_cache_policy(req, create=True).insert_snapshot = (
        MambaRadixInsertSnapshot(indices=indices, seqlen=seqlen)
    )


def get_req_mamba_radix_insert_snapshot(
    req: Any,
) -> Optional[MambaRadixInsertSnapshot]:
    """Return a worker-provided checkpoint snapshot for unfinished insert."""

    policy = _get_req_mamba_radix_cache_policy(req)
    return None if policy is None else policy.insert_snapshot


def get_unfinished_insert_plan(
    req: Any,
    *,
    enable_mamba_extra_buffer: bool,
    token_count: int,
) -> MambaRadixUnfinishedInsertPlan:
    """Resolve the checkpoint source for unfinished-request cache insertion.

    Normal decoding donates the request's current mamba ping-pong slot. Spec-v2
    overlap can instead attach a freshly materialized checkpoint snapshot; keep
    that ownership decision outside the radix tree insertion mechanics.
    """

    snapshot = get_req_mamba_radix_insert_snapshot(req)
    if (
        enable_mamba_extra_buffer
        and snapshot is not None
        and snapshot.indices is not None
        and snapshot.seqlen is not None
    ):
        return MambaRadixUnfinishedInsertPlan(
            cache_len=snapshot.seqlen,
            snapshot_indices=snapshot.indices,
        )

    cache_len = (
        req.mamba_last_track_seqlen if enable_mamba_extra_buffer else token_count
    )
    return MambaRadixUnfinishedInsertPlan(cache_len=cache_len)


def clear_req_mamba_radix_insert_snapshot(req: Any) -> None:
    policy = _get_req_mamba_radix_cache_policy(req)
    if policy is not None:
        policy.insert_snapshot = None
