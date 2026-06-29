from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

_MAMBA_RADIX_CACHE_POLICY_ATTR = "_mamba_radix_cache_policy"


@dataclass
class MambaRadixInsertSnapshot:
    indices: Any
    seqlen: int


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
    policy = getattr(req, _MAMBA_RADIX_CACHE_POLICY_ATTR, None)
    if policy is None and create:
        policy = MambaRadixCachePolicy()
        setattr(req, _MAMBA_RADIX_CACHE_POLICY_ATTR, policy)
    return policy


def mark_req_skip_mamba_radix_finished_insert(req: Any) -> None:
    _get_req_mamba_radix_cache_policy(
        req, create=True
    ).skip_finished_insert = True


def set_req_mamba_radix_insert_snapshot(
    req: Any, *, indices: Any, seqlen: int
) -> None:
    _get_req_mamba_radix_cache_policy(req, create=True).insert_snapshot = (
        MambaRadixInsertSnapshot(indices=indices, seqlen=seqlen)
    )


def get_req_mamba_radix_cache_policy(req: Any) -> MambaRadixCachePolicy:
    policy = _get_req_mamba_radix_cache_policy(req)
    return policy if policy is not None else MambaRadixCachePolicy()


def clear_req_mamba_radix_insert_snapshot(req: Any) -> None:
    policy = _get_req_mamba_radix_cache_policy(req)
    if policy is not None:
        policy.insert_snapshot = None


def reset_req_mamba_radix_cache_policy(req: Any) -> None:
    if hasattr(req, _MAMBA_RADIX_CACHE_POLICY_ATTR):
        delattr(req, _MAMBA_RADIX_CACHE_POLICY_ATTR)
