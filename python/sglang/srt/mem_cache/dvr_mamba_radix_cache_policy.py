from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class DVRMambaRadixInsertSnapshot:
    indices: Any
    seqlen: int


def mark_req_skip_mamba_radix_finished_insert(req: Any) -> None:
    req.mamba_radix_skip_finished_insert = True


def should_insert_finished_req(req: Any, *, default_is_insert: bool) -> bool:
    """Apply request-local cache policy to a finished-request insert decision."""

    return bool(
        default_is_insert
        and not getattr(req, "mamba_radix_skip_finished_insert", False)
    )


def set_req_mamba_radix_insert_snapshot(
    req: Any, *, indices: Any, seqlen: int
) -> None:
    req.mamba_radix_insert_snapshot = DVRMambaRadixInsertSnapshot(
        indices=indices, seqlen=seqlen
    )


def get_req_mamba_radix_insert_snapshot(
    req: Any,
) -> Optional[DVRMambaRadixInsertSnapshot]:
    """Return a worker-provided checkpoint snapshot for unfinished insert."""

    return getattr(req, "mamba_radix_insert_snapshot", None)


def get_unfinished_insert_state(
    req: Any,
    *,
    enable_mamba_extra_buffer: bool,
    token_count: int,
) -> Tuple[Optional[int], Optional[Any]]:
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
        return snapshot.seqlen, snapshot.indices

    cache_len = (
        req.mamba_last_track_seqlen if enable_mamba_extra_buffer else token_count
    )
    return cache_len, None


def clear_req_mamba_radix_insert_snapshot(req: Any) -> None:
    req.mamba_radix_insert_snapshot = None
