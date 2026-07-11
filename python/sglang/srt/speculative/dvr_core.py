from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping, Optional

import torch


DVRMambaCheckpoint = tuple[int, int]


@dataclass
class DVRRollbackActions:
    """DVR work deferred until the scheduler materializes verified tokens."""

    pending_mamba_checkpoints: Optional[list[Optional[DVRMambaCheckpoint]]] = None

    def cache_prefill_after_rollback(
        self,
        *,
        req: Any,
        batch: Any,
        req_index: int,
        tree_cache: Any,
        enable_hisparse: bool,
        hisparse_coordinator: Any,
    ) -> bool:
        should_cache_unfinished = (
            not batch.decoding_reqs or req not in batch.decoding_reqs
        )
        is_dvr_spec_v2 = batch.spec_algorithm.is_dvr_eagle() or getattr(
            batch, "enable_overlap", False
        )
        if is_dvr_spec_v2 and not should_cache_unfinished:
            scheduled_extend_len = (
                batch.extend_lens[req_index]
                if batch.extend_lens is not None
                else req.extend_input_len
            )
            should_cache_unfinished = scheduled_extend_len > 1

        if not should_cache_unfinished:
            return False

        from sglang.srt.mem_cache.common import maybe_cache_unfinished_req

        maybe_cache_unfinished_req(req, tree_cache)
        if enable_hisparse:
            hisparse_coordinator.admit_request_into_staging(req)
        return True

    def commit_checkpoint_after_decode(
        self,
        *,
        req: Any,
        batch: Any,
        req_index: int,
        tree_cache: Any,
    ) -> bool:
        """Commit a DVR-owned decode checkpoint after its tokens are visible."""

        if not (
            batch.spec_algorithm.is_dvr_eagle()
            or getattr(batch, "enable_overlap", False)
        ):
            return False

        if self.pending_mamba_checkpoints is None:
            raise RuntimeError("DVR decode result is missing Mamba checkpoint actions.")
        if req_index >= len(self.pending_mamba_checkpoints):
            raise RuntimeError(
                "DVR Mamba checkpoint actions do not match the request batch: "
                f"req_index={req_index}, actions={len(self.pending_mamba_checkpoints)}."
            )
        checkpoint = self.pending_mamba_checkpoints[req_index]
        if checkpoint is None:
            return True
        track_idx, seqlen = checkpoint
        if seqlen <= 0:
            raise RuntimeError(f"DVR produced invalid Mamba checkpoint length {seqlen}.")

        last_track_seqlen = getattr(req, "mamba_last_track_seqlen", None)
        if last_track_seqlen is not None and seqlen <= last_track_seqlen:
            return True

        materialized_len = len(req.origin_input_ids) + len(req.output_ids)
        if seqlen > materialized_len:
            raise RuntimeError(
                "DVR Mamba checkpoint precedes output materialization: "
                f"checkpoint={seqlen}, materialized={materialized_len}."
            )

        buffer = getattr(req, "mamba_ping_pong_track_buffer", None)
        if buffer is None or track_idx < 0 or track_idx >= buffer.numel():
            raise RuntimeError(
                "DVR Mamba checkpoint references an invalid tracking slot: "
                f"track_idx={track_idx}, slots={0 if buffer is None else buffer.numel()}."
            )
        if buffer[track_idx].item() == -1:
            raise RuntimeError(
                f"DVR Mamba checkpoint tracking slot {track_idx} is unallocated."
            )

        page_size = getattr(tree_cache, "page_size", 1)
        if page_size != 1 and seqlen % page_size != 0:
            raise RuntimeError(
                "DVR Mamba checkpoint is not page aligned: "
                f"checkpoint={seqlen}, page_size={page_size}."
            )

        req.mamba_last_track_seqlen = seqlen
        req.mamba_next_track_idx = batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
            track_idx
        )
        return True


def _dvr_output_stream(
    journal: MutableMapping[Any, list[int]],
    req: Any,
    *,
    error_prefix: str,
) -> list[int]:
    stream = journal.setdefault(req, [])
    output_ids = list(req.output_ids)
    if not output_ids:
        return stream

    common_len = min(len(stream), len(output_ids))
    if stream[:common_len] != output_ids[:common_len]:
        raise RuntimeError(
            f"{error_prefix} diverged from materialized output ids: "
            f"rid={req.rid}, tracked_tail={stream[-8:]}, "
            f"req_tail={output_ids[-8:]}, tracked_len={len(stream)}, "
            f"req_output_len={len(output_ids)}."
        )
    stream.extend(int(token_id) for token_id in output_ids[len(stream) :])
    return stream


def request_dvr_output_prefix_token_ids(
    journal: MutableMapping[Any, list[int]],
    req: Any,
    seq_len: int,
    *,
    error_prefix: str,
) -> list[int]:
    """Return a target-owned token prefix before overlap materializes outputs."""

    origin_input_ids = list(req.origin_input_ids)
    output_len = seq_len - len(origin_input_ids)
    if output_len <= 0:
        return origin_input_ids[:seq_len]

    stream = _dvr_output_stream(journal, req, error_prefix=error_prefix)
    if len(stream) >= output_len:
        return origin_input_ids + stream[:output_len]
    raise RuntimeError(
        f"{error_prefix} replay prefix is not yet owned by DVR: "
        f"rid={req.rid}, origin_tokens={len(req.origin_input_ids)}, "
        f"req_output_tokens={len(req.output_ids)}, "
        f"tracked_output_tokens={len(stream)}, seq_len={seq_len}."
    )


def append_dvr_batch_output_tokens(
    journal: MutableMapping[Any, list[int]],
    batch: Any,
    tokens_per_req,
    *,
    base_seq_lens_cpu: Optional[list[int]] = None,
    error_prefix: str = "DVR output prefix",
) -> None:
    """Advance the target-owned output journal for active requests."""

    if batch.reqs is None:
        journal.clear()
        return
    active_reqs = set(batch.reqs)
    for req in list(journal):
        if req not in active_reqs:
            journal.pop(req, None)

    if base_seq_lens_cpu is None and getattr(batch, "seq_lens", None) is not None:
        base_seq_lens_cpu = (
            batch.seq_lens_cpu.tolist()
            if getattr(batch, "seq_lens_cpu", None) is not None
            else batch.seq_lens.detach().cpu().tolist()
        )
    if base_seq_lens_cpu is None:
        base_seq_lens_cpu = [None] * len(batch.reqs)

    for req, base_seq_len, token_ids in zip(
        batch.reqs, base_seq_lens_cpu, tokens_per_req, strict=True
    ):
        stream = _dvr_output_stream(journal, req, error_prefix=error_prefix)
        if base_seq_len is not None:
            required_len = max(0, int(base_seq_len) - len(req.origin_input_ids))
            if len(stream) < required_len:
                raise RuntimeError(
                    f"{error_prefix} is behind the batch logical length: "
                    f"rid={req.rid}, tracked={len(stream)}, required={required_len}."
                )
        stream.extend(int(token_id) for token_id in token_ids)


def compact_dvr_output_rows(
    *,
    output_tokens: torch.Tensor,
    accept_lens,
    tokens_per_req: Optional[int] = None,
) -> tuple[list[int], list[list[int]]]:
    """Return accepted output rows in scheduler materialization order."""

    if torch.is_tensor(accept_lens):
        accept_lens_cpu = [int(x) for x in accept_lens.detach().cpu().tolist()]
    else:
        accept_lens_cpu = [int(x) for x in accept_lens]
    token_ids = output_tokens.detach().cpu().reshape(-1).tolist()

    token_ids_per_req = []
    if tokens_per_req is None:
        offset = 0
        for accept_len in accept_lens_cpu:
            end = offset + accept_len
            token_ids_per_req.append([int(x) for x in token_ids[offset:end]])
            offset = end
    else:
        for req_i, accept_len in enumerate(accept_lens_cpu):
            start = req_i * tokens_per_req
            end = start + accept_len
            token_ids_per_req.append([int(x) for x in token_ids[start:end]])
    return accept_lens_cpu, token_ids_per_req


def compact_dvr_accepted_input_tokens_and_cache_locs(
    *,
    batch: Any,
    verify_input_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    num_draft_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return model-committed verify-input tokens and cache slots."""

    valid_accept = torch.arange(
        num_draft_tokens, dtype=torch.long, device=accept_lens.device
    ).unsqueeze(0) < accept_lens.to(torch.long).unsqueeze(1)
    bs = accept_lens.shape[0]
    compact_input_indices = (
        torch.arange(bs, dtype=torch.long, device=accept_lens.device).unsqueeze(1)
        * int(num_draft_tokens)
        + torch.arange(
            num_draft_tokens, dtype=torch.long, device=accept_lens.device
        ).unsqueeze(0)
    )
    verify_input_tokens = verify_input_tokens.reshape(-1)
    return (
        verify_input_tokens[compact_input_indices[valid_accept]],
        batch.out_cache_loc[compact_input_indices[valid_accept]],
    )


def rollback_dvr_verify(
    *,
    batch: Any,
    linear_state: Any,
    linear_state_ctx: Any,
    accept_lens: torch.Tensor,
    accept_lens_cpu: Optional[list[int]],
    base_seq_lens_cpu: list[int],
    num_draft_tokens: int,
    accepted_input_tokens: Optional[torch.Tensor] = None,
    rollback_replay_kwargs: Optional[dict[str, Any]] = None,
    use_fast_self_draft_commit: bool = False,
) -> DVRRollbackActions:
    """Rollback target linear state after one verified speculative step."""

    if accept_lens_cpu is None:
        accept_lens_cpu = accept_lens.detach().cpu().tolist()

    pending_track_indices = None
    pending_track_seqlens = None
    if linear_state_ctx is not None:
        accepted_suffix_replay = None
        if rollback_replay_kwargs is not None and torch.any(
            accept_lens < num_draft_tokens
        ).item():
            if accepted_input_tokens is None:
                raise RuntimeError(
                    "DVR rollback replay requires accepted verify-input tokens."
                )
            accepted_ids, accepted_cache_locs = (
                compact_dvr_accepted_input_tokens_and_cache_locs(
                    batch=batch,
                    verify_input_tokens=accepted_input_tokens,
                    accept_lens=accept_lens,
                    num_draft_tokens=num_draft_tokens,
                )
            )
            if accepted_ids.numel() > 0:
                accepted_suffix_replay = (
                    linear_state.rollback_live_state_with_accepted_suffix(
                        batch=batch,
                        accepted_token_counts_cpu=accept_lens_cpu,
                        accepted_ids=accepted_ids,
                        accepted_cache_locs=accepted_cache_locs,
                        **rollback_replay_kwargs,
                    )
                )

        pending_track_indices, pending_track_seqlens = linear_state.commit_after_verify(
            batch=batch,
            accepted_token_counts=accept_lens.to(torch.long),
            accepted_steps=(accept_lens - 1).to(torch.long),
            accepted_token_counts_cpu=accept_lens_cpu,
            ctx=linear_state_ctx,
            seq_lens_cpu=base_seq_lens_cpu,
            live_state_already_replayed=(
                None
                if accepted_suffix_replay is None
                else accepted_suffix_replay[0]
            ),
            accepted_suffix_replay=accepted_suffix_replay,
            use_fast_self_draft_commit=use_fast_self_draft_commit,
        )

    actions = DVRRollbackActions()
    if pending_track_indices is None and pending_track_seqlens is None:
        return actions

    if pending_track_indices is None:
        pending_track_indices = [None] * len(pending_track_seqlens)
    if pending_track_seqlens is None:
        pending_track_seqlens = [None] * len(pending_track_indices)
    actions.pending_mamba_checkpoints = [
        (
            (int(track_idx), int(seqlen))
            if track_idx is not None and seqlen is not None
            else None
        )
        for track_idx, seqlen in zip(
            pending_track_indices, pending_track_seqlens, strict=True
        )
    ] or None
    return actions
