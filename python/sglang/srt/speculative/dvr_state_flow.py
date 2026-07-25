from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch


def copy_cached_prefix_boundary(
    *, forward_batch, req_to_token_pool, chunk_size: int
) -> None:
    """Copy a warm Radix boundary into DVR's request-owned keep slot.

    This runs eagerly after deferred Mamba COW and before target EXTEND enters a
    prefill graph. Keeping the whole-pool copy outside model layers avoids
    capturing data-dependent indexing from the dummy prefill batch.
    """

    if (
        not forward_batch.forward_mode.is_extend()
        or forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend_v2()
        or forward_batch.mamba_track_indices is None
        or forward_batch.extend_prefix_lens is None
    ):
        return

    batch_size = forward_batch._original_batch_size or forward_batch.batch_size
    prefix_lens = forward_batch.extend_prefix_lens[:batch_size]
    seq_lens = forward_batch.seq_lens[:batch_size]
    copy_mask = (prefix_lens > 0) & (prefix_lens == seq_lens // chunk_size * chunk_size)
    if forward_batch.mamba_track_mask is not None:
        copy_mask &= ~forward_batch.mamba_track_mask[:batch_size]

    translate = req_to_token_pool.translate_mamba_indices
    source = translate(
        req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices[:batch_size])
    )
    next_slots = forward_batch.mamba_track_indices[:batch_size]
    track_slots = req_to_token_pool.get_mamba_ping_pong_slots(
        forward_batch.req_pool_indices[:batch_size]
    )
    if track_slots.shape[1] == 2:
        destination = torch.where(
            track_slots[:, 0] == next_slots, track_slots[:, 1], track_slots[:, 0]
        )
    else:
        destination = next_slots
    destination = translate(destination)
    req_to_token_pool.mamba_pool.copy_from(
        source[copy_mask].to(torch.long), destination[copy_mask].to(torch.long)
    )


@dataclass
class DVRStateCommitPlan:
    request_rows: torch.Tensor
    accepted_conv_slots: torch.Tensor
    checkpoint_slots: torch.Tensor
    next_checkpoint_slots: torch.Tensor
    checkpoint_lanes: torch.Tensor
    accepted_tail_lens: torch.Tensor


class DVRStateLifecycle:
    """Own DVR's target boundary and self-draft rollback state.

    Temporal and convolution state intentionally have different authoritative
    positions. The large temporal state is checkpointed at the latest exact
    chunk boundary and replayed with the accepted transition tail. Convolution
    state is small and remains authoritative at the accepted endpoint. Self
    draft mutates a private copy of both. A radix miss is rebuilt by the ordinary
    match-prefix + EXTEND path; DVR does not maintain another checkpoint store.
    """

    def __init__(self, *, server_args, model_runner):
        self.server_args = server_args
        self.model_runner = model_runner
        self._state_adapter = None
        self.checkpoint_seq_lens = None

    def bind_state_adapter(self, state_adapter) -> None:
        self._state_adapter = state_adapter
        if state_adapter is None:
            if getattr(self.model_runner, "mambaish_config", None) is not None:
                raise RuntimeError(
                    "DVR does not support this hybrid linear-state backend: no "
                    "target state adapter was initialized."
                )
            return

        if self.server_args.mamba_track_interval != state_adapter.chunk_size:
            raise ValueError(
                "DVR linear-state verify requires mamba_track_interval to match "
                f"the adapter chunk size {state_adapter.chunk_size}, got "
                f"{self.server_args.mamba_track_interval}."
            )
        if self.server_args.mamba_cache_chunk_size != state_adapter.chunk_size:
            raise ValueError(
                "DVR linear-state verify requires mamba_cache_chunk_size to "
                f"match the adapter chunk size {state_adapter.chunk_size}, got "
                f"{self.server_args.mamba_cache_chunk_size}."
            )
        track_count = int(
            self.model_runner.req_to_token_pool.mamba_ping_pong_track_buffer_size
        )
        # Columns map directly to the request's ping-pong checkpoint slots. Keep
        # both lengths: stop/reasoning trimming may publish the older visible one.
        boundary_slots = state_adapter.verify_boundary_slots
        self.checkpoint_seq_lens = boundary_slots.new_full(
            (boundary_slots.numel(), track_count), -1
        )

    @property
    def chunk_size(self) -> int:
        if self._state_adapter is None:
            raise RuntimeError("DVR linear-state adapter is not initialized.")
        return self._state_adapter.chunk_size

    def clear_cache_state(self) -> None:
        if self.checkpoint_seq_lens is not None:
            self.checkpoint_seq_lens.fill_(-1)

    def prepare_target_extend(self, batch: ScheduleBatch) -> None:
        """Discard stale DVR ownership before target EXTEND publishes a boundary.

        ScheduleBatch owns Mamba tracking while Radix is enabled. ChunkCache
        intentionally disables that upstream path, so DVR supplies its single
        request-local checkpoint from the stable per-batch extend lengths.
        """

        if self._state_adapter is None or not batch.reqs:
            return

        request_rows = batch.req_pool_indices.to(
            device=self.checkpoint_seq_lens.device, dtype=torch.long
        )
        self.checkpoint_seq_lens.index_fill_(0, request_rows, -1)
        if not self.server_args.disable_radix_cache:
            return

        track_indices = []
        track_mask = []
        track_seqlens = []
        for req, extend_len in zip(batch.reqs, batch.extend_lens):
            if req.mamba_ping_pong_track_buffer is None:
                raise RuntimeError(
                    f"DVR request {req.rid} has no recurrent checkpoint slot."
                )
            track_indices.append(req.mamba_ping_pong_track_buffer[0])
            should_track = extend_len >= self.chunk_size
            track_mask.append(should_track)
            track_seqlen = len(req.prefix_indices) + extend_len
            track_seqlens.append(track_seqlen if should_track else -1)
            if should_track:
                req.mamba_last_track_seqlen = (
                    len(req.prefix_indices)
                    + extend_len // self.chunk_size * self.chunk_size
                )
        batch.mamba_track_indices = torch.stack(track_indices).to(
            device=batch.device, dtype=torch.int64
        )
        batch.mamba_track_mask = torch.tensor(
            track_mask, device=batch.device, dtype=torch.bool
        )
        batch.mamba_track_seqlens = torch.tensor(
            track_seqlens, device=batch.device, dtype=torch.int64
        )

    def prepare_for_cache_release(self, req) -> None:
        if self._state_adapter is None or req.req_pool_idx is None:
            return
        request_slot = int(req.req_pool_idx)
        try:
            if (
                self.server_args.disable_radix_cache
                or req.skip_radix_cache_insert
                or not self.server_args.enable_mamba_extra_buffer()
            ):
                return

            # Publish the newest exact boundary no later than the visible prefix.
            # Radix truncates to it; ordinary EXTEND rebuilds the remaining tail.
            track_lens = self.checkpoint_seq_lens[request_slot].tolist()
            committed_len = req._cache_commit_len()
            if req.finished_reason is not None:
                visible_kv_len = (
                    len(req.origin_input_ids) + len(req.output_ids_through_stop) - 1
                )
                committed_len = min(committed_len, max(visible_kv_len, 0))
            publish_len = committed_len // self.chunk_size * self.chunk_size
            candidates = [
                (length, track)
                for track, length in enumerate(track_lens)
                if 0 < length <= publish_len
            ]
            if not candidates:
                # Prefix caching is an optimization. A long stop/reasoning trim
                # may publish before both retained boundaries; discard this
                # request's uncached suffix instead of failing valid inference.
                req.skip_radix_cache_insert = True
                return
            checkpoint_len, checkpoint_track = max(candidates)
            req.mamba_last_track_seqlen = checkpoint_len
            req.mamba_next_track_idx = (
                self.model_runner.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    checkpoint_track
                )
            )
        finally:
            self.checkpoint_seq_lens[request_slot].fill_(-1)

    def prepare_for_draft(self, batch) -> Optional[DVRStateCommitPlan]:
        if self._state_adapter is None or batch.batch_size() == 0:
            return None

        request_rows, accepted_conv_slots = (
            self._state_adapter.resolve_request_slots(batch=batch)
        )
        track_slots = batch.req_to_token_pool.get_mamba_ping_pong_slots(
            batch.req_pool_indices
        ).to(device=accepted_conv_slots.device, dtype=torch.long)
        selected_checkpoint_lens, checkpoint_lanes = self.checkpoint_seq_lens[
            request_rows
        ].max(dim=1)
        accepted_tail_lens = batch.seq_lens.remainder(self.chunk_size).to(torch.long)
        torch._assert_async(
            selected_checkpoint_lens.eq(batch.seq_lens - accepted_tail_lens).all(),
            "DVR draft started without the latest exact recurrent checkpoint.",
        )
        checkpoint_slots = track_slots.gather(
            1, checkpoint_lanes.unsqueeze(1)
        ).squeeze(1)
        next_checkpoint_lanes = (
            batch.req_to_token_pool.get_mamba_ping_pong_other_idx(checkpoint_lanes)
        )
        next_checkpoint_slots = track_slots.gather(
            1, next_checkpoint_lanes.unsqueeze(1)
        ).squeeze(1)
        self._state_adapter.set_verify_boundaries(
            request_rows=request_rows,
            boundary_slots=checkpoint_slots,
        )
        return DVRStateCommitPlan(
            request_rows=request_rows,
            accepted_conv_slots=accepted_conv_slots,
            checkpoint_slots=checkpoint_slots,
            next_checkpoint_slots=next_checkpoint_slots,
            checkpoint_lanes=checkpoint_lanes,
            accepted_tail_lens=accepted_tail_lens,
        )

    def finish_target_extend(self, batch: ScheduleBatch) -> None:
        """Publish the exact recurrent boundary produced by target EXTEND."""

        if self._state_adapter is None or batch.batch_size() == 0:
            return
        request_rows, endpoint_slots = self._state_adapter.resolve_request_slots(
            batch=batch
        )
        if batch.seq_lens_cpu is None:
            raise RuntimeError(
                "DVR linear-state target EXTEND requires seq_lens_cpu; "
                "mixed chunk scheduling must remain disabled for this model."
            )
        seq_lens_cpu = [int(x) for x in batch.seq_lens_cpu.tolist()]
        prefix_lens = [int(x) for x in batch.prefix_lens]
        zero_indices = []
        boundary_lens = []
        boundary_tracks = []
        for i, req in enumerate(batch.reqs):
            seq_len = seq_lens_cpu[i]
            boundary_len = seq_len // self.chunk_size * self.chunk_size
            if (
                req.mamba_next_track_idx is None
                or req.mamba_ping_pong_track_buffer is None
            ):
                raise RuntimeError(
                    f"DVR request {req.rid} has no recurrent checkpoint slot."
                )
            keep_idx = batch.req_to_token_pool.get_mamba_ping_pong_keep_idx(req)
            boundary_index = req.mamba_ping_pong_track_buffer[keep_idx]

            if boundary_len == 0:
                zero_indices.append(boundary_index)
            elif req.mamba_last_track_seqlen != boundary_len and (
                self.server_args.disable_radix_cache or prefix_lens[i] != boundary_len
            ):
                raise RuntimeError(
                    "DVR target EXTEND did not restore the latest chunk boundary: "
                    f"rid={req.rid}, boundary={boundary_len}, "
                    f"last_track={req.mamba_last_track_seqlen}, "
                    f"prefix={prefix_lens[i]}."
                )
            # Ordinary tracking writes the latest boundary to keep. A short warm
            # EXTEND copies its Radix checkpoint there before model execution.

            boundary_lens.append(boundary_len)
            boundary_tracks.append(keep_idx)

        if zero_indices:
            self._state_adapter.zero_boundary_state(
                indices=torch.stack(zero_indices).to(
                    device=endpoint_slots.device, dtype=torch.long
                ),
            )
        boundary_tracks = torch.tensor(
            boundary_tracks, device=endpoint_slots.device, dtype=torch.int64
        )
        self.checkpoint_seq_lens[request_rows, boundary_tracks] = torch.tensor(
            boundary_lens, device=endpoint_slots.device, dtype=torch.int64
        )
        self._state_adapter.initialize_self_draft_state(
            request_rows=request_rows,
            endpoint_slots=endpoint_slots,
        )

    def commit_verified_state(
        self,
        *,
        batch: ScheduleBatch,
        plan: Optional[DVRStateCommitPlan],
        accept_lens: torch.Tensor,
    ) -> None:
        if plan is None:
            return
        if accept_lens.numel() > 0:
            crosses_chunk_boundary = self._state_adapter.commit_accepted_state(
                request_rows=plan.request_rows,
                endpoint_slots=plan.accepted_conv_slots,
                boundary_slots=plan.checkpoint_slots,
                alternate_boundary_slots=plan.next_checkpoint_slots,
                tail_lens_before=plan.accepted_tail_lens,
                accepted_token_counts=accept_lens.to(torch.long),
            )
            next_checkpoint_lanes = (
                batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    plan.checkpoint_lanes
                )
            )
            current_lens = self.checkpoint_seq_lens[
                plan.request_rows, plan.checkpoint_lanes
            ]
            next_checkpoint_lens = self.checkpoint_seq_lens[
                plan.request_rows, next_checkpoint_lanes
            ]
            self.checkpoint_seq_lens[plan.request_rows, next_checkpoint_lanes] = (
                torch.where(
                    crosses_chunk_boundary,
                    current_lens + self.chunk_size,
                    next_checkpoint_lens,
                )
            )
