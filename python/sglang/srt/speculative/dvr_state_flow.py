from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclass
class DVRLinearStateContext:
    state_cache: Any
    state_input_indices: torch.Tensor
    live_indices: torch.Tensor
    boundary_indices: torch.Tensor
    next_boundary_indices: torch.Tensor
    boundary_track_indices: torch.Tensor


class DVRLinearStateLifecycle:
    """Own DVR's target boundary and self-draft rollback state.

    The recurrent boundary itself stays in SGLang's existing Mamba state pools:
    an aligned live state is preferred, otherwise the latest tracked state is
    reused. A radix miss is rebuilt by the ordinary match-prefix + EXTEND path;
    DVR does not maintain a second temporal checkpoint store.
    """

    def __init__(self, *, server_args, model_runner):
        self.server_args = server_args
        self.model_runner = model_runner
        self._state_adapter = None
        self.draft_state_backup = None
        self.boundary_lens = None

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
        tail_lens = state_adapter.state_input_window().tail_lens
        track_count = int(
            self.model_runner.req_to_token_pool.mamba_ping_pong_track_buffer_size
        )
        self.boundary_lens = tail_lens.new_full(
            (tail_lens.numel(), track_count), -1, dtype=torch.int64
        )
        if state_adapter.draft_reuses_target_state:
            state_cache = self.model_runner.req_to_token_pool.get_speculative_mamba2_params_all_layers()
            # Self draft mutates only the target's live convolution state before
            # verify. Keep one request-pool-indexed copy; recurrent boundaries
            # remain in the existing ping-pong pool.
            self.draft_state_backup = state_adapter.allocate_draft_state_backup(
                state_cache=state_cache,
                backup_size=int(
                    self.model_runner.req_to_token_pool.req_to_token.shape[0]
                ),
            )

    @property
    def chunk_size(self) -> int:
        if self._state_adapter is None:
            raise RuntimeError("DVR linear-state adapter is not initialized.")
        return self._state_adapter.chunk_size

    def clear_cache_state(self) -> None:
        if self.boundary_lens is not None:
            self.boundary_lens.fill_(-1)

    def prepare_target_extend(self, batch: ScheduleBatch) -> None:
        """Track DVR's request-local chunk boundary without enabling Radix."""

        if self._state_adapter is None or not batch.reqs:
            return

        state_input_indices = batch.req_pool_indices.to(
            device=self.boundary_lens.device, dtype=torch.long
        )
        self.boundary_lens.index_fill_(0, state_input_indices, -1)
        if not self.server_args.disable_radix_cache:
            return

        track_indices = []
        track_mask = []
        track_seqlens = []
        for req in batch.reqs:
            if req.mamba_ping_pong_track_buffer is None:
                raise RuntimeError(
                    f"DVR request {req.rid} has no recurrent checkpoint slot."
                )
            track_indices.append(req.mamba_ping_pong_track_buffer[0])
            should_track = req.extend_input_len >= self.chunk_size
            track_mask.append(should_track)
            track_seqlen = len(req.prefix_indices) + req.extend_input_len
            track_seqlens.append(track_seqlen if should_track else -1)
            if should_track:
                req.mamba_last_track_seqlen = (
                    len(req.prefix_indices)
                    + req.extend_input_len // self.chunk_size * self.chunk_size
                )
        device = batch.device
        batch.mamba_track_indices = torch.stack(track_indices).to(
            device=device, dtype=torch.int64
        )
        batch.mamba_track_mask = torch.tensor(
            track_mask, device=device, dtype=torch.bool
        )
        batch.mamba_track_seqlens = torch.tensor(
            track_seqlens, device=device, dtype=torch.int64
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
            track_lens = [
                int(length) for length in self.boundary_lens[request_slot].tolist()
            ]
            committed_len = int(req.kv_committed_len)
            if self.server_args.strip_thinking_cache and req.reasoning_tokens > 0:
                committed_len = min(committed_len, len(req.origin_input_ids))
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
                if publish_len == 0:
                    req.skip_radix_cache_insert = True
                    return
                raise RuntimeError(
                    "DVR lost the recurrent checkpoint for a committed prefix: "
                    f"rid={req.rid}, publish={publish_len}, physical={track_lens}."
                )
            checkpoint_len, checkpoint_track = max(candidates)
            req.mamba_last_track_seqlen = checkpoint_len
            req.mamba_next_track_idx = (
                self.model_runner.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    checkpoint_track
                )
            )
        finally:
            self.boundary_lens[request_slot].fill_(-1)

    def prepare_for_draft(self, batch) -> Optional[DVRLinearStateContext]:
        if self._state_adapter is None or batch.batch_size() == 0:
            return None

        state_cache, state_input_indices, live_indices = (
            self._state_adapter.batch_state(batch=batch)
        )
        track_slots = batch.req_to_token_pool.get_mamba_ping_pong_slots(
            batch.req_pool_indices
        ).to(device=live_indices.device, dtype=torch.long)
        boundary_track_indices = self.boundary_lens[state_input_indices].argmax(dim=1)
        boundary_indices = track_slots.gather(
            1, boundary_track_indices.unsqueeze(1)
        ).squeeze(1)
        next_track_indices = batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
            boundary_track_indices
        )
        next_boundary_indices = track_slots.gather(
            1, next_track_indices.unsqueeze(1)
        ).squeeze(1)
        if self._state_adapter.draft_reuses_target_state:
            if self.draft_state_backup is None:
                raise RuntimeError("DVR self draft has no live-state backup storage.")
            self.draft_state_backup = self._state_adapter.backup_draft_state(
                state_cache=state_cache,
                indices=live_indices,
                backup_indices=state_input_indices,
                out=self.draft_state_backup,
            )
        return DVRLinearStateContext(
            state_cache=state_cache,
            state_input_indices=state_input_indices,
            live_indices=live_indices,
            boundary_indices=boundary_indices,
            next_boundary_indices=next_boundary_indices,
            boundary_track_indices=boundary_track_indices,
        )

    def finish_target_extend(self, batch: ScheduleBatch) -> None:
        """Publish the exact recurrent boundary produced by target EXTEND."""

        if self._state_adapter is None or batch.batch_size() == 0:
            return
        state_cache, state_input_indices, live_indices = (
            self._state_adapter.batch_state(batch=batch)
        )
        seq_lens_cpu = (
            [int(x) for x in batch.seq_lens_cpu.tolist()]
            if batch.seq_lens_cpu is not None
            else [int(x) for x in batch.seq_lens.detach().cpu().tolist()]
        )
        prefix_lens = [int(x) for x in batch.prefix_lens]
        zero_indices = []
        tail_lens = []
        boundary_lens = []
        boundary_tracks = []
        for i, req in enumerate(batch.reqs):
            seq_len = seq_lens_cpu[i]
            boundary_len = seq_len // self.chunk_size * self.chunk_size
            track_idx = req.mamba_next_track_idx
            if track_idx is None or req.mamba_ping_pong_track_buffer is None:
                raise RuntimeError(
                    f"DVR request {req.rid} has no recurrent checkpoint slot."
                )
            boundary_index = req.mamba_ping_pong_track_buffer[track_idx]

            if boundary_len == seq_len and boundary_len > 0:
                batch.req_to_token_pool.mamba_pool.copy_from(
                    live_indices[i].reshape(1), boundary_index.reshape(1)
                )
            elif req.mamba_last_track_seqlen == boundary_len and boundary_len > 0:
                keep_idx = batch.req_to_token_pool.get_mamba_ping_pong_keep_idx(req)
                source = req.mamba_ping_pong_track_buffer[keep_idx]
                if int(keep_idx) != int(track_idx):
                    batch.req_to_token_pool.mamba_pool.copy_from(
                        source.reshape(1), boundary_index.reshape(1)
                    )
            elif boundary_len == 0:
                zero_indices.append(boundary_index)
            elif self.server_args.disable_radix_cache or prefix_lens[i] != boundary_len:
                raise RuntimeError(
                    "DVR target EXTEND did not restore the latest chunk boundary: "
                    f"rid={req.rid}, boundary={boundary_len}, "
                    f"last_track={req.mamba_last_track_seqlen}, "
                    f"prefix={prefix_lens[i]}."
                )
            # Otherwise GDN captured the radix-owned prefix boundary before
            # target EXTEND consumed the unclosed tail.

            tail_lens.append(seq_len - boundary_len)
            boundary_lens.append(boundary_len)
            boundary_tracks.append(track_idx)

        if zero_indices:
            self._state_adapter.zero_recurrent_state(
                state_cache=state_cache,
                indices=torch.stack(zero_indices).to(
                    device=live_indices.device, dtype=torch.long
                ),
            )
        boundary_tracks = torch.tensor(
            boundary_tracks, device=live_indices.device, dtype=torch.int64
        )
        self.boundary_lens[state_input_indices, boundary_tracks] = torch.tensor(
            boundary_lens, device=live_indices.device, dtype=torch.int64
        )
        self._state_adapter.state_input_window().set_tail_lens(
            indices=state_input_indices,
            value=torch.tensor(tail_lens, device=live_indices.device),
        )

    def restore_for_verify(self, ctx: Optional[DVRLinearStateContext]) -> None:
        if ctx is None:
            return
        if (
            self._state_adapter.draft_reuses_target_state
            and self.draft_state_backup is None
        ):
            raise RuntimeError("DVR self draft is missing live-state backups.")
        self._state_adapter.prepare_recurrent_state_for_verify(
            state_cache=ctx.state_cache,
            live_indices=ctx.live_indices,
            boundary_indices=ctx.boundary_indices,
            draft_state_backup=self.draft_state_backup,
            backup_indices=ctx.state_input_indices,
        )

    def rollback_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        ctx: Optional[DVRLinearStateContext],
        accept_lens: torch.Tensor,
    ) -> None:
        if ctx is None:
            return
        if accept_lens.numel() > 0:
            crosses_chunk_boundary = self._state_adapter.commit_after_verify(
                state_cache=ctx.state_cache,
                state_input_indices=ctx.state_input_indices,
                live_indices=ctx.live_indices,
                boundary_indices=ctx.boundary_indices,
                next_boundary_indices=ctx.next_boundary_indices,
                accepted_token_counts=accept_lens.to(torch.long),
            )
            next_tracks = batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                ctx.boundary_track_indices
            )
            current_lens = self.boundary_lens[
                ctx.state_input_indices, ctx.boundary_track_indices
            ]
            next_lens = self.boundary_lens[ctx.state_input_indices, next_tracks]
            self.boundary_lens[ctx.state_input_indices, next_tracks] = torch.where(
                crosses_chunk_boundary,
                current_lens + self.chunk_size,
                next_lens,
            )
