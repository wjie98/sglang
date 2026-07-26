from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclass
class DVRStateCommitPlan:
    request_rows: torch.Tensor
    live_boundary_slots: torch.Tensor
    publish_boundary_slots: Optional[torch.Tensor]
    publish_boundary_lanes: Optional[torch.Tensor]
    accepted_tail_lens: torch.Tensor


class DVRStateLifecycle:
    """Own target boundaries and private self-draft state.

    A request's live Mamba slot has two deliberate timestamps: temporal state is
    the latest exact chunk boundary, while convolution state is the accepted
    endpoint. DVR replays the short accepted transition tail between them.
    Radix ping-pong slots remain ordinary cache-publication buffers and never
    serve as verify inputs.
    """

    def __init__(self, *, server_args, model_runner):
        self.server_args = server_args
        self.model_runner = model_runner
        self.state_adapter = None
        self.boundary_seq_lens = None
        self.published_boundary_lens = None

    def bind_state_adapter(self, state_adapter) -> None:
        self.state_adapter = state_adapter
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

        request_capacity = state_adapter.draft_state.shape[1]
        device = state_adapter.draft_state.device
        self.boundary_seq_lens = torch.full(
            (request_capacity,), -1, dtype=torch.int64, device=device
        )
        if not self.server_args.disable_radix_cache:
            track_count = int(
                self.model_runner.req_to_token_pool.mamba_ping_pong_track_buffer_size
            )
            self.published_boundary_lens = torch.full(
                (request_capacity, track_count),
                -1,
                dtype=torch.int64,
                device=device,
            )

    @property
    def chunk_size(self) -> int:
        if self.state_adapter is None:
            raise RuntimeError("DVR linear-state adapter is not initialized.")
        return self.state_adapter.chunk_size

    def clear_cache_state(self) -> None:
        if self.boundary_seq_lens is not None:
            self.boundary_seq_lens.fill_(-1)
        if self.published_boundary_lens is not None:
            self.published_boundary_lens.fill_(-1)

    def prepare_target_extend(self, batch: ScheduleBatch) -> None:
        if self.state_adapter is None or not batch.reqs:
            return
        if any(int(prefix_len) % self.chunk_size for prefix_len in batch.prefix_lens):
            raise ValueError(
                "DVR GDN target EXTEND must start from an exact chunk boundary."
            )
        request_rows = batch.req_pool_indices.to(
            device=self.boundary_seq_lens.device, dtype=torch.long
        )
        self.boundary_seq_lens.index_fill_(0, request_rows, -1)
        if self.published_boundary_lens is not None:
            self.published_boundary_lens.index_fill_(0, request_rows, -1)

    def prepare_for_cache_release(self, req) -> None:
        if (
            self.state_adapter is None
            or req.req_pool_idx is None
        ):
            return
        request_row = int(req.req_pool_idx)
        try:
            if self.published_boundary_lens is None or req.skip_radix_cache_insert:
                return

            committed_len = req._cache_commit_len()
            if req.finished_reason is not None:
                visible_kv_len = (
                    len(req.origin_input_ids) + len(req.output_ids_through_stop) - 1
                )
                committed_len = min(committed_len, max(visible_kv_len, 0))
            publish_len = committed_len // self.chunk_size * self.chunk_size
            candidates = [
                (length, lane)
                for lane, length in enumerate(
                    self.published_boundary_lens[request_row].tolist()
                )
                if 0 < length <= publish_len
            ]
            if not candidates:
                # There is no newer complete chunk to add. An existing warm
                # prefix remains cached; an uncached partial tail is discarded.
                req.skip_radix_cache_insert = True
                return

            checkpoint_len, checkpoint_lane = max(candidates)
            req.mamba_last_track_seqlen = checkpoint_len
            if self.published_boundary_lens.shape[1] == 2:
                req.mamba_next_track_idx = 1 - checkpoint_lane
            else:
                req.mamba_next_track_idx = checkpoint_lane
        finally:
            self.boundary_seq_lens[request_row] = -1
            if self.published_boundary_lens is not None:
                self.published_boundary_lens[request_row].fill_(-1)

    def prepare_for_draft(self, batch) -> Optional[DVRStateCommitPlan]:
        if self.state_adapter is None or batch.batch_size() == 0:
            return None

        request_rows, live_boundary_slots = self.state_adapter.resolve_request_slots(
            batch=batch
        )
        accepted_tail_lens = batch.seq_lens.remainder(self.chunk_size).to(torch.long)
        expected_boundaries = batch.seq_lens - accepted_tail_lens
        torch._assert_async(
            self.boundary_seq_lens[request_rows].eq(expected_boundaries).all(),
            "DVR draft started without the latest exact recurrent boundary.",
        )

        publish_boundary_slots = None
        publish_boundary_lanes = None
        if self.published_boundary_lens is not None:
            current_lens, current_lanes = self.published_boundary_lens[
                request_rows
            ].max(dim=1)
            scheduler_lanes = torch.tensor(
                [req.mamba_next_track_idx for req in batch.reqs],
                device=request_rows.device,
                dtype=torch.int64,
            )
            if self.published_boundary_lens.shape[1] == 2:
                publish_boundary_lanes = torch.where(
                    current_lens.eq(expected_boundaries),
                    1 - current_lanes,
                    scheduler_lanes,
                )
            else:
                publish_boundary_lanes = torch.zeros_like(current_lanes)
            track_slots = batch.req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[
                batch.req_pool_indices
            ]
            publish_boundary_slots = track_slots.gather(
                1, publish_boundary_lanes.unsqueeze(1)
            ).squeeze(1)

        return DVRStateCommitPlan(
            request_rows=request_rows,
            live_boundary_slots=live_boundary_slots,
            publish_boundary_slots=publish_boundary_slots,
            publish_boundary_lanes=publish_boundary_lanes,
            accepted_tail_lens=accepted_tail_lens,
        )

    def finish_target_extend(self, batch: ScheduleBatch) -> None:
        """Record the live boundary produced by target EXTEND."""

        if self.state_adapter is None or batch.batch_size() == 0:
            return
        request_rows, live_boundary_slots = self.state_adapter.resolve_request_slots(
            batch=batch
        )
        if batch.seq_lens_cpu is None:
            raise RuntimeError(
                "DVR linear-state target EXTEND requires seq_lens_cpu; "
                "mixed chunk scheduling must remain disabled for this model."
            )

        seq_lens = torch.tensor(
            [int(value) for value in batch.seq_lens_cpu.tolist()],
            device=live_boundary_slots.device,
            dtype=torch.int64,
        )
        prefix_lens = torch.tensor(
            [int(value) for value in batch.prefix_lens],
            device=live_boundary_slots.device,
            dtype=torch.int64,
        )
        boundary_lens = seq_lens // self.chunk_size * self.chunk_size
        zero_mask = boundary_lens == 0
        self.state_adapter.zero_boundary_state(indices=live_boundary_slots[zero_mask])
        self.boundary_seq_lens[request_rows] = boundary_lens

        if self.published_boundary_lens is not None:
            publish_lanes = []
            publish_slots = []
            for req in batch.reqs:
                lane = batch.req_to_token_pool.get_mamba_ping_pong_keep_idx(req)
                publish_lanes.append(lane)
                publish_slots.append(req.mamba_ping_pong_track_buffer[lane])
            publish_lanes = torch.tensor(
                publish_lanes, device=live_boundary_slots.device, dtype=torch.int64
            )
            publish_slots = torch.stack(publish_slots).to(
                device=live_boundary_slots.device, dtype=torch.int64
            )
            # Only a boundary created by this EXTEND has matching convolution
            # state in the tracking slot. A warm partial-tail EXTEND reuses the
            # existing Radix node and has nothing new to publish.
            publish_mask = boundary_lens > prefix_lens
            self.state_adapter.publish_boundary_state(
                source_slots=live_boundary_slots,
                destination_slots=publish_slots,
                publish_mask=publish_mask,
            )
            rows = request_rows[publish_mask]
            self.published_boundary_lens[rows, publish_lanes[publish_mask]] = (
                boundary_lens[publish_mask]
            )

        self.state_adapter.initialize_self_draft_state(
            request_rows=request_rows,
            accepted_conv_slots=live_boundary_slots,
            accepted_tail_lens=seq_lens.remainder(self.chunk_size),
        )

    def commit_verified_state(
        self,
        *,
        batch: ScheduleBatch,
        plan: Optional[DVRStateCommitPlan],
        accept_lens: torch.Tensor,
    ) -> None:
        if plan is None or accept_lens.numel() == 0:
            return

        crosses_boundary = self.state_adapter.commit_accepted_state(
            request_rows=plan.request_rows,
            accepted_conv_slots=plan.live_boundary_slots,
            publish_boundary_slots=plan.publish_boundary_slots,
            tail_lens_before=plan.accepted_tail_lens,
            accepted_token_counts=accept_lens.to(torch.long),
        )
        self.boundary_seq_lens[plan.request_rows] = torch.where(
            crosses_boundary,
            self.boundary_seq_lens[plan.request_rows] + self.chunk_size,
            self.boundary_seq_lens[plan.request_rows],
        )
        if plan.publish_boundary_lanes is not None:
            rows = plan.request_rows[crosses_boundary]
            lanes = plan.publish_boundary_lanes[crosses_boundary]
            self.published_boundary_lens[rows, lanes] = self.boundary_seq_lens[rows]
