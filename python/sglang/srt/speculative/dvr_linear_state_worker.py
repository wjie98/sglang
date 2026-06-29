from __future__ import annotations

from typing import List

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.dvr_target_replay import (
    build_boundary_replay_batch,
    build_boundary_replay_plan,
)
from sglang.srt.speculative.dvr_linear_state import DVRBoundaryReplayTask


class DVRLinearStateReplayMixin:
    """Worker helper for replaying missing DVR chunk-boundary state."""

    def _replay_linear_state_boundaries(
        self, batch: ScheduleBatch, tasks: List[DVRBoundaryReplayTask]
    ):
        if not tasks:
            return

        ctx = self.linear_state.state_context(batch)
        if ctx is None:
            return

        live_indices = torch.stack([task.live_idx for task in tasks]).to(
            device=ctx.live_indices.device, dtype=torch.long
        )
        live_backup = ctx.state_adapter.backup_recurrent_state(
            state_cache=ctx.state_cache,
            indices=live_indices,
        )

        try:
            zero_live_indices = [
                task.live_idx
                for task in tasks
                if task.source_state_indices is None or task.source_seqlen == 0
            ]
            if zero_live_indices:
                ctx.state_adapter.zero_recurrent_state(
                    state_cache=ctx.state_cache,
                    indices=torch.stack(zero_live_indices).to(
                        device=ctx.live_indices.device, dtype=torch.long
                    ),
                )

            replay_source_indices = [
                task.source_state_indices.reshape(-1)
                for task in tasks
                if task.source_state_indices is not None and task.source_seqlen > 0
            ]
            replay_live_indices = [
                task.live_idx
                for task in tasks
                if task.source_state_indices is not None and task.source_seqlen > 0
            ]
            if replay_source_indices:
                ctx.state_adapter.copy_state_indices(
                    batch=batch,
                    src_indices=torch.cat(replay_source_indices).to(
                        device=ctx.live_indices.device, dtype=torch.long
                    ),
                    dst_indices=torch.stack(replay_live_indices).to(
                        device=ctx.live_indices.device, dtype=torch.long
                    ),
                )
            zero_source_live_indices = [
                task.live_idx
                for task in tasks
                if task.source_state_indices is None and task.source_seqlen == 0
            ]
            if zero_source_live_indices:
                ctx.state_adapter.zero_recurrent_state(
                    state_cache=ctx.state_cache,
                    indices=torch.stack(zero_source_live_indices).to(
                        device=ctx.live_indices.device, dtype=torch.long
                    ),
                )

            replay_plan = build_boundary_replay_plan(
                batch=batch,
                tasks=tasks,
                state_adapter=ctx.state_adapter,
                request_token_ids_for_replay=self._request_token_ids_for_replay,
            )
            if replay_plan is None:
                return

            replay_batch = build_boundary_replay_batch(batch, replay_plan)
            forward_batch = ForwardBatch.init_new(replay_batch, self.model_runner)
            self.model_runner.forward(forward_batch)
        finally:
            ctx.state_adapter.restore_recurrent_state(
                state_cache=ctx.state_cache,
                backup=live_backup,
                indices=live_indices,
            )


class DVRSpecV2LinearStateMixin:
    """Worker helper for DVR spec-v2 linear-state commit bookkeeping."""

    @staticmethod
    def _batch_seq_lens_cpu_list(batch: ScheduleBatch) -> list[int]:
        if batch.seq_lens_cpu is not None:
            return [int(x) for x in batch.seq_lens_cpu.tolist()]
        return [int(x) for x in batch.seq_lens.detach().cpu().tolist()]

    def _commit_linear_state_after_verify_v2(
        self,
        *,
        batch: ScheduleBatch,
        accepted_token_counts: torch.Tensor,
        accepted_steps: torch.Tensor,
        ctx,
    ):
        pending_track_indices = [None] * len(batch.reqs)
        pending_track_seqlens = [None] * len(batch.reqs)
        if accepted_token_counts.numel() == 0:
            return pending_track_indices, pending_track_seqlens

        accepted_token_counts_cpu = accepted_token_counts.cpu().tolist()
        seq_lens_cpu = (
            batch.seq_lens_cpu.tolist()
            if batch.seq_lens_cpu is not None
            else batch.seq_lens.detach().cpu().tolist()
        )
        pre_verify_tail_lens_cpu = [
            int(seq_len) - self.linear_state.boundary_seqlen[req.rid]
            for req, seq_len in zip(batch.reqs, seq_lens_cpu, strict=True)
        ]
        verified_tail_lens = ctx.state_adapter.state_input_tail_lens(
            state_cache=ctx.state_cache,
            state_input_indices=ctx.state_input_indices,
        )
        if verified_tail_lens is None:
            verified_tail_lens = torch.tensor(
                pre_verify_tail_lens_cpu,
                dtype=torch.long,
                device=batch.seq_lens.device,
            )
        verified_tail_lens = verified_tail_lens.to(
            device=ctx.live_indices.device, dtype=torch.long
        )
        # The suffix oracle may have already asked the normal EXTEND tracker to
        # write the next boundary checkpoint. Let the lifecycle roll that back
        # for rejected tails or mark it as already materialized for crossings.
        boundary_already_tracked = (
            self.linear_state.prepare_suffix_replay_boundary_commit(
                ctx=ctx,
                verified_tail_lens_cpu=pre_verify_tail_lens_cpu,
                accepted_token_counts_cpu=accepted_token_counts_cpu,
            )
        )
        ctx.state_adapter.commit_after_verify(
            state_cache=ctx.state_cache,
            state_input_indices=ctx.state_input_indices,
            live_indices=ctx.live_indices,
            boundary_indices=ctx.boundary_indices,
            verified_tail_lens=verified_tail_lens,
            accepted_token_counts=accepted_token_counts,
            accepted_steps=accepted_steps,
            boundary_already_tracked=boundary_already_tracked,
        )

        for i, (req, verified_tail_len, accepted_token_num) in enumerate(
            zip(
                batch.reqs,
                pre_verify_tail_lens_cpu,
                accepted_token_counts_cpu,
                strict=True,
            )
        ):
            if verified_tail_len + accepted_token_num >= FLA_CHUNK_SIZE:
                new_boundary_seqlen = (
                    self.linear_state.boundary_seqlen[req.rid] + FLA_CHUNK_SIZE
                )
                self.linear_state.boundary_seqlen[req.rid] = new_boundary_seqlen
                # Overlap scheduling processes accepted output tokens after the
                # worker returns. Keep the newly written boundary state pending
                # until scheduler output processing has materialized those
                # tokens in req.output_ids, otherwise radix cache can observe a
                # checkpoint beyond the committed token prefix.
                pending_track_indices[i] = self.linear_state.boundary_track_idx[
                    req.rid
                ]
                pending_track_seqlens[i] = new_boundary_seqlen
        self.linear_state.boundary_backup = None
        self.linear_state.live_backup = None
        self.linear_state.suffix_replay_boundary_track_mask = None
        return pending_track_indices, pending_track_seqlens
