from __future__ import annotations

from typing import List

import torch

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
        live_state_already_replayed: torch.Tensor = None,
        use_fast_self_draft_commit: bool = False,
    ):
        return self.linear_state.commit_after_verify(
            batch=batch,
            accepted_token_counts=accepted_token_counts,
            accepted_steps=accepted_steps,
            accepted_token_counts_cpu=accepted_token_counts.cpu().tolist(),
            ctx=ctx,
            seq_lens_cpu=self._batch_seq_lens_cpu_list(batch),
            live_state_already_replayed=live_state_already_replayed,
            use_fast_self_draft_commit=use_fast_self_draft_commit,
            publish_boundary_checkpoint=False,
            return_pending_boundary=True,
        )
