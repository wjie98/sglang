from __future__ import annotations

from typing import List

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
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

        device = batch.device
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

            reqs = [task.req for task in tasks]
            input_ids = []
            out_cache_locs = []
            prefix_lens = []
            extend_lens = []
            seq_lens = []
            for task in tasks:
                token_ids = self._request_token_ids_for_replay(
                    task.req, task.boundary_seqlen
                )
                replay_token_ids = token_ids[
                    task.source_seqlen : task.boundary_seqlen
                ]
                input_ids.extend(replay_token_ids)
                out_cache_locs.append(
                    batch.req_to_token_pool.req_to_token[
                        task.req.req_pool_idx,
                        task.source_seqlen : task.boundary_seqlen,
                    ].to(torch.long)
                )
                prefix_lens.append(task.source_seqlen)
                extend_lens.append(task.boundary_seqlen - task.source_seqlen)
                seq_lens.append(task.boundary_seqlen)

            if not input_ids:
                return

            boundary_indices = ctx.state_adapter.get_boundary_indices_for_reqs(
                reqs=[task.req for task in tasks],
                track_indices=[task.boundary_track_idx for task in tasks],
                device=device,
            )
            replay_batch = ScheduleBatch(
                reqs=reqs,
                req_to_token_pool=batch.req_to_token_pool,
                token_to_kv_pool_allocator=batch.token_to_kv_pool_allocator,
                tree_cache=batch.tree_cache,
                model_config=batch.model_config,
                enable_overlap=batch.enable_overlap,
                device=batch.device,
                forward_mode=ForwardMode.EXTEND,
                input_ids=torch.tensor(input_ids, dtype=torch.int64, device=device),
                req_pool_indices=torch.tensor(
                    [req.req_pool_idx for req in reqs],
                    dtype=torch.int64,
                    device=device,
                ),
                seq_lens=torch.tensor(seq_lens, dtype=torch.int64, device=device),
                out_cache_loc=torch.cat(out_cache_locs).to(device=device),
                seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
                seq_lens_sum=sum(seq_lens),
                return_logprob=False,
                top_logprobs_nums=None,
                token_ids_logprobs=None,
                global_num_tokens=None,
                global_num_tokens_for_logprob=None,
                is_extend_in_batch=False,
                all_extend_in_batch=False,
                can_run_dp_cuda_graph=False,
                tbo_split_seq_index=None,
                global_forward_mode=None,
                extend_num_tokens=len(input_ids),
                extend_lens=extend_lens,
                prefix_lens=prefix_lens,
                extend_logprob_start_lens=prefix_lens,
                extend_input_logprob_token_ids=None,
                multimodal_inputs=[req.multimodal_inputs for req in reqs],
                encoder_cached=None,
                encoder_lens=None,
                encoder_lens_cpu=None,
                encoder_out_cache_loc=None,
                sampling_info=None,
                orig_seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
                input_embeds=None,
                ne_token_table=None,
                spec_algorithm=batch.spec_algorithm,
                spec_info=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                hicache_consumer_index=-1,
                is_prefill_only=True,
                dllm_config=batch.dllm_config,
                has_grammar=False,
                return_hidden_states_before_norm=False,
                mamba_track_indices=boundary_indices,
                mamba_track_mask=torch.ones(
                    len(tasks), dtype=torch.bool, device=device
                ),
                mamba_track_seqlens=torch.tensor(
                    seq_lens, dtype=torch.int64, device=device
                ),
            )
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
