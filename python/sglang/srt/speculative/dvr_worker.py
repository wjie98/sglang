from __future__ import annotations

import logging
from dataclasses import fields
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.model_executor.dvr_draft_cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
    dvr_self_draft_decode_context,
)
from sglang.srt.speculative.dvr_linear_state import (
    DVRBoundaryReplayTask,
    DVRLinearStateLifecycle,
)
from sglang.srt.speculative.dvr_target_replay import (
    DVRTargetReplaySpec,
    build_suffix_draft_replay_plan,
    draft_row_logits_from_replay_hidden_states,
    linear_state_replay_context,
    target_extend_replay_batch,
)
from sglang.srt.speculative.dvr_utils import chain_speculative_sampling
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.spec_utils import select_top_k_tokens
from sglang.srt.utils import is_cuda
from sglang.srt.utils.async_probe import maybe_detect_nan

if is_cuda():
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

logger = logging.getLogger(__name__)


class DVRTargetVerifyMixin:
    """DVR target-verify CUDA graph fixups shared by DVR draft variants."""

    @classmethod
    def from_eagle_verify_input(cls, verify_input: EagleVerifyInput):
        """Preserve EAGLE draft metadata while swapping in DVR verify hooks."""

        return cls(
            **{
                field.name: getattr(verify_input, field.name)
                for field in fields(EagleVerifyInput)
            }
        )

    def prepare_cuda_graph_replay_buffers(self, graph_runner, raw_num_token: int):
        if not graph_runner.capture_forward_mode.is_target_verify():
            return
        # The generic num_token_non_padded slot is enabled only for EP. GDN DVR
        # verify also needs the raw token count to mask padded graph rows.
        graph_runner.buffers.num_token_non_padded.fill_(raw_num_token)

    def cuda_graph_metadata_context(
        self,
        *,
        model_runner,
        attn_backend,
        forward_mode,
        fallback_custom_mask=None,
    ):
        from sglang.srt.speculative.dvr_utils import (
            dvr_causal_verify_cuda_graph_metadata,
        )

        return dvr_causal_verify_cuda_graph_metadata(
            model_runner,
            attn_backend,
            forward_mode,
            self,
            fallback_custom_mask,
        )


class DVREagleVerifyInput(DVRTargetVerifyMixin, EagleVerifyInput):
    """DVR target verify with a standard EAGLE target-only sampler."""


class DVRSelfDraftVerifyInput(DVRTargetVerifyMixin, EagleVerifyInput):
    """DVR verify input with classic chain speculative sampling.

    Shared EAGLE verification is target-only. DVR self-draft records the draft
    sampling distribution and must use the chain accept/reject kernel to keep
    the generated distribution and acceptance rate aligned with the reference
    branch.
    """

    def _sampling_fn_and_draft_probs(self, target_probs: torch.Tensor, batch):
        if self.draft_probs is None:
            return super()._sampling_fn_and_draft_probs(target_probs, batch)
        return chain_speculative_sampling, self.draft_probs


DVRVerifyInput = DVRSelfDraftVerifyInput


class DVRLinearBoundaryReplayMixin:
    """Replay missing chunk-boundary state through the target EXTEND path."""

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


class DecodeVerifyRollbackWorker(DVRLinearBoundaryReplayMixin):
    """DVR speculative worker using the target model as a self draft model.

    The control flow mirrors EAGLE: self-decode draft, target verify, then
    EAGLE-compatible postprocess. Linear-state rollback/commit is delegated to a
    lifecycle helper so this worker can stay focused on speculative scheduling.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        del gpu_id, dp_rank, moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port

        if server_args.speculative_eagle_topk != 1:
            raise ValueError("DVR currently supports only chain mode with topk == 1.")
        if server_args.page_size != 1 and (
            server_args.page_size > FLA_CHUNK_SIZE
            or FLA_CHUNK_SIZE % server_args.page_size != 0
        ):
            raise ValueError(
                "DVR page_size > 1 requires page_size no larger than and "
                "aligned to FLA_CHUNK_SIZE."
            )
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config
        self.tp_rank = tp_rank
        self.device = server_args.device
        self.page_size = server_args.page_size
        self.topk = 1
        self.max_batch_size = target_worker.max_running_requests
        self.num_draft_steps = server_args.speculative_num_steps
        self.num_draft_tokens = server_args.speculative_num_draft_tokens
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        self.linear_state = DVRLinearStateLifecycle(
            server_args=server_args,
            model_runner=self.model_runner,
        )
        self.cuda_graph_runner_for_draft_decode = None
        if (
            not server_args.disable_cuda_graph
            and not server_args.disable_draft_cuda_graph
        ):
            self.cuda_graph_runner_for_draft_decode = DVRDraftDecodeCudaGraphRunner(self)

        logger.info(
            "Initialized DVR self-decode worker: num_steps=%s, num_draft_tokens=%s",
            self.num_draft_steps,
            self.num_draft_tokens,
        )

    def _dummy_hidden_states(self, num_tokens: int, device=None) -> torch.Tensor:
        """Keep EAGLE indexing contracts without storing real DVR hidden states.

        DVR self-draft only needs accepted token ids for the next decode step.
        A zero-width tensor can still be indexed by accepted token positions, but
        avoids materializing [tokens, hidden_size] activations in target verify.
        """

        return torch.empty(
            (num_tokens, 0),
            dtype=self.model_config.dtype,
            device=device or self.device,
        )

    def _draft_anchor_tokens(self, draft_input: EagleDraftInput) -> torch.Tensor:
        """Return per-request token ids that seed the next DVR self-draft step."""

        return draft_input.bonus_tokens

    def _use_dummy_draft_hidden_states(self, draft_input: EagleDraftInput):
        """Normalize DVR draft input hidden states to the zero-width placeholder."""

        if (
            draft_input.hidden_states is not None
            and draft_input.hidden_states.shape[-1] == 0
        ):
            return
        if draft_input.hidden_states is not None:
            num_tokens = draft_input.hidden_states.shape[0]
            device = draft_input.hidden_states.device
        elif self._draft_anchor_tokens(draft_input) is not None:
            anchor_tokens = self._draft_anchor_tokens(draft_input)
            num_tokens = anchor_tokens.shape[0]
            device = anchor_tokens.device
        else:
            num_tokens = 0
            device = self.device
        draft_input.hidden_states = self._dummy_hidden_states(num_tokens, device=device)

    def __getattr__(self, name):
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        return None

    # Public worker entrypoints. The shape follows EAGLE: normal extend produces
    # the first verified token, then decode-verify-rollback handles generation.

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, can_run_cuda_graph = (
                self.forward_target_extend(batch)
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_correct_drafts=0,
                can_run_cuda_graph=can_run_cuda_graph,
            )

        spec_info = self.draft(batch)
        logits_output, verify_output, can_run_cuda_graph = self.verify(batch, spec_info)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.accept_tokens,
            num_correct_drafts=sum(verify_output.num_correct_drafts_per_req_cpu),
            num_correct_drafts_per_req_cpu=verify_output.num_correct_drafts_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, bool]:
        batch.capture_hidden_mode = CaptureHiddenMode.NULL
        batch_result = self.target_worker.forward_batch_generation(batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        topk_index = next_token_ids.to(torch.long).unsqueeze(-1)
        batch.spec_info = EagleDraftInput(
            hidden_states=(
                logits_output.hidden_states
                if logits_output.hidden_states is not None
                else self._dummy_hidden_states(
                    next_token_ids.shape[0], device=next_token_ids.device
                )
            ),
            bonus_tokens=next_token_ids,
            topk_p=torch.ones(
                (next_token_ids.shape[0], self.topk),
                dtype=torch.float32,
                device=next_token_ids.device,
            ),
            topk_index=topk_index,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        return logits_output, next_token_ids, batch_result.can_run_cuda_graph

    # Self-draft path. DVR uses the target model's decode path as a draft model,
    # but keeps EAGLE-compatible tree/retrieve metadata for verification.

    def _alloc_draft_cache_locs(self, batch: ScheduleBatch) -> torch.Tensor:
        num_seqs = batch.batch_size()
        if self.page_size == 1:
            return alloc_token_slots(
                batch.tree_cache,
                num_seqs * self.num_draft_tokens * self.topk,
            )

        prefix_lens = batch.seq_lens
        prefix_lens_cpu = batch.seq_lens_cpu
        end_lens = prefix_lens + self.num_draft_tokens
        end_lens_cpu = prefix_lens_cpu + self.num_draft_tokens
        last_loc = get_last_loc(
            batch.req_to_token_pool.req_to_token,
            batch.req_pool_indices,
            prefix_lens,
        )
        return alloc_paged_token_slots_extend(
            batch.tree_cache,
            prefix_lens,
            prefix_lens_cpu,
            end_lens,
            end_lens_cpu,
            last_loc,
            num_seqs * self.num_draft_tokens * self.topk,
        )

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        batch.maybe_evict_swa()
        for req in batch.reqs:
            req.decode_batch_idx += 1

        num_seqs = batch.batch_size()
        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        if batch.sampling_info.penalizer_orchestrator.is_required:
            # Keep draft sampling close to normal autoregressive decode by
            # accounting for the anchor token before sampling following tokens.
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                self._draft_anchor_tokens(spec_info).to(torch.int64)
            )

        # Self-draft decodes directly into the slots that target verify will
        # read. Do not allocate a second verify window, or KV ownership and
        # radix-cache rollback stop matching the normal speculative layout.
        out_cache_loc = self._alloc_draft_cache_locs(batch)
        self._assign_dvr_draft_cache_locs(batch, out_cache_loc)

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
        batch.return_hidden_states = False
        batch.mamba_track_indices = None
        batch.mamba_track_mask = None
        batch.mamba_track_seqlens = None
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

    def _assign_dvr_draft_cache_locs(
        self, batch: ScheduleBatch, out_cache_loc: torch.Tensor
    ):
        """Publish DVR self-draft KV slots into req_to_token for topk=1.

        The old generic EAGLE helper had a simple branch for ``topk == 1``:
        copy the freshly allocated draft slots into each request's token table
        starting at the current decode length. DVR only supports chain mode
        with topk=1, so keeping this local avoids reviving the broader tree
        kernel and keeps ownership identical to the original DVR path.
        """

        cols = batch.seq_lens.to(torch.long).unsqueeze(1) + torch.arange(
            self.num_draft_tokens, dtype=torch.long, device=batch.seq_lens.device
        ).unsqueeze(0)
        rows = batch.req_pool_indices.to(torch.long).unsqueeze(1)
        batch.req_to_token_pool.req_to_token[rows, cols] = out_cache_loc.reshape(
            batch.batch_size(), self.num_draft_tokens
        ).to(torch.int32)

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=0,
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    @staticmethod
    def _request_token_ids_for_replay(req, boundary_seqlen: int):
        token_ids = list(req.origin_input_ids) + list(req.output_ids)
        if len(token_ids) >= boundary_seqlen:
            return token_ids

        fill_ids = getattr(req, "fill_ids", None)
        if fill_ids is not None and len(fill_ids) >= boundary_seqlen:
            return fill_ids

        raise RuntimeError(
            "DVR boundary fallback replay cannot reconstruct the verified prefix: "
            f"rid={req.rid}, available_tokens={len(token_ids)}, "
            f"boundary_seqlen={boundary_seqlen}."
        )

    def draft(self, batch: ScheduleBatch) -> EagleVerifyInput:
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.num_draft_steps,
                self.num_draft_tokens,
            )

        replay_tasks = self.linear_state.prepare_for_draft(
            batch, use_request_seqlen=True
        )
        if replay_tasks:
            self._replay_linear_state_boundaries(batch, replay_tasks)
            self.linear_state.restore_tail_lens_after_replay(
                batch, replay_tasks, use_request_seqlen=True
            )
        self.linear_state.finish_prepare_for_draft(batch)
        self._draft_preprocess_decode(batch)
        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
        batch.return_hidden_states = False

        forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        parent_list, top_scores_index, draft_tokens, draft_probs = self.draft_forward(
            forward_batch
        )

        (
            _tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            self._draft_anchor_tokens(spec_info),
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.num_draft_steps,
            self.num_draft_tokens,
            tree_mask_mode=TreeMaskMode.QLEN_ONLY,
        )
        draft_tokens = draft_tokens.to(torch.long)

        return DVRSelfDraftVerifyInput(
            draft_token=draft_tokens,
            # DVR uses topk=1 chain verify. The tree builder is still reused
            # for token order/retrieve metadata, but attention itself should
            # stay on the ordinary causal path instead of a backend-specific
            # custom tree-mask path.
            custom_mask=None,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.num_draft_steps,
            topk=self.topk,
            draft_token_num=self.num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            draft_probs=draft_probs,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        out_cache_loc = forward_batch.out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.num_draft_tokens
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.num_draft_tokens, -1
        )

        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        draft_probs_list: List[torch.Tensor] = []
        scores = None
        topk_p = None
        topk_index = self._draft_anchor_tokens(spec_info)
        empty_hidden_states = torch.empty(
            (0, 0), dtype=torch.float32, device=topk_index.device
        )

        origin_seq_lens = forward_batch.seq_lens.clone()
        origin_seq_lens_cpu = forward_batch.seq_lens_cpu.clone()
        origin_seq_lens_sum = forward_batch.seq_lens_sum
        origin_spec_info = forward_batch.spec_info
        origin_positions = forward_batch.positions.clone()
        origin_out_cache_loc = forward_batch.out_cache_loc
        forward_batch.spec_info = None

        # Run the target model as its own draft model. The loop mutates
        # ForwardBatch fields to look like one-token decode steps, so every
        # scheduler-owned field is restored before target verify starts.
        for i in range(self.num_draft_steps + 1):
            if i == 0:
                input_ids = topk_index.flatten()
            else:
                input_ids, _, scores, tree_info = select_top_k_tokens(
                    i - 1,
                    topk_p,
                    topk_index,
                    empty_hidden_states,
                    scores,
                    self.topk,
                )
                score_list.append(tree_info[0])
                token_list.append(tree_info[1])
                parents_list.append(tree_info[2])
                forward_batch.positions.add_(1)

            if i == self.num_draft_steps:
                break

            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i].contiguous()
            forward_batch.seq_lens = origin_seq_lens + i + 1
            forward_batch.seq_lens_cpu = origin_seq_lens_cpu + i + 1
            forward_batch.seq_lens_sum = (
                origin_seq_lens_sum + (i + 1) * forward_batch.batch_size
            )
            logits_output = self._draft_decode_forward(forward_batch)
            maybe_detect_nan(logits_output.next_token_logits, f"dvr draft step {i}")

            next_token_ids = self.model_runner.sample(logits_output, forward_batch)
            if not forward_batch.sampling_info.is_all_greedy:
                draft_probs_list.append(
                    self.get_draft_sampling_probs(
                        forward_batch, logits_output.next_token_logits
                    )
                )
            topk_index = next_token_ids.to(torch.long).unsqueeze(-1)
            topk_p = torch.ones(
                (topk_index.shape[0], self.topk),
                dtype=torch.float32,
                device=topk_index.device,
            )

        forward_batch.seq_lens = origin_seq_lens
        forward_batch.seq_lens_cpu = origin_seq_lens_cpu
        forward_batch.seq_lens_sum = origin_seq_lens_sum
        forward_batch.spec_info = origin_spec_info
        forward_batch.positions.copy_(origin_positions)
        forward_batch.out_cache_loc = origin_out_cache_loc

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list,
            token_list,
            parents_list,
            self.num_draft_tokens,
        )
        draft_probs = (
            torch.stack(draft_probs_list, dim=1) if draft_probs_list else None
        )
        return parent_list, top_scores_index, draft_tokens, draft_probs

    def _draft_decode_forward(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_decode is not None
            and self.cuda_graph_runner_for_draft_decode.can_run(forward_batch)
        )
        if can_cuda_graph:
            return self.cuda_graph_runner_for_draft_decode.replay(forward_batch)

        with dvr_self_draft_decode_context(
            self.model_runner,
            disable_model_runner_graph=True,
            disable_batch_invariant_ops=True,
            clear_kernel_config_caches=True,
            disable_mamba_tracking=True,
        ):
            forward_batch.can_run_dp_cuda_graph = False
            return self.model_runner.forward(forward_batch).logits_output

    def get_draft_sampling_probs(
        self, forward_batch: ForwardBatch, sampling_probs: torch.Tensor
    ) -> torch.Tensor:
        # model_runner.sample() mutates next_token_logits into the probability
        # tensor used by the sampler. Build draft_probs from that tensor so
        # logit bias, grammar/custom processors, temperature, and draft
        # sampling stay on the same distribution.
        sampling_info = forward_batch.sampling_info
        if (
            self.model_runner.sampler.use_log_softmax_logprob
            and not sampling_info.need_top_p_sampling
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_min_p_sampling
        ):
            sampling_probs = torch.softmax(sampling_probs, dim=-1)
        sampling_probs = top_k_renorm_prob(sampling_probs, sampling_info.top_ks)
        if sampling_info.need_top_p_sampling:
            sampling_probs = top_p_renorm_prob(sampling_probs, sampling_info.top_ps)
        return sampling_probs

    # Accepted-token and verify-output helpers. These intentionally stay close
    # to EAGLE's postprocess contract so scheduler/radix-cache ownership remains
    # compatible with normal speculative decoding.

    def _accepted_token_counts_and_steps(
        self,
        verify_output: EagleVerifyOutput,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        accepted_token_counts_cpu = [
            x + 1 for x in verify_output.num_correct_drafts_per_req_cpu
        ]
        accepted_token_counts = torch.tensor(
            accepted_token_counts_cpu,
            dtype=torch.long,
            device=device,
        )
        if accepted_token_counts.numel() == 0:
            return accepted_token_counts, accepted_token_counts, accepted_token_counts_cpu

        # DVR enforces topk=1 chain verify, so the accepted draft step is just
        # the last accepted token's offset in each fixed verify window.
        accepted_steps = accepted_token_counts - 1
        return accepted_token_counts, accepted_steps, accepted_token_counts_cpu

    def _select_accepted_verify_outputs(
        self,
        logits_output: LogitsProcessorOutput,
        verify_output: EagleVerifyOutput,
    ):
        logits_output.next_token_logits = logits_output.next_token_logits[
            verify_output.accept_indices
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[
                verify_output.accept_indices
            ]

    def _prepare_next_draft_after_verify(
        self, batch: ScheduleBatch, verify_output: EagleVerifyOutput
    ):
        draft_extend_input = verify_output.draft_extend_input
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        if (
            batch.forward_mode.is_idle()
            or draft_extend_input.input_ids is None
            or draft_extend_input.input_ids.numel() == 0
        ):
            batch.spec_info = EagleDraftInput.create_idle_input(
                device=batch.device,
                hidden_size=0,
                dtype=self.model_config.dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            return

        num_accept_tokens_cpu = draft_extend_input.num_accept_tokens_cpu
        num_accept_tokens = torch.tensor(
            num_accept_tokens_cpu,
            dtype=torch.long,
            device=draft_extend_input.input_ids.device,
        )
        accept_end = torch.cumsum(num_accept_tokens, dim=0) - 1
        bonus_tokens = draft_extend_input.input_ids[accept_end]
        batch.spec_info = EagleDraftInput(
            hidden_states=self._dummy_hidden_states(
                bonus_tokens.shape[0], device=bonus_tokens.device
            ),
            bonus_tokens=bonus_tokens,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        # Keep the next-draft inputs in the same compatibility shape as EAGLE.
        batch.spec_info.topk_index = bonus_tokens.to(torch.long).unsqueeze(-1)
        batch.spec_info.topk_p = torch.ones(
            (bonus_tokens.shape[0], self.topk),
            dtype=torch.float32,
            device=bonus_tokens.device,
        )

    # Target verify. DVR keeps the forward call in TARGET_VERIFY mode like EAGLE,
    # then locally adapts GDN's physical window and state restore/commit.

    def _target_suffix_extend_verify_logits(
        self,
        *,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        linear_state_ctx,
        full_prefix_replay: bool = False,
    ) -> Optional[torch.Tensor]:
        """Compute verifier logits by replaying only the deterministic suffix.

        The ordinary TARGET_VERIFY forward still runs first so DVR can commit
        accepted recurrent states.  The oracle restores the live recurrent slot
        to the chunk-boundary checkpoint, then runs an ordinary EXTEND over the
        unclosed tail plus draft tokens.  This keeps verifier logits equivalent
        to target prefill without replaying the full prefix on every step.
        """

        if batch.forward_mode.is_idle() or linear_state_ctx is None:
            return None

        live_indices = linear_state_ctx.live_indices
        boundary_indices = linear_state_ctx.boundary_indices
        assert boundary_indices is not None

        base_seq_lens_cpu = (
            spec_info.seq_lens_cpu.tolist()
            if getattr(spec_info, "seq_lens_cpu", None) is not None
            else (
                batch.seq_lens_cpu.tolist()
                if batch.seq_lens_cpu is not None
                else batch.seq_lens.detach().cpu().tolist()
            )
        )
        boundary_lens = self.linear_state.boundary_lens_for_replay(
            batch, base_seq_lens_cpu
        )
        if full_prefix_replay:
            boundary_lens = [0 for _ in boundary_lens]
        replay_plan = build_suffix_draft_replay_plan(
            batch=batch,
            base_seq_lens_cpu=base_seq_lens_cpu,
            boundary_lens=boundary_lens,
            draft_tokens=spec_info.draft_token,
            draft_cache_locs=batch.out_cache_loc,
            request_token_ids_for_replay=self._request_token_ids_for_replay,
        )
        if replay_plan is None:
            return None

        # Suffix replay is an oracle for verifier logits.  Let DVR's commit path
        # materialize chunk-boundary checkpoints instead of publishing one as a
        # side effect of this temporary EXTEND replay.
        self.linear_state.set_suffix_replay_boundary_track_mask(None)
        replay_spec = DVRTargetReplaySpec(
            input_ids=replay_plan.input_ids,
            out_cache_locs=replay_plan.out_cache_locs,
            prefix_lens=[int(x) for x in replay_plan.boundary_lens],
            extend_lens=[int(x) for x in replay_plan.extend_lens_cpu],
            final_seq_lens=replay_plan.final_seq_lens_cpu,
            extend_logprob_start_lens=[int(x) for x in replay_plan.extend_lens_cpu],
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_logprob=False,
            # Cached-prefix prefill can leave deferred Mamba COW/clear tensors
            # on ScheduleBatch after the real forward consumed its ForwardBatch
            # copy.  Replay owns the restore operation explicitly below.
            mamba_cow_src_indices=None if full_prefix_replay else boundary_indices,
            mamba_cow_dst_indices=None if full_prefix_replay else live_indices,
            mamba_clear_indices=live_indices if full_prefix_replay else None,
        )

        with (
            linear_state_replay_context(
                linear_state_ctx,
                clear_state_input_window=full_prefix_replay,
                restore_live_state=True,
            ),
            target_extend_replay_batch(batch, replay_spec),
        ):
            # The suffix EXTEND must see draft KV at absolute positions
            # base_seq_len..base_seq_len+draft.  These are the same slots the
            # real verify path owns, so publishing them in req_to_token_pool is
            # intentional even though the rest of ScheduleBatch is restored.
            batch.req_to_token_pool.write(
                (replay_plan.draft_rows, replay_plan.draft_offsets),
                replay_plan.draft_cache_locs.to(torch.int32),
            )

            oracle_output = self.target_worker.forward_batch_generation(
                batch=batch,
                is_verify=True,
            )
            forward_batch = ForwardBatch.init_new(
                batch, self.target_worker.model_runner
            )
            draft_logits, _ = draft_row_logits_from_replay_hidden_states(
                target_worker=self.target_worker,
                forward_batch=forward_batch,
                hidden_states=oracle_output.logits_output.hidden_states,
                hidden_gather_indices=replay_plan.hidden_gather_indices,
            )
            return draft_logits

    def _replay_accepted_suffix_for_partial_verify(
        self,
        *,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        verify_output: EagleVerifyOutput,
        linear_state_ctx,
        accepted_token_counts_cpu: List[int],
    ) -> Optional[torch.Tensor]:
        """Refresh live GDN state with the exact accepted-token suffix.

        The hot TARGET_VERIFY path forwards the proposed draft tokens.  When a
        row is rejected, however, SGLang commits the target-predicted token at
        that row.  Replaying only "unclosed chunk tail + accepted tokens" keeps
        DVR's live recurrent state aligned with the actual scheduler-owned
        token sequence without replaying the full prefix.
        """

        if batch.forward_mode.is_idle() or linear_state_ctx is None:
            return None

        accepted_ids = verify_output.draft_extend_input.input_ids
        if accepted_ids is None or accepted_ids.numel() == 0:
            return None
        total_accepted = sum(int(x) for x in accepted_token_counts_cpu)
        if accepted_ids.numel() != total_accepted or batch.out_cache_loc.numel() != total_accepted:
            return None

        state_adapter = linear_state_ctx.state_adapter
        state_cache = linear_state_ctx.state_cache
        live_indices = linear_state_ctx.live_indices
        boundary_indices = linear_state_ctx.boundary_indices
        assert boundary_indices is not None

        base_seq_lens_cpu = (
            spec_info.seq_lens_cpu.tolist()
            if getattr(spec_info, "seq_lens_cpu", None) is not None
            else (
                batch.seq_lens_cpu.tolist()
                if batch.seq_lens_cpu is not None
                else batch.seq_lens.detach().cpu().tolist()
            )
        )
        boundary_lens = self.linear_state.boundary_lens_for_replay(
            batch, base_seq_lens_cpu
        )
        tail_lens_cpu = [
            int(seq_len) - int(boundary)
            for seq_len, boundary in zip(
                base_seq_lens_cpu, boundary_lens, strict=True
            )
        ]
        extend_lens_cpu = [
            int(tail) + int(accepted)
            for tail, accepted in zip(
                tail_lens_cpu, accepted_token_counts_cpu, strict=True
            )
        ]
        needs_exact_replay = any(
            int(accepted) < self.num_draft_tokens
            for accepted in accepted_token_counts_cpu
        )
        if not needs_exact_replay:
            return None
        final_seq_lens_cpu = [
            int(boundary) + int(extend_len)
            for boundary, extend_len in zip(
                boundary_lens, extend_lens_cpu, strict=True
            )
        ]

        accepted_ids_cpu = accepted_ids.detach().cpu().tolist()
        accepted_cache_locs = batch.out_cache_loc.to(torch.long)
        input_ids = []
        out_cache_locs = []
        accepted_rows = []
        accepted_offsets = []
        token_offset = 0
        for req_i, (req, seq_len, boundary, tail_len, accepted_count) in enumerate(
            zip(
                batch.reqs,
                base_seq_lens_cpu,
                boundary_lens,
                tail_lens_cpu,
                accepted_token_counts_cpu,
                strict=True,
            )
        ):
            accepted_count = int(accepted_count)
            token_ids = self._request_token_ids_for_replay(req, int(seq_len))
            input_ids.extend(token_ids[int(boundary) : int(seq_len)])
            accepted_slice = accepted_ids_cpu[token_offset : token_offset + accepted_count]
            input_ids.extend(accepted_slice)
            if int(tail_len) > 0:
                out_cache_locs.append(
                    batch.req_to_token_pool.req_to_token[
                        req.req_pool_idx, int(boundary) : int(seq_len)
                    ].to(torch.long)
                )
            accepted_locs = accepted_cache_locs[
                token_offset : token_offset + accepted_count
            ]
            out_cache_locs.append(accepted_locs)
            if accepted_count > 0:
                accepted_rows.append(
                    batch.req_pool_indices[req_i]
                    .to(dtype=torch.long)
                    .repeat(accepted_count)
                )
                accepted_offsets.append(
                    torch.arange(
                        int(seq_len),
                        int(seq_len) + accepted_count,
                        dtype=torch.long,
                        device=batch.seq_lens.device,
                    )
                )
            token_offset += accepted_count

        if not input_ids:
            return None

        saved_tail_lens = state_adapter.state_input_tail_lens(
            state_cache=state_cache,
            state_input_indices=linear_state_ctx.state_input_indices,
        )
        if saved_tail_lens is not None:
            saved_tail_lens = saved_tail_lens.clone()
        saved_state_input_window = state_adapter.backup_state_input_window(
            state_cache=state_cache,
            state_input_indices=linear_state_ctx.state_input_indices,
        )

        possible_boundary_track, boundary_track_seqlens = (
            self.linear_state.suffix_replay_boundary_track_info(
                boundary_lens,
                extend_lens_cpu,
                device=batch.seq_lens.device,
            )
        )
        possible_boundary_track = torch.zeros_like(possible_boundary_track)

        saved_fields = {
            "forward_mode": batch.forward_mode,
            "global_forward_mode": batch.global_forward_mode,
            "input_ids": batch.input_ids,
            "input_embeds": batch.input_embeds,
            "replace_embeds": batch.replace_embeds,
            "replace_positions": batch.replace_positions,
            "out_cache_loc": batch.out_cache_loc,
            "seq_lens": batch.seq_lens,
            "seq_lens_cpu": batch.seq_lens_cpu,
            "seq_lens_sum": batch.seq_lens_sum,
            "prefix_lens": batch.prefix_lens,
            "extend_lens": batch.extend_lens,
            "extend_num_tokens": batch.extend_num_tokens,
            "extend_logprob_start_lens": batch.extend_logprob_start_lens,
            "is_extend_in_batch": batch.is_extend_in_batch,
            "all_extend_in_batch": batch.all_extend_in_batch,
            "spec_info": batch.spec_info,
            "capture_hidden_mode": batch.capture_hidden_mode,
            "return_hidden_states": batch.return_hidden_states,
            "return_hidden_states_before_norm": batch.return_hidden_states_before_norm,
            "return_logprob": batch.return_logprob,
            "mamba_track_indices": batch.mamba_track_indices,
            "mamba_track_mask": batch.mamba_track_mask,
            "mamba_track_seqlens": batch.mamba_track_seqlens,
            "mamba_cow_src_indices": batch.mamba_cow_src_indices,
            "mamba_cow_dst_indices": batch.mamba_cow_dst_indices,
            "mamba_clear_indices": batch.mamba_clear_indices,
        }
        try:
            batch.forward_mode = ForwardMode.EXTEND
            batch.global_forward_mode = None
            batch.input_ids = torch.tensor(
                input_ids, dtype=torch.long, device=batch.seq_lens.device
            )
            batch.input_embeds = None
            batch.replace_embeds = None
            batch.replace_positions = None
            batch.out_cache_loc = torch.cat(out_cache_locs).to(
                device=batch.seq_lens.device
            )
            batch.prefix_lens = [int(x) for x in boundary_lens]
            batch.extend_lens = [int(x) for x in extend_lens_cpu]
            batch.extend_num_tokens = len(input_ids)
            batch.extend_logprob_start_lens = [int(x) for x in extend_lens_cpu]
            batch.seq_lens = torch.tensor(
                final_seq_lens_cpu,
                dtype=torch.long,
                device=saved_fields["seq_lens"].device,
            )
            batch.seq_lens_cpu = torch.tensor(
                final_seq_lens_cpu,
                dtype=torch.long,
            )
            batch.seq_lens_sum = sum(final_seq_lens_cpu)
            batch.is_extend_in_batch = True
            batch.all_extend_in_batch = True
            batch.spec_info = None
            batch.capture_hidden_mode = CaptureHiddenMode.NULL
            batch.return_hidden_states = False
            batch.return_hidden_states_before_norm = False
            batch.return_logprob = False
            if possible_boundary_track.any():
                batch.mamba_track_indices = boundary_indices.to(
                    device=batch.seq_lens.device, dtype=torch.long
                )
                batch.mamba_track_mask = possible_boundary_track
                batch.mamba_track_seqlens = boundary_track_seqlens
                self.linear_state.set_suffix_replay_boundary_track_mask(
                    possible_boundary_track
                )
            else:
                batch.mamba_track_indices = None
                batch.mamba_track_mask = None
                batch.mamba_track_seqlens = None
                self.linear_state.set_suffix_replay_boundary_track_mask(None)
            batch.mamba_cow_src_indices = boundary_indices
            batch.mamba_cow_dst_indices = live_indices
            batch.mamba_clear_indices = None
            if accepted_rows:
                batch.req_to_token_pool.write(
                    (torch.cat(accepted_rows), torch.cat(accepted_offsets)),
                    accepted_cache_locs.to(
                        device=batch.seq_lens.device, dtype=torch.int32
                    ),
                )

            self.target_worker.forward_batch_generation(batch=batch, is_verify=True)
            return torch.ones(
                len(batch.reqs), dtype=torch.bool, device=live_indices.device
            )
        finally:
            if saved_tail_lens is not None:
                state_adapter.set_state_input_tail_lens(
                    state_cache=state_cache,
                    state_input_indices=linear_state_ctx.state_input_indices,
                    tail_lens=saved_tail_lens,
                )
            state_adapter.restore_state_input_window(
                state_cache=state_cache,
                state_input_indices=linear_state_ctx.state_input_indices,
                backup=saved_state_input_window,
            )
            for name, value in saved_fields.items():
                setattr(batch, name, value)

    def verify(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
    ) -> Tuple[LogitsProcessorOutput, EagleVerifyOutput, bool]:
        # DVR reuses the cache locations populated by self-decode draft. This
        # matches the reference branch and avoids reallocating a second verify
        # window over the same tokens.
        if not batch.forward_mode.is_idle():
            batch.input_ids = spec_info.draft_token
        spec_info.num_tokens_per_req = self.num_draft_tokens
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info
        linear_state_ctx = self.linear_state.restore_for_verify(batch)

        batch.seq_lens_cpu_cache = spec_info.seq_lens_cpu
        batch.capture_hidden_mode = CaptureHiddenMode.NULL
        batch_result = self.target_worker.forward_batch_generation(batch, is_verify=True)
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
        oracle_logits = None
        if batch.return_logprob:
            # Strict returned logprobs are compared against a flush-cache full
            # prefill oracle.  The current Triton cached-prefix target-verify
            # path is deterministic but not bitwise identical to full prefill,
            # so only logprob-returning requests pay for a full-prefix oracle.
            oracle_logits = self._target_suffix_extend_verify_logits(
                batch=batch,
                spec_info=spec_info,
                linear_state_ctx=linear_state_ctx,
                full_prefix_replay=True,
            )
        if oracle_logits is not None:
            logits_output.next_token_logits = oracle_logits
        maybe_detect_nan(logits_output.next_token_logits, "dvr target verify")

        spec_info.hidden_states = (
            logits_output.hidden_states
            if logits_output.hidden_states is not None
            else self._dummy_hidden_states(
                spec_info.draft_token.numel(),
                device=logits_output.next_token_logits.device,
            )
        )
        verify_output: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask=None,
        )

        self._select_accepted_verify_outputs(logits_output, verify_output)
        if linear_state_ctx is not None:
            (
                accepted_token_counts,
                accepted_steps,
                accepted_token_counts_cpu,
            ) = self._accepted_token_counts_and_steps(
                verify_output, linear_state_ctx.live_indices.device
            )
            accepted_replay = None
            if not isinstance(spec_info, DVRSelfDraftVerifyInput):
                accepted_replay = self._replay_accepted_suffix_for_partial_verify(
                    batch=batch,
                    spec_info=spec_info,
                    verify_output=verify_output,
                    linear_state_ctx=linear_state_ctx,
                    accepted_token_counts_cpu=accepted_token_counts_cpu,
                )
            live_state_already_replayed = None
            if accepted_replay is not None:
                live_state_already_replayed = accepted_replay
            self.linear_state.commit_after_verify(
                batch=batch,
                accepted_token_counts=accepted_token_counts,
                accepted_steps=accepted_steps,
                accepted_token_counts_cpu=accepted_token_counts_cpu,
                ctx=linear_state_ctx,
                live_state_already_replayed=live_state_already_replayed,
                use_fast_self_draft_commit=isinstance(
                    spec_info, DVRSelfDraftVerifyInput
                ),
            )
        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, verify_output, logits_output)
        self.postprocess_for_verify(batch, verify_output)
        return logits_output, verify_output, can_run_cuda_graph

    def postprocess_for_verify(
        self, batch: ScheduleBatch, verify_output: EagleVerifyOutput
    ):
        self._prepare_next_draft_after_verify(batch, verify_output)
