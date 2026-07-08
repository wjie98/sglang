from __future__ import annotations

import logging
from dataclasses import dataclass, fields
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
    dvr_self_draft_eager_context,
    _min_seq_len_cpu,
)
from sglang.srt.speculative.dvr_linear_state import DVRLinearStateLifecycle
from sglang.srt.speculative.dvr_linear_state_worker import DVRLinearStateReplayMixin
from sglang.srt.speculative.dvr_logprob_repair import (
    defer_dvr_non_streaming_logprob_output_until_finish,
    score_deferred_dvr_final_logprob_repairs,
)
from sglang.srt.speculative.dvr_scheduler_utils import DVRReplayPrefixTracker
from sglang.srt.speculative.dvr_target_replay import (
    build_accepted_suffix_replay_plan,
    build_suffix_target_replay_batch,
    linear_state_replay_context,
    run_suffix_draft_replay_oracle,
    suffix_draft_replay_batch_context,
)
from sglang.srt.speculative.dvr_utils import (
    chain_speculative_sampling,
    dvr_chain_uniform_samples,
    dvr_has_graph_unsafe_short_prompt,
)
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
from sglang.srt.speculative.output_policy import (
    allow_req_non_streaming_logprob_output,
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

    def _sampling_uniforms(self, candidates: torch.Tensor, batch):
        # EAGLE target-only verification has rejection/final-sampling coins
        # outside the normal sampler. Honor request-level sampling_seed here so
        # DVR-EAGLE sync/overlap comparisons are reproducible under sampling.
        return dvr_chain_uniform_samples(candidates, batch)


@dataclass
class DVRSelfDraftVerifyInput(DVRTargetVerifyMixin, EagleVerifyInput):
    """DVR verify input with classic chain speculative sampling.

    Shared EAGLE verification is target-only. DVR self-draft records the draft
    sampling distribution and must use the chain accept/reject kernel to keep
    the generated distribution and acceptance rate aligned with the reference
    branch.
    """

    draft_probs: Optional[torch.Tensor] = None

    def _sampling_fn_and_draft_probs(self, target_probs: torch.Tensor, batch):
        if self.draft_probs is None:
            return super()._sampling_fn_and_draft_probs(target_probs, batch)
        return chain_speculative_sampling, self.draft_probs

    def _sampling_uniforms(self, candidates: torch.Tensor, batch):
        return dvr_chain_uniform_samples(candidates, batch)


class DecodeVerifyRollbackWorker(DVRLinearStateReplayMixin):
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
        self._logged_dvr_draft_graph_skip_reasons = set()

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

    def __getattr__(self, name):
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        self.linear_state.clear_cache_state()

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

    def _dvr_draft_rows_and_offsets(
        self, batch: ScheduleBatch, num_tokens: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return req_to_token rows/columns for the self-draft verify window."""

        num_tokens = self.num_draft_tokens if num_tokens is None else num_tokens
        offsets = batch.seq_lens.to(torch.long).unsqueeze(1) + torch.arange(
            num_tokens, dtype=torch.long, device=batch.seq_lens.device
        ).unsqueeze(0)
        rows = batch.req_pool_indices.to(torch.long).unsqueeze(1)
        return rows, offsets

    def _finish_dvr_draft_preprocess_decode(
        self, batch: ScheduleBatch, spec_info: EagleDraftInput
    ) -> None:
        """Apply decode metadata common to DVR self-draft v1 and v2."""

        batch.return_hidden_states = False
        batch.mamba_track_indices = None
        batch.mamba_track_mask = None
        batch.mamba_track_seqlens = None
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

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
        self._finish_dvr_draft_preprocess_decode(batch, spec_info)

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

        rows, cols = self._dvr_draft_rows_and_offsets(batch)
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

    def _prepare_dvr_draft_forward_batch(
        self, batch: ScheduleBatch, forward_batch: ForwardBatch
    ) -> None:
        skip_reason = None
        if dvr_has_graph_unsafe_short_prompt(batch):
            skip_reason = "one-token prompt GDN state-input graph boundary"
        forward_batch.dvr_draft_cuda_graph_skip_reason = skip_reason
        forward_batch.dvr_disable_draft_cuda_graph = skip_reason is not None

    def _log_dvr_draft_graph_skip_once(self, reason: str) -> None:
        if reason in self._logged_dvr_draft_graph_skip_reasons:
            return
        self._logged_dvr_draft_graph_skip_reasons.add(reason)
        logger.warning("DVR self-draft CUDA graph skipped: %s", reason)

    def _forward_target_verify_for_dvr(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        # DVR exact output logprobs are derived from the full-prefix oracle after
        # target verify. The ordinary target-verify pass only needs logits for
        # accept/reject and GDN state commit, so keep its logprob metadata path
        # disabled.
        need_return_logprob = batch.return_logprob
        disable_verify_graph = dvr_has_graph_unsafe_short_prompt(batch)
        saved_graph_runner = None
        batch.return_logprob = False
        if disable_verify_graph:
            # The same page-padded one-token prompt edge can crash TARGET_VERIFY
            # graph replay on GDN models. Keep the fallback local to DVR; normal
            # prompts and all non-DVR target paths still use the captured graph.
            saved_graph_runner = self.model_runner.graph_runner
            self.model_runner.graph_runner = None
        try:
            return self.target_worker.forward_batch_generation(batch, is_verify=True)
        finally:
            batch.return_logprob = need_return_logprob
            if disable_verify_graph:
                self.model_runner.graph_runner = saved_graph_runner

    def _build_self_draft_verify_input(
        self,
        *,
        batch: ScheduleBatch,
        spec_info: EagleDraftInput,
        parent_list,
        top_scores_index,
        draft_tokens: torch.Tensor,
        draft_probs: Optional[torch.Tensor],
        seq_lens_sum,
        seq_lens_cpu,
    ) -> DVRSelfDraftVerifyInput:
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
            draft_tokens.to(torch.long),
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.num_draft_steps,
            self.num_draft_tokens,
            tree_mask_mode=TreeMaskMode.QLEN_ONLY,
        )

        return DVRSelfDraftVerifyInput(
            draft_token=draft_tokens.to(torch.long),
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
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            draft_probs=draft_probs,
        )

    def _run_self_draft_and_build_verify_input(
        self,
        batch: ScheduleBatch,
        spec_info: EagleDraftInput,
        *,
        seq_lens_sum: Optional[int] = None,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        suppress_return_logprob: bool = False,
    ) -> EagleVerifyInput:
        """Run DVR self-draft and return the EAGLE-compatible verify input."""

        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
        batch.return_hidden_states = False

        saved_return_logprob = batch.return_logprob
        if suppress_return_logprob:
            batch.return_logprob = False
        try:
            forward_batch = ForwardBatch.init_new(batch, self.model_runner)
            self._prepare_dvr_draft_forward_batch(batch, forward_batch)
            parent_list, top_scores_index, draft_tokens, draft_probs = (
                self.draft_forward(forward_batch)
            )
        finally:
            if suppress_return_logprob:
                batch.return_logprob = saved_return_logprob

        return self._build_self_draft_verify_input(
            batch=batch,
            spec_info=spec_info,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            seq_lens_sum=(
                forward_batch.seq_lens_sum if seq_lens_sum is None else seq_lens_sum
            ),
            seq_lens_cpu=(
                forward_batch.seq_lens_cpu if seq_lens_cpu is None else seq_lens_cpu
            ),
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
        return self._run_self_draft_and_build_verify_input(batch, spec_info)

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
        graph_skip_reason = getattr(
            forward_batch, "dvr_draft_cuda_graph_skip_reason", None
        )
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_decode is not None
            and self.cuda_graph_runner_for_draft_decode.can_run(forward_batch)
        )
        if can_cuda_graph:
            return self.cuda_graph_runner_for_draft_decode.replay(forward_batch)

        min_seq_len = _min_seq_len_cpu(forward_batch)
        if graph_skip_reason is None and min_seq_len <= 2:
            graph_skip_reason = "seq_len<=2 initial GDN state-input graph boundary"

        if (
            getattr(self.model_runner, "hybrid_gdn_config", None) is not None
            and graph_skip_reason is None
        ):
            capture_bs = []
            if self.cuda_graph_runner_for_draft_decode is not None:
                capture_bs = self.cuda_graph_runner_for_draft_decode.capture_bs
            raise RuntimeError(
                "DVR self-draft decode requires the dedicated CUDA graph for "
                "gated linear-state models. The current batch cannot run it: "
                f"batch_size={forward_batch.batch_size}, "
                f"min_seq_len={min_seq_len}, capture_bs={capture_bs}. "
                "Use the default CUDA graph batch sizes or include the running "
                "batch size in --cuda-graph-bs/--cuda-graph-max-bs."
            )

        if graph_skip_reason is not None:
            self._log_dvr_draft_graph_skip_once(graph_skip_reason)

        with dvr_self_draft_eager_context(self.model_runner):
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

    def _next_self_draft_input_from_bonus_tokens(
        self, bonus_tokens: torch.Tensor
    ) -> EagleDraftInput:
        return EagleDraftInput(
            hidden_states=self._dummy_hidden_states(
                bonus_tokens.shape[0], device=bonus_tokens.device
            ),
            bonus_tokens=bonus_tokens,
            topk_p=torch.ones(
                (bonus_tokens.shape[0], self.topk),
                dtype=torch.float32,
                device=bonus_tokens.device,
            ),
            topk_index=bonus_tokens.to(torch.long).unsqueeze(-1),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

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
        batch.spec_info = self._next_self_draft_input_from_bonus_tokens(bonus_tokens)

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

        base_seq_lens_cpu = self._seq_lens_cpu_list_for_verify(batch, spec_info)
        with suffix_draft_replay_batch_context(
            batch=batch,
            linear_state=self.linear_state,
            linear_state_ctx=linear_state_ctx,
            base_seq_lens_cpu=base_seq_lens_cpu,
            draft_tokens=spec_info.draft_token,
            draft_cache_locs=batch.out_cache_loc,
            request_token_ids_for_replay=self._request_token_ids_for_replay,
            full_prefix_replay=full_prefix_replay,
            use_mamba_cow_from_boundary=True,
        ) as replay:
            if replay is None:
                return None
            replay_batch, replay_plan = replay
            draft_logits, _ = run_suffix_draft_replay_oracle(
                target_worker=self.target_worker,
                replay_batch=replay_batch,
                replay_plan=replay_plan,
            )
            return draft_logits

    def _replay_accepted_suffix_for_partial_verify(
        self,
        *,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        linear_state_ctx,
        accepted_token_counts_cpu: List[int],
        verify_output: Optional[EagleVerifyOutput] = None,
        accepted_ids: Optional[torch.Tensor] = None,
        accepted_cache_locs: Optional[torch.Tensor] = None,
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

        if accepted_ids is None:
            assert verify_output is not None
            accepted_ids = verify_output.draft_extend_input.input_ids
        if accepted_cache_locs is None:
            accepted_cache_locs = batch.out_cache_loc
        if accepted_ids is None or accepted_ids.numel() == 0:
            return None
        total_accepted = sum(int(x) for x in accepted_token_counts_cpu)
        if (
            accepted_ids.numel() != total_accepted
            or accepted_cache_locs.numel() != total_accepted
        ):
            return None

        live_indices = linear_state_ctx.live_indices
        boundary_indices = linear_state_ctx.boundary_indices
        assert boundary_indices is not None

        base_seq_lens_cpu = self._seq_lens_cpu_list_for_verify(batch, spec_info)
        boundary_lens = self.linear_state.boundary_lens_for_replay(
            batch, base_seq_lens_cpu
        )
        replay_plan = build_accepted_suffix_replay_plan(
            batch=batch,
            base_seq_lens_cpu=base_seq_lens_cpu,
            boundary_lens=boundary_lens,
            accepted_tokens=accepted_ids,
            accepted_cache_locs=accepted_cache_locs,
            accepted_token_counts_cpu=accepted_token_counts_cpu,
            num_draft_tokens=self.num_draft_tokens,
            request_token_ids_for_replay=self._request_token_ids_for_replay,
        )
        if replay_plan is None:
            return None

        # Accepted-suffix replay is a commit repair, not a checkpoint publisher.
        # It intentionally leaves the live recurrent slot updated by the replay.
        self.linear_state.set_suffix_replay_boundary_track_mask(None)
        replay_batch = build_suffix_target_replay_batch(
            batch,
            replay_plan,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            mamba_cow_src_indices=boundary_indices,
            mamba_cow_dst_indices=live_indices,
        )

        with linear_state_replay_context(
            linear_state_ctx, restore_live_state=False
        ):
            if replay_plan.append_rows is not None:
                assert replay_plan.append_offsets is not None
                replay_batch.req_to_token_pool.write(
                    (replay_plan.append_rows, replay_plan.append_offsets),
                    replay_plan.append_cache_locs.to(
                        device=replay_batch.seq_lens.device, dtype=torch.int32
                    ),
                )
            self.target_worker.forward_batch_generation(
                batch=replay_batch, is_verify=True
            )
            return torch.ones(
                len(batch.reqs), dtype=torch.bool, device=live_indices.device
            )

    @staticmethod
    def _seq_lens_cpu_list_for_verify(
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
    ) -> list[int]:
        if getattr(spec_info, "seq_lens_cpu", None) is not None:
            return [int(x) for x in spec_info.seq_lens_cpu.tolist()]
        if batch.seq_lens_cpu is not None:
            return [int(x) for x in batch.seq_lens_cpu.tolist()]
        return [int(x) for x in batch.seq_lens.detach().cpu().tolist()]

    @staticmethod
    def _compact_flat_tokens_by_accept_lens(
        flat_tokens: torch.Tensor,
        accept_lens_cpu: list[int],
        *,
        tokens_per_req: Optional[int] = None,
    ) -> list[list[int]]:
        token_ids = flat_tokens.detach().cpu().tolist()
        compact_tokens = []
        if tokens_per_req is None:
            offset = 0
            for accept_len in accept_lens_cpu:
                next_offset = offset + int(accept_len)
                compact_tokens.append([int(x) for x in token_ids[offset:next_offset]])
                offset = next_offset
        else:
            for req_i, accept_len in enumerate(accept_lens_cpu):
                start = req_i * tokens_per_req
                end = start + int(accept_len)
                compact_tokens.append([int(x) for x in token_ids[start:end]])
        return compact_tokens

    @staticmethod
    def _apply_spec_v1_final_logprob_repairs(batch: ScheduleBatch, repairs) -> None:
        if repairs is None:
            return

        for req_i, req in enumerate(batch.reqs):
            if req_i >= len(repairs):
                break
            repair = repairs[req_i]
            if repair is None or not req.return_logprob:
                continue
            output_len = len(repair.output_ids)
            if list(req.output_ids[:output_len]) != repair.output_ids:
                raise RuntimeError(
                    "DVR spec-v1 final logprob repair no longer matches "
                    f"materialized output ids: rid={req.rid}, "
                    f"repair_len={output_len}, req_output_len={len(req.output_ids)}."
                )
            req.logprob.output_token_logprobs_val[:] = repair.output_logprobs
            req.logprob.output_token_logprobs_idx[:] = repair.output_ids
            allow_req_non_streaming_logprob_output(req)

    def _repair_final_logprobs_for_spec_v1(
        self,
        *,
        batch: ScheduleBatch,
        linear_state_ctx,
        base_seq_lens_cpu: list[int],
        accept_lens_cpu: list[int],
        compact_output_token_ids_per_req: list[list[int]],
    ) -> None:
        """Repair final non-streaming DVR output logprobs with a prefill oracle."""

        if linear_state_ctx is None or not any(
            req.return_logprob and not req.stream and req.finished()
            for req in batch.reqs
        ):
            return

        repairs = score_deferred_dvr_final_logprob_repairs(
            batch=batch,
            target_worker=self.target_worker,
            replay_prefix=DVRReplayPrefixTracker(),
            linear_state_ctx=linear_state_ctx,
            base_seq_lens_cpu=base_seq_lens_cpu,
            accept_lens_cpu=accept_lens_cpu,
            compact_output_token_ids_per_req=compact_output_token_ids_per_req,
            error_prefix="DVR spec-v1 final logprob",
        )
        self._apply_spec_v1_final_logprob_repairs(batch, repairs)

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
        base_seq_lens_cpu = self._seq_lens_cpu_list_for_verify(batch, spec_info)
        if batch.return_logprob and linear_state_ctx is not None:
            # Spec-v1 may stream non-streaming chunks before final result assembly.
            # Hold them until the full-prefix repair below installs exact logprobs.
            defer_dvr_non_streaming_logprob_output_until_finish(
                batch,
                base_seq_lens_cpu=base_seq_lens_cpu,
            )

        batch.seq_lens_cpu_cache = spec_info.seq_lens_cpu
        batch.capture_hidden_mode = CaptureHiddenMode.NULL
        batch_result = self._forward_target_verify_for_dvr(batch)
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
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
        accept_lens_cpu = [
            int(num_correct) + 1
            for num_correct in verify_output.num_correct_drafts_per_req_cpu
        ]

        logits_output.next_token_logits = logits_output.next_token_logits[
            verify_output.accept_indices
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[
                verify_output.accept_indices
            ]
        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, verify_output, logits_output)
            self._repair_final_logprobs_for_spec_v1(
                batch=batch,
                linear_state_ctx=linear_state_ctx,
                base_seq_lens_cpu=base_seq_lens_cpu,
                accept_lens_cpu=accept_lens_cpu,
                compact_output_token_ids_per_req=(
                    self._compact_flat_tokens_by_accept_lens(
                        verify_output.accept_tokens,
                        accept_lens_cpu,
                    )
                ),
            )
        if linear_state_ctx is not None:
            accepted_token_counts = torch.tensor(
                accept_lens_cpu,
                dtype=torch.long,
                device=linear_state_ctx.live_indices.device,
            )
            # DVR enforces topk=1 chain verify, so the accepted draft step is
            # the last accepted token's offset in each fixed verify window.
            accepted_steps = accepted_token_counts - 1
            is_self_dvr = isinstance(spec_info, DVRSelfDraftVerifyInput)
            # return_logprob must not change the self-draft state lifecycle.
            # Exact logprobs are repaired separately; using accepted-suffix
            # replay as the commit path changes the next draft state and
            # regresses the self-DVR acceptance rate.
            live_state_already_replayed = None
            if not is_self_dvr:
                live_state_already_replayed = (
                    self._replay_accepted_suffix_for_partial_verify(
                        batch=batch,
                        spec_info=spec_info,
                        verify_output=verify_output,
                        linear_state_ctx=linear_state_ctx,
                        accepted_token_counts_cpu=accept_lens_cpu,
                    )
                )
            self.linear_state.commit_after_verify(
                batch=batch,
                accepted_token_counts=accepted_token_counts,
                accepted_steps=accepted_steps,
                accepted_token_counts_cpu=accept_lens_cpu,
                ctx=linear_state_ctx,
                live_state_already_replayed=live_state_already_replayed,
                use_fast_self_draft_commit=is_self_dvr,
            )
        self._prepare_next_draft_after_verify(batch, verify_output)
        return logits_output, verify_output, can_run_cuda_graph
