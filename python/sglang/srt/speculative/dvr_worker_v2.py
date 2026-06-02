from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.layers.utils.logprob import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dvr_utils import chain_speculative_sampling
from sglang.srt.speculative.dvr_worker import DecodeVerifyRollbackWorker
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    verify_tree_greedy_func,
)
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    TREE_SPEC_KERNEL_AVAILABLE,
    generate_simulated_accept_index,
    maybe_detect_nan,
)
from sglang.srt.utils import is_cuda

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )


class DecodeVerifyRollbackWorkerV2(DecodeVerifyRollbackWorker):
    """Overlap-scheduler DVR worker.

    DVR has no standalone draft model.  The scheduler still expects a spec-v2
    worker to expose a draft-worker-like object, so this class presents itself
    as that object while routing draft work through the target model runner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speculative_num_steps = self.num_draft_steps
        self.speculative_num_draft_tokens = self.num_draft_tokens
        self.draft_runner = self.model_runner

    @property
    def draft_worker(self):
        return self

    def _ensure_model_worker_batch_compat(self, batch: ModelWorkerBatch):
        """Attach ScheduleBatch-like fields used by DVR linear-state helpers."""

        batch.req_to_token_pool = self.req_to_token_pool
        batch.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
        batch.model_config = self.model_config
        batch.device = self.device
        batch.batch_size = lambda: len(batch.reqs) if batch.reqs is not None else 0
        return batch

    def _draft_cache_locs_from_req_to_token(
        self, batch: ModelWorkerBatch
    ) -> torch.Tensor:
        offsets = batch.seq_lens.to(torch.long).unsqueeze(1) + torch.arange(
            self.num_draft_tokens, dtype=torch.long, device=batch.seq_lens.device
        ).unsqueeze(0)
        req_pool_indices = batch.req_pool_indices.to(torch.long).unsqueeze(1)
        return self.req_to_token_pool.req_to_token[req_pool_indices, offsets].reshape(
            -1
        )

    @contextmanager
    def _req_seqlen_matches_batch_prefix(self, batch: ModelWorkerBatch):
        """Make V1 DVR linear-state helpers see Spec v2's logical length.

        In overlap scheduling, output processing has not necessarily appended
        accepted tokens to ``req.output_ids`` when the next forward starts.
        The existing DVR linear-state lifecycle derives chunk boundaries from
        ``req.seqlen - 1``. Temporarily padding request output length keeps
        those helpers on the same prefix length as V1 without changing forward
        metadata or user-visible outputs.
        """

        if batch.forward_mode.is_idle() or batch.reqs is None:
            yield
            return

        spec_info = batch.spec_info
        verified_id = getattr(spec_info, "verified_id", None)
        if verified_id is None or verified_id.numel() == 0:
            yield
            return

        seq_lens_cpu = (
            batch.seq_lens_cpu.tolist()
            if batch.seq_lens_cpu is not None
            else batch.seq_lens.detach().cpu().tolist()
        )
        verified_ids = verified_id.detach().cpu().tolist()
        appended_counts = []
        for req, seq_len, token_id in zip(
            batch.reqs, seq_lens_cpu, verified_ids, strict=True
        ):
            target_req_seqlen = int(seq_len) + 1
            append_count = max(0, target_req_seqlen - req.seqlen)
            if append_count:
                req.output_ids.extend([int(token_id)] * append_count)
            appended_counts.append(append_count)
        try:
            yield
        finally:
            for req, append_count in zip(
                batch.reqs, appended_counts, strict=True
            ):
                for _ in range(append_count):
                    req.output_ids.pop()

    def forward_batch_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> GenerationBatchResult:
        batch = self._ensure_model_worker_batch_compat(model_worker_batch)
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self.forward_target_extend_v2(batch)

        if batch.spec_info is None:
            batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=0,
                dtype=self.model_config.dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

        verify_input = self.draft_v2(batch)
        batch.spec_info = verify_input
        return self.verify_v2(batch, verify_input)

    def forward_target_extend_v2(
        self, batch: ModelWorkerBatch
    ) -> GenerationBatchResult:
        batch.capture_hidden_mode = CaptureHiddenMode.NULL
        batch_result = self.target_worker.forward_batch_generation(batch)
        next_token_ids = batch_result.next_token_ids
        topk_index = next_token_ids.to(torch.long).unsqueeze(-1)
        batch_result.next_draft_input = EagleDraftInput(
            hidden_states=self._dummy_hidden_states(
                next_token_ids.shape[0], device=next_token_ids.device
            ),
            verified_id=next_token_ids,
            topk_p=torch.ones(
                (next_token_ids.shape[0], self.topk),
                dtype=torch.float32,
                device=next_token_ids.device,
            ),
            topk_index=topk_index,
            new_seq_lens=batch.seq_lens,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        return batch_result

    def _draft_preprocess_decode_v2(self, batch: ModelWorkerBatch):
        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        penalizer_orchestrator = batch.sampling_info.penalizer_orchestrator
        if penalizer_orchestrator is not None and penalizer_orchestrator.is_required:
            penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        batch.out_cache_loc = self._draft_cache_locs_from_req_to_token(batch)
        batch.return_hidden_states = False
        batch.mamba_track_indices = None
        batch.mamba_track_mask = None
        batch.mamba_track_seqlens = None
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

    def draft_v2(self, batch: ModelWorkerBatch) -> EagleVerifyInput:
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.num_draft_steps,
                self.num_draft_tokens,
            )

        with self._req_seqlen_matches_batch_prefix(batch):
            replay_tasks = self.linear_state.prepare_for_draft(batch)
            if replay_tasks:
                self._replay_linear_state_boundaries(batch, replay_tasks)
                self.linear_state.restore_tail_lens_after_replay(batch, replay_tasks)
            self.linear_state.finish_prepare_for_draft(batch)
        self._draft_preprocess_decode_v2(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

        forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        parent_list, top_scores_index, draft_tokens, draft_probs = self.draft_forward(
            forward_batch
        )

        (
            _tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
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

        return EagleVerifyInput(
            draft_token=draft_tokens.to(torch.long),
            custom_mask=None,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.num_draft_steps,
            topk=self.topk,
            draft_token_num=self.num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
            draft_probs=draft_probs,
        )

    def verify_v2(
        self,
        batch: ModelWorkerBatch,
        spec_info: EagleVerifyInput,
    ) -> GenerationBatchResult:
        if not batch.forward_mode.is_idle():
            batch.input_ids = spec_info.draft_token
        spec_info.num_tokens_per_req = self.num_draft_tokens
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.capture_hidden_mode = CaptureHiddenMode.NULL
        batch.spec_info = spec_info

        scheduler_seq_lens = batch.seq_lens
        with self._req_seqlen_matches_batch_prefix(batch):
            linear_state_ctx = self.linear_state.restore_for_verify(batch)
        batch_result = self.target_worker.forward_batch_generation(
            batch, is_verify=True
        )
        logits_output = batch_result.logits_output
        maybe_detect_nan(logits_output.next_token_logits, "dvr v2 target verify")

        predict, accept_lens, accept_index = self._sample_verify_v2(
            batch, spec_info, logits_output
        )
        new_seq_lens = scheduler_seq_lens + accept_lens

        if linear_state_ctx is not None:
            self._commit_linear_state_after_verify_v2(
                batch=batch,
                accepted_token_counts=accept_lens.to(torch.long),
                accepted_steps=(accept_lens - 1).to(torch.long),
                ctx=linear_state_ctx,
            )

        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()

        if not batch.forward_mode.is_idle() and accept_lens.numel() > 0:
            select_index = (
                torch.arange(len(batch.seq_lens), device=self.device)
                * self.num_draft_tokens
                + accept_lens.to(torch.long)
                - 1
            )
            verified_id = predict[select_index]
        else:
            verified_id = torch.empty((0,), dtype=torch.int32, device=self.device)

        if batch.return_logprob and not batch.forward_mode.is_idle():
            self._compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index
            )

        next_draft_input = EagleDraftInput(
            hidden_states=self._dummy_hidden_states(
                verified_id.shape[0], device=verified_id.device
            ),
            verified_id=verified_id,
            topk_p=torch.ones(
                (verified_id.shape[0], self.topk),
                dtype=torch.float32,
                device=verified_id.device,
            ),
            topk_index=verified_id.to(torch.long).unsqueeze(-1),
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            routed_experts_output=batch_result.routed_experts_output,
        )

    def _sample_verify_v2(
        self,
        batch: ModelWorkerBatch,
        spec_info: EagleVerifyInput,
        logits_output: LogitsProcessorOutput,
        vocab_mask: Optional[torch.Tensor] = None,
    ):
        if batch.forward_mode.is_idle():
            predict = torch.empty(0, dtype=torch.int32, device=self.device)
            accept_lens = torch.empty(0, dtype=torch.int32, device=self.device)
            accept_index = torch.empty(0, dtype=torch.int32, device=self.device)
            return predict, accept_lens, accept_index

        bs = len(batch.seq_lens)
        candidates = spec_info.draft_token.reshape(bs, self.num_draft_tokens)
        sampling_info = batch.sampling_info
        next_token_logits = logits_output.next_token_logits
        predict = torch.zeros(
            (bs * self.num_draft_tokens,), dtype=torch.int32, device=self.device
        )
        accept_index = torch.full(
            (bs, self.num_draft_steps + 1),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        accept_lens = torch.empty((bs,), dtype=torch.int32, device=self.device)

        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.num_draft_tokens,
            )

        penalizer_orchestrator = sampling_info.penalizer_orchestrator
        if penalizer_orchestrator is not None and penalizer_orchestrator.is_required:
            penalizer_orchestrator.apply(next_token_logits, repeat=self.num_draft_tokens)

        if sampling_info.logit_bias is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.logit_bias, self.num_draft_tokens, dim=0
                )
            )

        if vocab_mask is not None:
            assert spec_info.grammar is not None
            spec_info.grammar.apply_vocab_mask(
                logits=next_token_logits, vocab_mask=vocab_mask
            )

        if sampling_info.is_all_greedy or not TREE_SPEC_KERNEL_AVAILABLE:
            target_predict = torch.argmax(next_token_logits, dim=-1).reshape(
                bs, self.num_draft_tokens
            )
            predict, accept_index, accept_lens = verify_tree_greedy_func(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=accept_lens,
                candidates=candidates,
                retrive_index=spec_info.retrive_index,
                retrive_next_token=spec_info.retrive_next_token,
                retrive_next_sibling=spec_info.retrive_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )
        else:
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.num_draft_tokens, dim=0
            )
            target_probs = F.softmax(
                next_token_logits / expanded_temperature, dim=-1
            )
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.num_draft_tokens, dim=0
                ),
            )
            if sampling_info.need_top_p_sampling:
                target_probs = top_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.top_ps, self.num_draft_tokens, dim=0
                    ),
                )
            target_probs = target_probs.reshape(bs, self.num_draft_tokens, -1)

            draft_probs = spec_info.draft_probs
            if draft_probs is None:
                draft_probs = torch.zeros_like(target_probs)
            sampling_fn = (
                chain_speculative_sampling
                if spec_info.draft_probs is not None
                else tree_speculative_sampling_target_only
            )
            sampling_fn(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=accept_lens,
                candidates=candidates,
                retrive_index=spec_info.retrive_index,
                retrive_next_token=spec_info.retrive_next_token,
                retrive_next_sibling=spec_info.retrive_next_sibling,
                uniform_samples=torch.rand_like(candidates, dtype=torch.float32),
                uniform_samples_for_final_sampling=torch.rand(
                    (bs,), dtype=torch.float32, device=self.device
                ),
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=self.server_args.speculative_accept_threshold_single,
                threshold_acc=self.server_args.speculative_accept_threshold_acc,
                deterministic=True,
            )

        if SIMULATE_ACC_LEN > 0.0:
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,
                accept_length=accept_lens,
                bs=bs,
                spec_steps=self.num_draft_steps,
            )

        accept_lens.add_(1)
        return predict, accept_lens, accept_index

    def _commit_linear_state_after_verify_v2(
        self,
        *,
        batch: ModelWorkerBatch,
        accepted_token_counts: torch.Tensor,
        accepted_steps: torch.Tensor,
        ctx,
    ):
        if accepted_token_counts.numel() == 0:
            return

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
        ctx.state_adapter.commit_after_verify(
            state_cache=ctx.state_cache,
            state_input_indices=ctx.state_input_indices,
            live_indices=ctx.live_indices,
            boundary_indices=ctx.boundary_indices,
            verified_tail_lens=verified_tail_lens,
            accepted_token_counts=accepted_token_counts,
            accepted_steps=accepted_steps,
        )

        for req, verified_tail_len, accepted_token_num in zip(
            batch.reqs,
            pre_verify_tail_lens_cpu,
            accepted_token_counts_cpu,
            strict=True,
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
                req.dvr_pending_mamba_track_idx = self.linear_state.boundary_track_idx[
                    req.rid
                ]
                req.dvr_pending_mamba_track_seqlen = new_boundary_seqlen
        self.linear_state.boundary_backup = None
        self.linear_state.live_backup = None

    def _compute_spec_v2_logprobs(
        self,
        batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
    ):
        bs = len(batch.seq_lens)
        max_accept = self.num_draft_steps + 1
        device = predict.device

        flat_output_idx = self._spec_v2_compact_output_indices(
            accept_index=accept_index,
            max_accept=max_accept,
            device=device,
        ).reshape(-1)
        gathered_logits = logits_output.next_token_logits[flat_output_idx]

        if (
            batch.sampling_info.is_all_greedy
            or envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get()
        ):
            gathered_logprobs = torch.nn.functional.log_softmax(
                gathered_logits, dim=-1
            )
        else:
            temperatures = torch.repeat_interleave(
                batch.sampling_info.temperatures,
                max_accept,
                dim=0,
            )
            gathered_logprobs = torch.nn.functional.log_softmax(
                gathered_logits / temperatures, dim=-1
            )
        gathered_logprobs.clamp_(min=torch.finfo(gathered_logprobs.dtype).min)

        accepted_token_ids = predict[flat_output_idx]
        token_logprobs = gathered_logprobs[
            torch.arange(bs * max_accept, device=device),
            accepted_token_ids.long(),
        ]
        logits_output.next_token_logprobs = token_logprobs.reshape(bs, max_accept)

        if batch.top_logprobs_nums and any(x > 0 for x in batch.top_logprobs_nums):
            top_logprobs_nums_expanded = [
                num for num in batch.top_logprobs_nums for _ in range(max_accept)
            ]
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(
                gathered_logprobs, top_logprobs_nums_expanded, no_copy_to_cpu=True
            )

        if batch.token_ids_logprobs and any(
            x is not None for x in batch.token_ids_logprobs
        ):
            token_ids_logprobs_expanded = [
                ids for ids in batch.token_ids_logprobs for _ in range(max_accept)
            ]
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(
                gathered_logprobs, token_ids_logprobs_expanded, no_copy_to_cpu=True
            )

    def _spec_v2_compact_output_indices(
        self,
        *,
        accept_index: torch.Tensor,
        max_accept: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the flat predict slots consumed by Spec v2 output processing.

        EAGLE V2 gathers logprobs through tree accept indices. DVR topk=1 is a
        chain and the scheduler emits tokens by compact per-request slices of
        ``predict``.  Keep logprob rows in that same compact order so returned
        logprobs line up with output token ids.
        """

        bs = accept_index.shape[0]
        base = (
            torch.arange(bs, dtype=torch.long, device=device).unsqueeze(1)
            * self.num_draft_tokens
        )
        offsets = torch.arange(max_accept, dtype=torch.long, device=device).unsqueeze(0)
        return base + offsets
