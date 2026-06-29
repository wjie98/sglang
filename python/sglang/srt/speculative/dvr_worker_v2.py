from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.layers.utils.logprob import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dvr_scheduler_utils import (
    DVRReplayPrefixTracker,
    DVRSpecResultAux,
)
from sglang.srt.speculative.dvr_linear_state_worker import DVRSpecV2LinearStateMixin
from sglang.srt.speculative.dvr_utils import chain_speculative_sampling
from sglang.srt.speculative.dvr_worker import (
    DVRSelfDraftVerifyInput,
    DecodeVerifyRollbackWorker,
)
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
)
from sglang.srt.utils import is_cuda
from sglang.srt.utils.async_probe import maybe_detect_nan

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )


class DecodeVerifyRollbackWorkerV2(
    DVRSpecV2LinearStateMixin, DecodeVerifyRollbackWorker
):
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
        self.dvr_replay_prefix = DVRReplayPrefixTracker()

    @property
    def draft_worker(self):
        return self

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        return None

    @staticmethod
    def _require_schedule_batch_interface(batch: ScheduleBatch):
        """Validate the ScheduleBatch fields used by DVR helpers.

        Spec-v2 calls the worker with the scheduler-owned ScheduleBatch.  Earlier
        revisions patched missing fields here, but that hid ownership bugs and
        made the worker mutate scheduler state outside its normal data flow.
        """

        missing = [
            name
            for name in (
                "req_to_token_pool",
                "token_to_kv_pool_allocator",
                "model_config",
                "device",
            )
            if getattr(batch, name, None) is None
        ]
        if not callable(getattr(batch, "batch_size", None)):
            missing.append("batch_size")
        if missing:
            raise RuntimeError(
                "DVR spec-v2 requires a scheduler ScheduleBatch with fields: "
                + ", ".join(missing)
            )
        return batch

    def _draft_cache_locs_from_req_to_token(
        self, batch: ScheduleBatch
    ) -> torch.Tensor:
        offsets = batch.seq_lens.to(torch.long).unsqueeze(1) + torch.arange(
            self.num_draft_tokens, dtype=torch.long, device=batch.seq_lens.device
        ).unsqueeze(0)
        req_pool_indices = batch.req_pool_indices.to(torch.long).unsqueeze(1)
        return self.req_to_token_pool.req_to_token[req_pool_indices, offsets].reshape(
            -1
        )

    def _request_token_ids_for_replay(self, req, boundary_seqlen: int):
        return self.dvr_replay_prefix.request_token_ids(
            req,
            boundary_seqlen,
            initialize_from_req_output=True,
            error_prefix="DVR spec-v2",
        )

    def _advance_v2_replay_prefix(self, batch: ScheduleBatch, tokens_per_req) -> None:
        if batch.forward_mode.is_idle() or batch.reqs is None:
            return

        self.dvr_replay_prefix.append_batch_output_tokens(
            batch,
            tokens_per_req,
            initialize_from_req_output=True,
        )

    def forward_batch_generation(
        self, model_worker_batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        batch = self._require_schedule_batch_interface(model_worker_batch)
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch_result = self.forward_target_extend_v2(batch)
            if on_publish is not None:
                on_publish(batch_result.new_seq_lens)
            return batch_result

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
        batch_result = self.verify_v2(batch, verify_input)
        if on_publish is not None:
            on_publish(batch_result.new_seq_lens)
        return batch_result

    def forward_target_extend_v2(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        batch.capture_hidden_mode = CaptureHiddenMode.NULL
        batch_result = self.target_worker.forward_batch_generation(batch)
        batch_result.new_seq_lens = batch.seq_lens
        next_token_ids = batch_result.next_token_ids
        self._advance_v2_replay_prefix(
            batch,
            [[token_id] for token_id in next_token_ids.detach().cpu().tolist()],
        )
        topk_index = next_token_ids.to(torch.long).unsqueeze(-1)
        batch_result.next_draft_input = EagleDraftInput(
            hidden_states=self._dummy_hidden_states(
                next_token_ids.shape[0], device=next_token_ids.device
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
        return batch_result

    def _draft_preprocess_decode_v2(self, batch: ScheduleBatch):
        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        penalizer_orchestrator = batch.sampling_info.penalizer_orchestrator
        if penalizer_orchestrator is not None and penalizer_orchestrator.is_required:
            penalizer_orchestrator.cumulate_output_tokens(
                self._draft_anchor_tokens(spec_info).to(torch.int64)
            )

        batch.out_cache_loc = self._draft_cache_locs_from_req_to_token(batch)
        batch.return_hidden_states = False
        batch.mamba_track_indices = None
        batch.mamba_track_mask = None
        batch.mamba_track_seqlens = None
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

    def draft_v2(self, batch: ScheduleBatch) -> EagleVerifyInput:
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.num_draft_steps,
                self.num_draft_tokens,
            )

        seq_lens_cpu = self._batch_seq_lens_cpu_list(batch)
        replay_tasks = self.linear_state.prepare_for_draft(
            batch, seq_lens_cpu=seq_lens_cpu
        )
        if replay_tasks:
            self._replay_linear_state_boundaries(batch, replay_tasks)
            self.linear_state.restore_tail_lens_after_replay(
                batch, replay_tasks, seq_lens_cpu=seq_lens_cpu
            )
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
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
            draft_probs=draft_probs,
        )

    def verify_v2(
        self,
        batch: ScheduleBatch,
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
        linear_state_ctx = self.linear_state.restore_for_verify(
            batch,
            seq_lens_cpu=self._batch_seq_lens_cpu_list(batch),
        )
        batch.seq_lens_cpu_cache = spec_info.seq_lens_cpu
        batch_result = self.target_worker.forward_batch_generation(
            batch, is_verify=True
        )
        logits_output = batch_result.logits_output
        oracle_logits = None
        if batch.return_logprob:
            oracle_logits = self._target_suffix_extend_verify_logits(
                batch=batch,
                spec_info=spec_info,
                linear_state_ctx=linear_state_ctx,
                full_prefix_replay=True,
            )
        if oracle_logits is not None:
            logits_output.next_token_logits = oracle_logits
        maybe_detect_nan(logits_output.next_token_logits, "dvr v2 target verify")

        predict, accept_lens, accept_index = self._sample_verify_v2(
            batch, spec_info, logits_output
        )
        new_seq_lens = scheduler_seq_lens + accept_lens
        if not batch.forward_mode.is_idle() and accept_lens.numel() > 0:
            predict_cpu = predict.detach().cpu().tolist()
            accept_lens_cpu = accept_lens.detach().cpu().tolist()
            self._advance_v2_replay_prefix(
                batch,
                [
                    predict_cpu[
                        req_i * self.num_draft_tokens : req_i * self.num_draft_tokens
                        + int(accept_len)
                    ]
                    for req_i, accept_len in enumerate(accept_lens_cpu)
                ],
            )

        pending_track_indices = None
        pending_track_seqlens = None
        if linear_state_ctx is not None:
            pending_track_indices, pending_track_seqlens = (
                self._commit_linear_state_after_verify_v2(
                    batch=batch,
                    accepted_token_counts=accept_lens.to(torch.long),
                    accepted_steps=(accept_lens - 1).to(torch.long),
                    ctx=linear_state_ctx,
                )
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
            bonus_tokens=verified_id,
            topk_p=torch.ones(
                (verified_id.shape[0], self.topk),
                dtype=torch.float32,
                device=verified_id.device,
            ),
            topk_index=verified_id.to(torch.long).unsqueeze(-1),
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
            new_seq_lens=new_seq_lens,
            speculative_num_draft_tokens=self.num_draft_tokens,
            spec_aux=DVRSpecResultAux.from_pending_mamba_checkpoint_lists(
                pending_track_indices,
                pending_track_seqlens,
            ),
            routed_experts_output=batch_result.routed_experts_output,
        )

    def _sample_verify_v2(
        self,
        batch: ScheduleBatch,
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
                retrieve_index=spec_info.retrieve_index,
                retrieve_next_token=spec_info.retrieve_next_token,
                retrieve_next_sibling=spec_info.retrieve_next_sibling,
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
                retrive_index=spec_info.retrieve_index,
                retrive_next_token=spec_info.retrieve_next_token,
                retrive_next_sibling=spec_info.retrieve_next_sibling,
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

    def _compute_spec_v2_logprobs(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
    ):
        bs = len(batch.seq_lens)
        max_accept = self.num_draft_steps + 1
        device = predict.device

        compact_output_idx = self._spec_v2_compact_output_indices(
            accept_index=accept_index,
            max_accept=max_accept,
            device=device,
        ).reshape(-1)
        flat_accept_idx = accept_index.clamp_min(0).long().reshape(-1)
        gathered_logits = logits_output.next_token_logits[flat_accept_idx]

        if (
            batch.sampling_info.is_all_greedy
            or envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get()
        ):
            gathered_logprobs = torch.nn.functional.log_softmax(
                gathered_logits, dim=-1
            )
        else:
            temperatures = batch.sampling_info.temperatures[
                flat_accept_idx // self.num_draft_tokens
            ]
            gathered_logprobs = torch.nn.functional.log_softmax(
                gathered_logits / temperatures, dim=-1
            )
        gathered_logprobs.clamp_(min=torch.finfo(gathered_logprobs.dtype).min)

        # Spec v2 output processing emits DVR tokens from the compact per-req
        # prefix of `predict`, while the correct target-model logprob row is
        # identified by `accept_index`. These differ as soon as a chain rejects
        # and samples the bonus token from the residual distribution.
        accepted_token_ids = predict[compact_output_idx]
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
