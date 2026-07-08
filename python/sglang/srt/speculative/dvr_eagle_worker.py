from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.dvr_eagle_verify_cuda_graph_runner import (
    DVREagleTargetVerifyCudaGraphRunner,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
)
from sglang.srt.speculative.dvr_scheduler_utils import (
    DVRReplayPrefixTracker,
    DVRSpecResultAux,
    compact_output_token_rows,
)
from sglang.srt.speculative.dvr_draft_decode_context import (
    draft_decode_performance_context,
)
from sglang.srt.speculative.dvr_logprob_repair import (
    score_deferred_dvr_final_logprob_repairs,
)
from sglang.srt.speculative.dvr_target_replay import (
    run_suffix_draft_replay_oracle,
    suffix_draft_replay_batch_context,
)
from sglang.srt.speculative.dvr_linear_state import DVRLinearStateLifecycle
from sglang.srt.speculative.dvr_worker import DVREagleVerifyInput
from sglang.srt.speculative.dvr_utils import (
    dvr_has_graph_unsafe_short_prompt,
    iter_dvr_attention_backends,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_info_v2 import fill_bonus_tokens
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2, EagleDraftWorker
from sglang.srt.speculative.spec_policy import get_spec_algorithm_policy
from sglang.srt.speculative.spec_utils import (
    generate_token_bitmask,
    record_stream_each,
    record_stream_for_v2_verify,
)
from sglang.srt.utils.async_probe import maybe_detect_inf, maybe_detect_nan
from sglang.srt.utils.common import is_npu, require_gathered_buffer

_is_npu = is_npu()


class DVREagleDraftWorker(EagleDraftWorker):
    """EAGLE draft worker with DVR's provisional decode policy.

    Standard EAGLE keeps the normal deterministic/runtime settings.  DVR-EAGLE
    verifies every draft with the target model, so its draft decode path should
    use the same performance-first context as self-DVR without making the
    upstream EagleDraftWorker depend on DVR internals.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._draft_extend_selected_logits = get_spec_algorithm_policy(
            self.speculative_algorithm
        ).uses_draft_extend_selected_logits(
            topk=self.topk,
            model=self.draft_runner.model,
            is_v2=True,
            requires_gathered_buffer=require_gathered_buffer(
                self.draft_runner.server_args
            ),
        )

    def _draft_decode_context(
        self,
        *,
        graph_capture: bool = False,
        clear_kernel_config_caches: bool = False,
    ):
        return draft_decode_performance_context(
            self.draft_runner,
            graph_capture=graph_capture,
            clear_kernel_config_caches=clear_kernel_config_caches,
            attn_backends=(
                getattr(self, "draft_attn_backend", None),
                getattr(self, "draft_extend_attn_backend", None),
            ),
        )


class DecodeVerifyRollbackEagleWorkerV2(EAGLEWorkerV2):
    """EAGLE draft with DVR target verify/rollback semantics.

    The draft model and draft-extend phases stay on the standard EAGLE v2
    implementation. DVR only owns target recurrent-state lifecycle around the
    verify forward, so this path remains isolated from the self-decode draft
    worker.
    """

    draft_worker_cls = DVREagleDraftWorker

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.topk != 1:
            raise ValueError("DVR EAGLE currently supports only topk == 1.")

        self.model_runner = self.target_worker.model_runner
        self.model_config = self.target_worker.model_config
        self.linear_state = DVRLinearStateLifecycle(
            server_args=self.server_args,
            model_runner=self.model_runner,
        )
        # EAGLE/MTP has two logical token streams:
        # - verifier prefix: accepted draft/input tokens used to reconstruct the
        #   next deterministic target verify replay.
        # - client output prefix: target predictions exposed to the user, used
        #   only by final exact logprob repair.
        self.dvr_verifier_replay_prefix = DVRReplayPrefixTracker()
        self.dvr_client_output_replay_prefix = DVRReplayPrefixTracker()
        self.cuda_graph_runner_for_target_verify = None
        if not self.server_args.disable_cuda_graph and (
            self.server_args.model_impl != "mindspore"
        ):
            self.cuda_graph_runner_for_target_verify = (
                DVREagleTargetVerifyCudaGraphRunner(self)
            )

    def clear_cache_pool(self):
        super().clear_cache_pool()
        self.linear_state.clear_cache_state()
        self.dvr_verifier_replay_prefix.clear()
        self.dvr_client_output_replay_prefix.clear()

    @contextmanager
    def _target_verify_context(
        self, *, disable_cuda_graph: bool = False, prepare: bool = False
    ):
        runner = None if disable_cuda_graph else self.cuda_graph_runner_for_target_verify
        use_plan_stream = prepare and self.plan_stream is not None and runner is None
        stream_ctx = self.plan_stream_ctx if use_plan_stream else nullcontext()

        target_runner = self.target_worker.model_runner
        patch_graph_runner = (
            runner is not None or self.plan_stream is not None or disable_cuda_graph
        )
        if not patch_graph_runner:
            with stream_ctx:
                yield use_plan_stream
            return

        saved_graph_runner = target_runner.graph_runner
        # Overlap target verify previously had to hide the generic EAGLE graph
        # because its padded metadata does not understand DVR GDN state-input
        # windows.  Prefer the DVR-EAGLE graph when captured; if capture is
        # disabled, keep the old eager fallback by installing None only inside
        # this scoped target-verify window.
        # During prepare, the dedicated DVR-EAGLE graph must be prepared on the
        # compute stream; only the eager fallback keeps the old plan-stream path.
        with stream_ctx:
            target_runner.graph_runner = runner
            try:
                yield use_plan_stream
            finally:
                target_runner.graph_runner = saved_graph_runner

    def _target_verify_graph_runner_bs(self, can_run_cuda_graph: bool):
        if not can_run_cuda_graph:
            return None
        runner = (
            self.cuda_graph_runner_for_target_verify
            or self.target_worker.model_runner.graph_runner
        )
        return None if runner is None else runner.bs

    def _update_verify_buffers_to_fill_after_draft(
        self, verify_input: DVREagleVerifyInput, can_run_cuda_graph: bool
    ):
        cuda_graph_bs = self._target_verify_graph_runner_bs(can_run_cuda_graph)
        for backend in iter_dvr_attention_backends(
            self.target_worker.model_runner.attn_backend
        ):
            try:
                backend.update_verify_buffers_to_fill_after_draft(
                    verify_input, cuda_graph_bs
                )
            except NotImplementedError:
                # Hybrid wrappers only route metadata calls to their children;
                # the real full-attention backend owns this EAGLE overlap hook.
                continue

    def _request_token_ids_for_replay(self, req, boundary_seqlen: int):
        return self.dvr_verifier_replay_prefix.request_verifier_prefix_token_ids(
            req,
            boundary_seqlen,
            error_prefix="DVR EAGLE",
        )

    def _prepare_dvr_boundary_for_verify(self, batch: ScheduleBatch) -> None:
        if batch.forward_mode.is_idle():
            return

        # EAGLE/MTP draft owns its draft KV state and does not run target GDN
        # state. DVR only checkpoints/restores target recurrent state for the
        # verifier. Use batch logical lengths explicitly because overlap can run
        # the next forward before Req.output_ids has been materialized.
        seq_lens_cpu = self.linear_state.batch_seq_lens_cpu(batch)
        replay_tasks = self.linear_state.prepare_for_draft(
            batch, seq_lens_cpu=seq_lens_cpu
        )
        if replay_tasks:
            self.linear_state.replay_boundary_tasks(
                batch,
                replay_tasks,
                request_token_ids_for_replay=self._request_token_ids_for_replay,
            )
            self.linear_state.restore_tail_lens_after_replay(
                batch, replay_tasks, seq_lens_cpu=seq_lens_cpu
            )
        self.linear_state.finish_prepare_for_draft(batch)

    def _target_suffix_extend_verify_output(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: DVREagleVerifyInput,
        linear_state_ctx,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Compute verifier outputs by replaying the deterministic suffix prefill.

        DVR correctness is defined by target prefill semantics.  For hybrid GDN
        models, a plain decode oracle can still diverge after EAGLE/MTP has
        prepared speculative state.  Replaying only the unclosed chunk tail plus
        draft tokens starts from the DVR boundary checkpoint and returns the
        prefill-equivalent logits and hidden states for draft-token rows.

        EAGLE uses target hidden states to seed the next MTP draft step.  The
        hidden states must come from the same replay as the verifier logits;
        otherwise KL can remain correct while the next draft chain drifts.
        """

        if self.topk != 1 or self.linear_state.boundary_backup is None:
            return None

        base_seq_lens_cpu = self.linear_state.batch_seq_lens_cpu(batch)
        with suffix_draft_replay_batch_context(
            batch=batch,
            linear_state=self.linear_state,
            linear_state_ctx=linear_state_ctx,
            base_seq_lens_cpu=base_seq_lens_cpu,
            draft_tokens=verify_input.draft_token,
            draft_cache_locs=batch.out_cache_loc,
            request_token_ids_for_replay=self._request_token_ids_for_replay,
            restore_boundary_state=True,
        ) as replay:
            if replay is None:
                return None
            replay_batch, replay_plan = replay
            draft_logits, draft_hidden_states = run_suffix_draft_replay_oracle(
                target_worker=self.target_worker,
                replay_batch=replay_batch,
                replay_plan=replay_plan,
                use_forward_batch=True,
            )
            return draft_logits, draft_hidden_states

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None):
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch.capture_hidden_mode = target_capture_mode
            batch_output = self.target_worker.forward_batch_generation(batch)
            batch_output.new_seq_lens = batch.seq_lens
            self._prepare_dvr_boundary_for_verify(batch)
            if batch.return_logprob:
                # This target token is the first client-visible output for a
                # prefill/extend step.  In overlap it can be consumed by the
                # first verify before Req.output_ids is materialized, so the
                # DVR-EAGLE final-logprob stream must learn it here.
                if batch_output.next_token_ids is not None:
                    next_token_ids_cpu = (
                        batch_output.next_token_ids.detach().cpu().tolist()
                    )
                    self.dvr_client_output_replay_prefix.append_batch_output_tokens(
                        batch,
                        [[token_id] for token_id in next_token_ids_cpu],
                    )
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )
                return batch_output

        self.activate_step_by_batch(batch.seq_lens.shape[0])

        if batch.spec_info is None:
            capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.LAST
            )
            batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=EagleDraftInput.hidden_size_for(self.draft_worker),
                dtype=EagleDraftInput.dtype_for(self.draft_worker),
                topk=self.topk,
                capture_hidden_mode=capture_mode,
            )

        self._prepare_dvr_boundary_for_verify(batch)
        with (
            self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
        assert verify_input.is_verify_input()
        batch.spec_info = verify_input
        batch_output = self.verify(batch)
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)
        with (
            self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker._draft_extend_for_decode(batch, batch_output)

        return batch_output

    def verify(self, batch: ScheduleBatch):
        fwd_stream = torch.get_device_module(self.device).current_stream()
        verify_input = batch.spec_info
        if not isinstance(verify_input, DVREagleVerifyInput):
            verify_input = DVREagleVerifyInput.from_eagle_verify_input(verify_input)
        batch.spec_info = verify_input
        record_stream_for_v2_verify(batch, verify_input, fwd_stream)

        verify_input.num_tokens_per_req = self.speculative_num_steps + 1
        bs = len(batch.seq_lens)

        disable_verify_graph = dvr_has_graph_unsafe_short_prompt(batch)
        with self._target_verify_context(
            disable_cuda_graph=disable_verify_graph,
            prepare=True,
        ) as prepared_on_plan_stream:
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)

        if prepared_on_plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
            if (
                _is_npu
                and self.target_worker.model_runner.model_is_mrope
                and batch.spec_info is not None
                and getattr(batch.spec_info, "positions", None) is not None
                and not batch.forward_mode.is_idle()
            ):
                verify_forward_batch.compute_spec_mrope_positions(
                    self.target_worker.model_runner, batch
                )

            self._update_verify_buffers_to_fill_after_draft(
                verify_input, can_run_cuda_graph
            )

        if batch.has_grammar:
            retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrieve_next_token.shape
            ).cpu()

        base_seq_lens_cpu = self.linear_state.batch_seq_lens_cpu(batch)
        linear_state_ctx = self.linear_state.restore_for_verify(
            batch,
            seq_lens_cpu=base_seq_lens_cpu,
        )

        with self._target_verify_context(
            disable_cuda_graph=disable_verify_graph
        ):
            forward_batch_output = self.target_worker.forward_batch_generation(
                batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
            )
        logits_output = forward_batch_output.logits_output
        oracle_output = None
        if (
            linear_state_ctx is not None
            or batch.return_logprob
            or logits_output.hidden_states is None
        ):
            # EAGLE needs replay logits and replay hidden states from the same
            # suffix EXTEND so the next MTP draft seed is paired with the
            # verifier logits. For GDN models this must not depend on
            # return_logprob: TARGET_VERIFY hidden states come from the
            # state-input verify window and are not a prefill-equivalent MTP
            # seed after chunk-boundary rollback. The replay is still only the
            # unclosed tail plus draft tokens, not the full prefix.
            oracle_output = self._target_suffix_extend_verify_output(
                batch=batch,
                verify_input=verify_input,
                linear_state_ctx=linear_state_ctx,
            )
        if oracle_output is not None:
            oracle_logits, oracle_hidden_states = oracle_output
            logits_output.next_token_logits = oracle_logits
            logits_output.hidden_states = oracle_hidden_states
        used_suffix_oracle = oracle_output is not None

        vocab_mask = None
        if batch.has_grammar:
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                batch.sampling_info.vocab_mask = None

        maybe_detect_nan(
            logits_output.next_token_logits, "dvr eagle verify: target model logits"
        )
        maybe_detect_inf(
            logits_output.next_token_logits, "dvr eagle verify: target model logits"
        )
        predict, accept_lens, accept_index = verify_input.sample(
            batch, logits_output, vocab_mask
        )
        new_seq_lens = batch.seq_lens + accept_lens
        # Replay-prefix tracking is needed whenever suffix replay may reconstruct
        # the next deterministic tail. It copies accepted draft tokens to CPU,
        # so non-GDN no-oracle EAGLE paths still keep it off the hot path.
        if batch.return_logprob or used_suffix_oracle or not can_run_cuda_graph:
            self.dvr_verifier_replay_prefix.advance_eagle_verifier_stream_from_draft_rows(
                batch=batch,
                draft_token=verify_input.draft_token,
                draft_token_num=verify_input.draft_token_num,
                accept_lens=accept_lens,
                error_prefix="DVR EAGLE replay prefix",
            )

        if not batch.forward_mode.is_idle():
            accept_tokens = predict[accept_index]
            bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
            fill_bonus_tokens[(bs,)](
                accept_tokens,
                accept_lens,
                bonus_tokens,
                accept_index.shape[1],
            )
        else:
            accept_tokens = None
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

        compact_output_token_ids_per_req = compact_output_token_rows(
            accept_tokens,
            accept_lens,
        )
        final_logprob_repairs = None
        if batch.return_logprob and not batch.forward_mode.is_idle():
            assert compact_output_token_ids_per_req is not None
            # DVR-EAGLE may rewrite previously collected output logprobs with a
            # final full-prefix oracle. Mark requests here so the generic
            # streamer only sees a request-level defer policy.
            self.dvr_client_output_replay_prefix.advance_output_stream_from_compact_rows(
                batch=batch,
                compact_output_token_ids_per_req=compact_output_token_ids_per_req,
                error_prefix="DVR EAGLE final logprob output replay prefix",
            )
            final_logprob_repairs = score_deferred_dvr_final_logprob_repairs(
                batch=batch,
                target_worker=self.target_worker,
                replay_prefix=self.dvr_client_output_replay_prefix,
                linear_state_ctx=linear_state_ctx,
                base_seq_lens_cpu=base_seq_lens_cpu,
                accept_lens_cpu=accept_lens.detach().cpu().tolist(),
                compact_output_token_ids_per_req=compact_output_token_ids_per_req,
                error_prefix="DVR EAGLE final logprob",
                allow_preclaimed_final_token=True,
            )

        pending_track_indices = None
        pending_track_seqlens = None
        if linear_state_ctx is not None:
            pending_track_indices, pending_track_seqlens = (
                self.linear_state.commit_after_verify_v2(
                    batch=batch,
                    accepted_token_counts=accept_lens.to(torch.long),
                    accepted_steps=(accept_lens - 1).to(torch.long),
                    ctx=linear_state_ctx,
                )
            )
            self.linear_state.backup_boundary_state(batch)
        elif (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
            or self.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            self._mamba_verify_update(batch, accept_lens, accept_index, bs)

        if batch.return_logprob and not batch.forward_mode.is_idle():
            compute_spec_v2_logprobs(
                batch,
                logits_output,
                predict,
                accept_index,
                self.speculative_num_steps,
            )

        next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            spec_aux=DVRSpecResultAux.from_pending_mamba_checkpoint_lists(
                pending_track_indices,
                pending_track_seqlens,
                final_logprob_repairs=final_logprob_repairs,
            ),
            routed_experts_output=forward_batch_output.routed_experts_output,
            indexer_topk_output=forward_batch_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )
