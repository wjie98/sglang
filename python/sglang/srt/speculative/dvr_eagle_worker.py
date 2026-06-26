from __future__ import annotations

import torch
from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dvr_linear_state import DVRLinearStateLifecycle
from sglang.srt.speculative.dvr_worker import (
    DVREagleVerifyInput,
    DecodeVerifyRollbackWorker,
)
from sglang.srt.speculative.dvr_worker_v2 import DecodeVerifyRollbackWorkerV2
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_info_v2 import fill_bonus_tokens
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.spec_utils import (
    generate_token_bitmask,
    record_stream_each,
    record_stream_for_v2_verify,
)
from sglang.srt.utils.async_probe import maybe_detect_inf, maybe_detect_nan
from sglang.srt.utils.common import is_npu

_is_npu = is_npu()


class DecodeVerifyRollbackEagleWorkerV2(EAGLEWorkerV2):
    """EAGLE draft with DVR target verify/rollback semantics.

    The draft model and draft-extend phases stay on the standard EAGLE v2
    implementation. DVR only owns target recurrent-state lifecycle around the
    verify forward, so this path remains isolated from the self-decode draft
    worker.
    """

    _request_token_ids_for_replay = (
        DecodeVerifyRollbackWorker._request_token_ids_for_replay
    )
    _replay_linear_state_boundaries = (
        DecodeVerifyRollbackWorker._replay_linear_state_boundaries
    )
    _req_seqlen_matches_batch_prefix = (
        DecodeVerifyRollbackWorkerV2._req_seqlen_matches_batch_prefix
    )
    _commit_linear_state_after_verify_v2 = (
        DecodeVerifyRollbackWorkerV2._commit_linear_state_after_verify_v2
    )

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
        self._dvr_eagle_replay_output_ids = {}

    def _as_dvr_verify_input(
        self, verify_input: EagleVerifyInput
    ) -> DVREagleVerifyInput:
        if isinstance(verify_input, DVREagleVerifyInput):
            return verify_input
        return DVREagleVerifyInput.from_eagle_verify_input(verify_input)

    def _request_token_ids_for_eagle_replay(self, req, seq_len: int):
        origin_input_ids = list(req.origin_input_ids)
        output_len = seq_len - len(origin_input_ids)
        if output_len <= 0:
            return origin_input_ids[:seq_len]

        replay_output_ids = self._dvr_eagle_replay_output_ids.get(req.rid)
        if replay_output_ids is not None and len(replay_output_ids) >= output_len:
            return origin_input_ids + replay_output_ids[:output_len]

        token_ids = origin_input_ids + req.output_ids
        if replay_output_ids is None and len(token_ids) >= seq_len:
            return token_ids

        if hasattr(req, "get_fill_ids"):
            fill_ids = req.get_fill_ids()
            if replay_output_ids is None and len(fill_ids) >= seq_len:
                return fill_ids

        fill_ids = getattr(req, "full_untruncated_fill_ids", None)
        if (
            replay_output_ids is None
            and fill_ids is not None
            and len(fill_ids) >= seq_len
        ):
            return fill_ids

        raise RuntimeError(
            "DVR EAGLE replay cannot reconstruct the verified prefix: "
            f"rid={req.rid}, tracked_output_tokens="
            f"{0 if replay_output_ids is None else len(replay_output_ids)}, "
            f"required_output_tokens={output_len}, seq_len={seq_len}."
        )

    def _request_token_ids_for_replay(self, req, boundary_seqlen: int):
        return self._request_token_ids_for_eagle_replay(req, boundary_seqlen)

    def _advance_eagle_replay_prefix(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: DVREagleVerifyInput,
        accept_lens: torch.Tensor,
    ) -> None:
        """Track the logical EAGLE prefix used by deterministic replay.

        EAGLE returns target predictions to the client, while the next verify
        prefix is advanced by the accepted draft/input tokens whose KV and
        recurrent states were computed by target verify.  The scheduler may
        start the next forward before those tokens are reflected in
        ``Req.output_ids``.  DVR-EAGLE therefore keeps its own prefix token
        stream and uses it only for verifier replay reconstruction.
        """

        if batch.forward_mode.is_idle() or batch.reqs is None:
            return

        active_rids = {req.rid for req in batch.reqs}
        for rid in list(self._dvr_eagle_replay_output_ids):
            if rid not in active_rids:
                self._dvr_eagle_replay_output_ids.pop(rid, None)

        bs = len(batch.seq_lens)
        draft_token_num = verify_input.draft_token_num
        seq_lens_cpu = (
            batch.seq_lens_cpu.tolist()
            if batch.seq_lens_cpu is not None
            else batch.seq_lens.detach().cpu().tolist()
        )
        accept_lens_cpu = accept_lens.detach().cpu().tolist()
        draft_tokens_cpu = (
            verify_input.draft_token.reshape(bs, draft_token_num)
            .detach()
            .cpu()
            .tolist()
        )

        for req, seq_len, accepted_len, draft_tokens in zip(
            batch.reqs,
            seq_lens_cpu,
            accept_lens_cpu,
            draft_tokens_cpu,
            strict=True,
        ):
            replay_output_ids = self._dvr_eagle_replay_output_ids.setdefault(
                req.rid, []
            )
            prefix_output_len = max(0, int(seq_len) - len(req.origin_input_ids))
            if len(replay_output_ids) > prefix_output_len:
                del replay_output_ids[prefix_output_len:]
            elif len(replay_output_ids) < prefix_output_len:
                missing = prefix_output_len - len(replay_output_ids)
                real_output_ids = req.output_ids[
                    len(replay_output_ids) : prefix_output_len
                ]
                if len(real_output_ids) != missing:
                    raise RuntimeError(
                        "DVR EAGLE replay prefix is behind the batch logical "
                        f"length: rid={req.rid}, tracked={len(replay_output_ids)}, "
                        f"required={prefix_output_len}."
                    )
                replay_output_ids.extend(int(token_id) for token_id in real_output_ids)

            accepted_token_ids = [
                int(token_id) for token_id in draft_tokens[: int(accepted_len)]
            ]
            replay_output_ids.extend(accepted_token_ids)

    def _prepare_dvr_boundary_for_verify(self, batch: ScheduleBatch) -> None:
        if batch.forward_mode.is_idle():
            return

        # EAGLE/MTP draft owns its draft KV state and does not run target GDN
        # state. DVR only checkpoints/restores target recurrent state for the
        # verifier; self-decode is the path that reuses the target live cache.
        # Spec v2 may start the next forward before output processing appends
        # accepted tokens to Req.output_ids. Reuse the self-draft compatibility
        # window so boundary lookup sees the same logical prefix as the batch.
        with self._req_seqlen_matches_batch_prefix(batch):
            replay_tasks = self.linear_state.prepare_for_draft(batch)
            if replay_tasks:
                self._replay_linear_state_boundaries(batch, replay_tasks)
                self.linear_state.restore_tail_lens_after_replay(
                    batch, replay_tasks
                )
            self.linear_state.finish_prepare_for_draft(batch)

    def _prepare_dvr_boundary_for_prefill_draft(
        self, batch: ScheduleBatch, bonus_tokens: torch.Tensor
    ) -> None:
        previous_spec_info = batch.spec_info
        batch.spec_info = EagleDraftInput(bonus_tokens=bonus_tokens)
        try:
            self._prepare_dvr_boundary_for_verify(batch)
        finally:
            batch.spec_info = previous_spec_info

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

        if batch.forward_mode.is_idle() or linear_state_ctx is None:
            return None
        if self.topk != 1 or self.linear_state.boundary_backup is None:
            return None

        state_adapter = linear_state_ctx.state_adapter
        state_cache = linear_state_ctx.state_cache
        live_indices = linear_state_ctx.live_indices
        boundary_indices = linear_state_ctx.boundary_indices
        if boundary_indices is None:
            return None

        bs = len(batch.seq_lens)
        draft_token_num = verify_input.draft_token_num
        base_seq_lens_cpu = (
            batch.seq_lens_cpu.tolist()
            if batch.seq_lens_cpu is not None
            else batch.seq_lens.detach().cpu().tolist()
        )
        # EAGLE keeps a bonus/root token one step ahead of the committed target
        # KV/recurrent prefix.  Until the chunk-boundary suffix oracle models
        # that lead exactly, use full prefill replay for the verifier logits.
        boundary_lens = [0 for _ in batch.reqs]
        tail_lens_cpu = [
            int(seq_len) - int(boundary)
            for seq_len, boundary in zip(
                base_seq_lens_cpu, boundary_lens, strict=True
            )
        ]
        extend_lens_cpu = [tail + draft_token_num for tail in tail_lens_cpu]
        final_seq_lens_cpu = [
            int(boundary) + int(extend_len)
            for boundary, extend_len in zip(
                boundary_lens, extend_lens_cpu, strict=True
            )
        ]

        input_ids = []
        out_cache_locs = []
        draft_tokens = verify_input.draft_token.reshape(bs, draft_token_num)
        draft_cache_locs = batch.out_cache_loc.reshape(bs, draft_token_num)
        for req_i, (req, seq_len, boundary, tail_len) in enumerate(
            zip(
                batch.reqs,
                base_seq_lens_cpu,
                boundary_lens,
                tail_lens_cpu,
                strict=True,
            )
        ):
            token_ids = self._request_token_ids_for_eagle_replay(req, int(seq_len))
            input_ids.extend(token_ids[int(boundary) : int(seq_len)])
            input_ids.extend(draft_tokens[req_i].detach().cpu().tolist())
            if tail_len > 0:
                out_cache_locs.append(
                    batch.req_to_token_pool.req_to_token[
                        req.req_pool_idx, int(boundary) : int(seq_len)
                    ].to(torch.long)
                )
            out_cache_locs.append(draft_cache_locs[req_i].to(torch.long))

        if not input_ids:
            return None

        saved_tail_lens = state_adapter.state_input_tail_lens(
            state_cache=state_cache,
            state_input_indices=linear_state_ctx.state_input_indices,
        )
        if saved_tail_lens is not None:
            saved_tail_lens = saved_tail_lens.clone()

        verify_ready_live_backup = state_adapter.backup_recurrent_state(
            state_cache=state_cache,
            indices=live_indices,
        )
        draft_offsets = batch.seq_lens.to(torch.long).unsqueeze(1) + torch.arange(
            draft_token_num, dtype=torch.long, device=batch.seq_lens.device
        ).unsqueeze(0)
        draft_rows = batch.req_pool_indices.to(torch.long).unsqueeze(1).expand_as(
            draft_offsets
        )

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
            "multimodal_inputs": batch.multimodal_inputs,
        }

        try:
            state_adapter.zero_recurrent_state(
                state_cache=state_cache, indices=live_indices
            )

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
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch.return_hidden_states = False
            batch.return_hidden_states_before_norm = False
            batch.return_logprob = False
            batch.mamba_track_indices = None
            batch.mamba_track_mask = None
            batch.mamba_track_seqlens = None
            # Full replay starts from DVR-managed recurrent state. Do not
            # re-apply cached-prefix deferred Mamba COW/clear ops that were
            # already consumed by the real prefill/verify forward.
            batch.mamba_cow_src_indices = None
            batch.mamba_cow_dst_indices = None
            batch.mamba_clear_indices = None
            batch.req_to_token_pool.write(
                (draft_rows, draft_offsets),
                draft_cache_locs.to(torch.int32),
            )

            forward_batch = ForwardBatch.init_new(batch, self.target_worker.model_runner)
            if self.target_worker.model_runner.model_is_mrope:
                mrope_chunks = []
                mm_inputs = saved_fields["multimodal_inputs"]
                for req_i, (seq_len, boundary, extend_len, tail_len) in enumerate(
                    zip(
                        base_seq_lens_cpu,
                        boundary_lens,
                        extend_lens_cpu,
                        tail_lens_cpu,
                        strict=True,
                    )
                ):
                    mm_input = (
                        None
                        if mm_inputs is None or req_i >= len(mm_inputs)
                        else mm_inputs[req_i]
                    )
                    mm_positions = getattr(mm_input, "mrope_positions", None)
                    chunk_parts = []
                    if mm_positions is not None:
                        tail_positions = mm_positions[
                            :, int(boundary) : int(seq_len)
                        ].to(device=batch.seq_lens.device, dtype=torch.long)
                        if tail_positions.shape[1] > 0:
                            chunk_parts.append(tail_positions)
                    filled_tail = (
                        sum(part.shape[1] for part in chunk_parts)
                        if chunk_parts
                        else 0
                    )
                    if filled_tail < int(tail_len):
                        start = int(boundary) + filled_tail
                        fallback_tail = torch.arange(
                            start,
                            int(seq_len),
                            dtype=torch.long,
                            device=batch.seq_lens.device,
                        ).unsqueeze(0).repeat(3, 1)
                        chunk_parts.append(fallback_tail)
                    draft_positions = torch.arange(
                        int(seq_len),
                        int(seq_len) + draft_token_num,
                        dtype=torch.long,
                        device=batch.seq_lens.device,
                    ).unsqueeze(0).repeat(3, 1)
                    chunk_parts.append(draft_positions)
                    mrope_chunks.append(torch.cat(chunk_parts, dim=1))
                forward_batch.mrope_positions = torch.cat(mrope_chunks, dim=1)
            oracle_output = self.target_worker.forward_batch_generation(
                batch=None,
                forward_batch=forward_batch,
                is_verify=True,
            )
            hidden_states = oracle_output.logits_output.hidden_states
            if hidden_states is None:
                raise RuntimeError(
                    "DVR EAGLE suffix EXTEND verifier did not return hidden states."
                )

            gather_indices = []
            offset = 0
            for tail_len, extend_len in zip(
                tail_lens_cpu, extend_lens_cpu, strict=True
            ):
                gather_indices.extend(
                    range(offset + tail_len, offset + tail_len + draft_token_num)
                )
                offset += extend_len
            gather_indices_t = torch.tensor(
                gather_indices, dtype=torch.long, device=hidden_states.device
            )
            draft_hidden_states = hidden_states[gather_indices_t]

            logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)
            logits_metadata.next_token_logits_buffer = None
            draft_logits = (
                self.target_worker.model_runner.model.logits_processor._get_logits(
                    draft_hidden_states,
                    self.target_worker.model_runner.model.lm_head,
                    logits_metadata,
                )
            )
            return (
                draft_logits,
                draft_hidden_states,
            )
        finally:
            state_adapter.restore_recurrent_state(
                state_cache=state_cache,
                backup=verify_ready_live_backup,
                indices=live_indices,
            )
            if saved_tail_lens is not None:
                state_adapter.set_state_input_tail_lens(
                    state_cache=state_cache,
                    state_input_indices=linear_state_ctx.state_input_indices,
                    tail_lens=saved_tail_lens,
                )
            for name, value in saved_fields.items():
                setattr(batch, name, value)

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
            self._prepare_dvr_boundary_for_prefill_draft(
                batch, batch_output.next_token_ids
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
        verify_input = self._as_dvr_verify_input(batch.spec_info)
        batch.spec_info = verify_input
        record_stream_for_v2_verify(batch, verify_input, fwd_stream)

        verify_input.num_tokens_per_req = self.speculative_num_steps + 1
        bs = len(batch.seq_lens)

        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)

        if self.plan_stream:
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

            target_runner = self.target_worker.model_runner
            target_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                target_runner.graph_runner.bs if can_run_cuda_graph else None,
            )

        if batch.has_grammar:
            retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrieve_next_token.shape
            ).cpu()

        linear_state_ctx = None
        oracle_output = None
        with self._req_seqlen_matches_batch_prefix(batch):
            linear_state_ctx = self.linear_state.restore_for_verify(batch)
            oracle_output = self._target_suffix_extend_verify_output(
                batch=batch,
                verify_input=verify_input,
                linear_state_ctx=linear_state_ctx,
            )

        forward_batch_output = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output = forward_batch_output.logits_output
        if oracle_output is not None:
            oracle_logits, oracle_hidden_states = oracle_output
            logits_output.next_token_logits = oracle_logits
            logits_output.hidden_states = oracle_hidden_states

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
        self._advance_eagle_replay_prefix(
            batch=batch,
            verify_input=verify_input,
            accept_lens=accept_lens,
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
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

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
            pending_mamba_checkpoint_track_indices=pending_track_indices,
            pending_mamba_checkpoint_seqlens=pending_track_seqlens,
            routed_experts_output=forward_batch_output.routed_experts_output,
            indexer_topk_output=forward_batch_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )
