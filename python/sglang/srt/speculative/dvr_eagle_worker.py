from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
from sglang.srt.layers.logits_processor import LogitsMetadata
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


_ATTN_BACKEND_CHILD_ATTRS = (
    "decode_backend",
    "prefill_backend",
    "full_attn_backend",
    "linear_attn_backend",
    "primary",
)
_ATTN_BACKEND_CHILD_LIST_ATTRS = (
    "attn_backend_list",
    "attn_backends",
    "backends",
    "children",
)
_TEMP_EXTEND_BATCH_FIELDS = (
    "forward_mode",
    "global_forward_mode",
    "input_ids",
    "input_embeds",
    "replace_embeds",
    "replace_positions",
    "out_cache_loc",
    "seq_lens",
    "seq_lens_cpu",
    "seq_lens_sum",
    "prefix_lens",
    "extend_lens",
    "extend_num_tokens",
    "extend_logprob_start_lens",
    "extend_input_logprob_token_ids",
    "global_num_tokens",
    "global_num_tokens_for_logprob",
    "is_extend_in_batch",
    "all_extend_in_batch",
    "spec_info",
    "capture_hidden_mode",
    "return_hidden_states",
    "return_hidden_states_before_norm",
    "return_logprob",
    "mamba_track_indices",
    "mamba_track_mask",
    "mamba_track_seqlens",
    "mamba_cow_src_indices",
    "mamba_cow_dst_indices",
    "mamba_clear_indices",
    "multimodal_inputs",
)


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
        self.cuda_graph_runner_for_target_verify = None
        if not self.server_args.disable_cuda_graph and (
            self.server_args.model_impl != "mindspore"
        ):
            self.cuda_graph_runner_for_target_verify = (
                DVREagleTargetVerifyCudaGraphRunner(self)
            )

    @contextmanager
    def _target_verify_graph_runner_context(self):
        target_runner = self.target_worker.model_runner
        runner = self.cuda_graph_runner_for_target_verify
        if runner is None and self.plan_stream is None:
            yield
            return

        saved_graph_runner = target_runner.graph_runner
        # Overlap target verify previously had to hide the generic EAGLE graph
        # because its padded metadata does not understand DVR GDN state-input
        # windows.  Prefer the DVR-EAGLE graph when captured; if capture is
        # disabled, keep the old eager fallback by installing None only inside
        # this scoped target-verify window.
        target_runner.graph_runner = runner
        try:
            yield
        finally:
            target_runner.graph_runner = saved_graph_runner

    @contextmanager
    def _target_verify_prepare_context(self):
        use_plan_stream = (
            self.plan_stream is not None
            and self.cuda_graph_runner_for_target_verify is None
        )
        # The dedicated DVR-EAGLE target-verify graph replays GDN state-input
        # and commit-buffer side effects.  Preparing its graph buffers on the
        # overlap plan stream can race with the graph replay on CUDA even after
        # a stream wait, so keep this graph's replay_prepare on the compute
        # stream.  If the dedicated graph is unavailable, preserve the previous
        # plan-stream eager fallback.
        stream_ctx = self.plan_stream_ctx if use_plan_stream else nullcontext()
        with stream_ctx, self._target_verify_graph_runner_context():
            yield use_plan_stream

    @contextmanager
    def _temporary_target_extend_batch(self, batch: ScheduleBatch):
        """Temporarily reinterpret a live verify batch as EXTEND.

        DVR-EAGLE uses target EXTEND replays as correctness oracles for GDN
        state and strict returned logprobs. ForwardBatch.init_new reads mutable
        ScheduleBatch fields, so every field touched by those temporary replays
        must be restored before the scheduler continues the real TARGET_VERIFY
        path.
        """

        saved_fields = {
            name: getattr(batch, name) for name in _TEMP_EXTEND_BATCH_FIELDS
        }
        try:
            yield saved_fields
        finally:
            for name, value in saved_fields.items():
                setattr(batch, name, value)

    def _target_verify_graph_runner_bs(self, can_run_cuda_graph: bool):
        if not can_run_cuda_graph:
            return None
        runner = (
            self.cuda_graph_runner_for_target_verify
            or self.target_worker.model_runner.graph_runner
        )
        return None if runner is None else runner.bs

    def _iter_target_attention_backends(self):
        seen = set()
        stack = [self.target_worker.model_runner.attn_backend]
        while stack:
            backend = stack.pop()
            if backend is None or id(backend) in seen:
                continue
            seen.add(id(backend))
            yield backend

            for attr_name in _ATTN_BACKEND_CHILD_ATTRS:
                stack.append(getattr(backend, attr_name, None))
            for attr_name in _ATTN_BACKEND_CHILD_LIST_ATTRS:
                stack.extend(getattr(backend, attr_name, None) or ())

    def _update_verify_buffers_to_fill_after_draft(
        self, verify_input: DVREagleVerifyInput, can_run_cuda_graph: bool
    ):
        cuda_graph_bs = self._target_verify_graph_runner_bs(can_run_cuda_graph)
        for backend in self._iter_target_attention_backends():
            try:
                backend.update_verify_buffers_to_fill_after_draft(
                    verify_input, cuda_graph_bs
                )
            except NotImplementedError:
                # Hybrid wrappers only route metadata calls to their children;
                # the real full-attention backend owns this EAGLE overlap hook.
                continue

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

        # The replay stream is a helper for overlap timing.  If it is behind
        # but the request already exposes a complete logical prefix, use that
        # instead of failing the strict oracle path.
        token_ids = origin_input_ids + list(req.output_ids)
        if len(token_ids) >= seq_len:
            return token_ids

        if hasattr(req, "get_fill_ids"):
            fill_ids = list(req.get_fill_ids())
            if len(fill_ids) >= seq_len:
                return fill_ids

        fill_ids = getattr(req, "full_untruncated_fill_ids", None)
        if fill_ids is not None:
            fill_ids = list(fill_ids)
            if len(fill_ids) >= seq_len:
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
        # EAGLE v2 emits the accepted draft-token prefix. Target's bonus token
        # seeds the next draft round, where it becomes a draft token before it
        # can be committed.  Tracking predict/accept_index here would shift the
        # replay prefix by one target prediction.
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
        # _req_seqlen_matches_batch_prefix reads spec_info.bonus_tokens to pad
        # req.seqlen-compatible metadata.  Prefill has the token separately, so
        # install a short-lived EagleDraftInput and restore the real spec_info.
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
        full_prefix_replay: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
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
        boundary_lens = self.linear_state.boundary_lens_for_replay(
            batch, base_seq_lens_cpu
        )
        if full_prefix_replay:
            boundary_lens = [0 for _ in boundary_lens]
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
        draft_tokens_cpu = draft_tokens.detach().cpu().tolist()
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
            replay_chunk_ids = token_ids[int(boundary) : int(seq_len)]
            replay_chunk_ids.extend(draft_tokens_cpu[req_i])
            input_ids.extend(replay_chunk_ids)
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
        saved_state_input_window = state_adapter.backup_state_input_window(
            state_cache=state_cache,
            state_input_indices=linear_state_ctx.state_input_indices,
        )
        if full_prefix_replay and saved_tail_lens is not None:
            # Strict returned-logprob replay is compared against a flush-cache
            # full prefill oracle.  Clear DVR's rolling state-input window so a
            # cached radix prefix cannot leak into that full-prefix replay.
            zero_tail_lens = torch.zeros_like(saved_tail_lens)
            state_adapter.set_state_input_tail_lens(
                state_cache=state_cache,
                state_input_indices=linear_state_ctx.state_input_indices,
                tail_lens=zero_tail_lens,
            )
            state_adapter.zero_state_input_after_lens(
                state_cache=state_cache,
                state_input_indices=linear_state_ctx.state_input_indices,
                keep_lens=zero_tail_lens,
            )

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
        possible_boundary_track, boundary_track_seqlens = (
            self.linear_state.suffix_replay_boundary_track_info(
                boundary_lens,
                extend_lens_cpu,
                device=batch.seq_lens.device,
            )
        )
        # The replay output replaces verifier logits/hidden states; checkpoint
        # ownership stays with DVR's commit path rather than this temporary
        # EXTEND forward.
        possible_boundary_track = torch.zeros_like(possible_boundary_track)

        with self._temporary_target_extend_batch(batch) as saved_fields:
            try:
                if not full_prefix_replay:
                    self.linear_state.restore_boundary_state_for_suffix_replay(
                        linear_state_ctx
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
                batch.extend_input_logprob_token_ids = None
                if saved_fields["global_num_tokens"] is not None:
                    dp_world = len(saved_fields["global_num_tokens"])
                    batch.global_num_tokens = [len(input_ids)] * dp_world
                    batch.global_num_tokens_for_logprob = [len(input_ids)] * dp_world
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
                if possible_boundary_track.any():
                    # Reuse the normal EXTEND mamba tracker to materialize the next
                    # chunk checkpoint while replaying the suffix.  Sampling may
                    # later reject before that boundary; commit_after_verify will
                    # roll back those speculative checkpoint writes.
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
                # Suffix replay starts from DVR-managed recurrent state. Do not
                # re-apply cached-prefix deferred Mamba COW/clear ops that were
                # already consumed by the real prefill/verify forward.
                batch.mamba_cow_src_indices = None
                batch.mamba_cow_dst_indices = None
                batch.mamba_clear_indices = live_indices if full_prefix_replay else None
                # The suffix EXTEND must see draft KV at absolute positions
                # base_seq_len..base_seq_len+draft.  These are the same slots the
                # real verify path owns, so publishing them in req_to_token_pool is
                # intentional even though the rest of ScheduleBatch is restored.
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

                hidden_gather_indices = []
                offset = 0
                for tail_len, extend_len in zip(
                    tail_lens_cpu, extend_lens_cpu, strict=True
                ):
                    hidden_gather_indices.extend(
                        range(offset + tail_len, offset + tail_len + draft_token_num)
                    )
                    offset += extend_len
                hidden_gather_indices_t = torch.tensor(
                    hidden_gather_indices, dtype=torch.long, device=hidden_states.device
                )
                draft_hidden_states = hidden_states[hidden_gather_indices_t]

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
                    None,
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
                state_adapter.restore_state_input_window(
                    state_cache=state_cache,
                    state_input_indices=linear_state_ctx.state_input_indices,
                    backup=saved_state_input_window,
                )

    def _target_full_prefix_score_accepted_logprobs(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: DVREagleVerifyInput,
        linear_state_ctx,
        predict: torch.Tensor,
        accept_lens: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor | None:
        """Score emitted EAGLE tokens with the same full-prefill oracle as KL tests.

        EAGLE's verify matrix mixes accepted draft tokens and a target bonus
        token.  Their row semantics differ enough that reconstructing returned
        logprobs from verify rows is fragile.  For strict return_logprob
        requests, replay the full prefix plus the actually emitted tokens and
        read SGLang's standard input-logprob slice.  This path is intentionally
        outside the no-logprob hot path.
        """

        if batch.forward_mode.is_idle() or linear_state_ctx is None:
            return None

        state_adapter = linear_state_ctx.state_adapter
        state_cache = linear_state_ctx.state_cache
        live_indices = linear_state_ctx.live_indices

        bs = len(batch.seq_lens)
        base_seq_lens_cpu = (
            batch.seq_lens_cpu.tolist()
            if batch.seq_lens_cpu is not None
            else batch.seq_lens.detach().cpu().tolist()
        )
        accept_lens_cpu = accept_lens.detach().cpu().tolist()
        predict_cpu = predict.detach().cpu().tolist()
        draft_tokens_cpu = (
            verify_input.draft_token.reshape(bs, draft_token_num)
            .detach()
            .cpu()
            .tolist()
        )

        input_ids = []
        logprob_token_ids = []
        out_cache_locs = []
        extend_lens_cpu = []
        final_seq_lens_cpu = []
        final_score_specs = []
        write_rows = []
        write_offsets = []
        write_locs = []
        flat_cache_locs = batch.out_cache_loc.to(torch.long)

        for req_i, (req, seq_len, accept_len) in enumerate(
            zip(batch.reqs, base_seq_lens_cpu, accept_lens_cpu, strict=True)
        ):
            accept_len = int(accept_len)
            compact_start = req_i * draft_token_num
            token_ids = self._request_token_ids_for_eagle_replay(req, int(seq_len))
            prev_output_len = len(req.output_ids)
            max_new_tokens = req.sampling_params.max_new_tokens
            final_output_len = None
            if (
                not req.stream
                and req.send_output_token_logprobs_offset == 0
                and max_new_tokens is not None
                and prev_output_len + accept_len >= max_new_tokens
            ):
                final_output_len = int(max_new_tokens)

            if final_output_len is not None:
                # For GDN models, logprobs inside an unclosed chunk are only
                # strictly comparable to the final full-prefill oracle once the
                # whole generated suffix is known.  Non-streaming requests have
                # not sent output logprobs yet, so patch the final response from
                # the exact prompt+output scoring pass instead of relying on
                # per-verify-round rows.
                final_curr_len = max(0, final_output_len - prev_output_len)
                emitted_tokens = [
                    int(token)
                    for token in predict_cpu[
                        compact_start : compact_start + final_curr_len
                    ]
                ]
                final_output_ids = list(req.output_ids) + emitted_tokens
                replay_ids = token_ids[: int(seq_len)] + emitted_tokens
                logprob_token_ids.extend(replay_ids[1:])
                logprob_token_ids.append(0)
                final_score_specs.append(
                    (req_i, req, prev_output_len, final_curr_len, final_output_ids)
                )
            else:
                # DVR-EAGLE's recurrent/KV state advances over the accepted
                # draft prefix, while Spec v2 emits the one-token-shifted
                # stream.  Replay the accepted draft prefix as context and ask
                # input-logprob to score the shifted tokens: draft[1:] plus the
                # target bonus.
                replay_tokens = [
                    int(token) for token in draft_tokens_cpu[req_i][:accept_len]
                ]
                bonus_token = int(predict_cpu[compact_start + accept_len - 1])
                replay_ids = token_ids[: int(seq_len)] + replay_tokens
                logprob_token_ids.extend(replay_ids[1:])
                logprob_token_ids.append(bonus_token)

            input_ids.extend(replay_ids)

            if int(seq_len) > 0:
                out_cache_locs.append(
                    batch.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : int(seq_len)
                    ].to(torch.long)
                )
            current_write_len = (
                accept_len if final_output_len is None else final_curr_len
            )
            accepted_locs = flat_cache_locs[
                compact_start : compact_start + current_write_len
            ]
            out_cache_locs.append(accepted_locs)

            req_row = torch.full(
                (current_write_len,),
                int(req.req_pool_idx),
                dtype=torch.long,
                device=batch.seq_lens.device,
            )
            req_offsets = torch.arange(
                int(seq_len),
                int(seq_len) + current_write_len,
                dtype=torch.long,
                device=batch.seq_lens.device,
            )
            if current_write_len > 0:
                write_rows.append(req_row)
                write_offsets.append(req_offsets)
                write_locs.append(accepted_locs.to(device=batch.seq_lens.device))

            extend_len = len(replay_ids)
            extend_lens_cpu.append(extend_len)
            final_seq_lens_cpu.append(extend_len)

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
        if saved_tail_lens is not None:
            zero_tail_lens = torch.zeros_like(saved_tail_lens)
            state_adapter.set_state_input_tail_lens(
                state_cache=state_cache,
                state_input_indices=linear_state_ctx.state_input_indices,
                tail_lens=zero_tail_lens,
            )
            state_adapter.zero_state_input_after_lens(
                state_cache=state_cache,
                state_input_indices=linear_state_ctx.state_input_indices,
                keep_lens=zero_tail_lens,
            )

        live_backup = state_adapter.backup_recurrent_state(
            state_cache=state_cache,
            indices=live_indices,
        )
        with self._temporary_target_extend_batch(batch) as saved_fields:
            try:
                batch.forward_mode = ForwardMode.EXTEND
                batch.global_forward_mode = None
                batch.input_ids = torch.tensor(
                    input_ids, dtype=torch.long, device=batch.seq_lens.device
                )
                batch.input_embeds = None
                batch.replace_embeds = None
                batch.replace_positions = None
                batch.out_cache_loc = torch.cat(out_cache_locs).to(batch.seq_lens.device)
                batch.prefix_lens = [0 for _ in extend_lens_cpu]
                batch.extend_lens = [int(x) for x in extend_lens_cpu]
                batch.extend_num_tokens = len(input_ids)
                batch.extend_logprob_start_lens = [0 for _ in extend_lens_cpu]
                batch.extend_input_logprob_token_ids = torch.tensor(
                    logprob_token_ids, dtype=torch.long, device=batch.seq_lens.device
                )
                if saved_fields["global_num_tokens"] is not None:
                    dp_world = len(saved_fields["global_num_tokens"])
                    batch.global_num_tokens = [len(input_ids)] * dp_world
                    batch.global_num_tokens_for_logprob = [len(input_ids)] * dp_world
                batch.seq_lens = torch.tensor(
                    final_seq_lens_cpu,
                    dtype=torch.long,
                    device=saved_fields["seq_lens"].device,
                )
                batch.seq_lens_cpu = torch.tensor(final_seq_lens_cpu, dtype=torch.long)
                batch.seq_lens_sum = sum(final_seq_lens_cpu)
                batch.is_extend_in_batch = True
                batch.all_extend_in_batch = True
                batch.spec_info = None
                batch.capture_hidden_mode = CaptureHiddenMode.NULL
                batch.return_hidden_states = False
                batch.return_hidden_states_before_norm = False
                batch.return_logprob = True
                batch.mamba_track_indices = None
                batch.mamba_track_mask = None
                batch.mamba_track_seqlens = None
                batch.mamba_cow_src_indices = None
                batch.mamba_cow_dst_indices = None
                batch.mamba_clear_indices = live_indices
                batch.multimodal_inputs = [None for _ in extend_lens_cpu]

                if write_rows:
                    batch.req_to_token_pool.write(
                        (torch.cat(write_rows), torch.cat(write_offsets)),
                        torch.cat(write_locs).to(torch.int32),
                    )

                forward_batch = ForwardBatch.init_new(batch, self.target_worker.model_runner)
                score_output = self.target_worker.forward_batch_generation(
                    batch=None,
                    forward_batch=forward_batch,
                    is_verify=True,
                )
                input_token_logprobs = score_output.logits_output.input_token_logprobs
                if input_token_logprobs is None:
                    raise RuntimeError(
                        "DVR EAGLE full-prefix accepted-token scoring did not return "
                        "input_token_logprobs."
                    )

                final_score_req_indices = {spec[0] for spec in final_score_specs}
                padded = torch.zeros(
                    (bs, draft_token_num),
                    dtype=input_token_logprobs.dtype,
                    device=input_token_logprobs.device,
                )
                offset = 0
                for req_i, (seq_len, extend_len, accept_len) in enumerate(
                    zip(
                        base_seq_lens_cpu,
                        extend_lens_cpu,
                        accept_lens_cpu,
                        strict=True,
                    )
                ):
                    if req_i in final_score_req_indices:
                        offset += int(extend_len)
                        continue
                    start = offset + int(seq_len)
                    end = start + int(accept_len)
                    padded[req_i, : int(accept_len)] = input_token_logprobs[start:end]
                    offset += int(extend_len)
                for (
                    req_i,
                    req,
                    prev_output_len,
                    final_curr_len,
                    final_output_ids,
                ) in final_score_specs:
                    prompt_len = len(req.origin_input_ids)
                    req_offset = sum(extend_lens_cpu[:req_i])
                    output_logprob_start = req_offset + prompt_len - 1
                    output_logprob_end = output_logprob_start + len(final_output_ids)
                    final_logprobs = input_token_logprobs[
                        output_logprob_start:output_logprob_end
                    ]
                    if req.logprob.output_token_logprobs_val is not None:
                        req.logprob.output_token_logprobs_val[:] = (
                            final_logprobs[:prev_output_len].detach().cpu().tolist()
                        )
                        req.logprob.output_token_logprobs_idx[:] = final_output_ids[
                            :prev_output_len
                        ]
                    if final_curr_len > 0:
                        padded[req_i, :final_curr_len] = final_logprobs[
                            prev_output_len : prev_output_len + final_curr_len
                        ]
                return padded
            finally:
                state_adapter.restore_recurrent_state(
                    state_cache=state_cache,
                    backup=live_backup,
                    indices=live_indices,
                )
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

        with self._target_verify_prepare_context() as prepared_on_plan_stream:
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

        linear_state_ctx = None
        with self._req_seqlen_matches_batch_prefix(batch):
            linear_state_ctx = self.linear_state.restore_for_verify(batch)

        with self._target_verify_graph_runner_context():
            forward_batch_output = self.target_worker.forward_batch_generation(
                batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
            )
        logits_output = forward_batch_output.logits_output
        oracle_output = None
        if batch.return_logprob or logits_output.hidden_states is None:
            # Strict returned-logprob requests need the full-prefix oracle used by
            # the self-DVR path.  Fast decode can consume the dedicated DVR-EAGLE
            # target-verify graph directly; keep suffix replay only as a hidden
            # state fallback for graph-disabled/eager configurations.
            oracle_output = self._target_suffix_extend_verify_output(
                batch=batch,
                verify_input=verify_input,
                linear_state_ctx=linear_state_ctx,
                full_prefix_replay=batch.return_logprob,
            )
        oracle_input_logprobs = None
        if oracle_output is not None:
            oracle_logits, oracle_hidden_states, oracle_input_logprobs = oracle_output
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
        # The fast no-logprob graph path consumes target verify hidden states
        # directly.  Replay-prefix tracking is only needed when strict logprob
        # requests or graph/eager fallback may reconstruct a suffix oracle; it
        # copies accepted draft tokens to CPU, so keep it off the hot path.
        if batch.return_logprob or used_suffix_oracle or not can_run_cuda_graph:
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

        exact_output_logprobs = None
        if batch.return_logprob and not batch.forward_mode.is_idle():
            exact_output_logprobs = self._target_full_prefix_score_accepted_logprobs(
                batch=batch,
                verify_input=verify_input,
                linear_state_ctx=linear_state_ctx,
                predict=predict,
                accept_lens=accept_lens,
                draft_token_num=verify_input.draft_token_num,
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
            if exact_output_logprobs is not None:
                pos = torch.arange(
                    verify_input.draft_token_num,
                    dtype=torch.long,
                    device=accept_lens.device,
                ).unsqueeze(0)
                accepted_mask = pos < accept_lens.to(torch.long).unsqueeze(1)
                logits_output.next_token_logprobs = torch.where(
                    accepted_mask.to(logits_output.next_token_logprobs.device),
                    exact_output_logprobs.to(logits_output.next_token_logprobs.device),
                    logits_output.next_token_logprobs,
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
