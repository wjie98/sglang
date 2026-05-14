from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    maybe_detect_nan,
    select_top_k_tokens,
)
from sglang.srt.utils import is_cuda, next_power_of_2

if is_cuda():
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

logger = logging.getLogger(__name__)


@contextmanager
def dvr_causal_verify_cuda_graph_metadata(
    model_runner, attn_backend, forward_mode, spec_info, fallback_custom_mask=None
):
    """Keep DVR cuda graph verify on causal attention without backend edits.

    Some attention backends still read `spec_info.custom_mask.shape` while
    building cuda-graph metadata. For DVR topk=1 the real mask is causal, so
    temporarily provide the graph buffer only to satisfy metadata shape code,
    then clear the captured metadata before graph capture/replay uses it.
    """

    old_custom_mask = getattr(spec_info, "custom_mask", None)
    should_clear = (
        model_runner.spec_algorithm.is_decode_verify_rollback()
        and forward_mode.is_target_verify()
        and spec_info is not None
    )
    if should_clear and old_custom_mask is None and fallback_custom_mask is not None:
        # DVR target verify is a topk=1 chain, so the real attention mask is
        # causal and `custom_mask` should stay None. Some cuda-graph metadata
        # builders still read `spec_info.custom_mask.shape` before producing
        # their metadata, so provide the fixed graph buffer only for that shape
        # bookkeeping. The metadata is cleared below before graph capture/replay.
        spec_info.custom_mask = fallback_custom_mask
    try:
        yield
    finally:
        if should_clear:
            spec_info.custom_mask = old_custom_mask
            metadata = getattr(attn_backend, "forward_metadata", None)
            if metadata is not None:
                # Restore DVR's causal semantics after metadata construction:
                # do not let the temporary tree-mask buffer select a custom-mask
                # attention path in the captured graph or replay metadata.
                if hasattr(metadata, "custom_mask"):
                    metadata.custom_mask = None
                if hasattr(metadata, "mask_indptr"):
                    metadata.mask_indptr = None


class DecodeVerifyRollbackWorker:
    """DVR speculative worker using the target model as a self draft model.

    Phase 2 implements only the self-decode draft provider and reuses the
    existing EAGLE chain verify structure to keep scheduler integration small.
    Later phases will replace the target verify internals with the
    prefill/extend-equivalent DVR verify path.
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

        if server_args.page_size != 1:
            raise ValueError("DVR currently requires page_size == 1.")
        if server_args.speculative_eagle_topk != 1:
            raise ValueError("DVR currently supports only chain mode with topk == 1.")

        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config
        self.tp_rank = tp_rank
        self.device = server_args.device
        self.page_size = server_args.page_size
        self.topk = 1
        self.max_batch_size = target_worker.max_running_requests
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        self._gdn_boundary_seqlen = {}

        logger.info(
            "Initialized DVR self-decode worker: num_steps=%s, num_draft_tokens=%s",
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

    def __getattr__(self, name):
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        return None

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, can_run_cuda_graph = (
                self.forward_target_extend(batch)
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=can_run_cuda_graph,
            )

        spec_info = self.draft(batch)
        logits_output, verify_output, can_run_cuda_graph = self.verify(batch, spec_info)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, bool]:
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        topk_index = next_token_ids.to(torch.long).unsqueeze(-1)
        batch.spec_info = EagleDraftInput(
            hidden_states=logits_output.hidden_states,
            verified_id=next_token_ids,
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
                spec_info.verified_id.to(torch.int64)
            )

        # Self-draft decodes directly into the slots that target verify will
        # read. Do not allocate a second verify window, or KV ownership and
        # radix-cache rollback stop matching the normal speculative layout.
        out_cache_loc = alloc_token_slots(
            batch.tree_cache,
            num_seqs * self.speculative_num_draft_tokens * self.topk,
        )
        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            self.num_new_pages_per_topk,
            out_cache_loc,
            None,
            None,
            None,
            0,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_draft_tokens,
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_draft_tokens + self.page_size),
        )

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=self.model_config.hidden_size,
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    def draft(self, batch: ScheduleBatch) -> EagleVerifyInput:
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        self._ensure_gdn_boundary_state(batch)
        self._draft_preprocess_decode(batch)
        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
        batch.return_hidden_states = False

        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        parent_list, top_scores_index, draft_tokens, draft_probs = self.draft_forward(
            forward_batch
        )

        (
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )
        draft_tokens = draft_tokens.to(torch.long)

        return EagleVerifyInput(
            draft_token=draft_tokens,
            # DVR uses topk=1 chain verify. The tree builder is still reused
            # for token order/retrieve metadata, but attention itself should
            # stay on the ordinary causal path instead of a backend-specific
            # custom tree-mask path.
            custom_mask=None,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            draft_probs=draft_probs,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        out_cache_loc = forward_batch.out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_draft_tokens
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_draft_tokens, -1
        )

        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        draft_probs_list: List[torch.Tensor] = []
        scores = None
        topk_p = None
        topk_index = spec_info.verified_id
        empty_hidden_states = torch.empty(
            (0, 0), dtype=torch.float32, device=topk_index.device
        )

        origin_seq_lens = forward_batch.seq_lens.clone()
        origin_seq_lens_cpu = forward_batch.seq_lens_cpu.clone()
        origin_seq_lens_sum = forward_batch.seq_lens_sum
        origin_spec_info = forward_batch.spec_info
        origin_positions = forward_batch.positions
        origin_out_cache_loc = forward_batch.out_cache_loc
        forward_batch.spec_info = None

        # Run the target model as its own draft model. The loop mutates
        # ForwardBatch fields to look like one-token decode steps, so every
        # scheduler-owned field is restored before target verify starts.
        for i in range(self.speculative_num_steps + 1):
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

            if i == self.speculative_num_steps:
                break

            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i].contiguous()
            forward_batch.seq_lens = origin_seq_lens + i + 1
            forward_batch.seq_lens_cpu = origin_seq_lens_cpu + i + 1
            forward_batch.seq_lens_sum = int(forward_batch.seq_lens.sum().item())
            logits_output = self.model_runner.forward(forward_batch).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"dvr draft step {i}")

            if not forward_batch.sampling_info.is_all_greedy:
                draft_probs_list.append(
                    self.get_draft_probs(forward_batch, logits_output.next_token_logits)
                )
            next_token_ids = self.model_runner.sample(logits_output, forward_batch)
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
        forward_batch.positions = origin_positions
        forward_batch.out_cache_loc = origin_out_cache_loc

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list,
            token_list,
            parents_list,
            self.speculative_num_draft_tokens,
        )
        draft_probs = (
            torch.stack(draft_probs_list, dim=1) if draft_probs_list else None
        )
        return parent_list, top_scores_index, draft_tokens, draft_probs

    def get_draft_probs(
        self, forward_batch: ForwardBatch, logits: torch.Tensor
    ) -> torch.Tensor:
        sampling_info = forward_batch.sampling_info
        probs = F.softmax(logits / sampling_info.temperatures, dim=-1)
        probs = top_k_renorm_prob(probs, sampling_info.top_ks)
        if sampling_info.need_top_p_sampling:
            probs = top_p_renorm_prob(probs, sampling_info.top_ps)
        return probs

    def _has_gdn_dvr_state(self, batch: ScheduleBatch) -> bool:
        return (
            self.model_runner.hybrid_gdn_config is not None
            and hasattr(batch.req_to_token_pool, "get_mamba_indices")
            and batch.batch_size() > 0
            and all(req.mamba_ping_pong_track_buffer is not None for req in batch.reqs)
        )

    def _mamba_indices_for_batch(
        self, batch: ScheduleBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        live_indices = batch.req_to_token_pool.get_mamba_indices(batch.req_pool_indices)
        boundary_indices = torch.stack(
            [req.mamba_ping_pong_track_buffer[0] for req in batch.reqs]
        ).to(device=live_indices.device, dtype=torch.long)
        return live_indices.to(torch.long), boundary_indices

    def _ensure_gdn_boundary_state(self, batch: ScheduleBatch):
        if not self._has_gdn_dvr_state(batch):
            return
        live_indices, boundary_indices = self._mamba_indices_for_batch(batch)
        init_src = []
        init_dst = []
        for i, req in enumerate(batch.reqs):
            if req.rid not in self._gdn_boundary_seqlen:
                self._gdn_boundary_seqlen[req.rid] = req.seqlen
                init_src.append(live_indices[i])
                init_dst.append(boundary_indices[i])
        if init_src:
            src = torch.stack(init_src)
            dst = torch.stack(init_dst)
            batch.req_to_token_pool.mamba_pool.copy_from(src, dst)
            mamba_cache = (
                batch.req_to_token_pool.get_speculative_mamba2_params_all_layers()
            )
            if getattr(mamba_cache, "dvr_qkvg_beta_pos", None) is not None:
                mamba_cache.dvr_qkvg_beta_pos[:, src] = 0

    def _restore_gdn_boundary_state_for_verify(self, batch: ScheduleBatch):
        if not self._has_gdn_dvr_state(batch):
            return
        self._ensure_gdn_boundary_state(batch)
        live_indices, boundary_indices = self._mamba_indices_for_batch(batch)
        batch.req_to_token_pool.mamba_pool.copy_from(boundary_indices, live_indices)

    def _commit_gdn_state_after_verify(
        self, batch: ScheduleBatch, verify_output: EagleVerifyOutput
    ):
        if not self._has_gdn_dvr_state(batch):
            return

        live_indices, boundary_indices = self._mamba_indices_for_batch(batch)
        accepted_tokens = torch.tensor(
            [x + 1 for x in verify_output.accept_length_per_req_cpu],
            dtype=torch.long,
            device=live_indices.device,
        )
        if accepted_tokens.numel() == 0:
            return

        attn_backend = self.model_runner.attn_backend
        linear_backend = getattr(attn_backend, "linear_attn_backend", None)
        if linear_backend is None:
            return
        dispatcher = linear_backend.kernel_dispatcher
        mamba_cache = batch.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        pos_before = mamba_cache.dvr_qkvg_beta_pos[0, live_indices].to(torch.long)
        pos_after = pos_before + accepted_tokens
        crossing = pos_after >= FLA_CHUNK_SIZE

        # Rebuild live tail state from the deterministic chunk-boundary state
        # plus the saved q/k/v/g/beta window. This repairs the live state dirtied
        # by draft/verify and lets the next self-decode continue from the last
        # accepted token.
        for layer_idx in range(mamba_cache.temporal.shape[0]):
            initial_state = mamba_cache.temporal[layer_idx, boundary_indices]
            tail_state = dispatcher.recurrent_state_from_qkvg_beta(
                mamba_cache.dvr_q_state_cache[layer_idx, live_indices],
                mamba_cache.dvr_k_state_cache[layer_idx, live_indices],
                mamba_cache.dvr_v_state_cache[layer_idx, live_indices],
                mamba_cache.dvr_g_state_cache[layer_idx, live_indices],
                mamba_cache.dvr_beta_state_cache[layer_idx, live_indices],
                initial_state=initial_state,
                token_count=pos_after,
            )
            mamba_cache.temporal[layer_idx, live_indices] = tail_state

        step_live = accepted_tokens - 1
        fused_mamba_state_scatter_with_mask(
            mamba_cache.conv[0],
            mamba_cache.intermediate_conv_window[0],
            live_indices,
            step_live,
        )

        if crossing.any():
            commit_step = torch.where(
                crossing, FLA_CHUNK_SIZE - pos_before - 1, torch.full_like(pos_before, -1)
            )
            for layer_idx in range(mamba_cache.temporal.shape[0]):
                boundary_state = dispatcher.chunkwise_boundary_state_from_qkvg_beta(
                    mamba_cache.dvr_q_state_cache[layer_idx, live_indices],
                    mamba_cache.dvr_k_state_cache[layer_idx, live_indices],
                    mamba_cache.dvr_v_state_cache[layer_idx, live_indices],
                    mamba_cache.dvr_g_state_cache[layer_idx, live_indices],
                    mamba_cache.dvr_beta_state_cache[layer_idx, live_indices],
                    state_pool=mamba_cache.temporal[layer_idx],
                    state_indices=boundary_indices,
                    boundary_token_count=FLA_CHUNK_SIZE,
                )
                mamba_cache.temporal[layer_idx, boundary_indices] = boundary_state
            fused_mamba_state_scatter_with_mask(
                mamba_cache.conv[0],
                mamba_cache.intermediate_conv_window[0],
                boundary_indices,
                commit_step,
            )

            new_pos = pos_after - FLA_CHUNK_SIZE
            for req_i, req in enumerate(batch.reqs):
                if not bool(crossing[req_i].item()):
                    continue
                remain = int(new_pos[req_i].item())
                slot = int(live_indices[req_i].item())
                if remain > 0:
                    for cache in (
                        mamba_cache.dvr_q_state_cache,
                        mamba_cache.dvr_k_state_cache,
                        mamba_cache.dvr_v_state_cache,
                        mamba_cache.dvr_g_state_cache,
                        mamba_cache.dvr_beta_state_cache,
                    ):
                        cache[:, slot, :remain] = cache[
                            :, slot, FLA_CHUNK_SIZE : FLA_CHUNK_SIZE + remain
                        ].clone()
                self._gdn_boundary_seqlen[req.rid] += FLA_CHUNK_SIZE
                req.mamba_last_track_seqlen = self._gdn_boundary_seqlen[req.rid]
                req.mamba_next_track_idx = 0
            pos_after = torch.where(crossing, new_pos, pos_after)

        mamba_cache.dvr_qkvg_beta_pos[:, live_indices] = pos_after.to(torch.int32)

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
        spec_info.num_tokens_per_req = self.speculative_num_draft_tokens
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info
        self._restore_gdn_boundary_state_for_verify(batch)

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
        maybe_detect_nan(logits_output.next_token_logits, "dvr target verify")

        spec_info.hidden_states = logits_output.hidden_states
        verify_output: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask=None,
        )

        logits_output.next_token_logits = logits_output.next_token_logits[
            verify_output.accepted_indices
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[
                verify_output.accepted_indices
            ]

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, verify_output, logits_output)

        self._commit_gdn_state_after_verify(batch, verify_output)
        self.postprocess_for_verify(batch, verify_output)
        return logits_output, verify_output, can_run_cuda_graph

    def postprocess_for_verify(
        self, batch: ScheduleBatch, verify_output: EagleVerifyOutput
    ):
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = verify_output.draft_input
        if batch.spec_info.topk_p is None and batch.spec_info.verified_id.numel() > 0:
            batch.spec_info.topk_p = torch.ones(
                (batch.spec_info.verified_id.shape[0], self.topk),
                dtype=torch.float32,
                device=batch.spec_info.verified_id.device,
            )
            batch.spec_info.topk_index = batch.spec_info.verified_id.to(
                torch.long
            ).unsqueeze(-1)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
        if not batch.forward_mode.is_idle():
            accept_end = torch.cumsum(batch.spec_info.accept_length + 1, dim=0) - 1
            batch.spec_info.verified_id = batch.spec_info.verified_id[accept_end]
            batch.spec_info.topk_p = batch.spec_info.topk_p[accept_end]
            batch.spec_info.topk_index = batch.spec_info.topk_index[accept_end]
