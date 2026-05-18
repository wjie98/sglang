from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.model_executor.dvr_draft_cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
)
from sglang.srt.speculative.dvr_linear_state import DVRLinearStateLifecycle
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
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    get_last_loc,
    maybe_detect_nan,
    select_top_k_tokens,
)
from sglang.srt.utils import is_cuda, next_power_of_2

if is_cuda():
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

logger = logging.getLogger(__name__)


class DecodeVerifyRollbackWorker:
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

        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        self.linear_state = DVRLinearStateLifecycle(
            server_args=server_args,
            model_runner=self.model_runner,
        )
        self.cuda_graph_runner_for_draft_decode = None
        if not server_args.disable_cuda_graph:
            self.cuda_graph_runner_for_draft_decode = DVRDraftDecodeCudaGraphRunner(self)

        logger.info(
            "Initialized DVR self-decode worker: num_steps=%s, num_draft_tokens=%s",
            self.num_draft_steps,
            self.num_draft_tokens,
        )

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
                spec_info.verified_id.to(torch.int64)
            )

        # Self-draft decodes directly into the slots that target verify will
        # read. Do not allocate a second verify window, or KV ownership and
        # radix-cache rollback stop matching the normal speculative layout.
        out_cache_loc = self._alloc_draft_cache_locs(batch)
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
            self.num_draft_tokens,
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.num_draft_tokens + self.page_size),
        )

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
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
                self.num_draft_steps,
                self.num_draft_tokens,
            )

        self.linear_state.prepare_for_draft(batch)
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
            self.num_draft_steps,
            self.num_draft_tokens,
            tree_mask_mode=TreeMaskMode.QLEN_ONLY,
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
            spec_steps=self.num_draft_steps,
            topk=self.topk,
            draft_token_num=self.num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
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

        forward_batch.can_run_dp_cuda_graph = False
        return self.model_runner.forward(forward_batch).logits_output

    def get_draft_probs(
        self, forward_batch: ForwardBatch, logits: torch.Tensor
    ) -> torch.Tensor:
        sampling_info = forward_batch.sampling_info
        probs = F.softmax(logits / sampling_info.temperatures, dim=-1)
        probs = top_k_renorm_prob(probs, sampling_info.top_ks)
        if sampling_info.need_top_p_sampling:
            probs = top_p_renorm_prob(probs, sampling_info.top_ps)
        return probs

    # Accepted-token and verify-output helpers. These intentionally stay close
    # to EAGLE's postprocess contract so scheduler/radix-cache ownership remains
    # compatible with normal speculative decoding.

    def _accepted_token_metadata(
        self,
        batch: ScheduleBatch,
        verify_output: EagleVerifyOutput,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        accepted_tokens = torch.tensor(
            [x + 1 for x in verify_output.accept_length_per_req_cpu],
            dtype=torch.long,
            device=device,
        )
        if accepted_tokens.numel() == 0:
            return accepted_tokens, accepted_tokens

        accepted_indices_offset = torch.arange(
            0,
            len(batch.seq_lens) * batch.spec_info.draft_token_num,
            step=batch.spec_info.draft_token_num,
            dtype=torch.long,
            device=device,
        )

        if (
            batch.spec_info.topk > 1
            and verify_output.accepted_indices.shape[0] > 0
        ):
            accepted_starts = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=device),
                    torch.cumsum(accepted_tokens, dim=0)[:-1],
                ]
            )
            accepted_steps = (
                verify_output.accepted_indices[
                    accepted_starts + accepted_tokens - 1
                ].to(device=device, dtype=torch.long)
                - accepted_indices_offset
            )
        else:
            accepted_steps = accepted_tokens - 1
        return accepted_tokens, accepted_steps

    def _select_accepted_verify_outputs(
        self,
        logits_output: LogitsProcessorOutput,
        verify_output: EagleVerifyOutput,
    ):
        logits_output.next_token_logits = logits_output.next_token_logits[
            verify_output.accepted_indices
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[
                verify_output.accepted_indices
            ]

    def _prepare_next_draft_after_verify(
        self, batch: ScheduleBatch, verify_output: EagleVerifyOutput
    ):
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = verify_output.draft_input
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
        if batch.forward_mode.is_idle():
            return

        accept_end = torch.cumsum(batch.spec_info.accept_length + 1, dim=0) - 1
        batch.spec_info.verified_id = batch.spec_info.verified_id[accept_end]
        # Keep the next-draft inputs in the same compatibility shape as EAGLE.
        batch.spec_info.topk_index = batch.spec_info.verified_id.to(torch.long).unsqueeze(
            -1
        )
        batch.spec_info.topk_p = torch.zeros(
            (batch.spec_info.verified_id.shape[0], self.topk),
            dtype=torch.float32,
            device=batch.spec_info.verified_id.device,
        )

    # Target verify. DVR keeps the forward call in TARGET_VERIFY mode like EAGLE,
    # then locally adapts GDN's physical window and state restore/commit.

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

        self._select_accepted_verify_outputs(logits_output, verify_output)
        if linear_state_ctx is not None:
            accepted_tokens, accepted_steps = self._accepted_token_metadata(
                batch, verify_output, linear_state_ctx.live_indices.device
            )
            self.linear_state.commit_after_verify(
                batch=batch,
                accepted_tokens=accepted_tokens,
                accepted_steps=accepted_steps,
                accepted_tokens_cpu=[
                    x + 1 for x in verify_output.accept_length_per_req_cpu
                ],
                ctx=linear_state_ctx,
            )
        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, verify_output, logits_output)
        self.postprocess_for_verify(batch, verify_output)
        return logits_output, verify_output, can_run_cuda_graph

    def postprocess_for_verify(
        self, batch: ScheduleBatch, verify_output: EagleVerifyOutput
    ):
        self._prepare_next_draft_after_verify(batch, verify_output)
