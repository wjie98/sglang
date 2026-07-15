from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from typing import List, Optional

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.dvr_draft_cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
    dvr_draft_decode_context,
    iter_dvr_attention_backends,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dvr_state_flow import DVRLinearStateLifecycle
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_info_v2 import fill_bonus_tokens
from sglang.srt.speculative.eagle_utils import (
    eagle_prepare_for_verify,
    eagle_sample,
)
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    generate_token_bitmask,
    record_stream_each,
    record_stream_for_v2_verify,
    renorm_sampling_probs,
    spec_stage_span,
)
from sglang.srt.utils.async_probe import maybe_detect_inf, maybe_detect_nan
from sglang.srt.utils.common import get_available_gpu_memory

logger = logging.getLogger(__name__)


class DecodeVerifyRollbackWorkerV2(BaseSpecWorker):
    """Self-decode draft worker with DVR target verify/rollback semantics.

    User-visible "spec v1" is a synchronous compatibility mode over this same
    worker.  Keep the implementation in one class so self-draft, target verify,
    GDN state, and v1/v2 glue are visible in a single execution flow.
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
        if server_args.speculative_eagle_topk != 1:
            raise ValueError("DVR currently supports only chain mode with topk == 1.")
        self.server_args = server_args
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.device = server_args.device
        self.topk = 1
        self.num_draft_steps = server_args.speculative_num_steps
        self.num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.is_dvr_eagle = self.speculative_algorithm.is_dvr_eagle()
        device_module = torch.get_device_module(self.device)
        self._rollback_ready_event = device_module.Event()
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        self.linear_state = DVRLinearStateLifecycle(
            server_args=server_args,
            model_runner=self.model_runner,
        )
        self.cuda_graph_runner_for_draft_decode = None
        if self.is_dvr_eagle:
            server_args.context_length = (
                target_worker.model_runner.model_config.context_len
            )
            self._draft_worker = EagleDraftWorker(
                server_args,
                gpu_id,
                tp_rank,
                dp_rank,
                moe_ep_rank,
                attn_cp_rank,
                moe_dp_rank,
                nccl_port,
                target_worker,
            )
            # Reuse the upstream EAGLE/MTP worker as the draft backend. DVR owns
            # target verify and rollback, not a second copy of EAGLE draft logic.
            if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
                device_module = torch.get_device_module(self.device)
                self.plan_stream = device_module.Stream()
                self.plan_stream_ctx = device_module.stream(self.plan_stream)
            else:
                self.plan_stream = None
                self.plan_stream_ctx = nullcontext()
            log_prefix = "DVR EAGLE"
        else:
            del dp_rank, moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port
            self._draft_worker = self
            self.plan_stream = None
            self.plan_stream_ctx = nullcontext()
            max_bs = max(
                server_args.cuda_graph_config.decode.max_bs or 0,
                server_args.max_running_requests or 0,
                1,
            )
            num_tokens = self.num_draft_tokens
            self._chain_retrieve_index = torch.arange(
                max_bs * num_tokens, dtype=torch.long, device=self.device
            ).view(max_bs, num_tokens)
            next_token = torch.arange(
                1, num_tokens + 1, dtype=torch.long, device=self.device
            )
            next_token[-1] = -1
            self._chain_retrieve_next = next_token.repeat(max_bs, 1)
            self._chain_retrieve_sibling = torch.full(
                (max_bs, num_tokens), -1, dtype=torch.long, device=self.device
            )
            self._chain_position_offsets = torch.arange(
                num_tokens, dtype=torch.long, device=self.device
            )
            log_prefix = "DVR self-decode"

        logger.info(
            "Initialized %s worker: num_steps=%s, num_draft_tokens=%s",
            log_prefix,
            self.num_draft_steps,
            self.num_draft_tokens,
        )

    @contextmanager
    def _draft_context(self):
        if not self.is_dvr_eagle:
            with dvr_draft_decode_context(self.model_runner, self_draft=True):
                yield
            return
        draft_worker = self._draft_worker
        extra_attn_backends = (
            draft_worker.draft_attn_backend,
            draft_worker.draft_extend_attn_backend
            or draft_worker.draft_runner.attn_backend,
        )
        with (
            draft_worker.draft_tp_context(draft_worker.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            dvr_draft_decode_context(
                draft_worker.draft_runner,
                extra_attn_backends=extra_attn_backends,
            ),
        ):
            yield

    def _idle_draft_input(self) -> EagleDraftInput:
        if not self.is_dvr_eagle:
            return EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=None,
                dtype=None,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
        return EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=EagleDraftInput.hidden_size_for(self._draft_worker),
            dtype=EagleDraftInput.dtype_for(self._draft_worker),
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            vocab_size=self.target_worker.model_config.vocab_size,
        )

    @staticmethod
    def _self_draft_input(bonus_tokens: torch.Tensor) -> EagleDraftInput:
        return EagleDraftInput(
            hidden_states=None,
            bonus_tokens=bonus_tokens,
            topk_p=torch.ones(
                (bonus_tokens.shape[0], 1),
                dtype=torch.float32,
                device=bonus_tokens.device,
            ),
            topk_index=bonus_tokens.to(torch.long).unsqueeze(-1),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

    @property
    def target_worker(self):
        return self._target_worker

    def init_attention_backends(self):
        if self.is_dvr_eagle:
            self._draft_worker.init_attention_backends()
        # Self-DVR target worker owns the model and attention backend. Scheduler
        # already initializes it before calling this self-draft worker hook.
        adapters = []
        for backend in iter_dvr_attention_backends(self.model_runner.attn_backend):
            adapter = getattr(backend, "dvr_state_adapter", None)
            if adapter is not None and all(adapter is not item for item in adapters):
                adapters.append(adapter)
        if len(adapters) > 1:
            raise RuntimeError("DVR target resolved multiple linear-state adapters.")
        self.linear_state.bind_state_adapter(adapters[0] if adapters else None)

    def init_cuda_graphs(self):
        if self.is_dvr_eagle:
            draft_worker = self._draft_worker
            extra_attn_backends = (
                draft_worker.draft_attn_backend,
                draft_worker.draft_extend_attn_backend
                or draft_worker.draft_runner.attn_backend,
            )
            with dvr_draft_decode_context(
                draft_worker.draft_runner,
                capture=True,
                extra_attn_backends=extra_attn_backends,
            ):
                draft_worker.init_cuda_graphs()
            return

        # Capture the dedicated self-draft decode graph after target attention
        # backends exist. This matches upstream's separated init order.
        if (
            self.cuda_graph_runner_for_draft_decode is None
            and self.server_args.cuda_graph_config.decode.backend != Backend.DISABLED
            and not self.server_args.disable_draft_cuda_graph
        ):
            before_mem = get_available_gpu_memory(
                self.device, self.model_runner.gpu_id
            )
            logger.info(
                "Capture DVR self-draft CUDA graph begin. avail mem=%.2f GB",
                before_mem,
            )
            with dvr_draft_decode_context(
                self.model_runner, capture=True, self_draft=True
            ):
                self.cuda_graph_runner_for_draft_decode = DVRDraftDecodeCudaGraphRunner(
                    self.model_runner
                )
            after_mem = get_available_gpu_memory(
                self.device, self.model_runner.gpu_id
            )
            logger.info(
                "Capture DVR self-draft CUDA graph end. mem usage=%.2f GB, "
                "avail mem=%.2f GB",
                before_mem - after_mem,
                after_mem,
            )

    def alloc_memory_pool(
        self,
        *,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        if self.is_dvr_eagle:
            self._draft_worker.alloc_memory_pool(
                memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator
            )
            self.req_to_token_pool = req_to_token_pool
            self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
            return

        # Self-DVR uses the target model and its pools.  Keep this hook explicit
        # so Scheduler's draft-worker initialization does not fall through
        # __getattr__ and allocate/profile the target pools a second time.
        del memory_pool_config
        if req_to_token_pool is None or token_to_kv_pool_allocator is None:
            req_to_token_pool, token_to_kv_pool_allocator = (
                self.target_worker.get_memory_pool()
            )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    # Self-draft path. DVR uses the target model's decode path as a draft model,
    # but keeps EAGLE-compatible tree/retrieve metadata for verification.

    def _draft_self(self, batch: ScheduleBatch) -> EagleVerifyInput:
        if batch.forward_mode.is_idle():
            batch.spec_info = self._idle_draft_input()
            return EagleVerifyInput.create_idle_input(
                self.topk, self.num_draft_steps, self.num_draft_tokens
            )

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        penalizer = batch.sampling_info.penalizer_orchestrator
        if penalizer is not None and penalizer.is_required:
            penalizer.cumulate_output_tokens(spec_info.bonus_tokens.to(torch.int64))

        # ScheduleBatch.prepare_for_decode already reserved the speculative
        # window. Self draft and target verify share it; allocating again would
        # split KV ownership between two paths.
        offsets = batch.seq_lens.to(torch.long).unsqueeze(1) + torch.arange(
            self.num_draft_tokens, dtype=torch.long, device=batch.seq_lens.device
        ).unsqueeze(0)
        rows = batch.req_pool_indices.to(torch.long).unsqueeze(1)
        batch.out_cache_loc = batch.req_to_token_pool.req_to_token[
            rows, offsets
        ].reshape(-1)
        batch.mamba_track_indices = None
        batch.mamba_track_mask = None
        batch.mamba_track_seqlens = None
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)
        spec_info.num_tokens_per_req = 1
        spec_info.num_tokens_for_logprob_per_req = 1
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
        batch.return_hidden_states = False

        saved_return_logprob = batch.return_logprob
        batch.return_logprob = False
        try:
            forward_batch = ForwardBatch.init_new(batch, self.model_runner)
            if forward_batch.seq_lens_cpu is not None and not torch.is_tensor(
                forward_batch.seq_lens_cpu
            ):
                forward_batch.seq_lens_cpu = torch.tensor(
                    forward_batch.seq_lens_cpu, dtype=torch.int64, device="cpu"
                )
            if (
                forward_batch.seq_lens_sum is None
                and forward_batch.seq_lens_cpu is not None
            ):
                forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum())
            draft_tokens, draft_probs = self._draft_self_tokens(forward_batch)
        finally:
            batch.return_logprob = saved_return_logprob

        batch_size = draft_tokens.shape[0]
        if batch_size > self._chain_retrieve_index.shape[0]:
            raise RuntimeError(
                "DVR self-draft batch exceeds its fixed chain buffers: "
                f"batch_size={batch_size}, capacity={self._chain_retrieve_index.shape[0]}."
            )
        positions = (
            batch.seq_lens[:, None] + self._chain_position_offsets[None, :]
        ).flatten()
        return EagleVerifyInput(
            draft_token=draft_tokens.flatten(),
            custom_mask=None,
            positions=positions,
            retrieve_index=self._chain_retrieve_index[:batch_size],
            retrieve_next_token=self._chain_retrieve_next[:batch_size],
            retrieve_next_sibling=self._chain_retrieve_sibling[:batch_size],
            retrieve_cum_len=None,
            spec_steps=self.num_draft_steps,
            topk=1,
            draft_token_num=self.num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            draft_probs=draft_probs,
        )

    def _draft_self_tokens(self, forward_batch: ForwardBatch):
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        out_cache_loc = forward_batch.out_cache_loc.reshape(
            forward_batch.batch_size, self.num_draft_tokens
        ).transpose(0, 1).contiguous()
        draft_tokens = [spec_info.bonus_tokens.to(torch.long)]
        draft_probs_list: List[torch.Tensor] = []

        origin_seq_lens = forward_batch.seq_lens.clone()
        origin_seq_lens_cpu = forward_batch.seq_lens_cpu
        origin_seq_lens_sum = forward_batch.seq_lens_sum
        origin_spec_info = forward_batch.spec_info
        origin_positions = forward_batch.positions.clone()
        origin_out_cache_loc = forward_batch.out_cache_loc
        forward_batch.spec_info = None

        try:
            # The target model drafts a fixed topk=1 chain; no tree ranking or
            # parent bookkeeping is needed between decode steps.
            for step in range(self.num_draft_steps):
                if step:
                    forward_batch.positions.add_(1)
                forward_batch.input_ids = draft_tokens[-1]
                forward_batch.out_cache_loc = out_cache_loc[step]
                forward_batch.seq_lens = origin_seq_lens + step + 1
                forward_batch.seq_lens_cpu = (
                    None
                    if origin_seq_lens_cpu is None
                    else origin_seq_lens_cpu + step + 1
                )
                forward_batch.seq_lens_sum = (
                    None
                    if origin_seq_lens_sum is None
                    else origin_seq_lens_sum
                    + (step + 1) * forward_batch.batch_size
                )
                logits_output = self._draft_decode_forward(forward_batch)
                maybe_detect_nan(
                    logits_output.next_token_logits, f"dvr draft step {step}"
                )
                next_token_ids = self.model_runner.sample(
                    logits_output, forward_batch
                )
                if not forward_batch.sampling_info.is_all_greedy:
                    sampling_probs = logits_output.next_token_logits
                    sampling_info = forward_batch.sampling_info
                    if (
                        self.model_runner.sampler.use_log_softmax_logprob
                        and not sampling_info.need_top_p_sampling
                        and not sampling_info.need_top_k_sampling
                        and not sampling_info.need_min_p_sampling
                    ):
                        sampling_probs = torch.softmax(sampling_probs, dim=-1)
                    draft_probs_list.append(
                        renorm_sampling_probs(sampling_probs, sampling_info)
                    )
                draft_tokens.append(next_token_ids.to(torch.long))
        finally:
            forward_batch.seq_lens = origin_seq_lens
            forward_batch.seq_lens_cpu = origin_seq_lens_cpu
            forward_batch.seq_lens_sum = origin_seq_lens_sum
            forward_batch.spec_info = origin_spec_info
            forward_batch.positions.copy_(origin_positions)
            forward_batch.out_cache_loc = origin_out_cache_loc

        draft_probs = (
            torch.stack(draft_probs_list, dim=1) if draft_probs_list else None
        )
        return torch.stack(draft_tokens, dim=1), draft_probs

    def _draft_decode_forward(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_decode is not None
            and self.cuda_graph_runner_for_draft_decode.can_run_graph(forward_batch)
        )
        if can_cuda_graph:
            return self.cuda_graph_runner_for_draft_decode.execute(forward_batch)

        seq_lens = forward_batch.seq_lens_cpu
        min_seq_len = (
            int(seq_lens.min()) if seq_lens is not None else "GPU-only"
        )
        capture_bs = []
        if self.cuda_graph_runner_for_draft_decode is not None:
            capture_bs = self.cuda_graph_runner_for_draft_decode.capture_bs
        raise RuntimeError(
            "DVR self-draft decode requires the dedicated CUDA graph; no eager "
            "fallback is used. The current batch cannot run it: "
            f"batch_size={forward_batch.batch_size}, min_seq_len={min_seq_len}, "
            f"capture_bs={capture_bs}. For batch-size misses, use the default "
            "CUDA graph batch sizes or include the running batch size "
            "in --cuda-graph-bs/--cuda-graph-max-bs."
        )

    # Target verify. DVR keeps the forward call in TARGET_VERIFY mode like EAGLE,
    # then locally adapts GDN's physical window and state restore/commit.

    def _prepare_dvr_boundary_for_verify(self, batch: ScheduleBatch) -> None:
        if batch.forward_mode.is_idle():
            return

        prefill_prefix_lens = None
        seq_lens_cpu = None
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            prefill_prefix_lens = [int(x) for x in batch.prefix_lens]
            seq_lens_cpu = self.linear_state.batch_seq_lens_cpu(batch)
        self.linear_state.prepare_for_draft(
            batch,
            seq_lens_cpu=seq_lens_cpu,
            prefill_prefix_lens=prefill_prefix_lens,
        )
        # EAGLE/MTP draft can run before the next target verify in overlap mode.
        # Preserve the target checkpoint authority across draft-side mutations.
        self.linear_state.backup_boundary_state(batch)

    @property
    def draft_worker(self):
        return self._draft_worker

    @property
    def war_fastpath_runner(self):
        if self.is_dvr_eagle:
            return self._draft_worker.draft_runner
        return self.target_worker.model_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        if not self.is_dvr_eagle:
            return (self.target_worker.model_runner.attn_backend,)
        return (
            self.target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
            self._draft_worker.draft_extend_attn_backend
            or self._draft_worker.draft_runner.attn_backend,
        )

    def iter_runners(self):
        if self.is_dvr_eagle:
            return [("draft", self._draft_worker.draft_runner)]
        return []

    def update_weights_from_disk(self, recv_req):
        if not self.is_dvr_eagle:
            # The scheduler has already updated the target model. Self-draft
            # shares those weights and must not load them a second time.
            return True, "Succeeded to update model weights."
        success, message = self._draft_worker.draft_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req):
        if not self.is_dvr_eagle:
            return True, "Succeeded to update model weights."
        success, message = self._draft_worker.draft_runner.update_weights_from_ipc(
            recv_req
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def clear_cache_pool(self):
        self.linear_state.clear_cache_state()

    def forward_batch_generation(
        self, model_worker_batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        batch = model_worker_batch
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch.capture_hidden_mode = (
                CaptureHiddenMode.FULL
                if self.is_dvr_eagle
                else CaptureHiddenMode.NULL
            )
            batch_result = self.target_worker.forward_batch_generation(batch)
            batch_result.new_seq_lens = batch.seq_lens
            self._prepare_dvr_boundary_for_verify(batch)
            if on_publish is not None:
                on_publish(batch_result.new_seq_lens)
            if self.is_dvr_eagle:
                with self._draft_context():
                    batch_result.next_draft_input = (
                        self.draft_worker._draft_extend_for_prefill(
                            batch,
                            batch_result.logits_output.hidden_states,
                            batch_result.next_token_ids,
                            batch_result.logits_output.mm_input_embeds,
                        )
                    )
            else:
                next_token_ids = batch_result.next_token_ids
                batch.spec_info = self._self_draft_input(next_token_ids)
                batch_result.next_draft_input = batch.spec_info
            return batch_result

        # DVR decode has one shared core: draft -> target verify -> rollback.
        if batch.spec_info is None:
            batch.spec_info = self._idle_draft_input()
        seq_lens_cpu = batch.seq_lens_cpu
        if seq_lens_cpu is not None:
            needs_one_root_verify = any(
                int(seq_len) <= 1 for seq_len in seq_lens_cpu
            )
        else:
            # Spec decode pre-claims the bonus slot before model execution, so
            # a committed length of two still represents a one-token prefix.
            needs_one_root_verify = any(
                req.kv_committed_len <= 2 for req in batch.reqs
            )
        with spec_stage_span("dvr_prepare"):
            self._prepare_dvr_boundary_for_verify(batch)
        if needs_one_root_verify:
            verify_input = self._build_one_root_verify_input(batch)
        else:
            with self._draft_context(), spec_stage_span("draft"):
                verify_input = (
                    self._draft_worker.draft(batch)
                    if self.is_dvr_eagle
                    else self._draft_self(batch)
                )
        assert verify_input.is_verify_input()
        batch.spec_info = verify_input
        batch_result = self.verify(batch, verify_input, on_publish=on_publish)
        if self.is_dvr_eagle:
            with self._draft_context(), spec_stage_span("draft_extend"):
                self.draft_worker._draft_extend_for_decode(batch, batch_result)
        return batch_result

    def _build_one_root_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        """Build a fixed-width verify input whose logical tree is only the root."""

        draft_input = batch.spec_info
        assert isinstance(draft_input, EagleDraftInput)
        batch_size = batch.seq_lens.shape[0]
        width = self.num_draft_tokens
        retrieve_index = torch.arange(
            batch_size * width, dtype=torch.long, device=self.device
        ).view(batch_size, width)
        terminal = torch.full_like(retrieve_index, -1)
        # Keep the physical verify shape identical to the captured DVR graph.
        # spec_steps=0 makes every padded node unreachable, so sampling accepts
        # only the root while attention/GDN retain their fixed-shape contract.
        return EagleVerifyInput(
            draft_token=(
                draft_input.bonus_tokens.to(torch.long).repeat_interleave(width)
            ),
            custom_mask=None,
            positions=(
                batch.seq_lens[:, None]
                + torch.arange(width, dtype=torch.long, device=self.device)[None, :]
            ).reshape(-1),
            retrieve_index=retrieve_index,
            retrieve_next_token=terminal,
            retrieve_next_sibling=terminal,
            retrieve_cum_len=None,
            spec_steps=0,
            topk=1,
            draft_token_num=width,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.is_dvr_eagle
                else CaptureHiddenMode.NULL
            ),
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def verify(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        on_publish=None,
    ) -> GenerationBatchResult:
        scheduler_seq_lens = batch.seq_lens
        vocab_mask = None

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrieve_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrieve_next_token.shape
            ).cpu()

        assert spec_info.is_verify_input()
        # DVR only supports topk=1 chains, whose tree mask is exactly the
        # backend's native causal mask. Both draft backends therefore enter the
        # same target-verify preparation and forward path.
        spec_info.custom_mask = None
        verify_tokens = spec_info.draft_token_num
        spec_info.num_tokens_per_req = verify_tokens
        if self.is_dvr_eagle:
            fwd_stream = torch.get_device_module(self.device).current_stream()
            record_stream_for_v2_verify(batch, spec_info, fwd_stream)
        else:
            # Self draft already resolved the host lengths needed by the target
            # graph. ForwardBatch.init_new consumes this one-shot mirror.
            batch.seq_lens_cpu_cache = spec_info.seq_lens_cpu

        prepared_on_plan_stream = self.plan_stream is not None
        need_return_logprob = batch.return_logprob
        batch.return_logprob = False
        try:
            with self.plan_stream_ctx, spec_stage_span("verify_prepare"):
                verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
                    spec_info,
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
        finally:
            batch.return_logprob = need_return_logprob

        if self.is_dvr_eagle:
            record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)

        if prepared_on_plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
            runner = self.target_worker.model_runner.decode_cuda_graph_runner
            cuda_graph_bs = (
                None if not can_run_cuda_graph or runner is None else runner.bs
            )
            for backend in iter_dvr_attention_backends(
                self.target_worker.model_runner.attn_backend
            ):
                try:
                    backend.update_verify_buffers_to_fill_after_draft(
                        spec_info, cuda_graph_bs
                    )
                except NotImplementedError:
                    continue

        with spec_stage_span("dvr_state_restore"):
            # Self-draft graph replay and target verify are both enqueued on
            # Scheduler's forward stream, so CUDA stream order is their fence.
            linear_state_ctx = self.linear_state.restore_for_verify(batch)
        with spec_stage_span("verify"):
            forward_output = self.target_worker.forward_batch_generation(
                batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
            )

        if batch.has_grammar:
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )
            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrieve_next_token.device)
                batch.sampling_info.vocab_mask = None

        logits_output = forward_output.logits_output
        if self.is_dvr_eagle and logits_output.hidden_states is None:
            raise RuntimeError(
                "DVR EAGLE target verify must return hidden states for the next "
                "draft step."
            )

        maybe_detect_nan(logits_output.next_token_logits, "DVR target verify")
        maybe_detect_inf(logits_output.next_token_logits, "DVR target verify")
        if not self.is_dvr_eagle:
            # EAGLE draft already applied request processors. Self draft uses a
            # target-only forward, so apply the same processors before entering
            # the shared EAGLE accept kernel.
            sampling_info = batch.sampling_info
            if sampling_info.has_custom_logit_processor:
                apply_custom_logit_processor(
                    logits_output.next_token_logits,
                    sampling_info,
                    num_tokens_in_batch=verify_tokens,
                )
            penalizer = sampling_info.penalizer_orchestrator
            if penalizer is not None and penalizer.is_required:
                penalizer.apply(
                    logits_output.next_token_logits, repeat=verify_tokens
                )
        with spec_stage_span("verify_sample"):
            predict, accept_lens, accept_index = eagle_sample(
                spec_info, batch, logits_output, vocab_mask
            )
        new_seq_lens = scheduler_seq_lens + accept_lens
        if on_publish is not None:
            # FutureMap publishes only next-iteration lengths. Shared-pool
            # mutation remains fenced by the WAR event recorded after DVR
            # rollback/checkpoint, so host scheduling may overlap this tail.
            on_publish(new_seq_lens)
        has_verify_tokens = (
            not batch.forward_mode.is_idle() and accept_lens.numel() > 0
        )

        if has_verify_tokens:
            accept_tokens = predict[accept_index]
            bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
            fill_bonus_tokens[(accept_lens.shape[0],)](
                accept_tokens,
                accept_lens,
                bonus_tokens,
                accept_index.shape[1],
            )
        else:
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

        if self.is_dvr_eagle:
            next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)
        else:
            next_draft_input = self._self_draft_input(bonus_tokens)

        with spec_stage_span("dvr_rollback"):
            rollback_actions = self.linear_state.rollback_after_verify(
                batch=batch,
                ctx=linear_state_ctx,
                accept_lens=accept_lens,
            )

        if linear_state_ctx is not None:
            # FutureMap readiness must cover the state consumed by the next
            # overlap forward, not only accepted sequence lengths.
            with spec_stage_span("dvr_checkpoint"):
                self.linear_state.backup_boundary_state(batch, ctx=linear_state_ctx)
        elif self.is_dvr_eagle:
            commit_mamba_states_after_verify(
                self.target_worker,
                batch,
                accept_lens,
                accept_index,
                verify_tokens,
            )
        # FutureMap intentionally publishes accepted lengths before this tail.
        # Fence the target state consumed by the next DVR iteration separately;
        # for self-draft the same event is Scheduler's final shared-pool reader.
        self._rollback_ready_event.record()
        # The overlap scheduler may process the preceding result as soon as
        # this worker returns. Defer its shared-pool writes until the current
        # target rollback/checkpoint has finished reading those pools.
        if not self.is_dvr_eagle:
            self.model_runner.war_fastpath_read_done_event = self._rollback_ready_event
        if has_verify_tokens and batch.return_logprob:
            with spec_stage_span("verify_logprob"):
                compute_spec_v2_logprobs(
                    batch,
                    logits_output,
                    predict,
                    accept_index,
                    spec_info.spec_steps,
                )

        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            speculative_num_draft_tokens=self.num_draft_tokens,
            dvr_rollback_actions=rollback_actions,
            result_process_ready_event=self._rollback_ready_event,
            routed_experts_output=forward_output.routed_experts_output,
            indexer_topk_output=forward_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )
        return batch_result
