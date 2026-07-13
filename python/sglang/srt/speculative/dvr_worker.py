from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from typing import List, Optional

import torch

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
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.dvr_draft_cuda_graph_runner import (
    dvr_draft_decode_context,
    iter_dvr_attention_backends,
)
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dvr_state_flow import (
    DVRLinearStateLifecycle,
    rollback_dvr_verify,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_info_v2 import fill_bonus_tokens
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.eagle_utils import (
    eagle_prepare_for_verify,
    eagle_sample,
)
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    generate_token_bitmask,
    record_stream_each,
    record_stream_for_v2_verify,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda
from sglang.srt.utils.async_probe import maybe_detect_nan
from sglang.srt.utils.async_probe import maybe_detect_inf
from sglang.srt.utils.common import is_npu

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
    )

logger = logging.getLogger(__name__)
_is_npu = is_npu()


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
        return EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=(
                EagleDraftInput.hidden_size_for(self._draft_worker)
                if self.is_dvr_eagle
                else None
            ),
            dtype=(
                EagleDraftInput.dtype_for(self._draft_worker)
                if self.is_dvr_eagle
                else None
            ),
            topk=self.topk,
            capture_hidden_mode=(
                CaptureHiddenMode.LAST
                if self.is_dvr_eagle
                else CaptureHiddenMode.NULL
            ),
            vocab_size=(
                self.target_worker.model_config.vocab_size
                if self.is_dvr_eagle
                else 0
            ),
        )

    def _sample_target_verify(
        self,
        *,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        logits_output: LogitsProcessorOutput,
        vocab_mask: Optional[torch.Tensor] = None,
    ):
        if not self.is_dvr_eagle:
            # Upstream EAGLE applies the shared penalties, bias, grammar, and
            # accept kernel. Self-DVR only needs to add processors that its
            # target-only forward does not run before entering that common path.
            sampling_info = batch.sampling_info
            if sampling_info.has_custom_logit_processor:
                apply_custom_logit_processor(
                    logits_output.next_token_logits,
                    sampling_info,
                    num_tokens_in_batch=self.num_draft_tokens,
                )
            penalizer = sampling_info.penalizer_orchestrator
            if penalizer is not None and penalizer.is_required:
                penalizer.apply(
                    logits_output.next_token_logits, repeat=self.num_draft_tokens
                )
        return eagle_sample(
            spec_info,
            batch,
            logits_output,
            vocab_mask,
            use_rejection_sampling=spec_info.draft_probs is not None,
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
            and not self.server_args.disable_cuda_graph
            and not self.server_args.disable_draft_cuda_graph
        ):
            with dvr_draft_decode_context(
                self.model_runner, capture=True, self_draft=True
            ):
                self.cuda_graph_runner_for_draft_decode = DecodeCudaGraphRunner(
                    self.model_runner
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

        self._draft_preprocess_decode_for_self_dvr(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        spec_info.num_tokens_per_req = 1
        spec_info.num_tokens_for_logprob_per_req = 1
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
        batch.return_hidden_states = False

        saved_return_logprob = batch.return_logprob
        batch.return_logprob = False
        try:
            forward_batch = ForwardBatch.init_new(batch, self.model_runner)
            if forward_batch.seq_lens_cpu is None:
                # Normal decode reuses the one-shot host mirror prepared with
                # the boundary oracle below. Keep this defensive path for
                # direct unit/integration callers that bypass that entrypoint.
                forward_batch.seq_lens_cpu = torch.tensor(
                    batch.seq_lens.detach().cpu().tolist(),
                    dtype=torch.int64,
                    device="cpu",
                )
            elif not torch.is_tensor(forward_batch.seq_lens_cpu):
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
                    draft_probs_list.append(
                        self.get_draft_sampling_probs(
                            forward_batch, logits_output.next_token_logits
                        )
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
            output = self.cuda_graph_runner_for_draft_decode.execute(forward_batch)
            # DecodeCudaGraphRunner treats an ordinary decode graph as the
            # step's last shared-pool reader. A self-draft graph is only a
            # provisional phase; target verify still reads req_to_token/state
            # pools. Do not let Scheduler's WAR barrier consume this early
            # event after the DVR worker returns.
            self.model_runner.war_fastpath_read_done_event = None
            return output

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
        if sampling_info.need_min_p_sampling:
            min_p_thresholds = (
                sampling_probs.amax(dim=-1, keepdim=True)
                * sampling_info.min_ps.unsqueeze(-1)
            )
            sampling_probs = torch.where(
                sampling_probs >= min_p_thresholds,
                sampling_probs,
                torch.zeros_like(sampling_probs),
            )
            sampling_probs /= sampling_probs.sum(dim=-1, keepdim=True)
        return sampling_probs

    # Target verify. DVR keeps the forward call in TARGET_VERIFY mode like EAGLE,
    # then locally adapts GDN's physical window and state restore/commit.

    def _prepare_dvr_boundary_for_verify(self, batch: ScheduleBatch) -> None:
        if batch.forward_mode.is_idle():
            return

        prefill_prefix_lens = None
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            prefill_prefix_lens = [int(x) for x in batch.prefix_lens]
        seq_lens_cpu = self.linear_state.batch_seq_lens_cpu(batch)
        if not self.is_dvr_eagle and prefill_prefix_lens is None:
            # ForwardBatch.init_new consumes this one-shot cache. Reuse the
            # boundary oracle's required D2H result instead of synchronizing a
            # second time when self draft builds its graph input.
            batch.seq_lens_cpu_cache = torch.tensor(
                seq_lens_cpu, dtype=torch.int64, device="cpu"
            )
        self.linear_state.prepare_for_draft(
            batch,
            seq_lens_cpu=seq_lens_cpu,
            prefill_prefix_lens=prefill_prefix_lens,
            defer_request_publish=prefill_prefix_lens is not None,
        )
        # EAGLE/MTP draft can run before the next target verify in overlap mode.
        # Preserve the target checkpoint authority across draft-side mutations.
        self.linear_state.backup_boundary_state(batch, preserve_existing=True)

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

    def _run_decode_draft_verify_rollback(
        self,
        batch: ScheduleBatch,
        on_publish=None,
    ) -> GenerationBatchResult:
        """Run one decode step through DVR's core draft -> verify -> rollback path."""

        if batch.spec_info is None:
            batch.spec_info = self._idle_draft_input()
        for req in batch.reqs:
            if len(req.origin_input_ids) <= 1:
                raise RuntimeError(
                    "DVR does not support one-token synthetic prompts on gated "
                    "linear-state models because the first draft/verify graph step "
                    "hits the seq_len<=2 state-input boundary. Use a normal "
                    "chat-template prompt or disable DVR for this tiny prompt test."
                )
        self._prepare_dvr_boundary_for_verify(batch)

        verify_input = self.draft(batch)
        assert verify_input.is_verify_input()
        batch.spec_info = verify_input
        return self.verify(batch, verify_input, on_publish=on_publish)

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
            if self.is_dvr_eagle:
                if on_publish is not None:
                    on_publish(batch_result.new_seq_lens)
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
                batch.spec_info = EagleDraftInput(
                    hidden_states=None,
                    bonus_tokens=next_token_ids,
                    topk_p=torch.ones(
                        (next_token_ids.shape[0], 1),
                        dtype=torch.float32,
                        device=next_token_ids.device,
                    ),
                    topk_index=next_token_ids.to(torch.long).unsqueeze(-1),
                    num_tokens_per_req=1,
                    num_tokens_for_logprob_per_req=1,
                    capture_hidden_mode=CaptureHiddenMode.NULL,
                )
                batch_result.next_draft_input = batch.spec_info
                if on_publish is not None:
                    on_publish(batch_result.new_seq_lens)
            return batch_result

        batch_result = self._run_decode_draft_verify_rollback(
            batch, on_publish=on_publish
        )
        if self.is_dvr_eagle:
            with self._draft_context():
                self.draft_worker._draft_extend_for_decode(batch, batch_result)
        return batch_result

    def _draft_preprocess_decode_for_self_dvr(self, batch: ScheduleBatch):
        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        penalizer_orchestrator = batch.sampling_info.penalizer_orchestrator
        if penalizer_orchestrator is not None and penalizer_orchestrator.is_required:
            penalizer_orchestrator.cumulate_output_tokens(
                spec_info.bonus_tokens.to(torch.int64)
            )

        # ScheduleBatch.prepare_for_decode already reserved the speculative
        # decode window and wrote it into req_to_token. Self-DVR draft and verify
        # share that window; allocating a second one leaks KV ownership.
        offsets = batch.seq_lens.to(torch.long).unsqueeze(1) + torch.arange(
            self.num_draft_tokens, dtype=torch.long, device=batch.seq_lens.device
        ).unsqueeze(0)
        rows = batch.req_pool_indices.to(torch.long).unsqueeze(1)
        batch.out_cache_loc = batch.req_to_token_pool.req_to_token[
            rows, offsets
        ].reshape(-1)
        batch.return_hidden_states = False
        batch.mamba_track_indices = None
        batch.mamba_track_mask = None
        batch.mamba_track_seqlens = None
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

    def draft(self, batch: ScheduleBatch) -> EagleVerifyInput:
        with self._draft_context():
            if self.is_dvr_eagle:
                return self._draft_worker.draft(batch)
            return self._draft_self(batch)

    def verify(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        on_publish=None,
    ) -> GenerationBatchResult:
        scheduler_seq_lens = batch.seq_lens
        vocab_mask = None
        extra_keep_alive_refs = None

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrieve_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrieve_next_token.shape
            ).cpu()

        if self.is_dvr_eagle:
            fwd_stream = torch.get_device_module(self.device).current_stream()
            assert spec_info.is_verify_input()
            # DVR only supports topk=1 chains, whose tree mask is exactly the
            # backend's native causal mask. Keep graph and eager verify on that
            # native path instead of carrying EAGLE tree-mask metadata.
            spec_info.custom_mask = None
            record_stream_for_v2_verify(batch, spec_info, fwd_stream)
            spec_info.num_tokens_per_req = self.num_draft_tokens
            bs = len(batch.seq_lens)

            prepared_on_plan_stream = self.plan_stream is not None
            with self.plan_stream_ctx:
                verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
                    spec_info,
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
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

            linear_state_ctx = self.linear_state.restore_for_verify(batch)
            forward_output = self.target_worker.forward_batch_generation(
                batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
            )

            extra_keep_alive_refs = [verify_forward_batch]
            error_prefix = "DVR EAGLE"
        else:
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

            linear_state_ctx = self.linear_state.restore_for_verify(batch)
            batch.seq_lens_cpu_cache = spec_info.seq_lens_cpu
            # Exact output logprobs are derived from target-verify logits below;
            # the forward itself only needs accept/reject and GDN commit data.
            need_return_logprob = batch.return_logprob
            batch.return_logprob = False
            try:
                forward_output = self.target_worker.forward_batch_generation(
                    batch, is_verify=True
                )
            finally:
                batch.return_logprob = need_return_logprob
            # The provisional decode graphs deliberately discard their
            # ordinary-DECODE WAR events. Target verify is self-DVR's final
            # shared-pool reader, so publish one authoritative event only after
            # its replay has completed. Later sampling, logprob, and state
            # commit can overlap with scheduling without exposing mutable pool
            # metadata to the verifier.
            read_done = torch.get_device_module(self.device).Event()
            read_done.record()
            self.model_runner.war_fastpath_read_done_event = read_done
            can_run_cuda_graph = forward_output.can_run_cuda_graph
            error_prefix = "DVR self"

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

        maybe_detect_nan(logits_output.next_token_logits, f"{error_prefix} verify")
        maybe_detect_inf(logits_output.next_token_logits, f"{error_prefix} verify")
        predict, accept_lens, accept_index = self._sample_target_verify(
            batch=batch,
            spec_info=spec_info,
            logits_output=logits_output,
            vocab_mask=vocab_mask,
        )
        new_seq_lens = scheduler_seq_lens + accept_lens
        if on_publish is not None:
            # Publish as soon as acceptance determines the next scheduler
            # lengths. Exact logprob, rollback, and recurrent-state commit stay
            # ordered later on the same forward stream, while host scheduling
            # can overlap with that post-verify work.
            on_publish(new_seq_lens)
        has_verify_tokens = (
            not batch.forward_mode.is_idle() and accept_lens.numel() > 0
        )

        if has_verify_tokens and batch.return_logprob:
            compute_spec_v2_logprobs(
                batch,
                logits_output,
                predict,
                accept_index,
                self.num_draft_steps,
            )

        if self.is_dvr_eagle:
            if has_verify_tokens:
                accept_tokens = predict[accept_index]
                bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
                fill_bonus_tokens[(bs,)](
                    accept_tokens,
                    accept_lens,
                    bonus_tokens,
                    accept_index.shape[1],
                )
            else:
                bonus_tokens = torch.empty(
                    (0,), device=self.device, dtype=torch.int32
                )
            next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)
        else:
            if has_verify_tokens:
                select_index = (
                    torch.arange(len(batch.seq_lens), device=self.device)
                    * self.num_draft_tokens
                    + accept_lens.to(torch.long)
                    - 1
                )
                verified_id = predict[select_index]
            else:
                verified_id = torch.empty(
                    (0,), dtype=torch.int32, device=self.device
                )
            next_draft_input = EagleDraftInput(
                hidden_states=None,
                bonus_tokens=verified_id,
                topk_p=torch.ones(
                    (verified_id.shape[0], 1),
                    dtype=torch.float32,
                    device=verified_id.device,
                ),
                topk_index=verified_id.to(torch.long).unsqueeze(-1),
                capture_hidden_mode=CaptureHiddenMode.NULL,
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
            )

        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            speculative_num_draft_tokens=self.num_draft_tokens,
            dvr_rollback_actions=rollback_dvr_verify(
                batch=batch,
                linear_state=self.linear_state,
                linear_state_ctx=linear_state_ctx,
                accept_lens=accept_lens,
            ),
            routed_experts_output=forward_output.routed_experts_output,
            indexer_topk_output=forward_output.indexer_topk_output,
            extra_keep_alive_refs=extra_keep_alive_refs,
        )
        if linear_state_ctx is not None:
            # Radix insertion may rebind a request's logical ping-pong slot
            # before the next overlap iteration. Keep the just-committed target
            # state authoritative and restore it into whichever physical slot
            # owns that logical boundary on the next verify.
            self.linear_state.backup_boundary_state(
                batch, preserve_existing=False, ctx=linear_state_ctx
            )
        elif self.is_dvr_eagle:
            commit_mamba_states_after_verify(
                self.target_worker,
                batch,
                accept_lens,
                accept_index,
                self.num_draft_tokens,
            )
        return batch_result
