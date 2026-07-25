from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.speculative.dvr_cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
    dvr_draft_decode_context,
    validate_dvr_attention_backend,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dvr_state_flow import (
    DVRStateCommitPlan,
    DVRStateLifecycle,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.triton_ops.eagle import fill_bonus_tokens
from sglang.srt.speculative.eagle_utils import (
    eagle_prepare_for_verify,
    eagle_sample,
)
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    record_stream_each,
    record_stream_for_v2_verify,
    spec_stage_span,
)
from sglang.srt.utils.async_probe import maybe_detect_inf, maybe_detect_nan
from sglang.srt.utils.common import get_available_gpu_memory

logger = logging.getLogger(__name__)


class SelfDraftBackend:
    """Self-draft operations around the common DVR target transaction."""

    target_capture_hidden_mode = CaptureHiddenMode.NULL

    def __init__(self, owner):
        self.owner = owner
        self.graph_runner = None
        self.proposal_prob_buffer = None

    def context(self):
        return dvr_draft_decode_context(
            self.owner.model_runner,
            self.owner.draft_graph_buffers,
            self_draft=True,
        )

    def idle_input(self):
        return EagleDraftInput.create_idle_input(
            device=self.owner.device,
            hidden_size=None,
            dtype=None,
            topk=1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    @staticmethod
    def input(bonus_tokens: torch.Tensor) -> EagleDraftInput:
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

    def finish_prefill(self, batch, batch_result):
        draft_input = self.input(batch_result.next_token_ids)
        batch.spec_info = draft_input
        return draft_input

    def propose(self, batch: ScheduleBatch) -> EagleVerifyInput:
        owner = self.owner
        if batch.forward_mode.is_idle():
            batch.spec_info = self.idle_input()
            return EagleVerifyInput.create_idle_input(
                1, owner.num_draft_steps, owner.num_draft_tokens
            )

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        # ScheduleBatch.prepare_for_decode already reserved the speculative
        # window. Self draft and target verify share it.
        offsets = batch.seq_lens.to(torch.long).unsqueeze(
            1
        ) + owner.chain_position_offsets.unsqueeze(0)
        rows = batch.req_pool_indices.to(torch.long).unsqueeze(1)
        batch.out_cache_loc = batch.req_to_token_pool.req_to_token[
            rows, offsets
        ].reshape(-1)
        batch.mamba_track_indices = None
        batch.mamba_track_mask = None
        batch.mamba_track_seqlens = None
        spec_info.positions = batch.seq_lens.clone()
        spec_info.num_tokens_per_req = 1
        spec_info.num_tokens_for_logprob_per_req = 1
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

        saved_return_logprob = batch.return_logprob
        saved_return_hidden_states = batch.return_hidden_states
        batch.return_logprob = False
        batch.return_hidden_states = False
        try:
            forward_batch = ForwardBatch.init_new(batch, owner.model_runner)
            draft_tokens, draft_probs = self.draft_tokens(forward_batch)
        finally:
            batch.return_logprob = saved_return_logprob
            batch.return_hidden_states = saved_return_hidden_states

        batch_size = draft_tokens.shape[0]
        positions = (
            batch.seq_lens[:, None] + owner.chain_position_offsets[None, :]
        ).flatten()
        return EagleVerifyInput(
            draft_token=draft_tokens.flatten(),
            custom_mask=None,
            positions=positions,
            retrieve_index=owner.chain_retrieve_index[:batch_size],
            retrieve_next_token=owner.chain_retrieve_next[:batch_size],
            retrieve_next_sibling=owner.chain_retrieve_sibling[:batch_size],
            retrieve_cum_len=None,
            spec_steps=owner.num_draft_steps,
            topk=1,
            draft_token_num=owner.num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            draft_probs=draft_probs,
        )

    def draft_tokens(self, forward_batch: ForwardBatch):
        owner = self.owner
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        out_cache_loc = (
            forward_batch.out_cache_loc.reshape(
                forward_batch.batch_size, owner.num_draft_tokens
            )
            .transpose(0, 1)
            .contiguous()
        )
        draft_tokens = [spec_info.bonus_tokens.to(torch.long)]
        draft_probs = None

        origin_seq_lens = forward_batch.seq_lens
        origin_seq_lens_cpu = forward_batch.seq_lens_cpu
        origin_seq_lens_sum = forward_batch.seq_lens_sum
        origin_spec_info = forward_batch.spec_info
        origin_out_cache_loc = forward_batch.out_cache_loc
        forward_batch.spec_info = None
        position_offset = 0

        try:
            # The target model drafts a fixed topk=1 chain; no tree ranking or
            # parent bookkeeping is needed between decode steps.
            for step in range(owner.num_draft_steps):
                if step:
                    forward_batch.positions.add_(1)
                    position_offset += 1
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
                    else origin_seq_lens_sum + (step + 1) * forward_batch.batch_size
                )
                logits_output = self.decode_forward(forward_batch)
                maybe_detect_nan(
                    logits_output.next_token_logits, f"dvr draft step {step}"
                )
                next_token_ids = owner.model_runner.sample(logits_output, forward_batch)
                if not forward_batch.sampling_info.is_all_greedy:
                    sampling_probs = logits_output.next_token_logits
                    sampling_info = forward_batch.sampling_info
                    if (
                        owner.model_runner.sampler.use_log_softmax_logprob
                        and not sampling_info.need_top_p_sampling
                        and not sampling_info.need_top_k_sampling
                        and not sampling_info.need_min_p_sampling
                    ):
                        sampling_probs = torch.softmax(sampling_probs, dim=-1)
                    proposal = owner.model_runner.sampler.normalize_probs(
                        sampling_probs,
                        sampling_info,
                    )
                    if self.proposal_prob_buffer is None:
                        self.proposal_prob_buffer = torch.empty(
                            (
                                owner.chain_retrieve_index.shape[0],
                                owner.num_draft_steps,
                                proposal.shape[-1],
                            ),
                            dtype=torch.float32,
                            device=proposal.device,
                        )
                    proposal_buffer = self.proposal_prob_buffer
                    proposal_buffer[: forward_batch.batch_size, step].copy_(proposal)
                    draft_probs = proposal_buffer[
                        : forward_batch.batch_size, : owner.num_draft_steps
                    ]
                draft_tokens.append(next_token_ids.to(torch.long))
        finally:
            forward_batch.seq_lens = origin_seq_lens
            forward_batch.seq_lens_cpu = origin_seq_lens_cpu
            forward_batch.seq_lens_sum = origin_seq_lens_sum
            forward_batch.spec_info = origin_spec_info
            if position_offset:
                forward_batch.positions.sub_(position_offset)
            forward_batch.out_cache_loc = origin_out_cache_loc

        return torch.stack(draft_tokens, dim=1), draft_probs

    def decode_forward(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        if self.graph_runner is not None and self.graph_runner.can_run_graph(
            forward_batch
        ):
            return self.graph_runner.execute(forward_batch)

        seq_lens = forward_batch.seq_lens_cpu
        min_seq_len = int(seq_lens.min()) if seq_lens is not None else "GPU-only"
        capture_bs = [] if self.graph_runner is None else self.graph_runner.capture_bs
        raise RuntimeError(
            "DVR self-draft decode requires the dedicated CUDA graph; no eager "
            "fallback is used. The current batch cannot run it: "
            f"batch_size={forward_batch.batch_size}, min_seq_len={min_seq_len}, "
            f"capture_bs={capture_bs}. For batch-size misses, use the default "
            "CUDA graph batch sizes or include the running batch size "
            "in --cuda-graph-bs/--cuda-graph-max-bs."
        )

    def finish_verify(self, batch, batch_result):
        del batch
        batch_result.next_draft_input = self.input(
            batch_result.next_draft_input.bonus_tokens
        )


class EagleDraftBackend:
    """Upstream EAGLE/MTP draft operations around the DVR target transaction."""

    target_capture_hidden_mode = CaptureHiddenMode.FULL

    def __init__(self, owner, worker):
        self.owner = owner
        self.worker = worker

    @contextmanager
    def context(self):
        with (
            self.worker.draft_tp_context(self.worker.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            dvr_draft_decode_context(
                self.worker.draft_runner,
                self.owner.draft_graph_buffers,
                extra_attn_backends=self.owner.spec_v2_attn_backends[1:],
            ),
        ):
            yield

    def idle_input(self):
        return EagleDraftInput.create_idle_input(
            device=self.owner.device,
            hidden_size=EagleDraftInput.hidden_size_for(self.worker),
            dtype=EagleDraftInput.dtype_for(self.worker),
            topk=1,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            vocab_size=self.owner.target_worker.model_config.vocab_size,
        )

    def finish_prefill(self, batch, batch_result):
        with self.context():
            return self.worker._draft_extend_for_prefill(
                batch,
                batch_result.logits_output.hidden_states,
                batch_result.next_token_ids,
                batch_result.logits_output.mm_input_embeds,
            )

    def propose(self, batch):
        return self.worker.draft(batch)

    def finish_verify(self, batch, batch_result):
        with self.context(), spec_stage_span("dvr_rollback_draft"):
            self.worker._draft_extend_for_decode(batch, batch_result)


class DecodeVerifyRollbackWorker(BaseSpecWorker):
    """DVR worker with pluggable self-decode or EAGLE/MTP draft execution.

    User-visible "spec v1" is a synchronous compatibility mode over this same
    worker. Both draft backends enter the same target verify, state rollback,
    output, and v1/v2 flow.
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
        self.server_args = server_args
        self.target_model_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.device = server_args.device
        self.num_draft_steps = server_args.speculative_num_steps
        self.num_draft_tokens = server_args.speculative_num_draft_tokens
        self.uses_eagle_draft = self.model_runner.spec_algorithm.is_dvr_eagle()
        device_module = torch.get_device_module(self.device)
        self.verify_plan_stream = None
        self.verify_plan_stream_ctx = nullcontext()
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        self.state_lifecycle = DVRStateLifecycle(
            server_args=server_args,
            model_runner=self.model_runner,
        )
        # A one-token prefill cannot seed every draft backend. Consume one
        # target-only verify before normal draft; pool slots are overwritten on
        # every prefill, so stale request identities cannot survive slot reuse.
        self.pending_seed_rows = set()
        self.draft_graph_buffers = {}
        max_bs = max(
            server_args.cuda_graph_config.decode.max_bs or 0,
            server_args.max_running_requests or 0,
            1,
        )
        num_tokens = self.num_draft_tokens
        self.chain_retrieve_index = torch.arange(
            max_bs * num_tokens, dtype=torch.long, device=self.device
        ).view(max_bs, num_tokens)
        next_token = torch.arange(
            1, num_tokens + 1, dtype=torch.long, device=self.device
        )
        next_token[-1] = -1
        self.chain_retrieve_next = next_token.repeat(max_bs, 1)
        self.chain_retrieve_sibling = torch.full(
            (max_bs, num_tokens), -1, dtype=torch.long, device=self.device
        )
        self.chain_position_offsets = torch.arange(
            num_tokens, dtype=torch.long, device=self.device
        )
        if self.uses_eagle_draft:
            server_args.context_length = (
                target_worker.model_runner.model_config.context_len
            )
            self.draft_model_worker = EagleDraftWorker(
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
            if any(
                isinstance(module, RadixLinearAttention)
                or callable(getattr(module, "_forward_mamba", None))
                for module in self.draft_worker.draft_runner.model.modules()
            ):
                raise NotImplementedError(
                    "DVR EAGLE requires a stateless/full-attention draft model; "
                    "recurrent draft-model state is not managed by DVR."
                )
            # Reuse the upstream EAGLE/MTP worker as the draft backend. DVR owns
            # target verify and rollback, not a second copy of EAGLE draft logic.
            if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
                self.verify_plan_stream = device_module.Stream()
                self.verify_plan_stream_ctx = device_module.stream(
                    self.verify_plan_stream
                )
            self.draft_backend = EagleDraftBackend(self, self.draft_worker)
            log_prefix = "DVR EAGLE"
        else:
            del dp_rank, moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port
            self.draft_model_worker = None
            self.draft_backend = SelfDraftBackend(self)
            log_prefix = "DVR self-decode"

        logger.info(
            "Initialized %s worker: num_steps=%s, num_draft_tokens=%s",
            log_prefix,
            self.num_draft_steps,
            self.num_draft_tokens,
        )

    @property
    def target_worker(self):
        return self.target_model_worker

    @property
    def draft_worker(self):
        return self.draft_model_worker

    def init_attention_backends(self):
        if self.uses_eagle_draft:
            self.draft_worker.init_attention_backends()
        # Self-DVR target worker owns the model and attention backend. Scheduler
        # already initializes it before calling this self-draft worker hook.
        target_verify_backends, state_adapter = validate_dvr_attention_backend(
            self.model_runner.attn_backend,
            ForwardMode.TARGET_VERIFY,
            phase="target verify",
        )
        self.target_verify_attn_backends = tuple(target_verify_backends)
        self.state_lifecycle.bind_state_adapter(state_adapter)

    def init_cuda_graphs(self):
        if self.uses_eagle_draft:
            draft_worker = self.draft_worker
            with dvr_draft_decode_context(
                draft_worker.draft_runner,
                self.draft_graph_buffers,
                capture=True,
                extra_attn_backends=self.spec_v2_attn_backends[1:],
            ):
                draft_worker.init_cuda_graphs()
            return

        # Capture the dedicated self-draft decode graph after target attention
        # backends exist. This matches upstream's separated init order.
        draft_backend = self.draft_backend
        if (
            draft_backend.graph_runner is None
            and self.server_args.cuda_graph_config.decode.backend != Backend.DISABLED
            and not self.server_args.disable_draft_cuda_graph
        ):
            before_mem = get_available_gpu_memory(self.device, self.model_runner.gpu_id)
            logger.info(
                "Capture DVR self-draft CUDA graph begin. avail mem=%.2f GB",
                before_mem,
            )
            with dvr_draft_decode_context(
                self.model_runner,
                self.draft_graph_buffers,
                capture=True,
                self_draft=True,
            ):
                draft_backend.graph_runner = DVRDraftDecodeCudaGraphRunner(
                    self.model_runner
                )
            after_mem = get_available_gpu_memory(self.device, self.model_runner.gpu_id)
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
        if self.uses_eagle_draft:
            self.draft_worker.alloc_memory_pool(
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
            raise RuntimeError(
                "DVR self-draft requires Scheduler to pass the target memory pools."
            )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    # Target verify. DVR keeps the forward call in TARGET_VERIFY mode like EAGLE,
    # then locally adapts GDN's physical window and state restore/commit.

    @property
    def war_fastpath_runner(self):
        if self.uses_eagle_draft:
            return self.draft_worker.draft_runner
        return self.target_worker.model_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        if not self.uses_eagle_draft:
            return (self.target_worker.model_runner.attn_backend,)
        return (
            self.target_worker.model_runner.attn_backend,
            self.draft_worker.draft_attn_backend,
            self.draft_worker.draft_extend_attn_backend
            or self.draft_worker.draft_runner.attn_backend,
        )

    def iter_runners(self):
        if self.uses_eagle_draft:
            return [("draft", self.draft_worker.draft_runner)]
        return []

    def update_weights_from_disk(self, recv_req):
        if not self.uses_eagle_draft:
            # The scheduler has already updated the target model. Self-draft
            # shares those weights and must not load them a second time.
            return True, "Succeeded to update model weights."
        success, message = self.draft_worker.draft_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req):
        if not self.uses_eagle_draft:
            return True, "Succeeded to update model weights."
        success, message = self.draft_worker.draft_runner.update_weights_from_ipc(
            recv_req
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def clear_cache_pool(self):
        self.state_lifecycle.clear_cache_state()
        self.pending_seed_rows.clear()

    def prepare_for_kv_cache_release(self, req) -> None:
        if req.req_pool_idx is not None:
            self.pending_seed_rows.discard(int(req.req_pool_idx))
        if getattr(self, "device", "cpu") != "cpu":
            rollback_done = self.war_fastpath_runner.war_fastpath_read_done_event
            if rollback_done is not None:
                # Overlap may already have launched one extra DVR round. Order
                # checkpoint selection and slot donation after that round without
                # synchronizing the host or changing the steady-state schedule.
                torch.get_device_module(self.device).current_stream().wait_event(
                    rollback_done
                )
        self.state_lifecycle.prepare_for_cache_release(req)

    def forward_batch_generation(
        self, model_worker_batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        batch = model_worker_batch
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch.capture_hidden_mode = self.draft_backend.target_capture_hidden_mode
            self.state_lifecycle.prepare_target_extend(batch)
            batch_result = self.target_worker.forward_batch_generation(batch)
            batch_result.new_seq_lens = batch.seq_lens
            self.state_lifecycle.finish_target_extend(batch)
            decoding_rids = {req.rid for req in batch.decoding_reqs or ()}
            for req in batch.reqs:
                request_slot = int(req.req_pool_idx)
                self.pending_seed_rows.discard(request_slot)
                if req.rid not in decoding_rids and len(req.origin_input_ids) <= 1:
                    self.pending_seed_rows.add(request_slot)
            if on_publish is not None:
                on_publish(batch_result.new_seq_lens)
            batch_result.next_draft_input = self.draft_backend.finish_prefill(
                batch, batch_result
            )
            return batch_result

        # DVR decode has one shared core: draft -> target verify -> rollback.
        if batch.spec_info is None:
            batch.spec_info = self.draft_backend.idle_input()
        if batch.batch_size() > self.chain_retrieve_index.shape[0]:
            raise RuntimeError(
                "DVR decode batch exceeds its fixed chain buffers: "
                f"batch_size={batch.batch_size()}, "
                f"capacity={self.chain_retrieve_index.shape[0]}."
            )
        needs_seed_verify = False
        for req in batch.reqs:
            request_slot = int(req.req_pool_idx)
            if request_slot in self.pending_seed_rows:
                needs_seed_verify = True
                self.pending_seed_rows.discard(request_slot)
        with spec_stage_span("dvr_prepare"):
            state_commit_plan = self.state_lifecycle.prepare_for_draft(batch)
        if needs_seed_verify:
            verify_input = self.build_root_only_verify_input(batch)
        else:
            with self.draft_backend.context(), spec_stage_span("draft"):
                verify_input = self.draft_backend.propose(batch)
        assert verify_input.is_verify_input()
        batch.spec_info = verify_input
        batch_result = self.verify(
            batch,
            verify_input,
            state_commit_plan=state_commit_plan,
            on_publish=on_publish,
        )
        final_reader = self.war_fastpath_runner
        # Discard a draft-phase or synchronous previous-iteration event. Only
        # the final rollback reader may release scheduler result writes.
        final_reader.war_fastpath_read_done_event = None
        # Self prepares its next chain input; EAGLE catches its private cache up
        # to the same accepted target endpoint.
        self.draft_backend.finish_verify(batch, batch_result)

        # Publish one fence for the complete rollback transaction. The EAGLE
        # graph records this as soon as draft-extend snapshots shared pools;
        # self-draft (and an eager EAGLE miss) records it here after its final
        # shared-pool reader has been enqueued.
        rollback_done = final_reader.war_fastpath_read_done_event
        if rollback_done is None:
            rollback_done = torch.get_device_module(self.device).Event()
            rollback_done.record()
            final_reader.war_fastpath_read_done_event = rollback_done
        return batch_result

    def build_root_only_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        """Build a fixed-width verify input whose logical tree is only the root."""

        draft_input = batch.spec_info
        assert isinstance(draft_input, EagleDraftInput)
        batch_size = batch.seq_lens.shape[0]
        width = self.num_draft_tokens
        retrieve_index = self.chain_retrieve_index[:batch_size]
        terminal = self.chain_retrieve_sibling[:batch_size]
        # Keep the physical verify shape identical to the captured DVR graph.
        # spec_steps=0 makes every padded node unreachable, so sampling accepts
        # only the root while attention/GDN retain their fixed-shape contract.
        return EagleVerifyInput(
            draft_token=(
                draft_input.bonus_tokens.to(torch.long).repeat_interleave(width)
            ),
            custom_mask=None,
            positions=(
                batch.seq_lens[:, None] + self.chain_position_offsets[None, :]
            ).reshape(-1),
            retrieve_index=retrieve_index,
            retrieve_next_token=terminal,
            retrieve_next_sibling=terminal,
            retrieve_cum_len=None,
            spec_steps=0,
            topk=1,
            draft_token_num=width,
            capture_hidden_mode=self.draft_backend.target_capture_hidden_mode,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def verify(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        state_commit_plan: Optional[DVRStateCommitPlan] = None,
        on_publish=None,
    ) -> GenerationBatchResult:
        scheduler_seq_lens = batch.seq_lens
        assert spec_info.is_verify_input()
        # DVR only supports topk=1 chains, whose tree mask is exactly the
        # backend's native causal mask. Both draft backends therefore enter the
        # same target-verify preparation and forward path.
        spec_info.custom_mask = None
        verify_tokens = spec_info.draft_token_num
        spec_info.num_tokens_per_req = verify_tokens
        if self.uses_eagle_draft:
            record_stream_for_v2_verify(
                batch,
                spec_info,
                torch.get_device_module(batch.device).current_stream(),
            )
        else:
            # ForwardBatch.init_new consumes this one-shot host-length mirror.
            batch.seq_lens_cpu_cache = spec_info.seq_lens_cpu

        with self.verify_plan_stream_ctx, spec_stage_span("verify_prepare"):
            verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
                spec_info,
                self.req_to_token_pool,
                batch,
                self.target_worker,
            )

        current_stream = torch.get_device_module(self.device).current_stream()
        if self.uses_eagle_draft:
            record_stream_each((batch.input_ids, batch.out_cache_loc), current_stream)
        if self.verify_plan_stream is not None:
            current_stream.wait_stream(self.verify_plan_stream)
            runner = self.model_runner.decode_cuda_graph_runner
            cuda_graph_bs = (
                None if not can_run_cuda_graph or runner is None else runner.bs
            )
            for backend in self.target_verify_attn_backends:
                backend.update_verify_buffers_to_fill_after_draft(
                    spec_info, cuda_graph_bs
                )

        with spec_stage_span("verify"):
            forward_output = self.target_worker.forward_batch_generation(
                batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
            )

        logits_output = forward_output.logits_output
        if self.uses_eagle_draft and logits_output.hidden_states is None:
            raise RuntimeError(
                "DVR EAGLE target verify must return hidden states for the next "
                "draft step."
            )

        # Target acceptance semantics are independent of the draft backend.
        # Accumulated penalties and logit bias are applied by eagle_sample.
        if spec_info.spec_steps > 0 and not batch.sampling_info.is_all_greedy:
            expected_shape = (
                len(batch.seq_lens),
                verify_tokens - 1,
                logits_output.next_token_logits.shape[-1],
            )
            if (
                spec_info.draft_probs is None
                or tuple(spec_info.draft_probs.shape) != expected_shape
            ):
                actual_shape = (
                    None
                    if spec_info.draft_probs is None
                    else tuple(spec_info.draft_probs.shape)
                )
                raise ValueError(
                    "DVR rejection sampling requires one target-vocabulary "
                    "proposal row per draft edge; got "
                    f"{actual_shape}, expected {expected_shape}."
                )
        with spec_stage_span("verify_sample"):
            maybe_detect_nan(
                logits_output.next_token_logits, "verify: target model logits"
            )
            maybe_detect_inf(
                logits_output.next_token_logits, "verify: target model logits"
            )
            predict, accept_lens, accept_index = eagle_sample(
                spec_info,
                batch,
                logits_output,
                target_prob_filter=self.model_runner.sampler.normalize_probs,
            )
            if not batch.forward_mode.is_idle() and accept_lens.numel() > 0:
                accept_tokens = predict[accept_index]
                bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
                fill_bonus_tokens[(accept_lens.shape[0],)](
                    accept_tokens,
                    accept_lens,
                    bonus_tokens,
                    accept_index.shape[1],
                )
            else:
                bonus_tokens = torch.empty(
                    (0,), device=predict.device, dtype=torch.int32
                )
        new_seq_lens = scheduler_seq_lens + accept_lens
        if on_publish is not None:
            # FutureMap publishes only next-iteration lengths. Shared-pool
            # mutation remains fenced by the WAR event recorded after DVR
            # rollback/checkpoint, so host scheduling may overlap this tail.
            on_publish(new_seq_lens)
        has_verify_tokens = not batch.forward_mode.is_idle() and accept_lens.numel() > 0

        next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)

        with spec_stage_span("dvr_rollback"):
            self.state_lifecycle.commit_verified_state(
                batch=batch,
                plan=state_commit_plan,
                accept_lens=accept_lens,
            )
        if state_commit_plan is None:
            commit_mamba_states_after_verify(
                self.target_worker,
                batch,
                accept_lens,
                accept_index,
                verify_tokens,
            )
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
            routed_experts_output=forward_output.routed_experts_output,
            indexer_topk_output=forward_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )
        return batch_result
