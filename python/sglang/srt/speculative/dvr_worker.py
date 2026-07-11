from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from typing import List, Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.layers.utils.hash import murmur_hash32
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.layers.utils.logprob import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.model_executor.dvr_draft_cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
    DVRTargetVerifyCudaGraphRunner,
    dvr_self_draft_eager_context,
    dvr_self_draft_graph_skip_reason,
    iter_dvr_attention_backends,
    _min_seq_len_cpu,
)
from sglang.srt.speculative.dvr_info import (
    DVRPendingOutputPrefix,
    dvr_compact_output_indices,
)
from sglang.srt.speculative.dvr_core import DVRVerifyOutput, finish_dvr_verify
from sglang.srt.speculative.dvr_state_flow import (
    DVRLinearStateLifecycle,
    dvr_suffix_replay_context,
    run_dvr_suffix_replay_oracle,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_info_v2 import fill_bonus_tokens
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    eagle_prepare_for_verify,
    eagle_sample,
    organize_draft_results,
    verify_tree_greedy_func,
)
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    TREE_SPEC_KERNEL_AVAILABLE,
    commit_mamba_states_after_verify,
    generate_simulated_accept_index,
    generate_token_bitmask,
    record_stream_each,
    record_stream_for_v2_verify,
    select_top_k_tokens,
)
from sglang.srt.speculative.reject_sampling import (
    chain_speculative_sampling_triton as chain_speculative_sampling,
)
from sglang.srt.speculative.spec_info import useful_spec_proposed_drafts
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
        tree_speculative_sampling_target_only,
    )

logger = logging.getLogger(__name__)
_is_npu = is_npu()


def _get_dvr_plan_stream(device: str):
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        return plan_stream, torch.get_device_module(device).stream(plan_stream)
    return None, nullcontext()


def dvr_has_graph_unsafe_short_prompt(batch) -> bool:
    """Return whether DVR/GDN graph replay should skip the one-token edge."""

    return any(len(req.origin_input_ids) <= 1 for req in batch.reqs)


class _DVRSelfDraftBackend:
    """Self-draft backend for the unified DVR worker.

    This first refactor stage keeps the existing self-draft implementation on
    ``DecodeVerifyRollbackWorkerV2`` and routes the DVR pipeline through a small
    backend boundary.  Later stages can move the implementation behind this
    boundary without changing the target verify/commit flow again.
    """

    def __init__(self, worker: "DecodeVerifyRollbackWorkerV2"):
        self.worker = worker

    def init_cuda_graphs(self) -> None:
        worker = self.worker
        if (
            worker.cuda_graph_runner_for_draft_decode is None
            and not worker.server_args.disable_cuda_graph
            and not worker.server_args.disable_draft_cuda_graph
        ):
            worker.cuda_graph_runner_for_draft_decode = DVRDraftDecodeCudaGraphRunner(
                worker
            )

    def idle_draft_input(self) -> EagleDraftInput:
        worker = self.worker
        return EagleDraftInput.create_idle_input(
            device=worker.device,
            hidden_size=0,
            dtype=worker.model_config.dtype,
            topk=worker.topk,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    def prefill_next_draft_input(
        self,
        *,
        logits_output: LogitsProcessorOutput,
        next_token_ids: torch.Tensor,
    ) -> EagleDraftInput:
        worker = self.worker
        return EagleDraftInput(
            hidden_states=(
                logits_output.hidden_states
                if logits_output.hidden_states is not None
                else worker._dummy_hidden_states(
                    next_token_ids.shape[0], device=next_token_ids.device
                )
            ),
            bonus_tokens=next_token_ids,
            topk_p=torch.ones(
                (next_token_ids.shape[0], worker.topk),
                dtype=torch.float32,
                device=next_token_ids.device,
            ),
            topk_index=next_token_ids.to(torch.long).unsqueeze(-1),
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    @staticmethod
    def record_prefill_output_prefix(
        *,
        batch: ScheduleBatch,
        output_prefix: DVRPendingOutputPrefix,
        logits_output: LogitsProcessorOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        token_logprobs_per_req = None
        if logits_output.next_token_logprobs is not None:
            logprob_values = logits_output.next_token_logprobs.detach().cpu().tolist()
            token_logprobs_per_req = [[float(value)] for value in logprob_values]
        if not batch.forward_mode.is_idle() and batch.reqs is not None:
            output_prefix.append_batch_output_tokens(
                batch,
                [[token_id] for token_id in next_token_ids.detach().cpu().tolist()],
                token_logprobs_per_req=token_logprobs_per_req,
            )

    def draft(self, batch: ScheduleBatch) -> EagleVerifyInput:
        worker = self.worker
        if batch.forward_mode.is_idle():
            batch.spec_info = self.idle_draft_input()
            return EagleVerifyInput.create_idle_input(
                worker.topk,
                worker.num_draft_steps,
                worker.num_draft_tokens,
            )

        seq_lens_cpu = worker.linear_state.batch_seq_lens_cpu(batch)
        worker.linear_state.prepare_for_draft(
            batch,
            seq_lens_cpu=seq_lens_cpu,
            request_token_ids_for_replay=worker._request_token_ids_for_replay,
        )
        worker.linear_state.backup_boundary_state(batch, preserve_existing=False)
        worker._draft_preprocess_decode_for_self_dvr(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        # Self-draft only proposes token ids. Keep user-visible exact logprobs
        # on the target verify path; carrying return_logprob into draft decode
        # adds unnecessary logits metadata and can perturb the overlap GDN state
        # lifecycle before verify repairs it.
        return worker._run_self_draft_and_build_verify_input(
            batch,
            spec_info,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            suppress_return_logprob=True,
        )

    def sample_verify(
        self,
        *,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        logits_output: LogitsProcessorOutput,
        vocab_mask: Optional[torch.Tensor] = None,
    ):
        return self.worker._sample_verify(batch, spec_info, logits_output, vocab_mask)

    def compute_compact_logprobs(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
    ) -> None:
        self.worker._compute_compact_logprobs(
            batch, logits_output, predict, accept_index
        )

    def next_draft_input(
        self,
        *,
        batch: ScheduleBatch,
        predict: torch.Tensor,
        accept_lens: torch.Tensor,
        has_verify_tokens: bool,
    ) -> EagleDraftInput:
        worker = self.worker
        if has_verify_tokens:
            select_index = (
                torch.arange(len(batch.seq_lens), device=worker.device)
                * worker.num_draft_tokens
                + accept_lens.to(torch.long)
                - 1
            )
            verified_id = predict[select_index]
        else:
            verified_id = torch.empty((0,), dtype=torch.int32, device=worker.device)

        return EagleDraftInput(
            hidden_states=worker._dummy_hidden_states(
                verified_id.shape[0], device=verified_id.device
            ),
            bonus_tokens=verified_id,
            topk_p=torch.ones(
                (verified_id.shape[0], worker.topk),
                dtype=torch.float32,
                device=verified_id.device,
            ),
            topk_index=verified_id.to(torch.long).unsqueeze(-1),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )


class _DVREagleDraftBackend:
    """EAGLE/MTP draft-side adapter for DVR.

    DVR owns target verify and GDN rollback; EAGLE/MTP still owns the draft
    model, draft KV, and draft-extend state. Keeping this boundary explicit
    lets self draft and EAGLE share the target-side DVR flow without copying
    EAGLE draft internals.
    """

    def __init__(self, worker: "DecodeVerifyRollbackWorkerV2"):
        self.worker = worker

    @contextmanager
    def draft_context(self):
        draft_worker = self.worker.draft_worker
        with (
            draft_worker.draft_tp_context(draft_worker.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            yield

    def idle_draft_input(self) -> EagleDraftInput:
        worker = self.worker
        capture_mode = (
            CaptureHiddenMode.NULL
            if worker.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        return EagleDraftInput.create_idle_input(
            device=worker.device,
            hidden_size=EagleDraftInput.hidden_size_for(worker.draft_worker),
            dtype=EagleDraftInput.dtype_for(worker.draft_worker),
            topk=worker.topk,
            capture_hidden_mode=capture_mode,
            vocab_size=worker.target_worker.model_config.vocab_size,
        )

    def draft(self, batch: ScheduleBatch) -> EagleVerifyInput:
        with self.draft_context():
            return self.worker.draft_worker.draft(batch)

    def draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        batch_output: GenerationBatchResult,
    ) -> EagleDraftInput:
        with self.draft_context():
            return self.worker.draft_worker._draft_extend_for_prefill(
                batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
                batch_output.logits_output.mm_input_embeds,
            )

    def draft_extend_for_decode(
        self,
        batch: ScheduleBatch,
        batch_output: GenerationBatchResult,
    ) -> None:
        with self.draft_context():
            self.worker.draft_worker._draft_extend_for_decode(batch, batch_output)

    def sample_verify(
        self,
        *,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        logits_output: LogitsProcessorOutput,
        vocab_mask: Optional[torch.Tensor] = None,
    ):
        return eagle_sample(spec_info, batch, logits_output, vocab_mask)


class DecodeVerifyRollbackWorkerV2:
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
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
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
        self.cuda_graph_runner_for_target_verify = None
        self._logged_dvr_draft_graph_skip_reasons = set()
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
            self.draft_backend = _DVREagleDraftBackend(self)
            self.draft_runner = self._draft_worker.draft_runner
            self.num_new_pages_per_topk = torch.empty(
                (), dtype=torch.int64, device=self.device
            )
            self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
            self.plan_stream, self.plan_stream_ctx = _get_dvr_plan_stream(self.device)
            log_prefix = "DVR EAGLE"
        else:
            del dp_rank, moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port
            self._draft_worker = self
            self.draft_backend = _DVRSelfDraftBackend(self)
            self.draft_runner = self.model_runner
            self.plan_stream = None
            self.plan_stream_ctx = nullcontext()
            log_prefix = "DVR self-decode"

        logger.info(
            "Initialized %s worker: num_steps=%s, num_draft_tokens=%s",
            log_prefix,
            self.num_draft_steps,
            self.num_draft_tokens,
        )
        self.speculative_num_steps = self.num_draft_steps
        self.speculative_num_draft_tokens = self.num_draft_tokens
        self.dvr_output_replay_prefix = DVRPendingOutputPrefix()

    def _dummy_hidden_states(self, num_tokens: int, device=None) -> torch.Tensor:
        """Keep EAGLE indexing contracts without storing real DVR hidden states.

        DVR self-draft only needs accepted token ids for the next decode step.
        A zero-width tensor can still be indexed by accepted token positions, but
        avoids materializing [tokens, hidden_size] activations in target verify.
        """

        return torch.empty(
            (num_tokens, 0),
            dtype=self.model_config.dtype,
            device=device or self.device,
        )

    def __getattr__(self, name):
        if name == "target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def init_attention_backends(self):
        if self.is_dvr_eagle:
            self._draft_worker.init_attention_backends()
        # Self-DVR target worker owns the model and attention backend. Scheduler
        # already initializes it before calling this self-draft worker hook.

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        pass

    def activate_step_by_batch(self, batch_size: int) -> None:
        pass

    def init_cuda_graphs(self):
        if self.is_dvr_eagle:
            self._draft_worker.init_cuda_graphs()
            if (
                self.cuda_graph_runner_for_target_verify is None
                and not self.server_args.disable_cuda_graph
                and self.server_args.model_impl != "mindspore"
            ):
                self.cuda_graph_runner_for_target_verify = (
                    DVRTargetVerifyCudaGraphRunner(
                        self.target_worker.model_runner,
                        skip_prefill_only_deterministic_for_capture=True,
                    )
                )
            return

        # Capture the dedicated self-draft decode graph after target attention
        # backends exist. This matches upstream's separated init order.
        self.draft_backend.init_cuda_graphs()

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

    def _log_dvr_draft_graph_skip_once(self, reason: str) -> None:
        if reason in self._logged_dvr_draft_graph_skip_reasons:
            return
        self._logged_dvr_draft_graph_skip_reasons.add(reason)
        logger.warning("DVR self-draft CUDA graph skipped: %s", reason)

    def _forward_target_verify_for_dvr(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        # DVR derives exact output logprobs from target-verify logits after
        # sampling. The ordinary target-verify pass only needs logits for
        # accept/reject and GDN state commit, so keep its logprob metadata path
        # disabled.
        need_return_logprob = batch.return_logprob
        disable_verify_graph = dvr_has_graph_unsafe_short_prompt(batch)
        saved_graph_runner = None
        batch.return_logprob = False
        if disable_verify_graph:
            # The same page-padded one-token prompt edge can crash TARGET_VERIFY
            # graph replay on GDN models. Keep the fallback local to DVR; normal
            # prompts and all non-DVR target paths still use the captured graph.
            saved_graph_runner = self.model_runner.decode_cuda_graph_runner
            self.model_runner.decode_cuda_graph_runner = None
        try:
            return self.target_worker.forward_batch_generation(batch, is_verify=True)
        finally:
            batch.return_logprob = need_return_logprob
            if disable_verify_graph:
                self.model_runner.decode_cuda_graph_runner = saved_graph_runner

    def _build_self_draft_verify_input(
        self,
        *,
        batch: ScheduleBatch,
        spec_info: EagleDraftInput,
        parent_list,
        top_scores_index,
        draft_tokens: torch.Tensor,
        draft_probs: Optional[torch.Tensor],
        seq_lens_sum,
        seq_lens_cpu,
    ) -> EagleVerifyInput:
        (
            _tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.bonus_tokens,
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
            # DVR uses topk=1 chain verify. The tree builder is still reused
            # for token order/retrieve metadata, but attention itself should
            # stay on the ordinary causal path instead of a backend-specific
            # custom tree-mask path.
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
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            draft_probs=draft_probs,
        )

    def _run_self_draft_and_build_verify_input(
        self,
        batch: ScheduleBatch,
        spec_info: EagleDraftInput,
        *,
        seq_lens_sum: Optional[int] = None,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        suppress_return_logprob: bool = False,
    ) -> EagleVerifyInput:
        """Run DVR self-draft and return the EAGLE-compatible verify input."""

        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
        batch.return_hidden_states = False

        saved_return_logprob = batch.return_logprob
        if suppress_return_logprob:
            batch.return_logprob = False
        try:
            forward_batch = ForwardBatch.init_new(batch, self.model_runner)
            if forward_batch.seq_lens_cpu is None:
                forward_batch.seq_lens_cpu = seq_lens_cpu
            if forward_batch.seq_lens_cpu is None:
                forward_batch.seq_lens_cpu = forward_batch.seq_lens.detach().cpu()
            elif not torch.is_tensor(forward_batch.seq_lens_cpu):
                forward_batch.seq_lens_cpu = torch.tensor(
                    forward_batch.seq_lens_cpu, dtype=torch.int64, device="cpu"
                )
            if forward_batch.seq_lens_sum is None:
                forward_batch.seq_lens_sum = (
                    seq_lens_sum
                    if seq_lens_sum is not None
                    else int(forward_batch.seq_lens_cpu.sum().item())
                )
            parent_list, top_scores_index, draft_tokens, draft_probs = (
                self.draft_forward(forward_batch)
            )
        finally:
            if suppress_return_logprob:
                batch.return_logprob = saved_return_logprob

        return self._build_self_draft_verify_input(
            batch=batch,
            spec_info=spec_info,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            seq_lens_sum=(
                forward_batch.seq_lens_sum if seq_lens_sum is None else seq_lens_sum
            ),
            seq_lens_cpu=forward_batch.seq_lens_cpu,
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
        topk_index = spec_info.bonus_tokens
        empty_hidden_states = torch.empty(
            (0, 0), dtype=torch.float32, device=topk_index.device
        )

        origin_seq_lens = forward_batch.seq_lens.clone()
        origin_seq_lens_cpu = forward_batch.seq_lens_cpu.clone()
        origin_seq_lens_sum = forward_batch.seq_lens_sum
        origin_spec_info = forward_batch.spec_info
        origin_positions = forward_batch.positions.clone()
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

            next_token_ids = self.model_runner.sample(logits_output, forward_batch)
            if not forward_batch.sampling_info.is_all_greedy:
                draft_probs_list.append(
                    self.get_draft_sampling_probs(
                        forward_batch, logits_output.next_token_logits
                    )
                )
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
        forward_batch.positions.copy_(origin_positions)
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
        graph_skip_reason = dvr_self_draft_graph_skip_reason(forward_batch)
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_decode is not None
            and self.cuda_graph_runner_for_draft_decode.can_run(forward_batch)
        )
        if can_cuda_graph:
            return self.cuda_graph_runner_for_draft_decode.replay(forward_batch)

        min_seq_len = _min_seq_len_cpu(forward_batch)
        if (
            getattr(self.model_runner, "hybrid_gdn_config", None) is not None
            and graph_skip_reason is None
        ):
            capture_bs = []
            if self.cuda_graph_runner_for_draft_decode is not None:
                capture_bs = self.cuda_graph_runner_for_draft_decode.capture_bs
            raise RuntimeError(
                "DVR self-draft decode requires the dedicated CUDA graph for "
                "gated linear-state models. The current batch cannot run it: "
                f"batch_size={forward_batch.batch_size}, "
                f"min_seq_len={min_seq_len}, capture_bs={capture_bs}. "
                "Use the default CUDA graph batch sizes or include the running "
                "batch size in --cuda-graph-bs/--cuda-graph-max-bs."
            )

        if graph_skip_reason is not None:
            self._log_dvr_draft_graph_skip_once(graph_skip_reason)

        with dvr_self_draft_eager_context(self.model_runner):
            forward_batch.can_run_dp_cuda_graph = False
            return self.model_runner.forward(forward_batch).logits_output

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
        return sampling_probs

    # Target verify. DVR keeps the forward call in TARGET_VERIFY mode like EAGLE,
    # then locally adapts GDN's physical window and state restore/commit.

    def _target_suffix_extend_verify_logits(
        self,
        *,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        linear_state_ctx,
        full_prefix_replay: bool = False,
    ) -> Optional[torch.Tensor]:
        """Compute verifier logits by replaying only the deterministic suffix.

        The ordinary TARGET_VERIFY forward still runs first so DVR can commit
        accepted recurrent states.  The oracle restores the live recurrent slot
        to the chunk-boundary checkpoint, then runs an ordinary EXTEND over the
        unclosed tail plus draft tokens.  This keeps verifier logits equivalent
        to target prefill without replaying the full prefix on every step.
        """

        if getattr(spec_info, "seq_lens_cpu", None) is not None:
            base_seq_lens_cpu = [int(x) for x in spec_info.seq_lens_cpu.tolist()]
        elif batch.seq_lens_cpu is not None:
            base_seq_lens_cpu = [int(x) for x in batch.seq_lens_cpu.tolist()]
        else:
            base_seq_lens_cpu = [int(x) for x in batch.seq_lens.detach().cpu().tolist()]
        with dvr_suffix_replay_context(
            batch=batch,
            linear_state=self.linear_state,
            linear_state_ctx=linear_state_ctx,
            base_seq_lens_cpu=base_seq_lens_cpu,
            draft_tokens=spec_info.draft_token,
            draft_cache_locs=batch.out_cache_loc,
            request_token_ids_for_replay=self._request_token_ids_for_replay,
            full_prefix_replay=full_prefix_replay,
            use_mamba_cow_from_boundary=True,
        ) as replay:
            if replay is None:
                return None
            replay_batch, replay_plan = replay
            draft_logits, _ = run_dvr_suffix_replay_oracle(
                target_worker=self.target_worker,
                replay_batch=replay_batch,
                replay_plan=replay_plan,
            )
            return draft_logits

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

        saved_graph_runner = target_runner.decode_cuda_graph_runner
        with stream_ctx:
            target_runner.decode_cuda_graph_runner = runner
            try:
                yield use_plan_stream
            finally:
                target_runner.decode_cuda_graph_runner = saved_graph_runner

    def _prepare_dvr_boundary_for_verify(self, batch: ScheduleBatch) -> None:
        if batch.forward_mode.is_idle():
            return

        seq_lens_cpu = self.linear_state.batch_seq_lens_cpu(batch)
        self.linear_state.prepare_for_draft(
            batch,
            seq_lens_cpu=seq_lens_cpu,
            request_token_ids_for_replay=self._request_token_ids_for_replay,
        )
        # EAGLE/MTP draft can run before the next target verify in overlap mode.
        # Preserve the target checkpoint authority across draft-side mutations.
        self.linear_state.backup_boundary_state(batch, preserve_existing=True)

    def _target_suffix_extend_verify_output(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: EagleVerifyInput,
        linear_state_ctx,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.topk != 1 or self.linear_state.boundary_backup is None:
            return None

        base_seq_lens_cpu = self.linear_state.batch_seq_lens_cpu(batch)
        with dvr_suffix_replay_context(
            batch=batch,
            linear_state=self.linear_state,
            linear_state_ctx=linear_state_ctx,
            base_seq_lens_cpu=base_seq_lens_cpu,
            draft_tokens=verify_input.draft_token,
            draft_cache_locs=batch.out_cache_loc,
            request_token_ids_for_replay=self._request_token_ids_for_replay,
            restore_boundary_state=True,
            track_replay_boundary_checkpoint=True,
        ) as replay:
            if replay is None:
                return None
            replay_batch, replay_plan = replay
            draft_logits, draft_hidden_states = run_dvr_suffix_replay_oracle(
                target_worker=self.target_worker,
                replay_batch=replay_batch,
                replay_plan=replay_plan,
                use_forward_batch=True,
            )
            return draft_logits, draft_hidden_states

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

    def clear_cache_pool(self):
        self.linear_state.clear_cache_state()
        self.dvr_output_replay_prefix.clear()

    def _request_token_ids_for_replay(self, req, boundary_seqlen: int):
        error_prefix = "DVR EAGLE" if self.is_dvr_eagle else "DVR spec-v2"
        return self.dvr_output_replay_prefix.request_output_prefix_token_ids(
            req,
            boundary_seqlen,
            error_prefix=error_prefix,
        )

    def _forward_batch_generation_eagle(
        self, batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(batch)
            batch_output.new_seq_lens = batch.seq_lens
            self._prepare_dvr_boundary_for_verify(batch)
            if batch_output.next_token_ids is not None:
                next_token_ids_cpu = batch_output.next_token_ids.detach().cpu().tolist()
                token_logprobs_per_req = None
                if batch_output.logits_output.next_token_logprobs is not None:
                    logprob_values = (
                        batch_output.logits_output.next_token_logprobs.detach()
                        .cpu()
                        .tolist()
                    )
                    token_logprobs_per_req = [
                        [float(value)] for value in logprob_values
                    ]
                self.dvr_output_replay_prefix.append_batch_output_tokens(
                    batch,
                    [[token_id] for token_id in next_token_ids_cpu],
                    token_logprobs_per_req=token_logprobs_per_req,
                    error_prefix="DVR EAGLE prefill output prefix",
                )
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            batch_output.next_draft_input = (
                self.draft_backend.draft_extend_for_prefill(batch, batch_output)
            )
            return batch_output

        self.activate_step_by_batch(batch.seq_lens.shape[0])
        if batch.spec_info is None:
            batch.spec_info = self.draft_backend.idle_draft_input()

        self._prepare_dvr_boundary_for_verify(batch)
        verify_input: EagleVerifyInput = self.draft_backend.draft(batch)
        assert verify_input.is_verify_input()
        batch.spec_info = verify_input
        batch_output = self.verify(batch, verify_input)
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)
        self.draft_backend.draft_extend_for_decode(batch, batch_output)
        return batch_output

    def _verify_eagle(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
    ) -> GenerationBatchResult:
        fwd_stream = torch.get_device_module(self.device).current_stream()
        assert spec_info.is_verify_input()
        record_stream_for_v2_verify(batch, spec_info, fwd_stream)

        spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        bs = len(batch.seq_lens)

        disable_verify_graph = dvr_has_graph_unsafe_short_prompt(batch)
        with self._target_verify_context(
            disable_cuda_graph=disable_verify_graph,
            prepare=True,
        ) as prepared_on_plan_stream:
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

            runner = (
                self.cuda_graph_runner_for_target_verify
                or self.target_worker.model_runner.decode_cuda_graph_runner
            )
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

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrieve_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrieve_next_token.shape
            ).cpu()

        base_seq_lens_cpu = self.linear_state.batch_seq_lens_cpu(batch)
        linear_state_ctx = self.linear_state.restore_for_verify(
            batch,
            seq_lens_cpu=base_seq_lens_cpu,
        )

        with self._target_verify_context(disable_cuda_graph=disable_verify_graph):
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
            oracle_output = self._target_suffix_extend_verify_output(
                batch=batch,
                verify_input=spec_info,
                linear_state_ctx=linear_state_ctx,
            )
        if oracle_output is not None:
            oracle_logits, oracle_hidden_states = oracle_output
            logits_output.next_token_logits = oracle_logits
            logits_output.hidden_states = oracle_hidden_states

        vocab_mask = None
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

        maybe_detect_nan(
            logits_output.next_token_logits, "dvr eagle verify: target model logits"
        )
        maybe_detect_inf(
            logits_output.next_token_logits, "dvr eagle verify: target model logits"
        )
        predict, accept_lens, accept_index = self.draft_backend.sample_verify(
            batch=batch,
            spec_info=spec_info,
            logits_output=logits_output,
            vocab_mask=vocab_mask,
        )
        new_seq_lens = batch.seq_lens + accept_lens
        has_verify_tokens = not batch.forward_mode.is_idle()

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
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

        if has_verify_tokens and batch.return_logprob:
            compute_spec_v2_logprobs(
                batch,
                logits_output,
                predict,
                accept_index,
                self.speculative_num_steps,
            )

        partial_suffix_replay_kwargs = None
        if linear_state_ctx is not None:
            partial_suffix_replay_kwargs = dict(
                target_worker=self.target_worker,
                linear_state=self.linear_state,
                linear_state_ctx=linear_state_ctx,
                base_seq_lens_cpu=base_seq_lens_cpu,
                num_draft_tokens=self.speculative_num_draft_tokens,
                request_token_ids_for_replay=self._request_token_ids_for_replay,
            )
        dvr_aux = finish_dvr_verify(
            batch=batch,
            linear_state=self.linear_state,
            linear_state_ctx=linear_state_ctx,
            accept_lens=accept_lens,
            accept_lens_cpu=None,
            num_draft_tokens=self.speculative_num_draft_tokens,
            output=(
                DVRVerifyOutput(
                    replay_prefix=self.dvr_output_replay_prefix,
                    tokens=predict,
                    token_logprobs=logits_output.next_token_logprobs,
                    tokens_per_req=spec_info.draft_token_num,
                    base_seq_lens_cpu=base_seq_lens_cpu,
                    error_prefix="DVR EAGLE",
                )
                if has_verify_tokens
                else None
            ),
            base_seq_lens_cpu=base_seq_lens_cpu,
            predict=predict if partial_suffix_replay_kwargs is not None else None,
            accept_index=(
                accept_index if partial_suffix_replay_kwargs is not None else None
            ),
            partial_suffix_replay_kwargs=partial_suffix_replay_kwargs,
        )
        if linear_state_ctx is not None:
            self.linear_state.backup_boundary_state(batch, preserve_existing=False)
        else:
            commit_mamba_states_after_verify(
                self.target_worker,
                batch,
                accept_lens,
                accept_index,
                self.speculative_num_draft_tokens,
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            num_proposed_drafts_per_req_cpu=[
                useful_spec_proposed_drafts(req, self.speculative_num_steps)
                for req in batch.reqs
            ],
            next_draft_input=EagleDraftInput(bonus_tokens=bonus_tokens),
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            dvr_aux=dvr_aux,
            routed_experts_output=forward_batch_output.routed_experts_output,
            indexer_topk_output=forward_batch_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )

    def forward_batch_generation(
        self, model_worker_batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        batch = model_worker_batch
        if self.is_dvr_eagle:
            return self._forward_batch_generation_eagle(batch, on_publish)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch.capture_hidden_mode = CaptureHiddenMode.NULL
            target_result = self.target_worker.forward_batch_generation(batch)
            logits_output, next_token_ids = (
                target_result.logits_output,
                target_result.next_token_ids,
            )
            batch.spec_info = self.draft_backend.prefill_next_draft_input(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
            )
            self.draft_backend.record_prefill_output_prefix(
                batch=batch,
                output_prefix=self.dvr_output_replay_prefix,
                logits_output=logits_output,
                next_token_ids=next_token_ids,
            )
            batch_result = GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                can_run_cuda_graph=target_result.can_run_cuda_graph,
                next_draft_input=batch.spec_info,
                new_seq_lens=batch.seq_lens,
            )
            if on_publish is not None:
                on_publish(batch_result.new_seq_lens)
            return batch_result

        if batch.spec_info is None:
            batch.spec_info = self.draft_backend.idle_draft_input()

        verify_input = self.draft_backend.draft(batch)
        batch.spec_info = verify_input
        batch_result = self.verify(batch, verify_input)
        if on_publish is not None:
            on_publish(batch_result.new_seq_lens)
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
        return self.draft_backend.draft(batch)

    def verify(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
    ) -> GenerationBatchResult:
        if self.is_dvr_eagle:
            return self._verify_eagle(batch, spec_info)

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
            seq_lens_cpu=self.linear_state.batch_seq_lens_cpu(batch),
        )
        batch.seq_lens_cpu_cache = spec_info.seq_lens_cpu
        batch_result = self._forward_target_verify_for_dvr(batch)
        logits_output = batch_result.logits_output
        oracle_logits = None
        streaming_return_logprob = any(
            req.return_logprob and req.stream for req in batch.reqs
        )
        if batch.return_logprob and streaming_return_logprob:
            # Streaming logprob chunks cannot be repaired after the final response
            # is materialized. Non-streaming requests use final-response repair
            # below; streaming keeps the conservative per-step oracle.
            oracle_logits = self._target_suffix_extend_verify_logits(
                batch=batch,
                spec_info=spec_info,
                linear_state_ctx=linear_state_ctx,
                full_prefix_replay=True,
            )
        if oracle_logits is not None:
            logits_output.next_token_logits = oracle_logits
        maybe_detect_nan(logits_output.next_token_logits, "dvr v2 target verify")

        predict, accept_lens, accept_index = self.draft_backend.sample_verify(
            batch=batch,
            spec_info=spec_info,
            logits_output=logits_output,
        )
        new_seq_lens = scheduler_seq_lens + accept_lens
        has_verify_tokens = (
            not batch.forward_mode.is_idle() and accept_lens.numel() > 0
        )
        base_seq_lens_cpu = self.linear_state.batch_seq_lens_cpu(batch)
        if has_verify_tokens:
            if batch.return_logprob:
                self.draft_backend.compute_compact_logprobs(
                    batch=batch,
                    logits_output=logits_output,
                    predict=predict,
                    accept_index=accept_index,
                )

        # Self draft reuses target KV/GDN state directly, so accepted suffix
        # repair is unnecessary; the fast commit path can copy verify state.
        dvr_aux = finish_dvr_verify(
            batch=batch,
            linear_state=self.linear_state,
            linear_state_ctx=linear_state_ctx,
            accept_lens=accept_lens,
            accept_lens_cpu=None,
            num_draft_tokens=self.num_draft_tokens,
            output=(
                DVRVerifyOutput(
                    replay_prefix=self.dvr_output_replay_prefix,
                    tokens=predict,
                    token_logprobs=logits_output.next_token_logprobs,
                    tokens_per_req=self.num_draft_tokens,
                    base_seq_lens_cpu=base_seq_lens_cpu,
                    error_prefix="DVR spec-v2",
                )
                if has_verify_tokens
                else None
            ),
            base_seq_lens_cpu=base_seq_lens_cpu,
            use_fast_self_draft_commit=True,
        )

        next_draft_input = self.draft_backend.next_draft_input(
            batch=batch,
            predict=predict,
            accept_lens=accept_lens,
            has_verify_tokens=has_verify_tokens,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            speculative_num_draft_tokens=self.num_draft_tokens,
            num_proposed_drafts_per_req_cpu=[
                useful_spec_proposed_drafts(req, self.num_draft_steps)
                for req in batch.reqs
            ],
            dvr_aux=dvr_aux,
            routed_experts_output=batch_result.routed_experts_output,
        )

    def _sample_verify(
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

        if sampling_info.acc_additive_penalties is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.acc_additive_penalties, self.num_draft_tokens, dim=0
                )
            )
        if sampling_info.acc_scaling_penalties is not None:
            apply_scaling_penalties(
                next_token_logits,
                torch.repeat_interleave(
                    sampling_info.acc_scaling_penalties, self.num_draft_tokens, dim=0
                ),
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
            if spec_info.draft_probs is not None:
                sampling_seed = getattr(sampling_info, "sampling_seed", None)
                if sampling_seed is None:
                    uniform_samples = torch.rand_like(candidates, dtype=torch.float32)
                    uniform_samples_for_final_sampling = torch.rand(
                        (bs,), dtype=torch.float32, device=self.device
                    )
                else:
                    num_slots = candidates.shape[1]
                    seed = sampling_seed.to(device=self.device).repeat_interleave(
                        num_slots + 1
                    )
                    slot_offsets = torch.arange(
                        num_slots + 1, dtype=torch.int64, device=self.device
                    )
                    positions = (
                        batch.seq_lens.to(device=self.device, dtype=torch.int64)
                        .unsqueeze(1)
                        + slot_offsets
                    ).reshape(-1)
                    cols = torch.zeros((1,), dtype=torch.int32, device=self.device)
                    uniforms = (
                        murmur_hash32(seed, positions, cols)
                        .reshape(bs, num_slots + 1)
                        .to(torch.float32)
                        / 4294967296.0
                    )
                    uniform_samples = uniforms[:, :num_slots]
                    uniform_samples_for_final_sampling = uniforms[
                        :, num_slots
                    ].contiguous()
            else:
                uniform_samples = torch.rand_like(candidates, dtype=torch.float32)
                uniform_samples_for_final_sampling = torch.rand(
                    (bs,), dtype=torch.float32, device=self.device
                )
            sampling_fn(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=accept_lens,
                candidates=candidates,
                retrive_index=spec_info.retrieve_index,
                retrive_next_token=spec_info.retrieve_next_token,
                retrive_next_sibling=spec_info.retrieve_next_sibling,
                uniform_samples=uniform_samples,
                uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
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

    def _compute_compact_logprobs(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
    ):
        bs = len(batch.seq_lens)
        max_accept = self.num_draft_steps + 1
        device = predict.device

        compact_output_idx = dvr_compact_output_indices(
            accept_index=accept_index,
            num_draft_tokens=self.num_draft_tokens,
            max_accept=max_accept,
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

        # DVR result processing emits tokens from the compact per-request
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
