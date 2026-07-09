from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.utils.hash import murmur_hash32
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
    dvr_self_draft_eager_context,
    dvr_self_draft_graph_skip_reason,
    _min_seq_len_cpu,
)
from sglang.srt.speculative.dvr_info import (
    DVRPendingOutputPrefix,
    DVRSelfDraftVerifyInput,
    DVRVerifyOutput,
)
from sglang.srt.speculative.dvr_linear_state import DVRLinearStateLifecycle
from sglang.srt.speculative.dvr_scheduler_utils import (
    apply_dvr_final_logprob_repairs,
)
from sglang.srt.speculative.dvr_target_replay import (
    defer_dvr_non_streaming_logprob_output_until_finish,
    replay_accepted_suffix_for_live_state,
    run_suffix_draft_replay_oracle,
    score_deferred_dvr_final_logprob_repairs,
    suffix_draft_replay_batch_context,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    organize_draft_results,
    eagle_sample,
)
from sglang.srt.speculative.spec_utils import select_top_k_tokens
from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda
from sglang.srt.utils.async_probe import maybe_detect_nan

if is_cuda():
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - non-CUDA builds use the torch fallback.
    triton = None
    tl = None

logger = logging.getLogger(__name__)


def dvr_has_graph_unsafe_short_prompt(batch) -> bool:
    """Return whether DVR/GDN graph replay should skip the one-token edge."""

    return any(len(req.origin_input_ids) <= 1 for req in batch.reqs)


def dvr_chain_uniform_samples(
    candidates: torch.Tensor,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return accept/reject uniforms for DVR chain sampling."""

    sampling_seed = getattr(batch.sampling_info, "sampling_seed", None)
    if sampling_seed is None:
        return (
            torch.rand_like(candidates, dtype=torch.float32),
            torch.rand((candidates.shape[0],), dtype=torch.float32, device=batch.device),
        )

    bs, num_slots = candidates.shape
    device = candidates.device
    seed = sampling_seed.to(device=device).repeat_interleave(num_slots + 1)
    slot_offsets = torch.arange(num_slots + 1, dtype=torch.int64, device=device)
    positions = (
        batch.seq_lens.to(device=device, dtype=torch.int64).unsqueeze(1) + slot_offsets
    ).reshape(-1)
    cols = torch.zeros((1,), dtype=torch.int32, device=device)
    uniforms = (
        murmur_hash32(seed, positions, cols)
        .reshape(bs, num_slots + 1)
        .to(torch.float32)
        / 4294967296.0
    )
    return uniforms[:, :num_slots], uniforms[:, num_slots].contiguous()


if triton is not None:

    @triton.jit
    def _chain_speculative_sampling_kernel(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        stride_cand_b: tl.constexpr,
        stride_cand_s: tl.constexpr,
        stride_acc_b: tl.constexpr,
        stride_acc_s: tl.constexpr,
        stride_idx_b: tl.constexpr,
        stride_idx_s: tl.constexpr,
        stride_uni_b: tl.constexpr,
        stride_uni_s: tl.constexpr,
        stride_tp_b: tl.constexpr,
        stride_tp_s: tl.constexpr,
        stride_tp_v: tl.constexpr,
        stride_dp_b: tl.constexpr,
        stride_dp_s: tl.constexpr,
        stride_dp_v: tl.constexpr,
        NUM_SLOTS: tl.constexpr,
        ACCEPT_COLS: tl.constexpr,
        PREDICT_NUMEL: tl.constexpr,
        TARGET_ROWS: tl.constexpr,
        DRAFT_ROWS: tl.constexpr,
        VOCAB_SIZE: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        bid = tl.program_id(0)
        cand_base = candidates + bid * stride_cand_b
        accept_base = accept_index + bid * stride_acc_b
        index_base = retrive_index + bid * stride_idx_b
        uniform_base = uniform_samples + bid * stride_uni_b

        prob_row = 0
        root_index = tl.load(index_base)
        valid_last = (root_index >= 0) & (root_index < PREDICT_NUMEL)
        if (ACCEPT_COLS > 0) and valid_last:
            tl.store(accept_base, root_index)
        last_index = root_index
        num_accepted = 0
        step = 1
        continue_verifying = 1
        while (step < NUM_SLOTS) and (continue_verifying == 1) and valid_last:
            draft_token = tl.load(cand_base + step * stride_cand_s)
            next_index = tl.load(index_base + step * stride_idx_s)
            valid_token = (
                (draft_token >= 0)
                & (draft_token < VOCAB_SIZE)
                & (prob_row < TARGET_ROWS)
            )
            if valid_token:
                target_prob = tl.load(
                    target_probs
                    + bid * stride_tp_b
                    + prob_row * stride_tp_s
                    + draft_token * stride_tp_v
                )
                valid_draft_row = prob_row < DRAFT_ROWS
                draft_prob = tl.load(
                    draft_probs
                    + bid * stride_dp_b
                    + prob_row * stride_dp_s
                    + draft_token * stride_dp_v,
                    mask=valid_draft_row,
                    other=0.0,
                )
                coin = tl.load(uniform_base + (step - 1) * stride_uni_s)
                can_accept = (
                    valid_draft_row
                    & (next_index >= 0)
                    & (next_index < PREDICT_NUMEL)
                    & ((num_accepted + 1) < ACCEPT_COLS)
                    & (coin * draft_prob < target_prob)
                )
            else:
                can_accept = False

            if can_accept:
                tl.store(predicts + last_index, draft_token)
                num_accepted += 1
                prob_row = step
                last_index = next_index
                tl.store(accept_base + num_accepted * stride_acc_s, last_index)
                step += 1
            else:
                continue_verifying = 0

        tl.store(accept_token_num + bid, num_accepted)
        all_accepted = continue_verifying

        target_base = target_probs + bid * stride_tp_b + prob_row * stride_tp_s
        draft_base = draft_probs + bid * stride_dp_b + prob_row * stride_dp_s

        norm_sum = 0.0
        if valid_last and (prob_row < TARGET_ROWS):
            for v_start in range(0, VOCAB_SIZE, BLOCK_V):
                offsets = v_start + tl.arange(0, BLOCK_V)
                mask = offsets < VOCAB_SIZE
                target_val = tl.load(
                    target_base + offsets * stride_tp_v, mask=mask, other=0.0
                )
                if all_accepted:
                    residual_val = target_val
                else:
                    draft_val = tl.load(
                        draft_base + offsets * stride_dp_v,
                        mask=mask & (prob_row < DRAFT_ROWS),
                        other=0.0,
                    )
                    residual_val = tl.maximum(target_val - draft_val, 0.0)
                norm_sum += tl.sum(residual_val)

            final_token = VOCAB_SIZE - 1
            if norm_sum <= 0.0:
                best_val = -float("inf")
                for v_start in range(0, VOCAB_SIZE, BLOCK_V):
                    offsets = v_start + tl.arange(0, BLOCK_V)
                    mask = offsets < VOCAB_SIZE
                    target_val = tl.load(
                        target_base + offsets * stride_tp_v,
                        mask=mask,
                        other=-float("inf"),
                    )
                    block_best = tl.max(target_val, axis=0)
                    if block_best > best_val:
                        best_val = block_best
                        final_token = v_start + tl.argmax(target_val, axis=0)
            else:
                target_u = tl.load(uniform_samples_for_final_sampling + bid) * norm_sum
                cdf = 0.0
                found = 0
                for v_start in range(0, VOCAB_SIZE, BLOCK_V):
                    if found == 0:
                        offsets = v_start + tl.arange(0, BLOCK_V)
                        mask = offsets < VOCAB_SIZE
                        target_val = tl.load(
                            target_base + offsets * stride_tp_v, mask=mask, other=0.0
                        )
                        if all_accepted:
                            residual_val = target_val
                        else:
                            draft_val = tl.load(
                                draft_base + offsets * stride_dp_v,
                                mask=mask & (prob_row < DRAFT_ROWS),
                                other=0.0,
                            )
                            residual_val = tl.maximum(target_val - draft_val, 0.0)
                        block_cdf = cdf + tl.cumsum(residual_val, axis=0)
                        matched = block_cdf > target_u
                        if tl.max(matched, axis=0):
                            final_token = v_start + tl.argmax(
                                matched.to(tl.int32), axis=0
                            )
                            found = 1
                        cdf += tl.sum(residual_val)

            tl.store(predicts + last_index, final_token)


def chain_speculative_sampling(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    deterministic: bool,
) -> None:
    """Classic topk=1 chain speculative sampling for DVR self draft."""

    del retrive_next_token, retrive_next_sibling
    del threshold_single, threshold_acc, deterministic

    if triton is None or not (candidates.is_cuda and target_probs.is_cuda):
        raise RuntimeError("DVR chain speculative sampling requires Triton CUDA.")

    batch_size, num_slots = candidates.shape
    _chain_speculative_sampling_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        candidates.stride(0),
        candidates.stride(1),
        accept_index.stride(0),
        accept_index.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        NUM_SLOTS=num_slots,
        ACCEPT_COLS=accept_index.shape[1],
        PREDICT_NUMEL=predicts.numel(),
        TARGET_ROWS=target_probs.shape[1],
        DRAFT_ROWS=draft_probs.shape[1],
        VOCAB_SIZE=target_probs.shape[-1],
        BLOCK_V=4096,
    )


def _add_output_logprobs_for_dvr_spec_v1(
    batch: ScheduleBatch,
    verify_output: Any,
    logits_output: LogitsProcessorOutput,
) -> None:
    """Populate request logprob buffers for DVR's spec-v1 worker.

    Upstream removed the old generic spec-v1 logprob helper when spec decoding
    moved to v2.  DVR keeps a v1 compatibility worker, so keep this narrow copy
    inside DVR instead of reintroducing a generic API that upstream no longer
    uses.
    """

    accept_indices = verify_output.accept_indices
    assert len(accept_indices) == len(logits_output.next_token_logits)

    temperatures = batch.sampling_info.temperatures
    num_draft_tokens = batch.spec_info.draft_token_num
    temperatures = temperatures[accept_indices // num_draft_tokens]
    if envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits, dim=-1
        )
    else:
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits / temperatures, dim=-1
        )

    num_tokens_per_req = [
        accept + 1 for accept in verify_output.num_correct_drafts_per_req_cpu
    ]
    top_logprobs_nums = batch.top_logprobs_nums or [0] * len(batch.reqs)
    token_ids_logprobs = batch.token_ids_logprobs or [None] * len(batch.reqs)
    top_logprobs_nums_expanded = [
        num
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req)
        for _ in range(num_tokens)
    ]
    token_ids_logprobs_expanded = [
        token_ids
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req)
        for _ in range(num_tokens)
    ]

    should_top_logprobs = any(x > 0 for x in top_logprobs_nums)
    should_token_ids_logprobs = any(x is not None for x in token_ids_logprobs)
    if should_top_logprobs:
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = get_top_logprobs(logprobs, top_logprobs_nums_expanded)

    if should_token_ids_logprobs:
        (
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(logprobs, token_ids_logprobs_expanded)

    batch_next_token_ids = verify_output.accept_tokens
    logits_output.next_token_logprobs = logprobs[
        torch.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
        batch_next_token_ids,
    ]

    pt = 0
    next_token_logprobs = logits_output.next_token_logprobs.tolist()
    accept_tokens_list = batch_next_token_ids.tolist()
    token_top_logprobs_val = logits_output.next_token_top_logprobs_val
    token_top_logprobs_idx = logits_output.next_token_top_logprobs_idx
    token_ids_logprobs_val = logits_output.next_token_token_ids_logprobs_val
    token_ids_logprobs_idx = logits_output.next_token_token_ids_logprobs_idx
    for req, num_tokens in zip(batch.reqs, num_tokens_per_req, strict=True):
        for _ in range(num_tokens):
            if req.return_logprob:
                req.logprob.output_token_logprobs_val.append(next_token_logprobs[pt])
                req.logprob.output_token_logprobs_idx.append(accept_tokens_list[pt])
                if req.logprob.top_logprobs_num > 0:
                    req.logprob.output_top_logprobs_val.append(
                        token_top_logprobs_val[pt]
                    )
                    req.logprob.output_top_logprobs_idx.append(
                        token_top_logprobs_idx[pt]
                    )
                if (
                    req.logprob.token_ids_logprob is not None
                    and should_token_ids_logprobs
                ):
                    req.logprob.output_token_ids_logprobs_val.append(
                        token_ids_logprobs_val[pt]
                    )
                    req.logprob.output_token_ids_logprobs_idx.append(
                        token_ids_logprobs_idx[pt]
                    )
            pt += 1


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
        self.max_batch_size = (
            getattr(target_worker, "max_running_requests", None)
            or getattr(target_worker.model_runner, "max_running_requests", None)
            or server_args.max_running_requests
        )
        self.num_draft_steps = server_args.speculative_num_steps
        self.num_draft_tokens = server_args.speculative_num_draft_tokens
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        self.linear_state = DVRLinearStateLifecycle(
            server_args=server_args,
            model_runner=self.model_runner,
        )
        self.cuda_graph_runner_for_draft_decode = None
        self._logged_dvr_draft_graph_skip_reasons = set()

        logger.info(
            "Initialized DVR self-decode worker: num_steps=%s, num_draft_tokens=%s",
            self.num_draft_steps,
            self.num_draft_tokens,
        )

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

    def _draft_anchor_tokens(self, draft_input: EagleDraftInput) -> torch.Tensor:
        """Return per-request token ids that seed the next DVR self-draft step."""

        return draft_input.bonus_tokens

    def __getattr__(self, name):
        return getattr(self.target_worker, name)

    def init_attention_backends(self):
        # Target worker owns the model and attention backend. Scheduler already
        # initializes it before calling this self-draft worker hook.
        pass

    def init_cuda_graphs(self):
        # Capture the dedicated self-draft decode graph after target attention
        # backends exist. This matches upstream's separated init order.
        if (
            self.cuda_graph_runner_for_draft_decode is None
            and not self.server_args.disable_cuda_graph
            and not self.server_args.disable_draft_cuda_graph
        ):
            self.cuda_graph_runner_for_draft_decode = DVRDraftDecodeCudaGraphRunner(self)

    def clear_cache_pool(self):
        self.linear_state.clear_cache_state()

    def alloc_memory_pool(
        self,
        *,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
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
                next_draft_input=batch.spec_info,
                num_correct_drafts=0,
                can_run_cuda_graph=can_run_cuda_graph,
            )

        spec_info = self.draft(batch)
        logits_output, verify_output, can_run_cuda_graph = self.verify(batch, spec_info)
        accept_lens = torch.tensor(
            [x + 1 for x in verify_output.num_correct_drafts_per_req_cpu],
            dtype=batch.seq_lens.dtype,
            device=batch.seq_lens.device,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.accept_tokens,
            num_correct_drafts=sum(verify_output.num_correct_drafts_per_req_cpu),
            num_correct_drafts_per_req_cpu=verify_output.num_correct_drafts_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=batch.spec_info,
            new_seq_lens=batch.seq_lens + accept_lens,
            speculative_num_draft_tokens=self.num_draft_tokens,
            num_proposed_drafts_per_req_cpu=[
                req.useful_spec_proposed_drafts(self.num_draft_steps)
                for req in batch.reqs
            ],
        )

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, bool]:
        batch.capture_hidden_mode = CaptureHiddenMode.NULL
        batch_result = self.target_worker.forward_batch_generation(batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        topk_index = next_token_ids.to(torch.long).unsqueeze(-1)
        batch.spec_info = EagleDraftInput(
            hidden_states=(
                logits_output.hidden_states
                if logits_output.hidden_states is not None
                else self._dummy_hidden_states(
                    next_token_ids.shape[0], device=next_token_ids.device
                )
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
        return logits_output, next_token_ids, batch_result.can_run_cuda_graph

    # Self-draft path. DVR uses the target model's decode path as a draft model,
    # but keeps EAGLE-compatible tree/retrieve metadata for verification.

    def _dvr_draft_rows_and_offsets(
        self, batch: ScheduleBatch, num_tokens: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return req_to_token rows/columns for the self-draft verify window."""

        num_tokens = self.num_draft_tokens if num_tokens is None else num_tokens
        offsets = batch.seq_lens.to(torch.long).unsqueeze(1) + torch.arange(
            num_tokens, dtype=torch.long, device=batch.seq_lens.device
        ).unsqueeze(0)
        rows = batch.req_pool_indices.to(torch.long).unsqueeze(1)
        return rows, offsets

    def _draft_cache_locs_from_req_to_token(
        self, batch: ScheduleBatch
    ) -> torch.Tensor:
        # ScheduleBatch.prepare_for_decode already reserved the speculative
        # decode window and wrote it into req_to_token. Self-DVR draft and verify
        # share that window; allocating a second one leaks KV ownership.
        rows, offsets = self._dvr_draft_rows_and_offsets(batch)
        return batch.req_to_token_pool.req_to_token[rows, offsets].reshape(-1)

    def _finish_dvr_draft_preprocess_decode(
        self, batch: ScheduleBatch, spec_info: EagleDraftInput
    ) -> None:
        """Apply decode metadata common to DVR self-draft v1 and v2."""

        batch.return_hidden_states = False
        batch.mamba_track_indices = None
        batch.mamba_track_mask = None
        batch.mamba_track_seqlens = None
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        batch.maybe_evict_swa()

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        penalizer_orchestrator = batch.sampling_info.penalizer_orchestrator
        if penalizer_orchestrator is not None and penalizer_orchestrator.is_required:
            # Keep draft sampling close to normal autoregressive decode by
            # accounting for the anchor token before sampling following tokens.
            penalizer_orchestrator.cumulate_output_tokens(
                self._draft_anchor_tokens(spec_info).to(torch.int64)
            )

        batch.out_cache_loc = self._draft_cache_locs_from_req_to_token(batch)
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
        self._finish_dvr_draft_preprocess_decode(batch, spec_info)

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=0,
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    @staticmethod
    def _request_token_ids_for_replay(req, boundary_seqlen: int):
        token_ids = list(req.origin_input_ids) + list(req.output_ids)
        if len(token_ids) >= boundary_seqlen:
            return token_ids

        fill_ids = getattr(req, "fill_ids", None)
        if fill_ids is not None and len(fill_ids) >= boundary_seqlen:
            return fill_ids

        raise RuntimeError(
            "DVR boundary fallback replay cannot reconstruct the verified prefix: "
            f"rid={req.rid}, available_tokens={len(token_ids)}, "
            f"boundary_seqlen={boundary_seqlen}."
        )

    def _log_dvr_draft_graph_skip_once(self, reason: str) -> None:
        if reason in self._logged_dvr_draft_graph_skip_reasons:
            return
        self._logged_dvr_draft_graph_skip_reasons.add(reason)
        logger.warning("DVR self-draft CUDA graph skipped: %s", reason)

    def _forward_target_verify_for_dvr(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        # DVR exact output logprobs are derived from the full-prefix oracle after
        # target verify. The ordinary target-verify pass only needs logits for
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
    ) -> DVRSelfDraftVerifyInput:
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

    def draft(self, batch: ScheduleBatch) -> EagleVerifyInput:
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.num_draft_steps,
                self.num_draft_tokens,
            )

        replay_tasks = self.linear_state.prepare_for_draft(
            batch, use_request_seqlen=True
        )
        if replay_tasks:
            self.linear_state.replay_boundary_tasks(
                batch,
                replay_tasks,
                request_token_ids_for_replay=self._request_token_ids_for_replay,
            )
            self.linear_state.restore_tail_lens_after_replay(
                batch, replay_tasks, use_request_seqlen=True
            )
        self.linear_state.finish_prepare_for_draft(batch)
        self._draft_preprocess_decode(batch)
        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        return self._run_self_draft_and_build_verify_input(batch, spec_info)

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
        topk_index = self._draft_anchor_tokens(spec_info)
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

    # Accepted-token and verify-output helpers. These intentionally stay close
    # to EAGLE's postprocess contract so scheduler/radix-cache ownership remains
    # compatible with normal speculative decoding.

    def _next_self_draft_input_from_bonus_tokens(
        self, bonus_tokens: torch.Tensor
    ) -> EagleDraftInput:
        return EagleDraftInput(
            hidden_states=self._dummy_hidden_states(
                bonus_tokens.shape[0], device=bonus_tokens.device
            ),
            bonus_tokens=bonus_tokens,
            topk_p=torch.ones(
                (bonus_tokens.shape[0], self.topk),
                dtype=torch.float32,
                device=bonus_tokens.device,
            ),
            topk_index=bonus_tokens.to(torch.long).unsqueeze(-1),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

    def _prepare_next_draft_after_verify(self, batch: ScheduleBatch, verify_output: Any):
        draft_extend_input = verify_output.draft_extend_input
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        if (
            batch.forward_mode.is_idle()
            or draft_extend_input.input_ids is None
            or draft_extend_input.input_ids.numel() == 0
        ):
            batch.spec_info = EagleDraftInput.create_idle_input(
                device=batch.device,
                hidden_size=0,
                dtype=self.model_config.dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            return

        num_accept_tokens_cpu = draft_extend_input.num_accept_tokens_cpu
        num_accept_tokens = torch.tensor(
            num_accept_tokens_cpu,
            dtype=torch.long,
            device=draft_extend_input.input_ids.device,
        )
        accept_end = torch.cumsum(num_accept_tokens, dim=0) - 1
        bonus_tokens = draft_extend_input.input_ids[accept_end]
        batch.spec_info = self._next_self_draft_input_from_bonus_tokens(bonus_tokens)

    def _sample_and_build_verify_output(
        self,
        *,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        logits_output: LogitsProcessorOutput,
    ) -> DVRVerifyOutput:
        predict, accept_lens, accept_index = eagle_sample(
            spec_info, batch, logits_output, vocab_mask=None
        )
        accept_lens_cpu = [int(x) for x in accept_lens.tolist()]
        accept_mask = torch.arange(
            accept_index.shape[1], dtype=torch.long, device=accept_index.device
        ).unsqueeze(0) < accept_lens.to(torch.long).unsqueeze(1)
        accept_indices = accept_index[accept_mask].to(torch.long)
        accept_tokens = predict[accept_indices].to(torch.long)
        return DVRVerifyOutput(
            accept_tokens=accept_tokens,
            accept_indices=accept_indices,
            num_correct_drafts_per_req_cpu=[
                max(num_accept - 1, 0) for num_accept in accept_lens_cpu
            ],
            draft_extend_input=EagleDraftExtendInput(
                input_ids=accept_tokens,
                num_accept_tokens_cpu=accept_lens_cpu,
            ),
        )

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

        base_seq_lens_cpu = self._seq_lens_cpu_list_for_verify(batch, spec_info)
        with suffix_draft_replay_batch_context(
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
            draft_logits, _ = run_suffix_draft_replay_oracle(
                target_worker=self.target_worker,
                replay_batch=replay_batch,
                replay_plan=replay_plan,
            )
            return draft_logits

    def _replay_accepted_suffix_for_partial_verify(
        self,
        *,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        linear_state_ctx,
        accepted_token_counts_cpu: List[int],
        verify_output: Optional[Any] = None,
        accepted_ids: Optional[torch.Tensor] = None,
        accepted_cache_locs: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Refresh live GDN state with the exact accepted-token suffix.

        The hot TARGET_VERIFY path forwards the proposed draft tokens.  When a
        row is rejected, however, SGLang commits the target-predicted token at
        that row.  Replaying only "unclosed chunk tail + accepted tokens" keeps
        DVR's live recurrent state aligned with the actual scheduler-owned
        token sequence without replaying the full prefix.
        """

        if batch.forward_mode.is_idle() or linear_state_ctx is None:
            return None

        if accepted_ids is None:
            assert verify_output is not None
            accepted_ids = verify_output.draft_extend_input.input_ids
        if accepted_cache_locs is None:
            accepted_cache_locs = batch.out_cache_loc
        if accepted_ids is None or accepted_ids.numel() == 0:
            return None
        base_seq_lens_cpu = self._seq_lens_cpu_list_for_verify(batch, spec_info)
        return replay_accepted_suffix_for_live_state(
            batch=batch,
            target_worker=self.target_worker,
            linear_state=self.linear_state,
            linear_state_ctx=linear_state_ctx,
            base_seq_lens_cpu=base_seq_lens_cpu,
            accepted_token_counts_cpu=accepted_token_counts_cpu,
            accepted_ids=accepted_ids,
            accepted_cache_locs=accepted_cache_locs,
            num_draft_tokens=self.num_draft_tokens,
            request_token_ids_for_replay=self._request_token_ids_for_replay,
        )

    @staticmethod
    def _seq_lens_cpu_list_for_verify(
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
    ) -> list[int]:
        if getattr(spec_info, "seq_lens_cpu", None) is not None:
            return [int(x) for x in spec_info.seq_lens_cpu.tolist()]
        if batch.seq_lens_cpu is not None:
            return [int(x) for x in batch.seq_lens_cpu.tolist()]
        return [int(x) for x in batch.seq_lens.detach().cpu().tolist()]

    @staticmethod
    def _compact_flat_tokens_by_accept_lens(
        flat_tokens: torch.Tensor,
        accept_lens_cpu: list[int],
        *,
        tokens_per_req: Optional[int] = None,
    ) -> list[list[int]]:
        token_ids = flat_tokens.detach().cpu().tolist()
        compact_tokens = []
        if tokens_per_req is None:
            offset = 0
            for accept_len in accept_lens_cpu:
                next_offset = offset + int(accept_len)
                compact_tokens.append([int(x) for x in token_ids[offset:next_offset]])
                offset = next_offset
        else:
            for req_i, accept_len in enumerate(accept_lens_cpu):
                start = req_i * tokens_per_req
                end = start + int(accept_len)
                compact_tokens.append([int(x) for x in token_ids[start:end]])
        return compact_tokens

    def _repair_final_logprobs_for_spec_v1(
        self,
        *,
        batch: ScheduleBatch,
        linear_state_ctx,
        base_seq_lens_cpu: list[int],
        accept_lens_cpu: list[int],
        compact_output_token_ids_per_req: list[list[int]],
    ) -> None:
        """Repair final non-streaming DVR output logprobs with a prefill oracle."""

        if linear_state_ctx is None or not any(
            req.return_logprob and not req.stream and req.finished()
            for req in batch.reqs
        ):
            return

        repairs = score_deferred_dvr_final_logprob_repairs(
            batch=batch,
            target_worker=self.target_worker,
            replay_prefix=DVRPendingOutputPrefix(),
            linear_state_ctx=linear_state_ctx,
            base_seq_lens_cpu=base_seq_lens_cpu,
            accept_lens_cpu=accept_lens_cpu,
            compact_output_token_ids_per_req=compact_output_token_ids_per_req,
            error_prefix="DVR spec-v1 final logprob",
        )
        apply_dvr_final_logprob_repairs(batch, repairs)

    def verify(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
    ) -> Tuple[LogitsProcessorOutput, Any, bool]:
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
        base_seq_lens_cpu = self._seq_lens_cpu_list_for_verify(batch, spec_info)
        if batch.return_logprob and linear_state_ctx is not None:
            # Spec-v1 may stream non-streaming chunks before final result assembly.
            # Hold them until the full-prefix repair below installs exact logprobs.
            defer_dvr_non_streaming_logprob_output_until_finish(
                batch,
                base_seq_lens_cpu=base_seq_lens_cpu,
            )

        batch.seq_lens_cpu_cache = spec_info.seq_lens_cpu
        batch.capture_hidden_mode = CaptureHiddenMode.NULL
        batch_result = self._forward_target_verify_for_dvr(batch)
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
        maybe_detect_nan(logits_output.next_token_logits, "dvr target verify")

        spec_info.hidden_states = (
            logits_output.hidden_states
            if logits_output.hidden_states is not None
            else self._dummy_hidden_states(
                spec_info.draft_token.numel(),
                device=logits_output.next_token_logits.device,
            )
        )
        verify_output = self._sample_and_build_verify_output(
            batch=batch,
            spec_info=spec_info,
            logits_output=logits_output,
        )
        accept_lens_cpu = [
            int(num_correct) + 1
            for num_correct in verify_output.num_correct_drafts_per_req_cpu
        ]

        logits_output.next_token_logits = logits_output.next_token_logits[
            verify_output.accept_indices
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[
                verify_output.accept_indices
            ]
        if batch.return_logprob:
            _add_output_logprobs_for_dvr_spec_v1(
                batch, verify_output, logits_output
            )
            self._repair_final_logprobs_for_spec_v1(
                batch=batch,
                linear_state_ctx=linear_state_ctx,
                base_seq_lens_cpu=base_seq_lens_cpu,
                accept_lens_cpu=accept_lens_cpu,
                compact_output_token_ids_per_req=(
                    self._compact_flat_tokens_by_accept_lens(
                        verify_output.accept_tokens,
                        accept_lens_cpu,
                    )
                ),
            )
        if linear_state_ctx is not None:
            accepted_token_counts = torch.tensor(
                accept_lens_cpu,
                dtype=torch.long,
                device=linear_state_ctx.live_indices.device,
            )
            # DVR enforces topk=1 chain verify, so the accepted draft step is
            # the last accepted token's offset in each fixed verify window.
            accepted_steps = accepted_token_counts - 1
            is_self_dvr = isinstance(spec_info, DVRSelfDraftVerifyInput)
            # return_logprob must not change the self-draft state lifecycle.
            # Exact logprobs are repaired separately; using accepted-suffix
            # replay as the commit path changes the next draft state and
            # regresses the self-DVR acceptance rate.
            live_state_already_replayed = None
            if not is_self_dvr:
                live_state_already_replayed = (
                    self._replay_accepted_suffix_for_partial_verify(
                        batch=batch,
                        spec_info=spec_info,
                        verify_output=verify_output,
                        linear_state_ctx=linear_state_ctx,
                        accepted_token_counts_cpu=accept_lens_cpu,
                    )
                )
            self.linear_state.commit_after_verify(
                batch=batch,
                accepted_token_counts=accepted_token_counts,
                accepted_steps=accepted_steps,
                accepted_token_counts_cpu=accept_lens_cpu,
                ctx=linear_state_ctx,
                live_state_already_replayed=live_state_already_replayed,
                use_fast_self_draft_commit=is_self_dvr,
            )
        self._prepare_next_draft_after_verify(batch, verify_output)
        return logits_output, verify_output, can_run_cuda_graph
