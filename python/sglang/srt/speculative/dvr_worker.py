from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

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
    get_last_loc,
    maybe_detect_nan,
    select_top_k_tokens,
)
from sglang.srt.speculative.dvr_utils import dvr_runtime_verify_window
from sglang.srt.utils import is_cuda, next_power_of_2

if is_cuda():
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

logger = logging.getLogger(__name__)


@dataclass
class _GDNFixedVerifyWindowState:
    input_ids: torch.Tensor
    out_cache_loc: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    seq_lens_sum: int
    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    num_tokens_per_req: int
    spec_seq_lens_cpu: torch.Tensor
    padding_locs: List[torch.Tensor]
    verified_tail_lens: torch.Tensor

    @classmethod
    def capture(cls, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        return cls(
            input_ids=batch.input_ids,
            out_cache_loc=batch.out_cache_loc,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            seq_lens_sum=batch.seq_lens_sum,
            draft_token=spec_info.draft_token,
            positions=spec_info.positions,
            draft_token_num=spec_info.draft_token_num,
            num_tokens_per_req=spec_info.num_tokens_per_req,
            spec_seq_lens_cpu=spec_info.seq_lens_cpu,
            padding_locs=[],
            verified_tail_lens=torch.empty(0, dtype=torch.long),
        )

    def restore(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        batch.input_ids = self.input_ids
        batch.out_cache_loc = self.out_cache_loc
        batch.seq_lens = self.seq_lens
        batch.seq_lens_cpu = self.seq_lens_cpu
        batch.seq_lens_sum = self.seq_lens_sum
        spec_info.draft_token = self.draft_token
        spec_info.positions = self.positions
        spec_info.draft_token_num = self.draft_token_num
        spec_info.num_tokens_per_req = self.num_tokens_per_req
        spec_info.seq_lens_cpu = self.spec_seq_lens_cpu


@dataclass
class _GDNDVRStateContext:
    mamba_cache: Any
    live_indices: torch.Tensor
    boundary_indices: Optional[torch.Tensor] = None


@dataclass
class _GDNFixedVerifyRequestWindow:
    input_ids: List[int]
    out_cache_locs: List[torch.Tensor]
    boundary_len: int
    padding_locs: List[torch.Tensor]


class DecodeVerifyRollbackWorker:
    """DVR speculative worker using the target model as a self draft model.

    The control flow mirrors EAGLE: self-decode draft, target verify, then
    EAGLE-compatible postprocess. DVR-specific work is kept local to this
    worker: causal chain verify, GDN fixed-window target verify, and GDN state
    rollback/commit around the verify window.
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
            not server_args.speculative_dvr_chunk_boundary_verify
            or FLA_CHUNK_SIZE % server_args.page_size != 0
            or server_args.speculative_num_draft_tokens % server_args.page_size != 0
        ):
            raise ValueError(
                "DVR page_size > 1 requires chunk-boundary verify and page_size "
                "aligned to both FLA_CHUNK_SIZE and num_draft_tokens."
            )
        if (
            target_worker.model_runner.hybrid_gdn_config is not None
            and server_args.mamba_track_interval != FLA_CHUNK_SIZE
        ):
            raise ValueError(
                "DVR GDN requires mamba_track_interval to match "
                f"FLA_CHUNK_SIZE={FLA_CHUNK_SIZE}, got "
                f"{server_args.mamba_track_interval}. Multiples larger than "
                "FLA_CHUNK_SIZE can miss the latest chunk boundary from the "
                "first prefill because the current extra_buffer path stores "
                "only one tracked prefill checkpoint."
            )
        if (
            target_worker.model_runner.hybrid_gdn_config is not None
            and server_args.mamba_ssm_dtype != "float32"
        ):
            raise ValueError("DVR GDN requires fp32 Mamba/GDN SSM state storage.")

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
        self.enable_chunk_boundary_verify = (
            server_args.speculative_dvr_chunk_boundary_verify
        )
        self.verify_window_size = FLA_CHUNK_SIZE + self.num_draft_tokens
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        self._gdn_boundary_seqlen = {}
        self._gdn_boundary_track_idx = {}
        self._gdn_boundary_backup = None

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
                self.num_draft_steps,
                self.num_draft_tokens,
            )

        self._ensure_gdn_boundary_state(batch)
        self._backup_gdn_boundary_state(batch)
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
            self.num_draft_tokens,
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

    # GDN state context helpers. These keep Mamba/GDN cache lookup local to DVR
    # and avoid passing speculative-worker semantics into generic GDN backends.

    def _has_gdn_dvr_state(self, batch: ScheduleBatch) -> bool:
        return (
            self.model_runner.hybrid_gdn_config is not None
            and hasattr(batch.req_to_token_pool, "get_mamba_indices")
            and batch.batch_size() > 0
            and all(req.mamba_ping_pong_track_buffer is not None for req in batch.reqs)
        )

    def _gdn_state_context(
        self, batch: ScheduleBatch, require_boundary: bool = False
    ) -> Optional[_GDNDVRStateContext]:
        if not self._has_gdn_dvr_state(batch):
            return None
        assert self.server_args.mamba_track_interval == FLA_CHUNK_SIZE, (
            "DVR GDN target verify must start from FLA chunk boundaries. "
            "The current prefill tracker only guarantees the latest boundary "
            "when mamba_track_interval equals FLA_CHUNK_SIZE."
        )
        live_indices = batch.req_to_token_pool.get_mamba_indices(
            batch.req_pool_indices
        ).to(torch.long)
        mamba_cache = batch.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        assert mamba_cache.temporal.dtype == torch.float32, (
            "DVR GDN requires fp32 temporal state checkpoints. bf16/fp16 "
            "checkpoints round the chunkwise scan state and can diverge from "
            "full prefill across chunks."
        )
        assert mamba_cache.intermediate_ssm.dtype == torch.float32, (
            "DVR GDN requires fp32 intermediate prefill states."
        )
        boundary_indices = None
        if require_boundary:
            boundary_indices = torch.stack(
                [
                    req.mamba_ping_pong_track_buffer[
                        self._gdn_boundary_track_idx[req.rid]
                    ]
                    for req in batch.reqs
                ]
            ).to(device=live_indices.device, dtype=torch.long)
        return _GDNDVRStateContext(
            mamba_cache=mamba_cache,
            live_indices=live_indices,
            boundary_indices=boundary_indices,
        )

    # Fixed physical verify window helpers. For GDN, target verify runs over
    # verified_tail + draft_token + padding_token, whose physical length is
    # FLA_CHUNK_SIZE + num_draft_tokens even though only the draft rows are
    # returned to speculative sampling.

    def _fixed_verify_draft_rows(
        self, verified_tail_lens: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        verified_tail_lens = verified_tail_lens.to(device=device, dtype=torch.long)
        row_starts = (
            torch.arange(
                verified_tail_lens.shape[0], dtype=torch.long, device=device
            )
            * self.verify_window_size
            + verified_tail_lens
        )
        draft_offsets = torch.arange(
            self.num_draft_tokens, dtype=torch.long, device=device
        )
        return (row_starts[:, None] + draft_offsets[None, :]).reshape(-1)

    def _alloc_padding_locs(self, batch: ScheduleBatch, num_tokens: int):
        if num_tokens == 0:
            empty = torch.empty(0, dtype=torch.int64, device=self.device)
            return empty, empty
        if self.page_size == 1:
            locs = alloc_token_slots(batch.tree_cache, num_tokens)
            return locs, locs

        alloc_len = ((num_tokens + self.page_size - 1) // self.page_size) * self.page_size
        owned_locs = alloc_token_slots(batch.tree_cache, alloc_len)
        return owned_locs[:num_tokens], owned_locs

    def _fixed_window_padding_locs(
        self,
        batch: ScheduleBatch,
        last_real_loc: torch.Tensor,
        num_padding_tokens: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if num_padding_tokens == 0:
            return None, None
        if self.page_size == 1:
            return self._alloc_padding_locs(batch, num_padding_tokens)

        last_real_loc_item = int(last_real_loc.item())
        page_tail_len = self.page_size - ((last_real_loc_item % self.page_size) + 1)
        reusable_len = min(num_padding_tokens, page_tail_len)
        pieces = []
        if reusable_len > 0:
            pieces.append(
                torch.arange(
                    last_real_loc_item + 1,
                    last_real_loc_item + 1 + reusable_len,
                    dtype=torch.long,
                    device=last_real_loc.device,
                )
            )

        remaining = num_padding_tokens - reusable_len
        owned_locs = None
        if remaining > 0:
            used_locs, owned_locs = self._alloc_padding_locs(batch, remaining)
            pieces.append(used_locs)

        return torch.cat(pieces) if len(pieces) > 1 else pieces[0], owned_locs

    @staticmethod
    def _gdn_boundary_and_tail(req) -> Tuple[int, int]:
        boundary_seqlen = ((req.seqlen - 1) // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
        verified_tail_len = (req.seqlen - 1) - boundary_seqlen
        return boundary_seqlen, verified_tail_len

    def _chunk_boundary_tail_lens(
        self, batch: ScheduleBatch, ctx: Optional[_GDNDVRStateContext] = None
    ) -> torch.Tensor:
        ctx = ctx or self._gdn_state_context(batch)
        if ctx is not None:
            return ctx.mamba_cache.dvr_qkvg_beta_pos[0, ctx.live_indices].to(
                torch.long
            )
        return torch.tensor(
            [self._gdn_boundary_and_tail(req)[1] for req in batch.reqs],
            dtype=torch.long,
            device=batch.seq_lens.device,
        )

    def _mamba_other_track_idx(self, batch: ScheduleBatch, track_idx: int) -> int:
        return batch.req_to_token_pool.get_mamba_ping_pong_other_idx(track_idx)

    def _current_prefill_checkpoint_track_idx(
        self, batch: ScheduleBatch, req
    ) -> Optional[int]:
        boundary_seqlen, _ = self._gdn_boundary_and_tail(req)
        assert boundary_seqlen % FLA_CHUNK_SIZE == 0
        last_track_seqlen = req.mamba_last_track_seqlen
        if last_track_seqlen is not None and last_track_seqlen > 0:
            assert last_track_seqlen % FLA_CHUNK_SIZE == 0, (
                "DVR GDN must not reuse non-chunk-boundary Mamba checkpoints."
            )
        if boundary_seqlen <= 0:
            return None
        if last_track_seqlen != boundary_seqlen:
            return None
        return self._mamba_other_track_idx(batch, req.mamba_next_track_idx)

    def _set_gdn_verified_tail_lens(
        self,
        mamba_cache,
        live_indices: torch.Tensor,
        verified_tail_lens: torch.Tensor,
    ):
        if getattr(mamba_cache, "dvr_qkvg_beta_pos", None) is None:
            return
        mamba_cache.dvr_qkvg_beta_pos[:, live_indices] = verified_tail_lens.to(
            device=mamba_cache.dvr_qkvg_beta_pos.device,
            dtype=mamba_cache.dvr_qkvg_beta_pos.dtype,
        ).unsqueeze(0)

    def _init_gdn_boundary_for_req(
        self, batch: ScheduleBatch, req, boundary_seqlen: int
    ) -> Optional[torch.Tensor]:
        assert boundary_seqlen % FLA_CHUNK_SIZE == 0
        checkpoint_track_idx = self._current_prefill_checkpoint_track_idx(batch, req)
        if checkpoint_track_idx is not None:
            # Normal prefill already wrote the chunk-aligned state into the
            # ping-pong checkpoint buffer. Reuse that slot instead of copying
            # from the live decode slot, which may no longer hold the
            # deterministic prefill checkpoint.
            self._gdn_boundary_track_idx[req.rid] = checkpoint_track_idx
            self._gdn_boundary_seqlen[req.rid] = boundary_seqlen
            req.mamba_last_track_seqlen = boundary_seqlen
            req.mamba_next_track_idx = self._mamba_other_track_idx(
                batch, checkpoint_track_idx
            )
            return None

        boundary_track_idx = req.mamba_next_track_idx
        self._gdn_boundary_track_idx[req.rid] = boundary_track_idx
        self._gdn_boundary_seqlen[req.rid] = boundary_seqlen
        req.mamba_last_track_seqlen = boundary_seqlen
        req.mamba_next_track_idx = self._mamba_other_track_idx(
            batch, boundary_track_idx
        )
        dst = req.mamba_ping_pong_track_buffer[boundary_track_idx]
        if boundary_seqlen == 0:
            return dst
        raise RuntimeError(
            "DVR GDN could not find a chunk-aligned prefill checkpoint "
            f"for boundary {boundary_seqlen}. mamba_track_interval must be "
            f"aligned to FLA_CHUNK_SIZE={FLA_CHUNK_SIZE}, and ordinary prefill "
            "must materialize that checkpoint before DVR target verify starts."
        )

    def _ensure_gdn_boundary_state(
        self, batch: ScheduleBatch, ctx: Optional[_GDNDVRStateContext] = None
    ):
        ctx = ctx or self._gdn_state_context(batch)
        if ctx is None:
            return
        zero_dst = []
        reset_pos_indices = []
        reset_pos_values = []
        for i, req in enumerate(batch.reqs):
            if req.rid not in self._gdn_boundary_seqlen:
                boundary_seqlen, verified_tail_len = self._gdn_boundary_and_tail(req)
                reset_pos_indices.append(ctx.live_indices[i])
                reset_pos_values.append(verified_tail_len)
                zero_dst_idx = self._init_gdn_boundary_for_req(
                    batch, req, boundary_seqlen
                )
                if zero_dst_idx is not None:
                    zero_dst.append(zero_dst_idx)
        if zero_dst:
            dst = torch.stack(zero_dst).to(
                device=ctx.live_indices.device, dtype=torch.long
            )
            for conv in ctx.mamba_cache.conv:
                conv[:, dst] = 0
            ctx.mamba_cache.temporal[:, dst] = 0
        if reset_pos_indices:
            self._set_gdn_verified_tail_lens(
                ctx.mamba_cache,
                torch.stack(reset_pos_indices),
                torch.tensor(reset_pos_values, device=ctx.live_indices.device),
            )

    def _restore_gdn_boundary_state_for_verify(
        self, batch: ScheduleBatch
    ) -> Optional[_GDNDVRStateContext]:
        self._ensure_gdn_boundary_state(batch)
        ctx = self._gdn_state_context(batch, require_boundary=True)
        if ctx is None:
            return None
        assert ctx.boundary_indices is not None
        if self._gdn_boundary_backup is not None:
            # Draft decode must not affect the verify starting state. Keep an
            # explicit snapshot because the shared extra-buffer slot can be
            # touched by generic Mamba tracking/cache code while draft runs.
            batch.req_to_token_pool.mamba_pool.restore_state(
                self._gdn_boundary_backup, ctx.boundary_indices
            )
            batch.req_to_token_pool.mamba_pool.restore_state(
                self._gdn_boundary_backup, ctx.live_indices
            )
        else:
            batch.req_to_token_pool.mamba_pool.copy_from(
                ctx.boundary_indices, ctx.live_indices
            )
        return ctx

    def _prepare_fixed_verify_window(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        ctx: Optional[_GDNDVRStateContext] = None,
    ):
        if (
            not self.enable_chunk_boundary_verify
            or batch.forward_mode.is_idle()
        ):
            return None

        bs = batch.batch_size()
        physical_verify_len = self.verify_window_size
        logical_draft_len = self.num_draft_tokens
        original = _GDNFixedVerifyWindowState.capture(batch, spec_info)

        draft_tokens = spec_info.draft_token.reshape(
            bs, logical_draft_len
        )
        draft_cache_locs = batch.out_cache_loc.reshape(
            bs, logical_draft_len
        )
        req_to_token = batch.req_to_token_pool.req_to_token
        verified_tail_lens = self._chunk_boundary_tail_lens(batch, ctx)

        input_ids: List[int] = []
        out_cache_locs = []
        boundary_lens = []
        padding_locs = []
        for req_i, req in enumerate(batch.reqs):
            verified_tail_len = int(verified_tail_lens[req_i].item())
            req_window = self._build_fixed_verify_req_window(
                batch=batch,
                spec_info=spec_info,
                req_i=req_i,
                req=req,
                verified_tail_len=verified_tail_len,
                draft_tokens=draft_tokens[req_i],
                draft_cache_locs=draft_cache_locs[req_i],
                req_to_token=req_to_token,
                logical_draft_len=logical_draft_len,
                physical_verify_len=physical_verify_len,
            )
            input_ids.extend(req_window.input_ids)
            out_cache_locs.extend(req_window.out_cache_locs)
            boundary_lens.append(req_window.boundary_len)
            padding_locs.extend(req_window.padding_locs)

        batch.input_ids = torch.tensor(
            input_ids, dtype=torch.long, device=spec_info.draft_token.device
        )
        batch.out_cache_loc = torch.cat(out_cache_locs).to(
            device=spec_info.draft_token.device, dtype=original.out_cache_loc.dtype
        )
        batch.seq_lens = torch.tensor(
            boundary_lens, dtype=original.seq_lens.dtype, device=batch.seq_lens.device
        )
        batch.seq_lens_cpu = torch.tensor(
            boundary_lens,
            dtype=original.seq_lens_cpu.dtype,
            device=original.seq_lens_cpu.device,
        )
        boundary_lens_gpu = batch.seq_lens.to(
            device=spec_info.positions.device, dtype=spec_info.positions.dtype
        )
        position_offsets = torch.arange(
            physical_verify_len,
            dtype=spec_info.positions.dtype,
            device=spec_info.positions.device,
        )
        batch.seq_lens_sum = sum(boundary_lens)
        spec_info.draft_token = batch.input_ids
        spec_info.positions = (
            boundary_lens_gpu[:, None] + position_offsets[None, :]
        ).reshape(-1)
        spec_info.draft_token_num = physical_verify_len
        spec_info.num_tokens_per_req = physical_verify_len
        spec_info.seq_lens_cpu = batch.seq_lens_cpu
        if ctx is not None:
            ctx.mamba_cache.dvr_qkvg_beta_pos[:, ctx.live_indices] = 0

        original.padding_locs.extend(padding_locs)
        original.verified_tail_lens = verified_tail_lens
        return original

    def _build_fixed_verify_req_window(
        self,
        *,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        req_i: int,
        req,
        verified_tail_len: int,
        draft_tokens: torch.Tensor,
        draft_cache_locs: torch.Tensor,
        req_to_token: torch.Tensor,
        logical_draft_len: int,
        physical_verify_len: int,
    ) -> _GDNFixedVerifyRequestWindow:
        boundary = (req.seqlen - 1) - verified_tail_len
        num_real_tokens = verified_tail_len + logical_draft_len
        num_padding_tokens = physical_verify_len - num_real_tokens
        if num_padding_tokens < 0:
            raise RuntimeError(
                f"DVR GDN verify window overflow: verified={verified_tail_len}, "
                f"draft={logical_draft_len}, window={physical_verify_len}."
            )

        # DVR chunk-boundary target verify uses a graphable fixed window:
        # verified_tail + draft_token + padding_token. The prompt/extend tail is
        # treated exactly like already accepted DVR tokens, so the rolling window
        # has one ownership model from prefill through target verify.
        all_ids = req.origin_input_ids + req.output_ids
        input_ids = list(all_ids[boundary : boundary + verified_tail_len])
        input_ids.extend(draft_tokens.tolist())
        input_ids.extend([0] * num_padding_tokens)

        out_cache_locs = []
        if verified_tail_len > 0:
            out_cache_locs.append(
                req_to_token[
                    batch.req_pool_indices[req_i],
                    boundary : boundary + verified_tail_len,
                ]
            )
        out_cache_locs.append(draft_cache_locs)
        padding_locs = []
        if num_padding_tokens > 0:
            pad_locs, owned_pad_locs = self._fixed_window_padding_locs(
                batch=batch,
                last_real_loc=draft_cache_locs[-1],
                num_padding_tokens=num_padding_tokens,
            )
            if owned_pad_locs is not None:
                padding_locs.append(owned_pad_locs)
            out_cache_locs.append(pad_locs)

        return _GDNFixedVerifyRequestWindow(
            input_ids=input_ids,
            out_cache_locs=out_cache_locs,
            boundary_len=boundary,
            padding_locs=padding_locs,
        )

    def _restore_after_fixed_verify_window(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        logits_output: LogitsProcessorOutput,
        fixed_window_state,
        ctx: Optional[_GDNDVRStateContext] = None,
    ):
        if fixed_window_state is None:
            return

        state: _GDNFixedVerifyWindowState = fixed_window_state
        keep = self._fixed_verify_draft_rows(
            state.verified_tail_lens, logits_output.next_token_logits.device
        )
        logits_output.next_token_logits = logits_output.next_token_logits[keep]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[keep]

        state.restore(batch, spec_info)

        if state.padding_locs:
            self.token_to_kv_pool_allocator.free(torch.cat(state.padding_locs))

        ctx = ctx or self._gdn_state_context(batch)
        if ctx is not None:
            self._set_gdn_verified_tail_lens(
                ctx.mamba_cache, ctx.live_indices, state.verified_tail_lens
            )

    # GDN boundary-state lifecycle. The boundary slot is the deterministic
    # chunk-aligned state used as the next target-verify starting point; the live
    # slot remains the autoregressive state consumed by following draft decode.

    def _backup_gdn_boundary_state(self, batch: ScheduleBatch):
        ctx = self._gdn_state_context(batch, require_boundary=True)
        if ctx is None:
            self._gdn_boundary_backup = None
            return
        assert ctx.boundary_indices is not None
        self._gdn_boundary_backup = batch.req_to_token_pool.mamba_pool.backup_state(
            ctx.boundary_indices
        )

    def _commit_gdn_state_after_verify(
        self,
        batch: ScheduleBatch,
        verify_output: EagleVerifyOutput,
        ctx: Optional[_GDNDVRStateContext] = None,
    ):
        ctx = ctx or self._gdn_state_context(batch, require_boundary=True)
        if ctx is None:
            return
        assert ctx.boundary_indices is not None

        accepted_tokens, accepted_steps = self._accepted_token_metadata(
            batch, verify_output, ctx.live_indices.device
        )
        if accepted_tokens.numel() == 0:
            return

        attn_backend = self.model_runner.attn_backend
        linear_backend = getattr(attn_backend, "linear_attn_backend", None)
        if linear_backend is None:
            return
        verified_tail_lens = self._chunk_boundary_tail_lens(batch, ctx).to(
            device=ctx.live_indices.device, dtype=torch.long
        )
        crossing = linear_backend.update_dvr_state_after_verify(
            live_indices=ctx.live_indices,
            boundary_indices=ctx.boundary_indices,
            verified_tail_lens=verified_tail_lens,
            accepted_tokens=accepted_tokens,
            accepted_steps=accepted_steps,
        )

        if crossing.any():
            for req_i, req in enumerate(batch.reqs):
                if not bool(crossing[req_i].item()):
                    continue
                self._gdn_boundary_seqlen[req.rid] += FLA_CHUNK_SIZE
                req.mamba_last_track_seqlen = self._gdn_boundary_seqlen[req.rid]
                req.mamba_next_track_idx = self._mamba_other_track_idx(
                    batch, self._gdn_boundary_track_idx[req.rid]
                )

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
        gdn_ctx = self._restore_gdn_boundary_state_for_verify(batch)
        fixed_window_state = self._prepare_fixed_verify_window(batch, spec_info, gdn_ctx)

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        with dvr_runtime_verify_window(
            self.model_runner.attn_backend, spec_info.num_tokens_per_req
        ):
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
        maybe_detect_nan(logits_output.next_token_logits, "dvr target verify")
        self._restore_after_fixed_verify_window(
            batch, spec_info, logits_output, fixed_window_state, gdn_ctx
        )

        spec_info.hidden_states = logits_output.hidden_states
        verify_output: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask=None,
        )

        self._select_accepted_verify_outputs(logits_output, verify_output)
        self._commit_gdn_state_after_verify(batch, verify_output, gdn_ctx)
        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, verify_output, logits_output)
        self.postprocess_for_verify(batch, verify_output)
        return logits_output, verify_output, can_run_cuda_graph

    def postprocess_for_verify(
        self, batch: ScheduleBatch, verify_output: EagleVerifyOutput
    ):
        self._prepare_next_draft_after_verify(batch, verify_output)
