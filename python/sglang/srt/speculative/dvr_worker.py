from __future__ import annotations

import logging
from typing import Optional

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class DecodeVerifyRollbackWorker:
    """DVR worker shell.

    Phase 1 only wires DVR into the speculative worker stack and validates
    launch-time constraints. Later phases will replace the transparent target
    forwarding below with self-decode draft and target prefill verification.
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
        self.page_size = server_args.page_size
        self.max_batch_size = target_worker.max_running_requests
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens

        logger.info(
            "Initialized DVR worker shell: num_steps=%s, num_draft_tokens=%s",
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

    def __getattr__(self, name):
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        return None

    def forward_batch_generation(self, batch, *args, **kwargs):
        if isinstance(batch, ScheduleBatch):
            # Spec v1 passes ScheduleBatch directly. A batch can contain extend
            # work while its global mode is DECODE, so the target worker needs a
            # model-worker view that uses the prefill/extend path.
            if batch.is_extend_in_batch and batch.forward_mode.is_decode():
                original_forward_mode = batch.forward_mode
                batch.forward_mode = ForwardMode.EXTEND
                try:
                    batch = batch.get_model_worker_batch()
                finally:
                    batch.forward_mode = original_forward_mode
            else:
                batch = batch.get_model_worker_batch()
        return self.target_worker.forward_batch_generation(batch, *args, **kwargs)
