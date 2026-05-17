from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclass
class DVRLinearStateContext:
    state_cache: Any
    live_indices: torch.Tensor
    boundary_indices: Optional[torch.Tensor] = None


class DVRLinearStateLifecycle:
    """Manage chunk-boundary state for DVR linear-state layers.

    The current implementation is backed by SGLang's Mamba/GDN state cache and
    ping-pong prefill checkpoints. Keeping it outside `dvr_worker.py` prevents
    the speculative control flow from depending on those backend details.
    """

    def __init__(self, *, server_args, model_runner):
        self.server_args = server_args
        self.model_runner = model_runner
        self.boundary_seqlen = {}
        self.boundary_track_idx = {}
        self.boundary_backup = None
        self.live_backup = None
        self.validate_args()

    def validate_args(self):
        if self.model_runner.hybrid_gdn_config is None:
            return
        if self.server_args.mamba_track_interval != FLA_CHUNK_SIZE:
            raise ValueError(
                "DVR GDN requires mamba_track_interval to match "
                f"FLA_CHUNK_SIZE={FLA_CHUNK_SIZE}, got "
                f"{self.server_args.mamba_track_interval}. Multiples larger than "
                "FLA_CHUNK_SIZE can miss the latest chunk boundary from the "
                "first prefill because the current extra_buffer path stores "
                "only one tracked prefill checkpoint."
            )
        if self.server_args.mamba_ssm_dtype != "float32":
            raise ValueError("DVR GDN requires fp32 Mamba/GDN SSM state storage.")

    def prepare_for_draft(self, batch: ScheduleBatch):
        self.ensure_boundary_state(batch)
        self.backup_boundary_state(batch)

    def restore_for_verify(
        self, batch: ScheduleBatch
    ) -> Optional[DVRLinearStateContext]:
        return self.restore_boundary_state_for_verify(batch)

    def commit_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        accepted_tokens: torch.Tensor,
        accepted_steps: torch.Tensor,
        ctx: Optional[DVRLinearStateContext] = None,
    ):
        ctx = ctx or self.state_context(batch, require_boundary=True)
        if ctx is None:
            return
        assert ctx.boundary_indices is not None
        if accepted_tokens.numel() == 0:
            return

        dvr_state_adapter = self.state_adapter()
        if dvr_state_adapter is None:
            return
        verified_tail_lens = self.chunk_boundary_tail_lens(batch, ctx).to(
            device=ctx.live_indices.device, dtype=torch.long
        )
        crossing = dvr_state_adapter.commit_after_verify(
            state_cache=ctx.state_cache,
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
                self.boundary_seqlen[req.rid] += FLA_CHUNK_SIZE
                req.mamba_last_track_seqlen = self.boundary_seqlen[req.rid]
                req.mamba_next_track_idx = self.mamba_other_track_idx(
                    batch, self.boundary_track_idx[req.rid]
                )
        self.boundary_backup = None
        self.live_backup = None

    def has_dvr_state(self, batch: ScheduleBatch) -> bool:
        return (
            self.model_runner.hybrid_gdn_config is not None
            and hasattr(batch.req_to_token_pool, "get_mamba_indices")
            and batch.batch_size() > 0
            and all(req.mamba_ping_pong_track_buffer is not None for req in batch.reqs)
        )

    def state_context(
        self, batch: ScheduleBatch, require_boundary: bool = False
    ) -> Optional[DVRLinearStateContext]:
        if not self.has_dvr_state(batch):
            return None
        assert self.server_args.mamba_track_interval == FLA_CHUNK_SIZE, (
            "DVR GDN target verify must start from FLA chunk boundaries. "
            "The current prefill tracker only guarantees the latest boundary "
            "when mamba_track_interval equals FLA_CHUNK_SIZE."
        )
        live_indices = batch.req_to_token_pool.get_mamba_indices(
            batch.req_pool_indices
        ).to(torch.long)
        state_cache = batch.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        assert state_cache.temporal.dtype == torch.float32, (
            "DVR GDN requires fp32 temporal state checkpoints. bf16/fp16 "
            "checkpoints round the chunkwise scan state and can diverge from "
            "full prefill across chunks."
        )
        assert state_cache.intermediate_ssm.dtype == torch.float32, (
            "DVR GDN requires fp32 intermediate prefill states."
        )
        boundary_indices = None
        if require_boundary:
            boundary_indices = torch.stack(
                [
                    req.mamba_ping_pong_track_buffer[
                        self.boundary_track_idx[req.rid]
                    ]
                    for req in batch.reqs
                ]
            ).to(device=live_indices.device, dtype=torch.long)
        return DVRLinearStateContext(
            state_cache=state_cache,
            live_indices=live_indices,
            boundary_indices=boundary_indices,
        )

    @staticmethod
    def boundary_and_tail(req) -> Tuple[int, int]:
        boundary_seqlen = ((req.seqlen - 1) // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
        verified_tail_len = (req.seqlen - 1) - boundary_seqlen
        return boundary_seqlen, verified_tail_len

    def chunk_boundary_tail_lens(
        self, batch: ScheduleBatch, ctx: Optional[DVRLinearStateContext] = None
    ) -> torch.Tensor:
        ctx = ctx or self.state_context(batch)
        if ctx is not None:
            dvr_state_adapter = self.state_adapter()
            if dvr_state_adapter is not None:
                tail_lens = dvr_state_adapter.state_input_tail_lens(
                    state_cache=ctx.state_cache,
                    live_indices=ctx.live_indices,
                )
                if tail_lens is not None:
                    return tail_lens.to(torch.long)
        return torch.tensor(
            [self.boundary_and_tail(req)[1] for req in batch.reqs],
            dtype=torch.long,
            device=batch.seq_lens.device,
        )

    def mamba_other_track_idx(self, batch: ScheduleBatch, track_idx: int) -> int:
        return batch.req_to_token_pool.get_mamba_ping_pong_other_idx(track_idx)

    def current_prefill_checkpoint_track_idx(
        self, batch: ScheduleBatch, req
    ) -> Optional[int]:
        boundary_seqlen, _ = self.boundary_and_tail(req)
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
        return self.mamba_other_track_idx(batch, req.mamba_next_track_idx)

    def set_verified_tail_lens(
        self,
        state_cache,
        live_indices: torch.Tensor,
        verified_tail_lens: torch.Tensor,
    ):
        dvr_state_adapter = self.state_adapter()
        if dvr_state_adapter is None:
            return
        dvr_state_adapter.set_state_input_tail_lens(
            state_cache=state_cache,
            live_indices=live_indices,
            tail_lens=verified_tail_lens,
        )

    def state_adapter(self):
        linear_backend = getattr(
            self.model_runner.attn_backend, "linear_attn_backend", None
        )
        if linear_backend is None:
            return None
        return getattr(linear_backend, "dvr_state_adapter", None)

    def init_boundary_for_req(
        self, batch: ScheduleBatch, req, boundary_seqlen: int
    ) -> Optional[torch.Tensor]:
        assert boundary_seqlen % FLA_CHUNK_SIZE == 0
        checkpoint_track_idx = self.current_prefill_checkpoint_track_idx(batch, req)
        if checkpoint_track_idx is not None:
            # Normal prefill already wrote the chunk-aligned state into the
            # ping-pong checkpoint buffer. Reuse that slot instead of copying
            # from the live decode slot, which may no longer hold the
            # deterministic prefill checkpoint.
            self.boundary_track_idx[req.rid] = checkpoint_track_idx
            self.boundary_seqlen[req.rid] = boundary_seqlen
            req.mamba_last_track_seqlen = boundary_seqlen
            req.mamba_next_track_idx = self.mamba_other_track_idx(
                batch, checkpoint_track_idx
            )
            return None

        boundary_track_idx = req.mamba_next_track_idx
        self.boundary_track_idx[req.rid] = boundary_track_idx
        self.boundary_seqlen[req.rid] = boundary_seqlen
        req.mamba_last_track_seqlen = boundary_seqlen
        req.mamba_next_track_idx = self.mamba_other_track_idx(batch, boundary_track_idx)
        dst = req.mamba_ping_pong_track_buffer[boundary_track_idx]
        if boundary_seqlen == 0:
            return dst
        raise RuntimeError(
            "DVR GDN could not find a chunk-aligned prefill checkpoint "
            f"for boundary {boundary_seqlen}. mamba_track_interval must be "
            f"aligned to FLA_CHUNK_SIZE={FLA_CHUNK_SIZE}, and ordinary prefill "
            "must materialize that checkpoint before DVR target verify starts."
        )

    def ensure_boundary_state(
        self, batch: ScheduleBatch, ctx: Optional[DVRLinearStateContext] = None
    ):
        ctx = ctx or self.state_context(batch)
        if ctx is None:
            return
        zero_dst = []
        reset_pos_indices = []
        reset_pos_values = []
        for i, req in enumerate(batch.reqs):
            if req.rid not in self.boundary_seqlen:
                boundary_seqlen, verified_tail_len = self.boundary_and_tail(req)
                reset_pos_indices.append(ctx.live_indices[i])
                reset_pos_values.append(verified_tail_len)
                zero_dst_idx = self.init_boundary_for_req(batch, req, boundary_seqlen)
                if zero_dst_idx is not None:
                    zero_dst.append(zero_dst_idx)
        if zero_dst:
            dst = torch.stack(zero_dst).to(
                device=ctx.live_indices.device, dtype=torch.long
            )
            for conv in ctx.state_cache.conv:
                conv[:, dst] = 0
            ctx.state_cache.temporal[:, dst] = 0
        if reset_pos_indices:
            self.set_verified_tail_lens(
                ctx.state_cache,
                torch.stack(reset_pos_indices),
                torch.tensor(reset_pos_values, device=ctx.live_indices.device),
            )

    def restore_boundary_state_for_verify(
        self, batch: ScheduleBatch
    ) -> Optional[DVRLinearStateContext]:
        self.ensure_boundary_state(batch)
        ctx = self.state_context(batch, require_boundary=True)
        if ctx is None:
            return None
        assert ctx.boundary_indices is not None
        if self.boundary_backup is not None:
            dvr_state_adapter = self.state_adapter()
            assert dvr_state_adapter is not None
            # Draft decode mutates the live recurrent slot. DVR target verify
            # needs the chunk-boundary SSM state for chunkwise scan, but the
            # draft-start conv state for producing q/k/v on the draft suffix.
            dvr_state_adapter.restore_recurrent_state(
                state_cache=ctx.state_cache,
                backup=self.boundary_backup,
                indices=ctx.boundary_indices,
            )
            ctx.state_cache.temporal[:, ctx.live_indices] = (
                self.boundary_backup.temporal.to(
                    ctx.state_cache.temporal.dtype, copy=False
                )
            )
            if self.live_backup is not None:
                for conv, saved_conv in zip(
                    ctx.state_cache.conv, self.live_backup.conv, strict=True
                ):
                    conv[:, ctx.live_indices] = saved_conv.to(conv.dtype, copy=False)
        else:
            ctx.state_cache.temporal[:, ctx.live_indices] = ctx.state_cache.temporal[
                :, ctx.boundary_indices
            ]
        return ctx

    def backup_boundary_state(self, batch: ScheduleBatch):
        ctx = self.state_context(batch, require_boundary=True)
        if ctx is None:
            self.boundary_backup = None
            self.live_backup = None
            return
        assert ctx.boundary_indices is not None
        dvr_state_adapter = self.state_adapter()
        assert dvr_state_adapter is not None
        self.boundary_backup = dvr_state_adapter.backup_recurrent_state(
            state_cache=ctx.state_cache,
            indices=ctx.boundary_indices,
        )
        self.live_backup = dvr_state_adapter.backup_recurrent_state(
            state_cache=ctx.state_cache,
            indices=ctx.live_indices,
        )
