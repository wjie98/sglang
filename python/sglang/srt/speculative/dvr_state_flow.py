from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dvr_server_args import is_dvr_eagle_enabled

_BoundaryReplayTask = tuple[Any, int, int, Optional[torch.Tensor], int, torch.Tensor]


def build_dvr_private_extend_batch(
    batch,
    *,
    reqs: list[Any],
    input_ids: list[int],
    out_cache_locs: Optional[list[torch.Tensor] | torch.Tensor],
    prefix_lens: list[int],
    extend_lens: list[int],
    final_seq_lens: list[int],
    extend_logprob_start_lens: list[int],
    is_prefill_only: bool,
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
) -> ScheduleBatch:
    """Create a DVR-owned EXTEND batch for target verify replay.

    Target verify replay runs EXTEND on a private token span: the unclosed GDN
    chunk tail plus draft tokens.  SGLang does not expose a smaller replay
    object, so this constructor mirrors only the ScheduleBatch fields consumed
    by ForwardBatch/EXTEND/GDN tracking and leaves sampling/output state off.
    Keep this explicit and close to ScheduleBatch when upstream fields change.
    """

    device = batch.seq_lens.device
    global_num_tokens = None
    global_num_tokens_for_logprob = None
    if batch.global_num_tokens is not None:
        dp_world = len(batch.global_num_tokens)
        global_num_tokens = [len(input_ids)] * dp_world
        global_num_tokens_for_logprob = [len(input_ids)] * dp_world

    req_pool_indices = torch.tensor(
        [req.req_pool_idx for req in reqs],
        dtype=torch.int64,
        device=device,
    )
    final_seq_lens_tensor = torch.tensor(
        final_seq_lens, dtype=torch.int64, device=device
    )
    out_cache_loc = None
    if out_cache_locs is not None:
        out_cache_loc = (
            torch.cat(out_cache_locs)
            if isinstance(out_cache_locs, list)
            else out_cache_locs
        ).to(device=device)
    replay_batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=batch.req_to_token_pool,
        token_to_kv_pool_allocator=batch.token_to_kv_pool_allocator,
        tree_cache=batch.tree_cache,
        model_config=batch.model_config,
        enable_overlap=batch.enable_overlap,
        spec_algorithm=batch.spec_algorithm,
        dllm_config=batch.dllm_config,
    )

    # Request and token layout: ForwardBatch consumes these fields to build the
    # physical EXTEND positions and write replay KV rows back to the live pool.
    replay_batch.forward_mode = ForwardMode.EXTEND
    replay_batch.input_ids = torch.tensor(
        input_ids, dtype=torch.int64, device=device
    )
    replay_batch.req_pool_indices = req_pool_indices
    replay_batch.req_pool_indices_cpu = req_pool_indices.cpu()
    replay_batch.seq_lens = final_seq_lens_tensor
    replay_batch.seq_lens_cpu = final_seq_lens_tensor.cpu()
    replay_batch.seq_lens_sum = sum(final_seq_lens)
    replay_batch.out_cache_loc = out_cache_loc
    replay_batch.orig_seq_lens = final_seq_lens_tensor.to(dtype=torch.int32)
    replay_batch.extend_num_tokens = len(input_ids)
    replay_batch.extend_lens = extend_lens
    replay_batch.prefix_lens = prefix_lens

    # Replay is an internal verifier oracle. It may request hidden states, but
    # never owns sampling or user-visible output.
    replay_batch.extend_logprob_start_lens = extend_logprob_start_lens
    replay_batch.multimodal_inputs = [req.multimodal_inputs for req in reqs]
    replay_batch.return_logprob = False
    replay_batch.global_num_tokens = global_num_tokens
    replay_batch.global_num_tokens_for_logprob = global_num_tokens_for_logprob
    replay_batch.is_extend_in_batch = True
    replay_batch.all_extend_in_batch = True
    replay_batch.capture_hidden_mode = capture_hidden_mode
    replay_batch.is_prefill_only = is_prefill_only
    replay_batch.has_grammar = False
    replay_batch.return_hidden_states = False
    return replay_batch


@contextmanager
def dvr_suffix_replay_context(
    *,
    batch,
    linear_state,
    linear_state_ctx,
    base_seq_lens_cpu: list[int],
    append_tokens_cpu_by_req: list[list[int]],
    append_cache_locs_by_req: list[torch.Tensor],
    request_token_ids_for_replay,
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
    hidden_gather_append_count: Optional[int] = None,
    restore_boundary_state: bool = False,
    use_mamba_cow_from_boundary: bool = False,
    restore_live_state: bool = True,
):
    """Prepare one boundary-tail plus appended-token replay.

    Verify oracles append every draft row and gather hidden states. Partial
    rollback appends only accepted verify-input rows and publishes the replayed
    live state. Both use the same physical EXTEND layout and cache ownership.
    """

    if batch.forward_mode.is_idle() or linear_state_ctx is None:
        yield None
        return

    live_indices = linear_state_ctx.live_indices
    boundary_indices = linear_state_ctx.boundary_indices
    if boundary_indices is None:
        raise RuntimeError("DVR suffix replay requires a chunk-boundary state slot.")

    boundary_lens = linear_state.boundary_lens_for_replay(batch, base_seq_lens_cpu)
    extend_lens = []
    final_seq_lens = []
    input_ids = []
    out_cache_locs = []
    append_locs = []
    append_rows = []
    append_offsets = []
    hidden_gather_indices = []
    token_offset = 0

    for req_i, (req, seq_len, boundary, tokens, cache_locs) in enumerate(
        zip(
            batch.reqs,
            base_seq_lens_cpu,
            boundary_lens,
            append_tokens_cpu_by_req,
            append_cache_locs_by_req,
            strict=True,
        )
    ):
        seq_len = int(seq_len)
        boundary = int(boundary)
        tail_len = seq_len - boundary
        append_count = len(tokens)
        extend_len = tail_len + append_count
        extend_lens.append(extend_len)
        final_seq_lens.append(seq_len + append_count)

        replay_tokens = request_token_ids_for_replay(req, seq_len)
        input_ids.extend(replay_tokens[boundary:seq_len])
        input_ids.extend(tokens)
        if tail_len:
            out_cache_locs.append(
                batch.req_to_token_pool.req_to_token[
                    req.req_pool_idx, boundary:seq_len
                ].to(torch.long)
            )
        if append_count:
            cache_locs = cache_locs.to(torch.long)
            out_cache_locs.append(cache_locs)
            append_locs.append(cache_locs)
            append_rows.append(
                batch.req_pool_indices[req_i].to(torch.long).repeat(append_count)
            )
            append_offsets.append(
                torch.arange(
                    seq_len,
                    seq_len + append_count,
                    dtype=torch.long,
                    device=batch.seq_lens.device,
                )
            )
        if hidden_gather_append_count is not None:
            hidden_gather_indices.extend(
                range(
                    token_offset + tail_len,
                    token_offset + tail_len + hidden_gather_append_count,
                )
            )
        token_offset += extend_len

    if not input_ids:
        raise RuntimeError("DVR suffix replay received no tail or append tokens.")
    replay_batch = build_dvr_private_extend_batch(
        batch,
        reqs=batch.reqs,
        input_ids=input_ids,
        out_cache_locs=out_cache_locs,
        prefix_lens=boundary_lens,
        extend_lens=extend_lens,
        final_seq_lens=final_seq_lens,
        extend_logprob_start_lens=extend_lens,
        capture_hidden_mode=capture_hidden_mode,
        is_prefill_only=batch.is_prefill_only,
    )
    if use_mamba_cow_from_boundary:
        replay_batch.mamba_cow_src_indices = boundary_indices
        replay_batch.mamba_cow_dst_indices = live_indices

    with linear_state.replay_state_context(
        linear_state_ctx,
        restore_live_state=restore_live_state,
    ):
        if restore_boundary_state:
            boundary_backup = linear_state.boundary_backup
            if boundary_backup is None:
                raise RuntimeError(
                    "DVR target replay requires the preserved boundary state."
                )
            linear_state_ctx.state_adapter.restore_recurrent_state(
                state_cache=linear_state_ctx.state_cache,
                backup=boundary_backup,
                indices=live_indices,
            )

        # Draft KV rows must be visible at absolute positions
        # base_seq_len..base_seq_len+draft during the oracle forward.
        if append_rows:
            replay_batch.req_to_token_pool.write(
                (torch.cat(append_rows), torch.cat(append_offsets)),
                torch.cat(append_locs).to(
                    device=replay_batch.seq_lens.device,
                    dtype=torch.int32,
                ),
            )
        gather_indices = (
            torch.tensor(
                hidden_gather_indices,
                dtype=torch.long,
                device=batch.seq_lens.device,
            )
            if hidden_gather_append_count is not None
            else None
        )
        yield replay_batch, gather_indices


def run_target_verify_replay(
    *,
    target_worker,
    replay_batch: ScheduleBatch,
    hidden_gather_indices: torch.Tensor,
    append_token_count: int,
    use_forward_batch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run target verify replay and return only draft-row outputs."""

    model_runner = target_worker.model_runner
    if use_forward_batch:
        forward_batch = ForwardBatch.init_new(replay_batch, model_runner)
        if model_runner.model_is_mrope:
            device = replay_batch.seq_lens.device
            mrope_chunks = []
            mm_inputs = replay_batch.multimodal_inputs
            for req_i, (final_seq_len, boundary, extend_len) in enumerate(
                zip(
                    replay_batch.seq_lens_cpu.tolist(),
                    replay_batch.prefix_lens,
                    replay_batch.extend_lens,
                    strict=True,
                )
            ):
                seq_len = int(final_seq_len) - append_token_count
                tail_len = int(extend_len) - append_token_count
                mm_input = (
                    None
                    if mm_inputs is None or req_i >= len(mm_inputs)
                    else mm_inputs[req_i]
                )
                mm_positions = getattr(mm_input, "mrope_positions", None)
                chunk_parts = []
                if mm_positions is not None:
                    tail_positions = mm_positions[:, int(boundary) : int(seq_len)].to(
                        device=device, dtype=torch.long
                    )
                    if tail_positions.shape[1] > 0:
                        chunk_parts.append(tail_positions)
                filled_tail = sum(part.shape[1] for part in chunk_parts)
                if filled_tail < int(tail_len):
                    chunk_parts.append(
                        torch.arange(
                            int(boundary) + filled_tail,
                            int(seq_len),
                            dtype=torch.long,
                            device=device,
                        )
                        .unsqueeze(0)
                        .repeat(3, 1)
                    )
                chunk_parts.append(
                    torch.arange(
                        int(seq_len),
                        int(seq_len) + append_token_count,
                        dtype=torch.long,
                        device=device,
                    )
                    .unsqueeze(0)
                    .repeat(3, 1)
                )
                mrope_chunks.append(torch.cat(chunk_parts, dim=1))
            forward_batch.mrope_positions = torch.cat(mrope_chunks, dim=1)
        oracle_output = target_worker.forward_batch_generation(
            batch=None,
            forward_batch=forward_batch,
            is_verify=True,
        )
    else:
        oracle_output = target_worker.forward_batch_generation(
            batch=replay_batch,
            is_verify=True,
        )
        forward_batch = ForwardBatch.init_new(replay_batch, model_runner)

    hidden_states = oracle_output.logits_output.hidden_states
    if hidden_states is None:
        raise RuntimeError("DVR target replay did not return hidden states.")
    draft_hidden_states = hidden_states[
        hidden_gather_indices.to(
            device=hidden_states.device,
            dtype=torch.long,
        )
    ]
    logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)
    logits_metadata.next_token_logits_buffer = None
    draft_logits = target_worker.model_runner.model.logits_processor._get_logits(
        draft_hidden_states,
        target_worker.model_runner.model.lm_head,
        logits_metadata,
    )
    return draft_logits, draft_hidden_states


@dataclass
class DVRLinearStateContext:
    state_cache: Any
    state_adapter: Any
    state_input_cache: Any
    state_input_indices: torch.Tensor
    live_indices: torch.Tensor
    boundary_indices: Optional[torch.Tensor] = None


class DVRLinearStateLifecycle:
    """Manage chunk-boundary state for DVR linear-state layers.

    The current implementation is backed by SGLang's linear-state cache and
    ping-pong prefill checkpoints. Keeping it outside `dvr_worker.py` prevents
    the speculative control flow from depending on those backend details.
    """

    def __init__(self, *, server_args, model_runner):
        self.server_args = server_args
        self.model_runner = model_runner
        self.boundary_seqlen = {}
        self.boundary_track_idx = {}
        self.boundary_backup = None
        # Identifies the worker-local target boundary snapshot.  Separate
        # EAGLE/MTP draft phases can mutate shared linear-state slots before
        # the next target verify, so DVR-EAGLE preserves this snapshot when the
        # request and chunk boundary are unchanged.
        self.boundary_backup_keys = None
        self.live_backup = None
        if self.state_adapter() is None:
            return
        if self.server_args.mamba_track_interval != FLA_CHUNK_SIZE:
            raise ValueError(
                "DVR linear-state verify requires mamba_track_interval to match "
                f"FLA_CHUNK_SIZE={FLA_CHUNK_SIZE}, got "
                f"{self.server_args.mamba_track_interval}. Multiples larger than "
                "FLA_CHUNK_SIZE can miss the latest chunk boundary from the "
                "first prefill because the current extra_buffer path stores "
                "only one tracked prefill checkpoint."
            )
        if self.server_args.mamba_ssm_dtype != "float32":
            raise ValueError(
                "DVR linear-state verify requires fp32 recurrent state storage."
            )

    def clear_cache_state(self):
        self.boundary_seqlen.clear()
        self.boundary_track_idx.clear()
        self.boundary_backup = None
        self.boundary_backup_keys = None
        self.live_backup = None

    @contextmanager
    def replay_state_context(
        self,
        linear_state_ctx: DVRLinearStateContext,
        *,
        restore_live_state: bool = True,
    ):
        """Save/restore linear-state side effects around a replay oracle."""

        state_adapter = linear_state_ctx.state_adapter
        state_cache = linear_state_ctx.state_cache
        state_input_cache = linear_state_ctx.state_input_cache
        saved_tail_lens = state_input_cache.get_tail_lens(
            indices=linear_state_ctx.state_input_indices
        ).clone()
        saved_state_input_window = state_input_cache.backup_rows(
            indices=linear_state_ctx.state_input_indices
        )

        live_backup = None
        if restore_live_state:
            live_backup = state_adapter.backup_recurrent_state(
                state_cache=state_cache,
                indices=linear_state_ctx.live_indices,
            )

        try:
            yield
        finally:
            if live_backup is not None:
                state_adapter.restore_recurrent_state(
                    state_cache=state_cache,
                    backup=live_backup,
                    indices=linear_state_ctx.live_indices,
                )
            state_input_cache.set_tail_lens(
                indices=linear_state_ctx.state_input_indices,
                value=saved_tail_lens,
            )
            state_input_cache.restore_rows(
                indices=linear_state_ctx.state_input_indices,
                backup=saved_state_input_window,
            )

    def prepare_for_draft(
        self,
        batch: ScheduleBatch,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
        request_token_ids_for_replay: Optional[Callable] = None,
    ) -> None:
        active_rids = {req.rid for req in batch.reqs}
        for rid in list(self.boundary_seqlen):
            if rid not in active_rids:
                self.boundary_seqlen.pop(rid, None)
                self.boundary_track_idx.pop(rid, None)

        seq_lens_cpu = seq_lens_cpu or self.batch_seq_lens_cpu(batch)
        for i, req in enumerate(batch.reqs):
            recorded_boundary = self.boundary_seqlen.get(req.rid)
            if recorded_boundary is None:
                continue
            current_boundary, _ = self.boundary_and_tail_for_seq_len(
                int(seq_lens_cpu[i])
            )
            if (
                req.rid not in self.boundary_track_idx
                or recorded_boundary != current_boundary
                or recorded_boundary % FLA_CHUNK_SIZE != 0
            ):
                self.boundary_seqlen.pop(req.rid, None)
                self.boundary_track_idx.pop(req.rid, None)

        tasks = self.ensure_boundary_state(
            batch,
            seq_lens_cpu=seq_lens_cpu,
        )
        if not tasks:
            return
        if request_token_ids_for_replay is None:
            raise RuntimeError("DVR boundary replay requires a token replay source.")

        ctx = self.state_context(batch)
        if ctx is None:
            return

        live_indices = torch.stack([task[-1] for task in tasks]).to(
            device=ctx.live_indices.device, dtype=torch.long
        )
        live_backup = ctx.state_adapter.backup_recurrent_state(
            state_cache=ctx.state_cache,
            indices=live_indices,
        )

        try:
            zero_live_indices = [
                live_idx
                for _, source_seqlen, _, source_state_indices, _, live_idx in tasks
                if source_state_indices is None or source_seqlen == 0
            ]
            if zero_live_indices:
                ctx.state_adapter.zero_recurrent_state(
                    state_cache=ctx.state_cache,
                    indices=torch.stack(zero_live_indices).to(
                        device=ctx.live_indices.device, dtype=torch.long
                    ),
                )

            replay_source_indices = [
                source_state_indices.reshape(-1)
                for _, source_seqlen, _, source_state_indices, _, _ in tasks
                if source_state_indices is not None and source_seqlen > 0
            ]
            replay_live_indices = [
                live_idx
                for _, source_seqlen, _, source_state_indices, _, live_idx in tasks
                if source_state_indices is not None and source_seqlen > 0
            ]
            if replay_source_indices:
                self.copy_state_indices(
                    batch=batch,
                    src_indices=torch.cat(replay_source_indices).to(
                        device=ctx.live_indices.device, dtype=torch.long
                    ),
                    dst_indices=torch.stack(replay_live_indices).to(
                        device=ctx.live_indices.device, dtype=torch.long
                    ),
                )
            input_ids = []
            out_cache_locs = []
            prefix_lens = []
            extend_lens = []
            final_seq_lens = []
            for (
                req,
                source_seqlen,
                boundary_seqlen,
                _source_state_indices,
                _boundary_track_idx,
                _live_idx,
            ) in tasks:
                token_ids = request_token_ids_for_replay(req, boundary_seqlen)
                input_ids.extend(token_ids[source_seqlen:boundary_seqlen])
                out_cache_locs.append(
                    batch.req_to_token_pool.req_to_token[
                        req.req_pool_idx,
                        source_seqlen:boundary_seqlen,
                    ].to(torch.long)
                )
                prefix_lens.append(source_seqlen)
                extend_lens.append(boundary_seqlen - source_seqlen)
                final_seq_lens.append(boundary_seqlen)
            if input_ids:
                reqs = [task[0] for task in tasks]
                replay_batch = build_dvr_private_extend_batch(
                    batch,
                    reqs=reqs,
                    input_ids=input_ids,
                    out_cache_locs=out_cache_locs,
                    prefix_lens=prefix_lens,
                    extend_lens=extend_lens,
                    final_seq_lens=final_seq_lens,
                    extend_logprob_start_lens=prefix_lens,
                    is_prefill_only=True,
                )
                replay_batch.mamba_track_indices = self.boundary_indices_for_reqs(
                    reqs=reqs,
                    track_indices=[task[4] for task in tasks],
                    device=batch.device,
                )
                replay_batch.mamba_track_mask = torch.ones(
                    len(reqs), dtype=torch.bool, device=batch.device
                )
                replay_batch.mamba_track_seqlens = torch.tensor(
                    final_seq_lens, dtype=torch.int64, device=batch.device
                )
                replay_batch.is_extend_in_batch = False
                replay_batch.all_extend_in_batch = False
                forward_batch = ForwardBatch.init_new(replay_batch, self.model_runner)
                self.model_runner.forward(forward_batch)
        finally:
            ctx.state_adapter.restore_recurrent_state(
                state_cache=ctx.state_cache,
                backup=live_backup,
                indices=live_indices,
            )
        task_rids = {task[0].rid for task in tasks}
        seq_lens_by_rid = {
            req.rid: int(seq_len)
            for req, seq_len in zip(batch.reqs, seq_lens_cpu, strict=True)
        }
        state_input_indices = []
        tail_lens = []
        for i, req in enumerate(batch.reqs):
            if req.rid not in task_rids:
                continue
            seq_len = seq_lens_by_rid[req.rid]
            boundary = self.boundary_seqlen[req.rid]
            state_input_indices.append(ctx.state_input_indices[i])
            tail_lens.append(seq_len - boundary)
        if state_input_indices:
            ctx.state_input_cache.set_tail_lens(
                indices=torch.stack(state_input_indices),
                value=torch.tensor(tail_lens, device=ctx.live_indices.device),
            )

    def backup_boundary_state(
        self, batch: ScheduleBatch, *, preserve_existing: bool = False
    ):
        ctx = self.state_context(batch, require_boundary=True)
        if ctx is None:
            self.boundary_backup = None
            self.boundary_backup_keys = None
            self.live_backup = None
            return
        assert ctx.boundary_indices is not None
        backup_keys = [
            (req.rid, int(self.boundary_seqlen.get(req.rid, -1)))
            for req in batch.reqs
        ]
        if (
            preserve_existing
            and self.boundary_backup is not None
            and self.boundary_backup_keys == backup_keys
        ):
            return
        self.boundary_backup = ctx.state_adapter.backup_recurrent_state(
            state_cache=ctx.state_cache,
            indices=ctx.boundary_indices,
        )
        self.live_backup = ctx.state_adapter.backup_recurrent_state(
            state_cache=ctx.state_cache,
            indices=ctx.live_indices,
        )
        self.boundary_backup_keys = backup_keys

    def restore_for_verify(
        self,
        batch: ScheduleBatch,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
    ) -> Optional[DVRLinearStateContext]:
        replay_tasks = self.ensure_boundary_state(batch, seq_lens_cpu=seq_lens_cpu)
        if replay_tasks:
            raise RuntimeError(
                "DVR boundary replay tasks must be materialized before target verify."
            )
        ctx = self.state_context(batch, require_boundary=True)
        if ctx is None:
            return None
        assert ctx.boundary_indices is not None
        ctx.state_adapter.prepare_recurrent_state_for_verify(
            state_cache=ctx.state_cache,
            live_indices=ctx.live_indices,
            boundary_indices=ctx.boundary_indices,
            boundary_backup=self.boundary_backup,
            live_backup=self.live_backup,
        )
        return ctx

    def commit_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        accepted_token_counts: torch.Tensor,
        accepted_steps: torch.Tensor,
        accepted_token_counts_cpu,
        ctx: DVRLinearStateContext,
        seq_lens_cpu: List[int],
        accepted_suffix_replay=None,
        use_fast_self_draft_commit: bool = False,
    ):
        pending_track_indices = [None] * len(batch.reqs)
        pending_track_seqlens = [None] * len(batch.reqs)
        assert ctx.boundary_indices is not None
        if accepted_token_counts.numel() == 0:
            return pending_track_indices, pending_track_seqlens

        # Use immutable pre-verify lengths. Request metadata can already include
        # accepted tokens when the synchronous result reaches this method.
        verified_tail_lens_cpu = [
            int(seq_len) - self.boundary_seqlen[req.rid]
            for req, seq_len in zip(batch.reqs, seq_lens_cpu, strict=True)
        ]
        verified_tail_lens = ctx.state_input_cache.get_tail_lens(
            indices=ctx.state_input_indices
        )
        verified_tail_lens = verified_tail_lens.to(
            device=ctx.live_indices.device, dtype=torch.long
        )
        ctx.state_adapter.commit_after_verify(
            state_cache=ctx.state_cache,
            state_input_indices=ctx.state_input_indices,
            live_indices=ctx.live_indices,
            boundary_indices=ctx.boundary_indices,
            verified_tail_lens=verified_tail_lens,
            accepted_token_counts=accepted_token_counts,
            accepted_steps=accepted_steps,
            live_state_already_replayed=(
                None
                if accepted_suffix_replay is None
                else accepted_suffix_replay[0]
            ),
            use_fast_self_draft_commit=use_fast_self_draft_commit,
        )
        if accepted_suffix_replay is not None:
            _, replay_tail_lens, replay_state_input_window = accepted_suffix_replay
            ctx.state_input_cache.set_tail_lens(
                indices=ctx.state_input_indices,
                value=replay_tail_lens,
            )
            ctx.state_input_cache.restore_rows(
                indices=ctx.state_input_indices,
                backup=replay_state_input_window,
            )

        for i, (req, verified_tail_len, accepted_token_num) in enumerate(
            zip(
                batch.reqs,
                verified_tail_lens_cpu,
                accepted_token_counts_cpu,
                strict=True,
            )
        ):
            if verified_tail_len + accepted_token_num >= FLA_CHUNK_SIZE:
                new_boundary_seqlen = self.boundary_seqlen[req.rid] + FLA_CHUNK_SIZE
                self.boundary_seqlen[req.rid] = new_boundary_seqlen
                track_idx = self.boundary_track_idx[req.rid]
                # Scheduler materializes accepted tokens after the worker
                # returns. Keep the checkpoint pending until result processing
                # has committed those tokens to Req.output_ids.
                pending_track_indices[i] = track_idx
                pending_track_seqlens[i] = new_boundary_seqlen
        self.boundary_backup = None
        self.boundary_backup_keys = None
        self.live_backup = None
        return pending_track_indices, pending_track_seqlens

    def rollback_live_state_with_accepted_suffix(
        self,
        *,
        batch: ScheduleBatch,
        target_worker,
        linear_state_ctx: DVRLinearStateContext,
        base_seq_lens_cpu: list[int],
        accepted_token_counts_cpu: list[int],
        accepted_ids: torch.Tensor,
        accepted_cache_locs: torch.Tensor,
        num_draft_tokens: int,
        request_token_ids_for_replay,
    ):
        """Replay the actual accepted suffix after a partial verify acceptance."""

        if batch.forward_mode.is_idle():
            return None
        if linear_state_ctx is None:
            raise RuntimeError("DVR accepted-suffix replay requires linear state.")
        if accepted_ids.numel() == 0:
            raise RuntimeError("DVR accepted-suffix replay requires accepted tokens.")

        live_indices = linear_state_ctx.live_indices
        boundary_indices = linear_state_ctx.boundary_indices
        assert boundary_indices is not None

        total_accepted = sum(int(x) for x in accepted_token_counts_cpu)
        if (
            accepted_ids.numel() != total_accepted
            or accepted_cache_locs.numel() != total_accepted
        ):
            raise RuntimeError(
                "DVR accepted-suffix token/cache shape mismatch: "
                f"accepted={total_accepted}, tokens={accepted_ids.numel()}, "
                f"cache_locs={accepted_cache_locs.numel()}."
            )
        if all(
            int(accepted) >= num_draft_tokens
            for accepted in accepted_token_counts_cpu
        ):
            raise RuntimeError(
                "DVR accepted-suffix replay was requested without a rejection."
            )

        accepted_cache_locs = accepted_cache_locs.to(torch.long)
        accepted_tokens_cpu = accepted_ids.detach().cpu().tolist()
        accepted_tokens_by_req = []
        accepted_cache_locs_by_req = []
        token_offset = 0
        for accepted_count in accepted_token_counts_cpu:
            accepted_count = int(accepted_count)
            accepted_tokens_by_req.append(
                accepted_tokens_cpu[token_offset : token_offset + accepted_count]
            )
            accepted_cache_locs_by_req.append(
                accepted_cache_locs[token_offset : token_offset + accepted_count]
            )
            token_offset += accepted_count
        # Accepted-suffix rollback advances target-owned state, not the
        # client-visible output stream.  EAGLE/MTP output tokens are shifted by
        # one verifier row, so callers pass the accepted verify-input path here.
        # This replay publishes the live recurrent state and rolling
        # state-input tail for the next verify.
        with dvr_suffix_replay_context(
            batch=batch,
            linear_state=self,
            linear_state_ctx=linear_state_ctx,
            base_seq_lens_cpu=base_seq_lens_cpu,
            append_tokens_cpu_by_req=accepted_tokens_by_req,
            append_cache_locs_by_req=accepted_cache_locs_by_req,
            request_token_ids_for_replay=request_token_ids_for_replay,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            use_mamba_cow_from_boundary=True,
            restore_live_state=False,
        ) as replay:
            if replay is None:
                raise RuntimeError("Active DVR accepted-suffix replay was skipped.")
            replay_batch, _hidden_gather_indices = replay
            target_worker.forward_batch_generation(batch=replay_batch, is_verify=True)
            accepted_tail_lens = linear_state_ctx.state_input_cache.get_tail_lens(
                indices=linear_state_ctx.state_input_indices
            ).clone()
            accepted_state_input_window = linear_state_ctx.state_input_cache.backup_rows(
                indices=linear_state_ctx.state_input_indices
            )
        return (
            torch.ones(
                len(batch.reqs), dtype=torch.bool, device=live_indices.device
            ),
            accepted_tail_lens,
            accepted_state_input_window,
        )

    def state_context(
        self, batch: ScheduleBatch, require_boundary: bool = False
    ) -> Optional[DVRLinearStateContext]:
        state_adapter = self.state_adapter()
        if state_adapter is None or not state_adapter.has_dvr_state(batch=batch):
            return None
        assert self.server_args.mamba_track_interval == FLA_CHUNK_SIZE, (
            "DVR linear-state target verify must start from FLA chunk boundaries. "
            "The current prefill tracker only guarantees the latest boundary "
            "when mamba_track_interval equals FLA_CHUNK_SIZE."
        )
        live_indices = state_adapter.get_live_indices(batch=batch)
        state_input_indices = state_adapter.get_state_input_indices(
            batch=batch, device=live_indices.device
        )
        state_cache = state_adapter.get_state_cache(batch=batch)
        state_adapter.validate_state_cache(state_cache=state_cache)
        boundary_indices = None
        if require_boundary:
            boundary_indices = self.boundary_indices_for_reqs(
                reqs=batch.reqs,
                track_indices=[
                    self.boundary_track_idx[req.rid] for req in batch.reqs
                ],
                device=live_indices.device,
            )
        return DVRLinearStateContext(
            state_cache=state_cache,
            state_adapter=state_adapter,
            state_input_cache=state_adapter.state_input_window(),
            state_input_indices=state_input_indices,
            live_indices=live_indices,
            boundary_indices=boundary_indices,
        )

    @staticmethod
    def boundary_and_tail_for_seq_len(seq_len: int) -> Tuple[int, int]:
        boundary_seqlen = (seq_len // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
        verified_tail_len = seq_len - boundary_seqlen
        return boundary_seqlen, verified_tail_len

    @staticmethod
    def batch_seq_lens_cpu(batch: ScheduleBatch) -> List[int]:
        if batch.seq_lens_cpu is not None:
            return [int(x) for x in batch.seq_lens_cpu.tolist()]
        return [int(x) for x in batch.seq_lens.detach().cpu().tolist()]

    def state_adapter(self):
        # Scheduler constructs the DVR worker before attention backends are
        # initialized. Treat a missing backend as "not ready" and resolve the
        # adapter lazily when verify/draft state is actually used.
        attn_backend = getattr(self.model_runner, "attn_backend", None)
        linear_backend = getattr(attn_backend, "linear_attn_backend", None)
        if linear_backend is None:
            return None
        return getattr(linear_backend, "dvr_state_adapter", None)

    def set_boundary_checkpoint(
        self,
        batch: ScheduleBatch,
        req,
        track_idx: int,
        boundary_seqlen: int,
    ):
        self.boundary_track_idx[req.rid] = track_idx
        self.boundary_seqlen[req.rid] = boundary_seqlen
        req.mamba_last_track_seqlen = boundary_seqlen
        req.mamba_next_track_idx = batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
            track_idx
        )

    @staticmethod
    def copy_state_indices(
        *, batch: ScheduleBatch, src_indices: torch.Tensor, dst_indices: torch.Tensor
    ):
        batch.req_to_token_pool.mamba_pool.copy_from(
            src_indices.reshape(-1), dst_indices.reshape(-1)
        )

    @staticmethod
    def boundary_indices_for_reqs(*, reqs, track_indices, device) -> torch.Tensor:
        return torch.stack(
            [
                req.mamba_ping_pong_track_buffer[track_idx]
                for req, track_idx in zip(reqs, track_indices, strict=True)
            ]
        ).to(device=device, dtype=torch.long)

    @staticmethod
    def radix_node_state_indices(node) -> Optional[torch.Tensor]:
        return None if node is None else getattr(node, "mamba_value", None)

    @staticmethod
    def radix_node_seqlen(node) -> int:
        seqlen = 0
        while node is not None:
            key = getattr(node, "key", None)
            if key is not None:
                seqlen += len(key)
            node = getattr(node, "parent", None)
        return seqlen

    def find_nearest_radix_state_node(self, *, req, boundary_seqlen: int):
        node = getattr(req, "last_node", None)
        while node is not None:
            node_seqlen = self.radix_node_seqlen(node)
            if (
                self.radix_node_state_indices(node) is not None
                and node_seqlen < boundary_seqlen
                and node_seqlen % FLA_CHUNK_SIZE == 0
            ):
                return node, node_seqlen
            node = getattr(node, "parent", None)
        return None, 0

    def init_boundary_for_req(
        self,
        batch: ScheduleBatch,
        req,
        boundary_seqlen: int,
        live_idx: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[_BoundaryReplayTask]]:
        assert boundary_seqlen % FLA_CHUNK_SIZE == 0
        last_track_seqlen = req.mamba_last_track_seqlen
        if last_track_seqlen is not None and last_track_seqlen > 0:
            assert last_track_seqlen % FLA_CHUNK_SIZE == 0, (
                "DVR linear-state verify must not reuse non-chunk-boundary "
                "checkpoints."
            )
        checkpoint_track_idx = (
            batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                req.mamba_next_track_idx
            )
            if boundary_seqlen > 0 and last_track_seqlen == boundary_seqlen
            else None
        )
        if checkpoint_track_idx is not None:
            # Normal prefill already wrote the chunk-aligned state into the
            # ping-pong checkpoint buffer. DVR verify mutates its boundary
            # slot after every accepted chunk, so copy-on-write the prefill
            # checkpoint into the request's next writable ping-pong slot before
            # registering it as the DVR boundary.
            boundary_track_idx = req.mamba_next_track_idx
            dst = req.mamba_ping_pong_track_buffer[boundary_track_idx]
            src = req.mamba_ping_pong_track_buffer[checkpoint_track_idx]
            self.copy_state_indices(
                batch=batch,
                src_indices=src.unsqueeze(0),
                dst_indices=dst.unsqueeze(0),
            )
            self.set_boundary_checkpoint(
                batch,
                req,
                boundary_track_idx,
                boundary_seqlen,
            )
            return None, None
        boundary_track_idx = req.mamba_next_track_idx
        dst = req.mamba_ping_pong_track_buffer[boundary_track_idx]
        if boundary_seqlen == 0:
            self.set_boundary_checkpoint(
                batch, req, boundary_track_idx, boundary_seqlen
            )
            return dst, None
        exact_state_indices = None
        if not is_dvr_eagle_enabled(self.server_args):
            exact_node = getattr(req, "last_node", None)
            while exact_node is not None:
                if self.radix_node_seqlen(exact_node) == boundary_seqlen:
                    exact_state_indices = self.radix_node_state_indices(exact_node)
                    break
                exact_node = getattr(exact_node, "parent", None)
        if exact_state_indices is not None:
            self.copy_state_indices(
                batch=batch,
                src_indices=exact_state_indices,
                dst_indices=dst,
            )
            self.set_boundary_checkpoint(
                batch, req, boundary_track_idx, boundary_seqlen
            )
            return None, None

        source_node, source_seqlen = self.find_nearest_radix_state_node(
            req=req, boundary_seqlen=boundary_seqlen
        )
        self.set_boundary_checkpoint(
            batch, req, boundary_track_idx, boundary_seqlen
        )
        return None, (
            req,
            source_seqlen,
            boundary_seqlen,
            self.radix_node_state_indices(source_node),
            boundary_track_idx,
            live_idx,
        )

    def ensure_boundary_state(
        self,
        batch: ScheduleBatch,
        ctx: Optional[DVRLinearStateContext] = None,
        *,
        seq_lens_cpu: Optional[List[int]] = None,
    ) -> List[_BoundaryReplayTask]:
        ctx = ctx or self.state_context(batch)
        if ctx is None:
            return []
        replay_tasks = []
        zero_boundary_indices = []
        reset_pos_indices = []
        reset_pos_values = []
        seq_lens_cpu = seq_lens_cpu or self.batch_seq_lens_cpu(batch)
        for i, req in enumerate(batch.reqs):
            if req.rid not in self.boundary_seqlen:
                boundary_seqlen, verified_tail_len = (
                    self.boundary_and_tail_for_seq_len(int(seq_lens_cpu[i]))
                )
                reset_pos_indices.append(ctx.state_input_indices[i])
                reset_pos_values.append(verified_tail_len)
                zero_boundary_idx, replay_task = self.init_boundary_for_req(
                    batch,
                    req,
                    boundary_seqlen,
                    ctx.live_indices[i],
                )
                if zero_boundary_idx is not None:
                    zero_boundary_indices.append(zero_boundary_idx)
                if replay_task is not None:
                    replay_tasks.append(replay_task)
        if zero_boundary_indices:
            boundary_indices_to_zero = torch.stack(zero_boundary_indices).to(
                device=ctx.live_indices.device, dtype=torch.long
            )
            ctx.state_adapter.zero_recurrent_state(
                state_cache=ctx.state_cache, indices=boundary_indices_to_zero
            )
        if reset_pos_indices:
            ctx.state_input_cache.set_tail_lens(
                indices=torch.stack(reset_pos_indices),
                value=torch.tensor(reset_pos_values, device=ctx.live_indices.device),
            )
        return replay_tasks

    def boundary_lens_for_replay(self, batch: ScheduleBatch, seq_lens_cpu) -> List[int]:
        boundary_lens = []
        for req, seq_len in zip(batch.reqs, seq_lens_cpu, strict=True):
            boundary = self.boundary_seqlen.get(req.rid)
            if boundary is None:
                raise RuntimeError(
                    "DVR suffix replay requires a prepared chunk-boundary "
                    f"checkpoint: rid={req.rid}, seq_len={int(seq_len)}."
                )
            if boundary > int(seq_len):
                raise RuntimeError(
                    "DVR suffix replay boundary is ahead of the batch prefix: "
                    f"rid={req.rid}, boundary={boundary}, seq_len={int(seq_len)}."
                )
            boundary_lens.append(int(boundary))
        return boundary_lens
