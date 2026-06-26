"""Fixed-window DVR verify helpers for gated linear-state layers.

Called by dvr_state_adapter only. The helpers build the chunk commit plan, run
chunkwise verify, export boundary state/conv windows, and rebuild live
recurrent state for the next self-draft decode.
"""

import os
from typing import Optional

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.linear.dvr_state import (
    DVRStateInputs,
    DVRStateInputWindow,
    DVRStateOps,
)


def write_dvr_chunk_boundary_state(
    *,
    h: Optional[torch.Tensor],
    intermediate_state_cache: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    batch_size: int,
):
    """Copy the first chunk-boundary state exported by chunkwise scan.

    DVR verifies a fixed physical window of `CHUNK_SIZE + draft_tokens`. Only
    the state at the first `CHUNK_SIZE` boundary can become the next deterministic
    prefill-equivalent checkpoint. DVR always calls the chunkwise kernel with an
    equal-length [batch, chunk+draft, ...] window, so the boundary state is h[:, 1].
    """

    if h is None or h.shape[1] <= 1:
        return

    boundary_state = h[:batch_size, 1]
    # DVR only commits the first chunk-boundary state, stored in the one-slot
    # boundary buffer allocated for DVR speculative state.
    intermediate_state_cache[
        intermediate_state_indices[:batch_size].to(torch.long),
        0,
    ] = boundary_state.to(intermediate_state_cache.dtype)


def write_dvr_conv_windows(
    *,
    intermediate_conv_window_cache: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    initial_conv_windows: torch.Tensor,
    conv_input_reshaped: torch.Tensor,
    num_draft_tokens: int,
):
    """Export per-token conv windows for later live-state commits.

    The GDN short convolution is part of the recurrent state lifecycle. During
    DVR verify we rebuild conv windows from the verified input sequence:
    previous conv state + current verify-window inputs. The resulting windows
    are indexed by accepted draft step, matching the normal EAGLE/mamba state
    scatter path.
    """

    conv_source = torch.cat([initial_conv_windows, conv_input_reshaped], dim=2)
    state_len = initial_conv_windows.shape[-1]
    conv_windows = conv_source.unfold(
        dimension=2, size=state_len, step=1
    )[:, :, 1 : num_draft_tokens + 1].transpose(1, 2)
    rows = (
        intermediate_state_indices[: initial_conv_windows.shape[0]]
        .to(torch.long)
        .unsqueeze(1)
        .expand(-1, num_draft_tokens)
    )
    cols = torch.arange(
        num_draft_tokens,
        dtype=torch.long,
        device=intermediate_state_indices.device,
    ).unsqueeze(0)
    intermediate_conv_window_cache[rows, cols] = conv_windows


def select_dvr_draft_suffix(
    core_attn_out: torch.Tensor,
    *,
    tail_lens: torch.Tensor,
    batch_size: int,
    verify_window_size: int,
    draft_token_num: int,
) -> torch.Tensor:
    """Select the newly drafted suffix from a physical chunk+draft output.

    The chunkwise scan computes outputs for the whole physical window. The
    verifier should only consume logits for the draft tokens, whose columns are
    offset by the already-verified tail length of each request.
    """

    value_shape = core_attn_out.shape[-2:]
    core_attn_out = core_attn_out.view(
        batch_size, verify_window_size, *value_shape
    )
    rows = (
        torch.arange(batch_size, dtype=torch.long, device=core_attn_out.device)
        .unsqueeze(1)
        .expand(-1, draft_token_num)
    )
    cols = (
        torch.arange(draft_token_num, dtype=torch.long, device=core_attn_out.device)
        .unsqueeze(0)
        .add(tail_lens.unsqueeze(1))
    )
    return core_attn_out[rows, cols].reshape(
        1, batch_size * draft_token_num, *value_shape
    ).contiguous()


def _debug_pack_dvr_verify_window(
    state_inputs: DVRStateInputs,
    *,
    tail_lens: torch.Tensor,
    draft_token_num: int,
) -> tuple[DVRStateInputs, torch.Tensor]:
    lens = tail_lens + draft_token_num
    pieces = []
    for tensor in state_inputs.tensors():
        rows = [tensor[i, : int(lens[i].item())] for i in range(tensor.shape[0])]
        pieces.append(torch.cat(rows, dim=0).unsqueeze(0))
    cu_seqlens = torch.zeros(
        lens.numel() + 1, dtype=torch.long, device=tail_lens.device
    )
    cu_seqlens[1:] = torch.cumsum(lens, dim=0)
    return type(state_inputs).from_tensors(tuple(pieces)), cu_seqlens


def run_dvr_chunkwise_verify(
    *,
    state_ops: DVRStateOps,
    state_window: DVRStateInputWindow,
    draft_state_inputs: DVRStateInputs,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    state_input_indices: torch.Tensor,
    intermediate_state_cache: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    chunk_size: int = FLA_CHUNK_SIZE,
    tail_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run DVR's fixed chunk+draft linear-state verify window.

    The backend passes only the new draft-token state inputs produced by this
    forward. DVR first appends those rows to the rolling window, then runs
    chunkwise scan over the full `CHUNK_SIZE + draft_tokens` physical window so
    the verify path uses the same scan semantics as deterministic prefill. The
    function caches the chunk-boundary state for post-accept commits and returns
    only the draft suffix, preserving the ordinary target-verify logits shape.
    """

    indices = cache_indices[:batch_size].to(torch.long)
    input_indices = state_input_indices[:batch_size].to(torch.long)
    if tail_lens is None:
        tail_lens = state_window.tail_lens(indices=input_indices).to(torch.long)
    else:
        tail_lens = tail_lens[:batch_size].to(device=indices.device, dtype=torch.long)
    tail_lens = tail_lens.clamp(min=0, max=chunk_size)
    draft_state_inputs.write_draft_rows(
        state_window,
        indices=input_indices,
        col_start=tail_lens,
        draft_token_num=draft_token_num,
    )
    keep_lens = tail_lens + draft_token_num
    if os.environ.get("SGLANG_DVR_DEBUG_ZERO_FUTURE_DRAFTS") == "1":
        keep_lens = tail_lens + 1
    state_window.zero_after_lens(
        indices=input_indices,
        keep_lens=keep_lens,
    )
    verify_window_size = chunk_size + draft_token_num
    window_inputs = state_window.read_window(indices=input_indices)
    if os.environ.get("SGLANG_DVR_DEBUG_VARLEN_VERIFY") == "1":
        window_inputs, packed_query_start_loc = _debug_pack_dvr_verify_window(
            window_inputs,
            tail_lens=tail_lens,
            draft_token_num=draft_token_num,
        )
        core_attn_out, _, h = state_ops.scan_chunkwise(
            state_inputs=window_inputs,
            ssm_states=ssm_states,
            cache_indices=indices,
            query_start_loc=packed_query_start_loc,
        )
        value_shape = core_attn_out.shape[-2:]
        cols = tail_lens.to(torch.long)
        rows = torch.arange(
            batch_size, dtype=torch.long, device=core_attn_out.device
        )
        starts = packed_query_start_loc[:-1]
        gather_cols = (
            starts.unsqueeze(1)
            + cols.unsqueeze(1)
            + torch.arange(
                draft_token_num, dtype=torch.long, device=core_attn_out.device
            ).unsqueeze(0)
        )
        packed_suffix = core_attn_out[0, gather_cols.reshape(-1)]
        return packed_suffix.reshape(
            1, batch_size * draft_token_num, *value_shape
        ).contiguous()
    core_attn_out, _, h = state_ops.scan_chunkwise(
        state_inputs=window_inputs,
        ssm_states=ssm_states,
        cache_indices=indices,
        # Equal-length [bs, chunk+draft, ...] avoids variable-length chunk
        # preparation, which can synchronize CPU/GPU and is not graph-safe.
        query_start_loc=None,
    )
    write_dvr_chunk_boundary_state(
        h=h,
        intermediate_state_cache=intermediate_state_cache,
        intermediate_state_indices=intermediate_state_indices,
        batch_size=batch_size,
    )
    return select_dvr_draft_suffix(
        core_attn_out,
        tail_lens=tail_lens,
        batch_size=batch_size,
        verify_window_size=verify_window_size,
        draft_token_num=draft_token_num,
    )


def rebuild_dvr_live_state_grouped(
    *,
    state_ops: DVRStateOps,
    state_window: DVRStateInputWindow,
    temporal_state: torch.Tensor,
    state_input_indices: torch.Tensor,
    live_indices: torch.Tensor,
    boundary_indices: torch.Tensor,
    req_indices: torch.Tensor,
    token_count: torch.Tensor,
):
    """Rebuild DVR live recurrent state from chunk-boundary checkpoints.

    The live state is consumed by the next self-draft decode, so it should use
    the recurrent semantics of normal decode. Grouping flattens
    layers x requests and calls the recurrent state kernel once per distinct
    accepted length instead of once per layer/request.
    """

    if req_indices.numel() == 0:
        return

    selected_state_input_indices = state_input_indices[req_indices]
    state_live_indices = live_indices[req_indices]
    state_boundary_indices = boundary_indices[req_indices]
    window_inputs = state_window.read_all_layers_window(
        indices=selected_state_input_indices
    )
    num_layers = temporal_state.shape[0]
    num_reqs = state_live_indices.numel()
    token_count = token_count.to(device=temporal_state.device, dtype=torch.long)

    initial_state = temporal_state[:, state_boundary_indices]

    flat_dim = num_layers * num_reqs
    window_inputs = window_inputs.flatten_leading_dims(flat_dim, keep_from_dim=2)
    initial_state = initial_state.reshape(flat_dim, *initial_state.shape[2:])
    flat_token_count = (
        token_count.unsqueeze(0).expand(num_layers, -1).reshape(-1).contiguous()
    )

    rebuilt_state = state_ops.rebuild_recurrent_state(
        window_inputs,
        initial_state=initial_state,
        token_count=flat_token_count,
    )

    temporal_state[:, state_live_indices] = rebuilt_state.reshape(
        num_layers, num_reqs, *rebuilt_state.shape[1:]
    )
