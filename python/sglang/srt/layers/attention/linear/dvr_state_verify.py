"""Fixed-window DVR verify helpers for gated linear-state layers.

Called by dvr_state_adapter only. The helpers build the chunk commit plan, run
chunkwise verify, export boundary state/conv windows, and rebuild live
recurrent state for the next self-draft decode.
"""

from typing import Optional

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.linear.dvr_gdn_state import DVRGDNStateInputWindow
from sglang.srt.layers.attention.linear.dvr_state import DVRStateOps


def write_dvr_chunk_boundary_state(
    *,
    h: Optional[torch.Tensor],
    intermediate_state_cache: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    batch_size: int,
    verify_window_size: int,
    chunk_size: int = FLA_CHUNK_SIZE,
):
    """Copy the first chunk-boundary state exported by chunkwise scan.

    DVR verifies a fixed physical window of `CHUNK_SIZE + draft_tokens`. Only
    the state at the first `CHUNK_SIZE` boundary can become the next deterministic
    prefill-equivalent checkpoint. The chunkwise kernel returns `h` in two
    layouts depending on whether the input is graph-friendly equal length or
    packed variable length; this helper normalizes both layouts and writes a
    single boundary state into the speculative state cache.
    """

    if h is None or h.shape[1] <= 1:
        return

    chunks_per_req = (verify_window_size - 1) // chunk_size + 1
    if h.shape[0] == batch_size:
        # Equal-length graph-friendly path: DVR passes q/k/v/g/beta as
        # [batch, chunk+draft, ...], so the first chunk boundary is h[:, 1].
        boundary_state = h[:, 1]
    else:
        # Variable-length path packs all request chunks into h[0, total_chunks].
        # DVR's physical window is chunk + draft, so each request contributes
        # two chunks and the first boundary is at offsets 1, 3, 5, ...
        boundary_h_indices = (
            torch.arange(batch_size, dtype=torch.long, device=h.device)
            * chunks_per_req
            + 1
        )
        boundary_state = h.squeeze(0)[boundary_h_indices]
    # DVR only commits the first chunk-boundary state. Newer DVR cache layouts
    # store that state in a one-slot boundary buffer; the fallback keeps
    # compatibility with the older per-token speculative state cache.
    boundary_slot = 0 if intermediate_state_cache.shape[1] == 1 else chunk_size - 1
    intermediate_state_cache[
        intermediate_state_indices[:batch_size].to(torch.long),
        boundary_slot,
    ] = boundary_state.to(intermediate_state_cache.dtype)


def write_dvr_conv_windows(
    *,
    intermediate_conv_window_cache: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    initial_conv_windows: torch.Tensor,
    mixed_qkv_reshaped: torch.Tensor,
    verify_window_size: int,
):
    """Export per-token conv windows for later live-state commits.

    The GDN short convolution is part of the recurrent state lifecycle. During
    DVR verify we rebuild conv windows from the verified input sequence:
    previous conv state + current verify-window inputs. The resulting windows
    are indexed by accepted draft step, matching the normal EAGLE/mamba state
    scatter path.
    """

    conv_source = torch.cat([initial_conv_windows, mixed_qkv_reshaped], dim=2)
    state_len = initial_conv_windows.shape[-1]
    conv_windows = conv_source.unfold(
        dimension=2, size=state_len, step=1
    )[:, :, 1 : verify_window_size + 1].transpose(1, 2)
    rows = (
        intermediate_state_indices[: initial_conv_windows.shape[0]]
        .to(torch.long)
        .unsqueeze(1)
        .expand(-1, verify_window_size)
    )
    cols = torch.arange(
        verify_window_size,
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
    num_v_heads: int,
    head_v_dim: int,
) -> torch.Tensor:
    """Select the newly drafted suffix from a physical chunk+draft output.

    The chunkwise scan computes outputs for the whole physical window. The
    verifier should only consume logits for the draft tokens, whose columns are
    offset by the already-verified tail length of each request.
    """

    core_attn_out = core_attn_out.view(
        batch_size, verify_window_size, num_v_heads, head_v_dim
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
        1, batch_size * draft_token_num, num_v_heads, head_v_dim
    ).contiguous()


def run_dvr_chunkwise_verify(
    *,
    state_ops: DVRStateOps,
    state_window: DVRGDNStateInputWindow,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_state_cache: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    num_q_heads: int,
    head_q_dim: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    chunk_size: int = FLA_CHUNK_SIZE,
    tail_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run DVR's fixed chunk+draft linear-state verify window.

    The GDN backend passes only the draft-token q/k/v/g/beta rows produced by
    this forward. DVR first appends those rows to the rolling window, then runs
    chunkwise scan over the full `CHUNK_SIZE + draft_tokens` physical window so
    the verify path uses the same scan semantics as deterministic prefill. The
    function caches the chunk-boundary state for post-accept commits and returns
    only the draft suffix, preserving the ordinary target-verify logits shape.
    """

    indices = cache_indices[:batch_size].to(torch.long)
    if tail_lens is None:
        tail_lens = state_window.tail_lens(indices=indices).to(torch.long)
    else:
        tail_lens = tail_lens[:batch_size].to(
            device=indices.device, dtype=torch.long
        )
    tail_lens = tail_lens.clamp(min=0, max=chunk_size)
    state_window.write_draft_rows(
        indices=indices,
        col_start=tail_lens,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        batch_size=batch_size,
        draft_token_num=draft_token_num,
        num_q_heads=num_q_heads,
        head_q_dim=head_q_dim,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
    )
    verify_window_size = chunk_size + draft_token_num
    q_window, k_window, v_window, g_window, beta_window = state_window.read_window(
        indices=indices
    )
    core_attn_out, _, h = state_ops.scan_chunkwise(
        q=q_window,
        k=k_window,
        v=v_window,
        g=g_window,
        beta=beta_window,
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
        verify_window_size=verify_window_size,
        chunk_size=chunk_size,
    )
    return select_dvr_draft_suffix(
        core_attn_out,
        tail_lens=tail_lens,
        batch_size=batch_size,
        verify_window_size=verify_window_size,
        draft_token_num=draft_token_num,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
    )


def rebuild_dvr_live_state_grouped(
    *,
    state_ops: DVRStateOps,
    state_window: DVRGDNStateInputWindow,
    temporal_state: torch.Tensor,
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

    state_live_indices = live_indices[req_indices]
    state_boundary_indices = boundary_indices[req_indices]
    q_cache, k_cache, v_cache, g_cache, beta_cache = state_window.tensors()
    num_layers = temporal_state.shape[0]
    num_reqs = state_live_indices.numel()
    token_count = token_count.to(device=temporal_state.device, dtype=torch.long)

    q = q_cache[:, state_live_indices]
    k = k_cache[:, state_live_indices]
    v = v_cache[:, state_live_indices]
    g = g_cache[:, state_live_indices]
    beta = beta_cache[:, state_live_indices]
    initial_state = temporal_state[:, state_boundary_indices]

    flat_shape = (num_layers * num_reqs,)
    q = q.reshape(*flat_shape, *q.shape[2:])
    k = k.reshape(*flat_shape, *k.shape[2:])
    v = v.reshape(*flat_shape, *v.shape[2:])
    g = g.reshape(*flat_shape, *g.shape[2:])
    beta = beta.reshape(*flat_shape, *beta.shape[2:])
    initial_state = initial_state.reshape(*flat_shape, *initial_state.shape[2:])
    flat_token_count = (
        token_count.unsqueeze(0).expand(num_layers, -1).reshape(-1).contiguous()
    )

    rebuilt_state = state_ops.rebuild_recurrent_state(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        token_count=flat_token_count,
    )

    temporal_state[:, state_live_indices] = rebuilt_state.reshape(
        num_layers, num_reqs, *rebuilt_state.shape[1:]
    )
