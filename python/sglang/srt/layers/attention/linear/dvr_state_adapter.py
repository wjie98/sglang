from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.linear.dvr_state_ops import DVRStateOps


@dataclass(frozen=True)
class DVRStateCommitPlan:
    pos_before: torch.Tensor
    pos_after: torch.Tensor
    crossing: torch.Tensor
    accepted_window_steps: torch.Tensor
    boundary_conv_steps: torch.Tensor


@dataclass(frozen=True)
class DVRGatedForwardContext:
    """Layer-local DVR state context for one gated linear-state forward."""

    layer: Any
    forward_batch: Any
    state_cache: Any
    cache_indices: torch.Tensor
    query_start_loc: Optional[torch.Tensor]
    conv_states: torch.Tensor
    ssm_states: torch.Tensor
    seq_len: int
    is_target_verify: bool

    @property
    def spec_info(self):
        return self.forward_batch.spec_info


@dataclass(frozen=True)
class DVRRecurrentStateBackup:
    conv: Tuple[torch.Tensor, ...]
    temporal: torch.Tensor
    indices: torch.Tensor


@dataclass(frozen=True)
class DVRStateInputWindow:
    q: Optional[torch.Tensor]
    k: Optional[torch.Tensor]
    v: Optional[torch.Tensor]
    g: Optional[torch.Tensor]
    beta: Optional[torch.Tensor]
    pos: Optional[torch.Tensor]

    @classmethod
    def from_cache(cls, state_cache):
        inputs = getattr(state_cache, "dvr_state_inputs", None)
        tail_lens = getattr(state_cache, "dvr_state_input_tail_lens", None)
        if inputs is None or tail_lens is None:
            return cls(q=None, k=None, v=None, g=None, beta=None, pos=None)
        q, k, v, g, beta = inputs
        return cls(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            pos=tail_lens,
        )

    @property
    def enabled(self) -> bool:
        return self.q is not None

    def tensors(self) -> Tuple[torch.Tensor, ...]:
        assert self.enabled
        assert self.q is not None
        assert self.k is not None
        assert self.v is not None
        assert self.g is not None
        assert self.beta is not None
        return self.q, self.k, self.v, self.g, self.beta

    @property
    def capacity(self) -> int:
        assert self.enabled
        assert self.q is not None
        # Full-cache layout is [layers, slots, window, ...]. Layer-local cache
        # layout is [slots, window, ...].
        return self.q.shape[2] if self.q.dim() >= 5 else self.q.shape[1]

    def tail_lens(self, *, indices: torch.Tensor) -> torch.Tensor:
        assert self.pos is not None
        indices = indices.to(device=self.pos.device, dtype=torch.long)
        if self.pos.dim() == 2:
            return self.pos[0, indices]
        return self.pos[indices]

    def set_tail_lens(self, *, indices: torch.Tensor, value: Union[int, torch.Tensor]):
        assert self.pos is not None
        indices = indices.to(device=self.pos.device, dtype=torch.long)
        value = torch.as_tensor(value, device=self.pos.device, dtype=self.pos.dtype)
        if self.pos.dim() == 2:
            self.pos[:, indices] = value
        else:
            self.pos[indices] = value

    def write_tail(
        self,
        *,
        dst: torch.Tensor,
        cols: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ):
        q_cache, k_cache, v_cache, g_cache, beta_cache = self.tensors()
        q_cache[dst, cols] = q
        k_cache[dst, cols] = k
        v_cache[dst, cols] = v
        g_cache[dst, cols] = g
        beta_cache[dst, cols] = beta

    def write_draft_rows(
        self,
        *,
        indices: torch.Tensor,
        col_start: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        batch_size: int,
        draft_token_num: int,
        num_q_heads: int,
        head_q_dim: int,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
    ):
        cols = (
            torch.arange(draft_token_num, dtype=torch.long, device=indices.device)
            .unsqueeze(0)
            .add(col_start.to(torch.long).unsqueeze(1))
        )
        rows = indices.unsqueeze(1).expand(-1, draft_token_num)
        q_cache, k_cache, v_cache, g_cache, beta_cache = self.tensors()
        q_cache[rows, cols] = q.reshape(
            batch_size, draft_token_num, num_q_heads, head_q_dim
        )
        k_cache[rows, cols] = k.reshape(
            batch_size, draft_token_num, num_k_heads, head_k_dim
        )
        v_cache[rows, cols] = v.reshape(
            batch_size, draft_token_num, num_v_heads, head_v_dim
        )
        g_cache[rows, cols] = g.reshape(batch_size, draft_token_num, num_v_heads)
        beta_cache[rows, cols] = beta.reshape(
            batch_size, draft_token_num, num_v_heads
        )

    def read_window(self, *, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        q_cache, k_cache, v_cache, g_cache, beta_cache = self.tensors()
        return (
            q_cache[indices],
            k_cache[indices],
            v_cache[indices],
            g_cache[indices],
            beta_cache[indices],
        )

    def shift_suffix(self, *, slots: torch.Tensor, start: int, length: int):
        if length <= 0 or slots.numel() == 0:
            return
        has_layer_dim = self.pos is not None and self.pos.dim() == 2
        for cache in self.tensors():
            if has_layer_dim:
                cache[:, slots, :length] = cache[:, slots, start : start + length].clone()
            else:
                cache[slots, :length] = cache[slots, start : start + length].clone()

    def shift_after_boundary(
        self,
        *,
        live_indices: torch.Tensor,
        crossing: torch.Tensor,
        new_pos: torch.Tensor,
        chunk_size: int = FLA_CHUNK_SIZE,
    ):
        crossing_idx = crossing.nonzero(as_tuple=True)[0]
        crossing_slots = live_indices[crossing_idx]
        if crossing_slots.numel() == 0:
            return

        tail_capacity = self.capacity - chunk_size
        tail_len = min(int(new_pos[crossing_idx].max().item()), tail_capacity)
        if tail_len > 0:
            self.shift_suffix(
                slots=crossing_slots,
                start=chunk_size,
                length=tail_len,
            )

    def write_extend_tail(
        self,
        *,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        extend_prefix_lens_cpu,
        extend_seq_lens_cpu,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = FLA_CHUNK_SIZE,
    ):
        for req_i, (prefix_len, extend_len) in enumerate(
            zip(extend_prefix_lens_cpu, extend_seq_lens_cpu, strict=True)
        ):
            seq_len = prefix_len + extend_len
            boundary = (seq_len // chunk_size) * chunk_size
            num_verified_tokens = seq_len - boundary
            dst = cache_indices[req_i].to(torch.long)
            self.set_tail_lens(indices=dst, value=num_verified_tokens)
            if num_verified_tokens == 0:
                continue

            write_start = max(prefix_len, boundary)
            write_end = seq_len
            if write_start >= write_end:
                continue

            src_start = int(query_start_loc[req_i].item()) + (write_start - prefix_len)
            src_end = src_start + (write_end - write_start)
            cols = torch.arange(
                write_start - boundary,
                write_end - boundary,
                dtype=torch.long,
                device=cache_indices.device,
            )
            self.write_tail(
                dst=dst,
                cols=cols,
                q=q[src_start:src_end],
                k=k[src_start:src_end],
                v=v[src_start:src_end],
                g=g[src_start:src_end],
                beta=beta[src_start:src_end],
            )


def has_dvr_state_window(state_cache) -> bool:
    """Return whether a cache exposes DVR's rolling state-input window."""

    return DVRStateInputWindow.from_cache(state_cache).enabled


def check_dvr_state_input_position(
    *,
    pos_before: torch.Tensor,
    pos_after: torch.Tensor,
    accepted_tokens: torch.Tensor,
    window_capacity: int,
):
    if (
        torch.any(pos_before < 0).item()
        or torch.any(pos_before >= FLA_CHUNK_SIZE).item()
        or torch.any(pos_after > window_capacity).item()
    ):
        raise RuntimeError(
            "Invalid DVR GDN state-input tail position: "
            f"pos_before={pos_before.tolist()}, "
            f"accepted_tokens={accepted_tokens.tolist()}, "
            f"capacity={window_capacity}, chunk_size={FLA_CHUNK_SIZE}."
        )


def check_dvr_state_steps(
    *, accepted_window_steps: torch.Tensor, window_capacity: int
):
    if torch.any(accepted_window_steps >= window_capacity).item():
        raise RuntimeError(
            "Invalid DVR GDN accepted state step: "
            f"steps={accepted_window_steps.tolist()}, "
            f"capacity={window_capacity}."
        )


def check_dvr_conv_steps(
    *,
    accepted_steps: torch.Tensor,
    boundary_steps: torch.Tensor,
    crossing: torch.Tensor,
    conv_capacity: int,
):
    if torch.any(accepted_steps >= conv_capacity).item():
        raise RuntimeError(
            "Invalid DVR GDN accepted conv step: "
            f"steps={accepted_steps.tolist()}, capacity={conv_capacity}."
        )
    if crossing.any():
        crossing_boundary_steps = boundary_steps[crossing]
        if (
            torch.any(crossing_boundary_steps < 0).item()
            or torch.any(crossing_boundary_steps >= conv_capacity).item()
        ):
            raise RuntimeError(
                "Invalid DVR GDN boundary conv step: "
                f"steps={crossing_boundary_steps.tolist()}, "
                f"capacity={conv_capacity}."
            )


def build_dvr_state_commit_plan(
    *,
    verified_tail_lens: torch.Tensor,
    accepted_tokens: torch.Tensor,
    accepted_steps: torch.Tensor,
    window_capacity: int,
    conv_capacity: int,
    device: torch.device,
    chunk_size: int = FLA_CHUNK_SIZE,
) -> DVRStateCommitPlan:
    pos_before = verified_tail_lens.to(device=device, dtype=torch.long)
    pos_after = pos_before + accepted_tokens
    check_dvr_state_input_position(
        pos_before=pos_before,
        pos_after=pos_after,
        accepted_tokens=accepted_tokens,
        window_capacity=window_capacity,
    )
    crossing = pos_after >= chunk_size
    accepted_window_steps = pos_before + accepted_steps
    check_dvr_state_steps(
        accepted_window_steps=accepted_window_steps,
        window_capacity=window_capacity,
    )
    boundary_conv_steps = chunk_size - 1 - pos_before
    check_dvr_conv_steps(
        accepted_steps=accepted_steps,
        boundary_steps=boundary_conv_steps,
        crossing=crossing,
        conv_capacity=conv_capacity,
    )
    return DVRStateCommitPlan(
        pos_before=pos_before,
        pos_after=pos_after,
        crossing=crossing,
        accepted_window_steps=accepted_window_steps,
        boundary_conv_steps=boundary_conv_steps,
    )


def write_dvr_chunk_boundary_state(
    *,
    h: Optional[torch.Tensor],
    intermediate_state_cache: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    batch_size: int,
    verify_window_size: int,
    chunk_size: int = FLA_CHUNK_SIZE,
):
    if h is None or h.shape[1] <= 1:
        return

    chunks_per_req = (verify_window_size - 1) // chunk_size + 1
    if h.shape[0] == batch_size:
        # Equal-length graph-friendly path: GDN DVR passes q/k/v/g/beta as
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


def build_dvr_conv_windows(
    *,
    initial_conv_windows: torch.Tensor,
    mixed_qkv_reshaped: torch.Tensor,
    verify_window_size: int,
) -> torch.Tensor:
    conv_source = torch.cat([initial_conv_windows, mixed_qkv_reshaped], dim=2)
    state_len = initial_conv_windows.shape[-1]
    return conv_source.unfold(
        dimension=2, size=state_len, step=1
    )[:, :, 1 : verify_window_size + 1].transpose(1, 2)


def write_dvr_conv_windows(
    *,
    intermediate_conv_window_cache: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    initial_conv_windows: torch.Tensor,
    mixed_qkv_reshaped: torch.Tensor,
    verify_window_size: int,
):
    conv_windows = build_dvr_conv_windows(
        initial_conv_windows=initial_conv_windows,
        mixed_qkv_reshaped=mixed_qkv_reshaped,
        verify_window_size=verify_window_size,
    )
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
    state_window: DVRStateInputWindow,
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
) -> torch.Tensor:
    """Run DVR's fixed chunk+draft linear-state verify window.

    Backends provide state-input tensors and kernels; the adapter owns the DVR
    rolling-window mechanics and only returns the draft suffix consumed by the
    normal target-verify path.
    """

    indices = cache_indices[:batch_size].to(torch.long)
    tail_lens = state_window.tail_lens(indices=indices).to(torch.long)
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
    state_window: DVRStateInputWindow,
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

    rebuilt_state = initial_state.clone()
    for count in torch.unique(flat_token_count).tolist():
        count = int(count)
        if count == 0:
            continue
        group = (flat_token_count == count).nonzero(as_tuple=True)[0]
        rebuilt_state[group] = state_ops.rebuild_recurrent_state(
            q[group, :count],
            k[group, :count],
            v[group, :count],
            g[group, :count],
            beta[group, :count],
            initial_state=initial_state[group],
            token_count=count,
        )

    temporal_state[:, state_live_indices] = rebuilt_state.reshape(
        num_layers, num_reqs, *rebuilt_state.shape[1:]
    )


@dataclass
class DVRGatedStateAdapter:
    """Adapter for DVR state replay in gated linear-state layers.

    The backend still produces layer-specific tensors such as q/k/v/g/beta and
    conv windows. This adapter owns the DVR rolling-window and commit mechanics
    so a model backend can opt in through a small set of calls.
    """

    ops: DVRStateOps
    chunk_size: int = FLA_CHUNK_SIZE

    def has_window(self, state_cache) -> bool:
        return has_dvr_state_window(state_cache)

    def is_verify_enabled(self, *, state_cache, is_target_verify: bool) -> bool:
        return is_target_verify and self.has_window(state_cache)

    def state_input_tail_lens(
        self, *, state_cache, live_indices: torch.Tensor
    ) -> Optional[torch.Tensor]:
        state_window = DVRStateInputWindow.from_cache(state_cache)
        if not state_window.enabled:
            return None
        return state_window.tail_lens(indices=live_indices)

    def set_state_input_tail_lens(
        self,
        *,
        state_cache,
        live_indices: torch.Tensor,
        tail_lens: torch.Tensor,
    ):
        state_window = DVRStateInputWindow.from_cache(state_cache)
        if not state_window.enabled:
            return
        state_window.set_tail_lens(indices=live_indices, value=tail_lens)

    def backup_recurrent_state(
        self, *, state_cache, indices: torch.Tensor
    ) -> DVRRecurrentStateBackup:
        indices = indices.to(device=state_cache.temporal.device, dtype=torch.long)
        return DVRRecurrentStateBackup(
            conv=tuple(conv[:, indices].clone() for conv in state_cache.conv),
            temporal=state_cache.temporal[:, indices].clone(),
            indices=indices.clone(),
        )

    def restore_recurrent_state(
        self,
        *,
        state_cache,
        backup: DVRRecurrentStateBackup,
        indices: Optional[torch.Tensor] = None,
    ):
        dst_indices = backup.indices if indices is None else indices
        dst_indices = dst_indices.to(device=state_cache.temporal.device, dtype=torch.long)
        for conv, saved_conv in zip(state_cache.conv, backup.conv, strict=True):
            conv[:, dst_indices] = saved_conv.to(conv.dtype, copy=False)
        state_cache.temporal[:, dst_indices] = backup.temporal.to(
            state_cache.temporal.dtype, copy=False
        )

    def make_forward_context(
        self,
        *,
        layer,
        forward_batch,
        state_cache,
        cache_indices: torch.Tensor,
        query_start_loc: Optional[torch.Tensor],
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        seq_len: int,
    ) -> DVRGatedForwardContext:
        return DVRGatedForwardContext(
            layer=layer,
            forward_batch=forward_batch,
            state_cache=state_cache,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            conv_states=conv_states,
            ssm_states=ssm_states,
            seq_len=seq_len,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
        )

    @staticmethod
    def verify_shape(*, seq_len: int, spec_info) -> Tuple[int, int]:
        draft_token_num = spec_info.draft_token_num
        return seq_len // draft_token_num, draft_token_num

    def target_verify_shape(
        self, context: DVRGatedForwardContext
    ) -> Tuple[int, int]:
        return self.verify_shape(
            seq_len=context.seq_len, spec_info=context.spec_info
        )

    def target_verify_has_initial_states(
        self, context: DVRGatedForwardContext
    ) -> torch.Tensor:
        batch_size, _ = self.target_verify_shape(context)
        forward_batch = context.forward_batch
        return (forward_batch.seq_lens[:batch_size] > 0).to(
            dtype=torch.bool,
            device=forward_batch.input_ids.device,
        )

    def cache_extend_state_inputs(
        self,
        *,
        context: DVRGatedForwardContext,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ):
        state_window = DVRStateInputWindow.from_cache(context.state_cache)
        if not state_window.enabled:
            return
        forward_batch = context.forward_batch
        if (
            forward_batch.extend_prefix_lens_cpu is None
            or forward_batch.extend_seq_lens_cpu is None
            or context.query_start_loc is None
        ):
            return

        q = q.reshape(q.shape[1], q.shape[2], q.shape[3])
        k = k.reshape(k.shape[1], k.shape[2], k.shape[3])
        v = v.reshape(v.shape[1], v.shape[2], v.shape[3])
        g = g.reshape(-1, g.shape[-1])
        beta = beta.reshape(-1, beta.shape[-1])

        state_window.write_extend_tail(
            cache_indices=context.cache_indices,
            query_start_loc=context.query_start_loc,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            chunk_size=self.chunk_size,
        )

    def cache_extend_state_inputs_from_forward(
        self,
        *,
        layer,
        forward_batch,
        state_cache,
        cache_indices: torch.Tensor,
        query_start_loc: Optional[torch.Tensor],
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        seq_len: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ):
        self.cache_extend_state_inputs(
            context=self.make_forward_context(
                layer=layer,
                forward_batch=forward_batch,
                state_cache=state_cache,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                conv_states=conv_states,
                ssm_states=ssm_states,
                seq_len=seq_len,
            ),
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
        )

    def process_target_verify_conv(
        self,
        *,
        context: DVRGatedForwardContext,
        mixed_qkv: torch.Tensor,
    ) -> torch.Tensor:
        """Run DVR draft conv and export absolute-offset conv windows."""

        assert self.is_verify_enabled(
            state_cache=context.state_cache, is_target_verify=context.is_target_verify
        )

        batch_size, draft_token_num = self.target_verify_shape(context)
        has_initial_states = self.target_verify_has_initial_states(context)
        mixed_qkv_linear = mixed_qkv
        mixed_qkv_reshaped = mixed_qkv_linear.view(
            batch_size, draft_token_num, -1
        ).transpose(1, 2)
        dvr_indices = context.cache_indices[:batch_size].to(torch.long)
        initial_conv_windows = context.conv_states[dvr_indices].clone()
        mixed_qkv = self.ops.run_verify_conv(
            mixed_qkv_linear.transpose(0, 1),
            context.layer.conv_weights,
            context.layer.bias,
            activation=context.layer.activation,
            conv_states=context.conv_states,
            has_initial_state=has_initial_states,
            cache_indices=dvr_indices,
            query_start_loc=context.query_start_loc,
            seq_lens_cpu=[draft_token_num] * batch_size,
        ).transpose(0, 1)[: mixed_qkv.shape[0]]

        write_dvr_conv_windows(
            intermediate_conv_window_cache=context.state_cache.intermediate_conv_window[
                0
            ],
            intermediate_state_indices=torch.arange(
                context.cache_indices.shape[0],
                dtype=torch.int32,
                device=context.cache_indices.device,
            ),
            initial_conv_windows=initial_conv_windows,
            mixed_qkv_reshaped=mixed_qkv_reshaped,
            verify_window_size=draft_token_num,
        )
        return mixed_qkv

    def process_target_verify_state(
        self,
        *,
        context: DVRGatedForwardContext,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        assert self.is_verify_enabled(
            state_cache=context.state_cache, is_target_verify=context.is_target_verify
        )

        batch_size, draft_token_num = self.target_verify_shape(context)
        return run_dvr_chunkwise_verify(
            state_ops=self.ops,
            state_window=DVRStateInputWindow.from_cache(context.state_cache),
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=context.ssm_states,
            cache_indices=context.cache_indices,
            intermediate_state_cache=context.state_cache.intermediate_ssm,
            intermediate_state_indices=torch.arange(
                context.cache_indices.shape[0],
                dtype=torch.int32,
                device=context.cache_indices.device,
            ),
            batch_size=batch_size,
            draft_token_num=draft_token_num,
            num_q_heads=context.layer.num_q_heads,
            head_q_dim=context.layer.head_q_dim,
            num_k_heads=context.layer.num_k_heads,
            head_k_dim=context.layer.head_k_dim,
            num_v_heads=context.layer.num_v_heads,
            head_v_dim=context.layer.head_v_dim,
            chunk_size=self.chunk_size,
        )

    def commit_after_verify(
        self,
        *,
        state_cache,
        live_indices: torch.Tensor,
        boundary_indices: torch.Tensor,
        verified_tail_lens: torch.Tensor,
        accepted_tokens: torch.Tensor,
        accepted_steps: torch.Tensor,
    ) -> torch.Tensor:
        state_window = DVRStateInputWindow.from_cache(state_cache)
        commit_plan = build_dvr_state_commit_plan(
            verified_tail_lens=verified_tail_lens,
            accepted_tokens=accepted_tokens,
            accepted_steps=accepted_steps,
            window_capacity=state_window.capacity,
            conv_capacity=state_cache.intermediate_conv_window[0].shape[2],
            device=live_indices.device,
            chunk_size=self.chunk_size,
        )
        crossing = commit_plan.crossing

        rebuild_dvr_live_state_grouped(
            state_ops=self.ops,
            state_window=state_window,
            temporal_state=state_cache.temporal,
            live_indices=live_indices,
            boundary_indices=boundary_indices,
            req_indices=(~crossing).nonzero(as_tuple=True)[0],
            token_count=commit_plan.pos_after[~crossing],
        )

        self.ops.scatter_state(
            state_cache.conv[0],
            state_cache.intermediate_conv_window[0],
            live_indices,
            accepted_steps,
        )

        if crossing.any():
            boundary_state_step = (
                0
                if state_cache.intermediate_ssm.shape[2] == 1
                else self.chunk_size - 1
            )
            commit_step = torch.where(
                crossing,
                torch.full_like(commit_plan.pos_before, boundary_state_step),
                torch.full_like(commit_plan.pos_before, -1),
            )
            self.ops.scatter_state(
                state_cache.temporal,
                state_cache.intermediate_ssm,
                boundary_indices,
                commit_step,
            )
            self.ops.scatter_state(
                state_cache.conv[0],
                state_cache.intermediate_conv_window[0],
                boundary_indices,
                torch.where(crossing, commit_plan.boundary_conv_steps, commit_step),
            )

            new_pos = commit_plan.pos_after - self.chunk_size
            crossing_idx = crossing.nonzero(as_tuple=True)[0]
            state_window.shift_after_boundary(
                live_indices=live_indices,
                crossing=crossing,
                new_pos=new_pos,
                chunk_size=self.chunk_size,
            )
            rebuild_dvr_live_state_grouped(
                state_ops=self.ops,
                state_window=state_window,
                temporal_state=state_cache.temporal,
                live_indices=live_indices,
                boundary_indices=boundary_indices,
                req_indices=crossing_idx,
                token_count=new_pos[crossing_idx],
            )
            pos_after = torch.where(crossing, new_pos, commit_plan.pos_after)
        else:
            pos_after = commit_plan.pos_after

        state_window.set_tail_lens(
            indices=live_indices, value=pos_after.to(torch.int32)
        )
        return crossing
