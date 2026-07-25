# DVR-specific GDN kernels. The recurrent producer is adapted from FLA:
# https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_intra,
)
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.fla.op import exp, safe_exp
from sglang.srt.layers.attention.fla.utils import (
    autotune_cache_kwargs,
    is_nvidia_hopper,
)

NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]
CHUNK_SIZE = 64
GDN_CHUNK_H_BV = int(os.getenv("SGLANG_GDN_CHUNK_H_BV", "32"))
GDN_CHUNK_H_NUM_WARPS = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_WARPS", "4"))
GDN_CHUNK_H_NUM_STAGES = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_STAGES", "2"))


@triton.autotune(
    # Single hardcoded config. The kernel writes ht (final state) back into
    # initial_state in-place; with multiple configs, triton's autotune benchmark
    # phase invokes the kernel many times for timing and corrupts the cache pool,
    # producing silently wrong output on the first user request. Restoring via
    # `restore_value=["initial_state"]` works for unit tests but OOMs on
    # production-scale models (e.g. Kimi-Linear-48B at default mem_fraction)
    # because cloning the cache pool for each benchmark exceeds available memory.
    # NT_BUCKET is kept in the autotune key for forward-compatibility (allows
    # future per-bucket configs once the kernel is refactored to write final
    # state to a separate output buffer). The env knobs keep this single-config
    # property while allowing model/hardware-local validation of the selected
    # tile without corrupting the state pool through multi-config autotune.
    configs=[
        triton.Config(
            {"BV": GDN_CHUNK_H_BV},
            num_warps=GDN_CHUNK_H_NUM_WARPS,
            num_stages=GDN_CHUNK_H_NUM_STAGES,
        )
    ],
    key=["H", "K", "V", "BT", "USE_GK", "NT_BUCKET"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _dvr_chunk_gated_delta_rule_fwd_kernel_h(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    initial_state,
    initial_state_indices,
    boundary_state,
    boundary_state_indices,
    boundary_state_steps,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_UPDATE: tl.constexpr,
    WRITE_BOUNDARY_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_BUCKET: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, BK]
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    index = tl.load(initial_state_indices + i_n).to(tl.int32)
    h0 = initial_state + index * stride_h
    ht = initial_state + index * stride_h
    if USE_INITIAL_STATE:
        h0 = h0 + i_h * V * K
    if INPLACE_UPDATE:
        ht = ht + i_h * V * K

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        if WRITE_BOUNDARY_STATE:
            boundary_index = tl.load(boundary_state_indices + i_n).to(tl.int64)
            boundary_step = tl.load(boundary_state_steps + i_n).to(tl.int32)
            boundary_ptr = (
                boundary_state + (boundary_index * H + i_h) * V * K + i_v * BV * K
            )
            boundary_mask = i_t == boundary_step
            tl.store(
                boundary_ptr
                + tl.arange(0, BV)[:, None] * K
                + tl.arange(0, 64)[None, :],
                b_h1,
                mask=boundary_mask
                & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                & (tl.arange(0, 64)[None, :] < K),
            )
            if K > 64:
                tl.store(
                    boundary_ptr
                    + tl.arange(0, BV)[:, None] * K
                    + 64
                    + tl.arange(0, 64)[None, :],
                    b_h2,
                    mask=boundary_mask
                    & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                    & (64 + tl.arange(0, 64)[None, :] < K),
                )
            if K > 128:
                tl.store(
                    boundary_ptr
                    + tl.arange(0, BV)[:, None] * K
                    + 128
                    + tl.arange(0, 64)[None, :],
                    b_h3,
                    mask=boundary_mask
                    & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                    & (128 + tl.arange(0, 64)[None, :] < K),
                )
            if K > 192:
                tl.store(
                    boundary_ptr
                    + tl.arange(0, BV)[:, None] * K
                    + 192
                    + tl.arange(0, 64)[None, :],
                    b_h4,
                    mask=boundary_mask
                    & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                    & (192 + tl.arange(0, 64)[None, :] < K),
                )

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                b_h4 *= exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    # epilogue
    if WRITE_BOUNDARY_STATE:
        boundary_index = tl.load(boundary_state_indices + i_n).to(tl.int64)
        boundary_step = tl.load(boundary_state_steps + i_n).to(tl.int32)
        boundary_ptr = (
            boundary_state + (boundary_index * H + i_h) * V * K + i_v * BV * K
        )
        boundary_mask = boundary_step == NT
        tl.store(
            boundary_ptr + tl.arange(0, BV)[:, None] * K + tl.arange(0, 64)[None, :],
            b_h1,
            mask=boundary_mask
            & (i_v * BV + tl.arange(0, BV)[:, None] < V)
            & (tl.arange(0, 64)[None, :] < K),
        )
        if K > 64:
            tl.store(
                boundary_ptr
                + tl.arange(0, BV)[:, None] * K
                + 64
                + tl.arange(0, 64)[None, :],
                b_h2,
                mask=boundary_mask
                & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                & (64 + tl.arange(0, 64)[None, :] < K),
            )
        if K > 128:
            tl.store(
                boundary_ptr
                + tl.arange(0, BV)[:, None] * K
                + 128
                + tl.arange(0, 64)[None, :],
                b_h3,
                mask=boundary_mask
                & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                & (128 + tl.arange(0, 64)[None, :] < K),
            )
        if K > 192:
            tl.store(
                boundary_ptr
                + tl.arange(0, BV)[:, None] * K
                + 192
                + tl.arange(0, 64)[None, :],
                b_h4,
                mask=boundary_mask
                & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                & (192 + tl.arange(0, 64)[None, :] < K),
            )

    if INPLACE_UPDATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def _dvr_chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    inplace_update: bool = True,
    boundary_state: Optional[torch.Tensor] = None,
    boundary_state_indices: Optional[torch.Tensor] = None,
    boundary_state_steps: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = CHUNK_SIZE

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    write_boundary_state = boundary_state is not None
    h = k.new_empty(B, NT, H, V, K)

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    _dvr_chunk_gated_delta_rule_fwd_kernel_h[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        boundary_state=boundary_state,
        boundary_state_indices=boundary_state_indices,
        boundary_state_steps=boundary_state_steps,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_INITIAL_STATE=initial_state is not None,
        INPLACE_UPDATE=inplace_update,
        WRITE_BOUNDARY_STATE=write_boundary_state,
        SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None,
        NT_BUCKET=(0 if NT <= 32 else (1 if NT <= 128 else 2)),
    )
    return h, v_new


@torch.compiler.disable
def dvr_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    scale: Optional[float] = None,
    boundary_state: Optional[torch.Tensor] = None,
    boundary_state_indices: Optional[torch.Tensor] = None,
    boundary_state_steps: Optional[torch.Tensor] = None,
):
    """Run the FLA GDN chunk path without mutating the authoritative state.

    DVR verify reads an exact recurrent checkpoint, emits a possible 64-token
    boundary into request-owned storage, and leaves the source checkpoint
    untouched. The ordinary FLA path remains in-place and keeps its upstream API.
    """

    boundary_outputs = (
        boundary_state,
        boundary_state_indices,
        boundary_state_steps,
    )
    if any(value is not None for value in boundary_outputs) and not all(
        value is not None for value in boundary_outputs
    ):
        raise ValueError(
            "boundary_state, boundary_state_indices, and boundary_state_steps "
            "must be provided together."
        )
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype == torch.float32:
        raise ValueError("DVR GDN q, k, and v must share a non-fp32 dtype.")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError("Variable-length DVR GDN expects a flattened batch.")
        if initial_state_indices.shape[0] != len(cu_seqlens) - 1:
            raise ValueError("initial_state_indices must contain one row per sequence.")

    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
        if cu_seqlens is not None
        else None
    )
    g = chunk_local_cumsum(
        g,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    w, u, _ = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new = _dvr_chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        inplace_update=False,
        boundary_state=boundary_state,
        boundary_state_indices=boundary_state_indices,
        boundary_state_steps=boundary_state_steps,
    )
    output = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=k.shape[-1] ** -0.5 if scale is None else scale,
        cu_seqlens=cu_seqlens,
    )
    return output.to(q.dtype), None, h


@triton.jit
def _dvr_scatter_state_kernel(
    src,
    dst,
    source_rows,
    destination_rows,
    source_steps,
    elements_per_row: tl.constexpr,
    src_layer_stride,
    src_request_stride,
    src_step_stride,
    dst_layer_stride,
    dst_request_stride,
    src_request_count,
    src_step_count,
    dst_request_count,
    BLOCK_SIZE: tl.constexpr,
):
    request = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    block = tl.program_id(2).to(tl.int64)
    step = tl.load(source_steps + request).to(tl.int64)
    if step < 0:
        return

    source = tl.load(source_rows + request).to(tl.int64)
    destination = tl.load(destination_rows + request).to(tl.int64)
    if not (
        (source >= 0)
        & (source < src_request_count)
        & (step < src_step_count)
        & (destination >= 0)
        & (destination < dst_request_count)
    ):
        return

    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements_per_row
    src_offset = (
        layer * src_layer_stride
        + source * src_request_stride
        + step * src_step_stride
        + offsets
    )
    dst_offset = layer * dst_layer_stride + destination * dst_request_stride + offsets
    tl.store(dst + dst_offset, tl.load(src + src_offset, mask=mask), mask=mask)


def dvr_scatter_state(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    source_rows: torch.Tensor,
    destination_rows: torch.Tensor,
    source_steps: torch.Tensor,
) -> None:
    """Publish selected request-owned recurrent states into cache slots."""

    request_count = source_steps.shape[0]
    if request_count == 0:
        return
    if (
        destination.ndim < 2
        or source.ndim < 3
        or destination.shape[0] != source.shape[0]
        or destination.shape[2:] != source.shape[3:]
    ):
        raise ValueError("DVR recurrent source and destination shapes do not match.")
    if not destination.is_contiguous() or not source.is_contiguous():
        raise ValueError("DVR recurrent-state scatter requires contiguous tensors.")

    source_rows = source_rows.to(device=source.device, dtype=torch.int32).contiguous()
    destination_rows = destination_rows.to(
        device=source.device, dtype=torch.int32
    ).contiguous()
    source_steps = source_steps.to(device=source.device, dtype=torch.int32).contiguous()
    elements_per_row = destination[0, 0].numel()
    block_size = 1024
    _dvr_scatter_state_kernel[
        (
            request_count,
            destination.shape[0],
            triton.cdiv(elements_per_row, block_size),
        )
    ](
        source,
        destination,
        source_rows,
        destination_rows,
        source_steps,
        elements_per_row,
        source.stride(0),
        source.stride(1),
        source.stride(2),
        destination.stride(0),
        destination.stride(1),
        source.shape[1],
        source.shape[2],
        destination.shape[1],
        BLOCK_SIZE=block_size,
    )


@triton.jit
def _dvr_scatter_conv_window_kernel(
    src,
    dst,
    source_rows,
    destination_rows,
    source_steps,
    elements_per_row: tl.constexpr,
    window_size: tl.constexpr,
    src_layer_stride,
    src_request_stride,
    src_step_stride,
    src_dim_stride,
    src_window_stride,
    dst_layer_stride,
    dst_request_stride,
    src_request_count,
    src_step_count,
    dst_request_count,
    BLOCK_SIZE: tl.constexpr,
):
    request = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    block = tl.program_id(2).to(tl.int64)
    step = tl.load(source_steps + request).to(tl.int64)
    if step < 0:
        return

    source = tl.load(source_rows + request).to(tl.int64)
    destination = tl.load(destination_rows + request).to(tl.int64)
    if not (
        (source >= 0)
        & (source < src_request_count)
        & (step < src_step_count)
        & (destination >= 0)
        & (destination < dst_request_count)
    ):
        return

    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements_per_row
    dim = offsets // window_size
    window = offsets % window_size
    src_offset = (
        layer * src_layer_stride
        + source * src_request_stride
        + step * src_step_stride
        + dim * src_dim_stride
        + window * src_window_stride
    )
    dst_offset = layer * dst_layer_stride + destination * dst_request_stride + offsets
    tl.store(dst + dst_offset, tl.load(src + src_offset, mask=mask), mask=mask)


def dvr_scatter_conv_window(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    source_rows: torch.Tensor,
    destination_rows: torch.Tensor,
    source_steps: torch.Tensor,
) -> None:
    """Publish selected overlapping conv windows into contiguous cache rows."""

    request_count = source_steps.shape[0]
    if request_count == 0:
        return
    if (
        destination.ndim != 4
        or source.ndim != 5
        or destination.shape[0] != source.shape[0]
        or destination.shape[2:] != source.shape[3:]
    ):
        raise ValueError("DVR conv-window source and destination shapes do not match.")
    if not destination.is_contiguous():
        raise ValueError("DVR conv-state destination must be contiguous.")

    source_rows = source_rows.to(device=source.device, dtype=torch.int32).contiguous()
    destination_rows = destination_rows.to(
        device=source.device, dtype=torch.int32
    ).contiguous()
    source_steps = source_steps.to(device=source.device, dtype=torch.int32).contiguous()
    window_size = destination.shape[-1]
    elements_per_row = destination.shape[-2] * window_size
    block_size = 1024
    _dvr_scatter_conv_window_kernel[
        (
            request_count,
            destination.shape[0],
            triton.cdiv(elements_per_row, block_size),
        )
    ](
        source,
        destination,
        source_rows,
        destination_rows,
        source_steps,
        elements_per_row,
        window_size,
        source.stride(0),
        source.stride(1),
        source.stride(2),
        source.stride(3),
        source.stride(4),
        destination.stride(0),
        destination.stride(1),
        source.shape[1],
        source.shape[2],
        destination.shape[1],
        BLOCK_SIZE=block_size,
    )
