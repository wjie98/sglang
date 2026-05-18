"""Manual unit check for DVR GDN recurrent-state rebuild.

DVR target verify stores the chunkwise-scan inputs q/k/v/g/beta, not every
token's full SSM state. After sampling decides how many draft tokens are
accepted, the live recurrent state for the next self-draft decode is rebuilt
from:

  chunk-boundary state + q/k/v/g/beta rows up to accepted length.

This test compares the DVR Triton rebuild kernel with FLA's recurrent GDN
reference. The script intentionally uses .tolist()/.item() for diagnostics; it
is a manual test and not part of the latency-sensitive serving path.

Example:

PYTHONPATH=python conda run -n dvr_dev python \
  test/manual/dvr/test_dvr_gdn_recurrent_state.py \
  --device cuda --dtype bfloat16 --max-count 16
"""

from __future__ import annotations

import argparse
from typing import List

import torch

from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from sglang.srt.layers.attention.linear.dvr_gdn_state import (
    _rebuild_gdn_state_from_qkvg_beta_triton,
)


def make_token_counts(batch: int, max_count: int, device: str) -> torch.Tensor:
    # Include powers/near-powers that have repeatedly exposed off-by-one bugs in
    # DVR state handling: 0, 1, 2, 3, 5, 8, 13, 16, then the configured max.
    seeds = [0, 1, 2, 3, 5, 8, 13, 16, max_count]
    values: List[int] = []
    for value in seeds:
        if 0 <= value <= max_count and value not in values:
            values.append(value)
    while len(values) < batch:
        values.append((len(values) * 7) % (max_count + 1))
    return torch.tensor(values[:batch], dtype=torch.long, device=device)


def rebuild_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    token_count: torch.Tensor,
) -> torch.Tensor:
    states = []
    for i, end in enumerate(token_count.tolist()):
        if end == 0:
            states.append(initial_state[i : i + 1])
            continue
        _, final_state = fused_recurrent_gated_delta_rule(
            q=q[i : i + 1, :end],
            k=k[i : i + 1, :end],
            v=v[i : i + 1, :end],
            g=g[i : i + 1, :end],
            beta=beta[i : i + 1, :end],
            initial_state=initial_state[i : i + 1],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        states.append(final_state)
    return torch.cat(states, dim=0)


def run_case(
    *,
    device: str,
    dtype: torch.dtype,
    batch: int,
    max_count: int,
    num_q_heads: int,
    num_v_heads: int,
    head_dim: int,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    q = torch.randn(batch, max_count, num_q_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, max_count, num_q_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, max_count, num_v_heads, head_dim, dtype=dtype, device=device)
    g = torch.randn(batch, max_count, num_v_heads, dtype=dtype, device=device)
    beta = torch.rand(batch, max_count, num_v_heads, dtype=dtype, device=device)
    initial_state = torch.randn(
        batch,
        num_v_heads,
        head_dim,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    token_count = make_token_counts(batch, max_count, device)

    actual = _rebuild_gdn_state_from_qkvg_beta_triton(
        q, k, v, g, beta, initial_state=initial_state, token_count=token_count
    )
    expected = rebuild_reference(q, k, v, g, beta, initial_state, token_count)
    max_diff = (actual - expected).abs().max().item()
    print(
        "batch={} max_count={} q_heads={} v_heads={} head_dim={} "
        "token_count={} max_diff={}".format(
            batch,
            max_count,
            num_q_heads,
            num_v_heads,
            head_dim,
            token_count.tolist(),
            max_diff,
        ),
        flush=True,
    )
    return max_diff


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--max-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the DVR GDN Triton rebuild test.")

    dtype = getattr(torch, args.dtype)
    max_diffs = []
    for i, batch in enumerate(parse_int_list(args.batch_sizes)):
        max_diffs.append(
            run_case(
                device=args.device,
                dtype=dtype,
                batch=batch,
                max_count=args.max_count,
                num_q_heads=4,
                num_v_heads=8,
                head_dim=64,
                seed=args.seed + i,
            )
        )

    worst = max(max_diffs or [0.0])
    print(f"WORST_MAX_DIFF {worst}", flush=True)
    if worst > args.tolerance:
        raise SystemExit(
            f"DVR GDN recurrent rebuild mismatch: {worst} > {args.tolerance}"
        )


if __name__ == "__main__":
    main()
