import argparse

import torch

from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from sglang.srt.layers.attention.linear.dvr_state_ops import (
    rebuild_gdn_state_from_qkvg_beta,
    rebuild_gdn_state_from_qkvg_beta_triton,
)


def rebuild_reference(q, k, v, g, beta, initial_state, token_count):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(20260516)
    batch, max_count, num_q_heads, num_v_heads, head_dim = 8, 16, 4, 8, 64
    q = torch.randn(
        batch, max_count, num_q_heads, head_dim, dtype=dtype, device=args.device
    )
    k = torch.randn(
        batch, max_count, num_q_heads, head_dim, dtype=dtype, device=args.device
    )
    v = torch.randn(
        batch, max_count, num_v_heads, head_dim, dtype=dtype, device=args.device
    )
    g = torch.randn(batch, max_count, num_v_heads, dtype=dtype, device=args.device)
    beta = torch.rand(batch, max_count, num_v_heads, dtype=dtype, device=args.device)
    initial_state = torch.randn(
        batch, num_v_heads, head_dim, head_dim, dtype=torch.float32, device=args.device
    )
    token_count = torch.tensor([0, 1, 2, 3, 5, 8, 13, 16], dtype=torch.long, device=args.device)

    actual = rebuild_gdn_state_from_qkvg_beta(
        q, k, v, g, beta, initial_state=initial_state, token_count=token_count
    )
    expected = rebuild_reference(q, k, v, g, beta, initial_state, token_count)
    max_diff = (actual - expected).abs().max().item()
    triton_actual = rebuild_gdn_state_from_qkvg_beta_triton(
        q, k, v, g, beta, initial_state=initial_state, token_count=token_count
    )
    triton_max_diff = (triton_actual - actual).abs().max().item()
    print(f"max_diff={max_diff} triton_max_diff={triton_max_diff}")
    if max_diff != 0.0:
        raise SystemExit(f"DVR recurrent rebuild mismatch: max_diff={max_diff}")
    if triton_max_diff != 0.0:
        raise SystemExit(
            f"DVR triton recurrent rebuild mismatch: max_diff={triton_max_diff}"
        )


if __name__ == "__main__":
    main()
