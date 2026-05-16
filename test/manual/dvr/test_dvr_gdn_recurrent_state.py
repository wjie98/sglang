import argparse

import torch

from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_update,
)


def rebuild_varlen(q, k, v, g, beta, initial_state, token_count):
    max_count = q.shape[1]
    row_mask = (
        torch.arange(max_count, device=q.device, dtype=torch.long).unsqueeze(0)
        < token_count.unsqueeze(1)
    )
    row_idx, col_idx = row_mask.nonzero(as_tuple=True)
    q_packed = q[row_idx, col_idx].unsqueeze(0).contiguous()
    k_packed = k[row_idx, col_idx].unsqueeze(0).contiguous()
    v_packed = v[row_idx, col_idx].unsqueeze(0).contiguous()
    g_packed = g[row_idx, col_idx].unsqueeze(0).contiguous()
    beta_packed = beta[row_idx, col_idx].unsqueeze(0).contiguous()
    cu_seqlens = torch.empty(
        token_count.shape[0] + 1, dtype=torch.int32, device=q.device
    )
    cu_seqlens[0] = 0
    cu_seqlens[1:] = torch.cumsum(token_count.to(torch.int32), dim=0)
    state_indices = torch.arange(token_count.shape[0], dtype=torch.int32, device=q.device)
    final_state = initial_state.clone()
    fused_recurrent_gated_delta_rule_update(
        q=q_packed,
        k=k_packed,
        v=v_packed,
        g=g_packed,
        beta=beta_packed,
        initial_state_source=final_state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=False,
        disable_output_calculation=True,
        intermediate_state_indices=state_indices,
    )
    return final_state


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

    actual = rebuild_varlen(q, k, v, g, beta, initial_state, token_count)
    expected = rebuild_reference(q, k, v, g, beta, initial_state, token_count)
    max_diff = (actual - expected).abs().max().item()
    print(f"max_diff={max_diff}")
    if max_diff != 0.0:
        raise SystemExit(f"DVR recurrent rebuild mismatch: max_diff={max_diff}")


if __name__ == "__main__":
    main()
