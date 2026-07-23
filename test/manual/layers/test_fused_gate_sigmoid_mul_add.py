import itertools

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers import elementwise
from sglang.srt.layers.elementwise import fused_gate_sigmoid_mul_add

DTYPES = [torch.float16, torch.bfloat16]
TOKEN_COUNTS = [1, 2, 4, 8, 16, 64, 512, 1024, 2048, 4096, 8192]
HIDDEN_DIMS = [2048, 3072, 4096, 6144]


def _reference(hidden_states, gate_weight, shared_output, final_hidden_states):
    gate = hidden_states @ gate_weight
    final_hidden_states += torch.sigmoid(gate).unsqueeze(1) * shared_output


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


@pytest.mark.parametrize(
    "num_tokens, hidden_dim, dtype",
    list(itertools.product(TOKEN_COUNTS, HIDDEN_DIMS, DTYPES)),
)
def test_correctness(num_tokens, hidden_dim, dtype):
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    gate_weight = torch.randn(hidden_dim, dtype=dtype, device="cuda")
    shared_output = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    final_ref = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    final_test = final_ref.clone()

    _reference(hidden_states, gate_weight, shared_output, final_ref)
    fused_gate_sigmoid_mul_add(hidden_states, gate_weight, shared_output, final_test)

    torch.testing.assert_close(final_test, final_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gate_near_zero(dtype):
    num_tokens, hidden_dim = 16, 2048
    hs = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    gw = torch.zeros(hidden_dim, dtype=dtype, device="cuda")
    so = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    f_ref = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    f_test = f_ref.clone()

    _reference(hs, gw, so, f_ref)
    fused_gate_sigmoid_mul_add(hs, gw, so, f_test)

    torch.testing.assert_close(f_test, f_ref, rtol=1e-2, atol=1e-2)


def test_inplace_semantics():
    num_tokens, hidden_dim = 32, 2048
    hs = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device="cuda")
    gw = torch.randn(hidden_dim, dtype=torch.float16, device="cuda")
    so = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device="cuda")
    fhs = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device="cuda")
    original_ptr = fhs.data_ptr()

    fused_gate_sigmoid_mul_add(hs, gw, so, fhs)

    assert fhs.data_ptr() == original_ptr


def test_token_count_does_not_change_reduction():
    """The same row must not change when prefill crosses 1024 tokens."""
    num_tokens, hidden_dim = 1024, 4096
    hs = (torch.randn(num_tokens, hidden_dim, device="cuda") * 0.05).to(torch.bfloat16)
    gw = (torch.randn(hidden_dim, device="cuda") * 0.05).to(torch.bfloat16)
    so = (torch.randn(num_tokens, hidden_dim, device="cuda") * 100).to(torch.bfloat16)
    initial = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda")
    below_boundary = initial[:-1].clone()
    at_boundary = initial.clone()

    with envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(True):
        fused_gate_sigmoid_mul_add(hs[:-1], gw, so[:-1], below_boundary)
        fused_gate_sigmoid_mul_add(hs, gw, so, at_boundary)

    torch.testing.assert_close(below_boundary, at_boundary[:-1], rtol=0, atol=0)


def test_large_batch_warp_policy_follows_deterministic_mode(monkeypatch):
    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(kwargs["num_warps"])

            return launch

    monkeypatch.setattr(elementwise, "_fused_gate_sigmoid_mul_add_kernel", FakeKernel())
    monkeypatch.setattr(elementwise, "is_arch_support_pdl", lambda: False)

    hidden_states = torch.empty(1024, 4096)
    gate_weight = torch.empty(4096)
    shared_output = torch.empty_like(hidden_states)
    final_hidden_states = torch.empty_like(hidden_states)

    with envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False):
        fused_gate_sigmoid_mul_add(
            hidden_states, gate_weight, shared_output, final_hidden_states
        )
    with envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(True):
        fused_gate_sigmoid_mul_add(
            hidden_states, gate_weight, shared_output, final_hidden_states
        )

    assert launches == [8, 16]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
