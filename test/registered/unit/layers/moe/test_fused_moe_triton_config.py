import pytest

from sglang.srt.layers.moe.moe_runner.triton_utils import (
    fused_moe_triton_config as config_module,
)


@pytest.mark.parametrize(
    ("dtype", "low_smem", "expected"),
    [
        (
            "fp8_w8a8",
            False,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            },
        ),
        (
            "fp8_w8a8",
            True,
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            },
        ),
        (
            None,
            False,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            },
        ),
    ],
)
def test_batch_invariant_moe_configs(monkeypatch, dtype, low_smem, expected):
    monkeypatch.setattr(
        config_module, "is_batch_invariant_mode_enabled", lambda: True
    )
    monkeypatch.setattr(
        config_module, "_use_low_smem_fp8_default", lambda: low_smem
    )

    assert (
        config_module.get_default_config(
            M=16,
            E=512,
            N=512,
            K=4096,
            topk=10,
            dtype=dtype,
            is_marlin=False,
            block_shape=None,
        )
        == expected
    )
