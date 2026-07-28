import pytest

from sglang.srt.layers.moe.moe_runner.triton_utils import (
    fused_moe_triton_config as config_module,
)
from sglang.srt.layers.moe.moe_runner.triton_utils import (
    fused_moe_triton_kernels as kernel_module,
)


@pytest.mark.parametrize(
    ("low_smem", "is_hip", "expected"),
    [
        (
            False,
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
            True,
            False,
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
            False,
            True,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 2,
            },
        ),
    ],
)
def test_deterministic_fp8_moe_configs(monkeypatch, low_smem, is_hip, expected):
    monkeypatch.setattr(
        config_module,
        "get_global_server_args",
        lambda: type("Args", (), {"enable_deterministic_inference": True})(),
    )
    monkeypatch.setattr(
        config_module, "_use_low_smem_fp8_default", lambda: low_smem
    )
    monkeypatch.setattr(config_module, "_is_hip", is_hip)

    for num_tokens in (16, 80):
        assert (
            config_module.get_default_config(
                M=num_tokens,
                E=512,
                N=512,
                K=4096,
                topk=10,
                dtype="fp8_w8a8",
                is_marlin=False,
                block_shape=None,
            )
            == expected
        )


def test_deterministic_non_fp8_uses_upstream_default(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "get_global_server_args",
        lambda: type("Args", (), {"enable_deterministic_inference": True})(),
    )

    assert config_module.get_default_config(
        M=16,
        E=512,
        N=512,
        K=4096,
        topk=10,
        dtype=None,
        is_marlin=False,
        block_shape=None,
    ) == {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }


def test_swap_ab_cache_tracks_active_invariant_mode(monkeypatch):
    invariant = {"enabled": True}
    monkeypatch.setattr(kernel_module, "_is_cuda", True)
    monkeypatch.setattr(kernel_module, "is_sm90_supported", lambda: True)
    monkeypatch.setattr(
        kernel_module,
        "is_batch_invariant_mode_enabled",
        lambda: invariant["enabled"],
    )
    kernel_module._should_enable_swap_ab.cache_clear()
    try:
        assert not kernel_module.should_enable_swap_ab(32, 64)
        invariant["enabled"] = False
        assert kernel_module.should_enable_swap_ab(32, 64)
        invariant["enabled"] = True
        assert not kernel_module.should_enable_swap_ab(32, 64)
    finally:
        kernel_module._should_enable_swap_ab.cache_clear()
