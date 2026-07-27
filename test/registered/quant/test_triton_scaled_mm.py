import unittest
from typing import Optional
from unittest.mock import patch

import torch
import torch.testing

import sglang.srt.layers.quantization.fp8_utils as fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_kernel import triton_scaled_mm
from sglang.srt.utils.common import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=12, suite="stage-b-test-1-gpu-small-amd")


def torch_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation using float32 for stability"""
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a.to(torch.float32) * out * scale_b.to(torch.float32).T
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(out_dtype)


class TestScaledMM(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not (torch.cuda.is_available() or torch.xpu.is_available()):
            raise unittest.SkipTest("No CUDA or XPU device available")
        cls._device = get_device()
        torch.set_default_device(cls._device)

    def _make_inputs(self, M, K, N, in_dtype):
        if in_dtype == torch.int8:
            a = torch.randint(-8, 8, (M, K), dtype=in_dtype, device=self._device)
            b = torch.randint(-8, 8, (K, N), dtype=in_dtype, device=self._device)
        else:  # fp8
            a = torch.clamp(
                0.1 * torch.randn((M, K), dtype=torch.float16, device=self._device),
                -0.3,
                0.3,
            ).to(in_dtype)
            b = torch.clamp(
                0.1 * torch.randn((K, N), dtype=torch.float16, device=self._device),
                -0.3,
                0.3,
            ).to(in_dtype)
        return a, b

    def test_basic_cases(self):
        """Test core functionality with reduced precision requirements"""
        test_configs = [
            (32, 32, 32, torch.int8, torch.float16, False),
            (17, 64, 96, torch.int8, torch.float16, False),
            (64, 64, 64, torch.int8, torch.float16, True),
        ]

        try:
            torch.tensor([1.0], dtype=torch.float8_e4m3fn, device=self._device)
            test_configs.append((32, 32, 32, torch.float8_e4m3fn, torch.float16, False))
            test_configs.append((17, 64, 96, torch.float8_e4m3fn, torch.float16, False))
        except:
            print("FP8 not supported, skipping")

        for M, K, N, in_dtype, out_dtype, with_bias in test_configs:
            with self.subTest(M=M, K=K, N=N, dtype=in_dtype, bias=with_bias):
                print(f"Currently testing with in_dtype: {in_dtype}")
                torch.manual_seed(42)

                input, weight = self._make_inputs(M, K, N, in_dtype)
                scale_a = 0.1 + 0.05 * torch.rand(
                    (M, 1), dtype=torch.float32, device=self._device
                )
                scale_b = 0.1 + 0.05 * torch.rand(
                    (N, 1), dtype=torch.float32, device=self._device
                )
                bias = (
                    0.01 * torch.randn((M, N), dtype=out_dtype, device=self._device)
                    if with_bias
                    else None
                )

                triton_out = triton_scaled_mm(
                    input, weight, scale_a, scale_b, out_dtype, bias
                )
                ref_out = torch_scaled_mm(
                    input, weight, scale_a, scale_b, out_dtype, bias
                )

                # Use relaxed tolerances
                rtol = 0.15 if in_dtype == torch.int8 else 0.25
                atol = 0.1 if in_dtype == torch.int8 else 0.15

                torch.testing.assert_close(triton_out, ref_out, rtol=rtol, atol=atol)

                scale_b_row = scale_b.t().contiguous()
                triton_out_row_scale = triton_scaled_mm(
                    input, weight, scale_a, scale_b_row, out_dtype, bias
                )
                torch.testing.assert_close(
                    triton_out_row_scale, ref_out, rtol=rtol, atol=atol
                )

    def test_explicit_tile(self):
        torch.manual_seed(42)
        input, weight = self._make_inputs(17, 64, 96, torch.int8)
        scale_a = 0.1 + 0.05 * torch.rand(
            (17, 1), dtype=torch.float32, device=self._device
        )
        scale_b = 0.1 + 0.05 * torch.rand(
            (96, 1), dtype=torch.float32, device=self._device
        )

        output = triton_scaled_mm(
            input,
            weight,
            scale_a,
            scale_b,
            torch.float16,
            block_size_m=64,
            block_size_n=64,
            block_size_k=32,
            use_heuristic=False,
        )
        reference = torch_scaled_mm(
            input, weight, scale_a, scale_b, torch.float16
        )

        torch.testing.assert_close(output, reference, rtol=0.15, atol=0.1)

    def test_fixed_w8a8_dispatch_uses_explicit_tile(self):
        input = torch.ones((3, 4), dtype=torch.bfloat16, device=self._device)
        weight = torch.ones((4, 5), dtype=torch.bfloat16, device=self._device)
        weight_scale = torch.ones((5, 1), dtype=torch.float32, device=self._device)
        qinput = torch.ones((3, 4), dtype=torch.int8, device=self._device)
        input_scale = torch.ones((3, 1), dtype=torch.float32, device=self._device)
        captured = {}

        def fixed_scaled_mm(*args, **kwargs):
            captured.update(kwargs)
            return torch.zeros((3, 5), dtype=torch.bfloat16, device=self._device)

        with (
            patch.object(
                fp8_utils,
                "sglang_per_token_quant_fp8",
                return_value=(qinput, input_scale),
            ),
            patch.object(fp8_utils, "triton_scaled_mm", side_effect=fixed_scaled_mm),
        ):
            output = fp8_utils.apply_fp8_linear(
                input,
                weight,
                weight_scale,
                cutlass_fp8_supported=True,
                use_fixed_w8a8_triton=True,
            )

        self.assertEqual(output.shape, (3, 5))
        self.assertEqual(
            captured,
            {
                "block_size_m": 64,
                "block_size_n": 64,
                "block_size_k": 256,
                "use_heuristic": False,
            },
        )

    def test_fp8_method_captures_deterministic_policy(self):
        config = type(
            "Config",
            (),
            {
                "weight_block_size": None,
                "use_mxfp8": False,
                "is_checkpoint_fp8_serialized": True,
            },
        )()

        with (
            patch(
                "sglang.srt.layers.quantization.fp8.cutlass_fp8_supported",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.quantization.fp8."
                "dispatch_w8a8_block_fp8_linear",
                return_value=None,
            ),
            patch(
                "sglang.srt.layers.quantization.fp8.envs."
                "SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get",
                return_value=True,
            ),
        ):
            method = Fp8LinearMethod(config)

        self.assertTrue(method.use_fixed_w8a8_triton)


if __name__ == "__main__":
    unittest.main(verbosity=2)
