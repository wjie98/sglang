import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dvr_server_args import (
    DVR_EAGLE_SPECULATIVE_ALGORITHM,
    DVR_SPECULATIVE_ALGORITHM,
    handle_dvr_defaults,
    handle_dvr_speculative_decoding,
)


class _Args:
    speculative_algorithm = DVR_SPECULATIVE_ALGORITHM
    device = "cuda"
    enable_dp_attention = False
    enable_pdmux = False
    disaggregation_mode = "null"
    speculative_draft_model_path = None
    speculative_num_draft_tokens = 16
    speculative_num_steps = None
    page_size = 64
    speculative_eagle_topk = None
    max_running_requests = 4
    disable_overlap_schedule = False
    enable_mixed_chunk = True
    disable_cuda_graph = False
    disable_draft_cuda_graph = False
    disable_cuda_graph_padding = False
    disable_radix_cache = False
    disable_custom_all_reduce = False
    enable_deterministic_inference = False
    mamba_radix_cache_strategy = "auto"
    mamba_track_interval = 1
    mamba_ssm_dtype = "float32"
    linear_attn_backend = "triton"
    linear_attn_prefill_backend = None

    def __init__(self):
        self.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.FULL, bs=[1, 2], max_bs=2)
        )

    def get_model_config(self):
        return SimpleNamespace(
            hf_config=SimpleNamespace(get_text_config=lambda: object())
        )


class TestDVRServerArgs(unittest.TestCase):
    def test_dvr_rejects_zero_step_chain_mode(self):
        args = _Args()
        args.speculative_num_draft_tokens = 1
        with self.assertRaisesRegex(ValueError, "speculative_num_draft_tokens >= 2"):
            handle_dvr_speculative_decoding(args)

    def test_dvr_self_draft_extends_cuda_graph_max_with_padding(self):
        args = _Args()

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            handle_dvr_speculative_decoding(args)

        self.assertEqual(args.cuda_graph_config.decode.bs, [1, 2, 4])
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 4)

    def test_dvr_self_draft_extends_exact_cuda_graph_bs_without_padding(self):
        args = _Args()
        args.disable_cuda_graph_padding = True
        args.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.FULL, bs=[1, 4], max_bs=4)
        )

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            handle_dvr_speculative_decoding(args)

        self.assertEqual(args.cuda_graph_config.decode.bs, [1, 2, 3, 4])
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 4)

    def test_dvr_self_draft_extends_cuda_graph_max_before_default_bs(self):
        args = _Args()
        args.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.FULL, bs=None, max_bs=2)
        )

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            handle_dvr_speculative_decoding(args)

        self.assertIsNone(args.cuda_graph_config.decode.bs)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 4)

    def test_dvr_self_draft_gdn_requires_draft_cuda_graph(self):
        args = _Args()
        args.disable_draft_cuda_graph = True

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "requires draft CUDA graphs"):
                handle_dvr_speculative_decoding(args)

    def test_dvr_self_draft_gdn_uses_phase_graph_config(self):
        args = _Args()
        args.cuda_graph_config.decode.backend = Backend.DISABLED

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "requires draft CUDA graphs"):
                handle_dvr_speculative_decoding(args)

    def test_dvr_self_draft_plain_transformer_requires_draft_cuda_graph(self):
        args = _Args()
        args.disable_draft_cuda_graph = True
        args.cuda_graph_config.decode.backend = Backend.DISABLED

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=False,
        ):
            with self.assertRaisesRegex(ValueError, "requires draft CUDA graphs"):
                handle_dvr_speculative_decoding(args)

    def test_dvr_eagle_keeps_radix_cache(self):
        args = _Args()
        args.speculative_algorithm = DVR_EAGLE_SPECULATIVE_ALGORITHM
        args.speculative_draft_model_path = "draft"

        handle_dvr_speculative_decoding(args)

        self.assertFalse(args.disable_radix_cache)

    def test_dvr_preserves_disabled_radix_cache_with_request_checkpoints(self):
        args = _Args()
        args.disable_radix_cache = True

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            handle_dvr_defaults(args)

        self.assertTrue(args.disable_radix_cache)
        self.assertEqual(args.mamba_radix_cache_strategy, "extra_buffer")
        self.assertTrue(ServerArgs.enable_mamba_extra_buffer(args))

    def test_dvr_gdn_requires_boundary_exporting_linear_prefill(self):
        args = _Args()
        args.linear_attn_prefill_backend = "flashinfer"

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "linear-attn-prefill-backend"):
                handle_dvr_speculative_decoding(args)

    def test_dvr_rejects_pdmux_attention(self):
        args = _Args()
        args.enable_pdmux = True

        with self.assertRaisesRegex(ValueError, "PDMux"):
            handle_dvr_speculative_decoding(args)

    def test_dvr_preserves_draft_custom_all_reduce_intent(self):
        args = _Args()
        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=False,
        ):
            handle_dvr_defaults(args)

        self.assertTrue(args._dvr_enable_draft_custom_all_reduce)

        args = _Args()
        args.disable_custom_all_reduce = True
        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=False,
        ):
            handle_dvr_defaults(args)

        self.assertFalse(args._dvr_enable_draft_custom_all_reduce)

    def test_dvr_enables_deterministic_target_execution(self):
        args = _Args()
        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=False,
        ):
            handle_dvr_defaults(args)

        self.assertTrue(args.enable_deterministic_inference)
        self.assertTrue(args._dvr_enable_draft_custom_all_reduce)

    def test_dvr_defaults_normalize_algorithm_before_generic_spec_handling(self):
        args = _Args()
        args.speculative_algorithm = DVR_SPECULATIVE_ALGORITHM.lower()

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=False,
        ):
            handle_dvr_defaults(args)

        self.assertEqual(args.speculative_algorithm, DVR_SPECULATIVE_ALGORITHM)
        self.assertTrue(args._dvr_enable_draft_custom_all_reduce)

    def test_non_gdn_dvr_does_not_inherit_fla_limits(self):
        args = _Args()
        args.page_size = 32
        args.speculative_num_draft_tokens = 65
        args.speculative_num_steps = 64

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=False,
        ):
            handle_dvr_defaults(args)
            handle_dvr_speculative_decoding(args)

        self.assertEqual(args.page_size, 32)
        self.assertEqual(args.speculative_num_draft_tokens, 65)


if __name__ == "__main__":
    unittest.main()
