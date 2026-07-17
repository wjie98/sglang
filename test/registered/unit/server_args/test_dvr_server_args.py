import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
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
    pp_size = 1
    speculative_adaptive = False
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
    attention_backend = "triton"
    prefill_attention_backend = None
    decode_attention_backend = None
    speculative_use_rejection_sampling = False
    speculative_accept_threshold_single = 1.0
    speculative_accept_threshold_acc = 1.0
    enable_multi_layer_eagle = False
    sampling_backend = "flashinfer"
    enable_streaming_session = False
    enable_int8_mamba_checkpoint = False

    def __init__(self):
        self.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.FULL, bs=[1, 2], max_bs=2)
        )
        self._cuda_graph_config_locked = set()

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

    def test_dvr_rejects_pipeline_parallelism(self):
        args = _Args()
        args.pp_size = 2

        with self.assertRaisesRegex(ValueError, "pp_size == 1"):
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

    def test_dvr_self_draft_does_not_mutate_explicit_cuda_graph_bs(self):
        args = _Args()
        args._cuda_graph_config_locked = {(Phase.DECODE, "bs")}

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "Explicit.*decode.*bs"):
                handle_dvr_speculative_decoding(args)

        self.assertEqual(args.cuda_graph_config.decode.bs, [1, 2])

    def test_dvr_self_draft_does_not_mutate_explicit_cuda_graph_max_bs(self):
        args = _Args()
        args.cuda_graph_config.decode.bs = None
        args._cuda_graph_config_locked = {(Phase.DECODE, "max_bs")}

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "Explicit.*decode.*max_bs"):
                handle_dvr_speculative_decoding(args)

        self.assertEqual(args.cuda_graph_config.decode.max_bs, 2)

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

    def test_dvr_eagle_defaults_to_one_mtp_draft_step(self):
        args = _Args()
        args.speculative_algorithm = DVR_EAGLE_SPECULATIVE_ALGORITHM
        args.speculative_draft_model_path = "draft"
        args.speculative_num_draft_tokens = None

        handle_dvr_speculative_decoding(args)

        self.assertEqual(args.speculative_num_draft_tokens, 2)
        self.assertEqual(args.speculative_num_steps, 1)

    def test_dvr_rejection_sampling_is_default_and_can_be_disabled(self):
        args = _Args()
        handle_dvr_speculative_decoding(args)
        self.assertTrue(args.speculative_use_rejection_sampling)

        args = _Args()
        with envs.SGLANG_DVR_USE_REJECTION_SAMPLING.override(False):
            handle_dvr_speculative_decoding(args)
        self.assertFalse(args.speculative_use_rejection_sampling)

    def test_dvr_rejection_sampling_rejects_ignored_acceptance_thresholds(self):
        args = _Args()
        args.speculative_accept_threshold_single = 0.9

        with self.assertRaisesRegex(ValueError, "thresholds must remain 1.0"):
            handle_dvr_speculative_decoding(args)

    def test_dvr_rejection_sampling_rejects_tree_and_multi_layer_eagle(self):
        args = _Args()
        args.speculative_eagle_topk = 2
        with self.assertRaisesRegex(ValueError, "chain mode with topk == 1"):
            handle_dvr_speculative_decoding(args)

        args = _Args()
        args.speculative_algorithm = DVR_EAGLE_SPECULATIVE_ALGORITHM
        args.speculative_draft_model_path = "draft"
        args.enable_multi_layer_eagle = True
        with self.assertRaisesRegex(NotImplementedError, "multi-layer EAGLE"):
            handle_dvr_speculative_decoding(args)

    def test_dvr_rejects_custom_sampler(self):
        for rejection_sampling in (True, False):
            args = _Args()
            args.sampling_backend = "custom"
            with (
                envs.SGLANG_DVR_USE_REJECTION_SAMPLING.override(rejection_sampling),
                self.assertRaisesRegex(
                    ValueError, "built-in flashinfer and pytorch"
                ),
            ):
                handle_dvr_speculative_decoding(args)

    def test_dvr_preserves_disabled_radix_cache_with_request_checkpoints(self):
        for strategy in ("auto", "no_buffer", "extra_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                args = _Args()
                args.disable_radix_cache = True
                args.mamba_radix_cache_strategy = strategy

                with patch(
                    "sglang.srt.speculative.dvr_server_args."
                    "_is_dvr_gated_linear_state_model",
                    return_value=True,
                ):
                    handle_dvr_defaults(args)

                self.assertTrue(args.disable_radix_cache)
                self.assertEqual(args.mamba_radix_cache_strategy, strategy)
                self.assertTrue(ServerArgs.enable_mamba_extra_buffer(args))

    def test_dvr_gdn_preserves_supported_page_sizes(self):
        for page_size in (1, 8, 16, 32, 64):
            with self.subTest(page_size=page_size):
                args = _Args()
                args.page_size = page_size
                with patch(
                    "sglang.srt.speculative.dvr_server_args."
                    "_is_dvr_gated_linear_state_model",
                    return_value=True,
                ):
                    handle_dvr_defaults(args)
                self.assertEqual(args.page_size, page_size)

    def test_dvr_gdn_rejects_page_sizes_incompatible_with_fla_chunks(self):
        for page_size in (0, 48, 128):
            with self.subTest(page_size=page_size):
                args = _Args()
                args.page_size = page_size
                with patch(
                    "sglang.srt.speculative.dvr_server_args."
                    "_is_dvr_gated_linear_state_model",
                    return_value=True,
                ):
                    with self.assertRaisesRegex(ValueError, "positive divisor"):
                        handle_dvr_defaults(args)

    def test_dvr_gdn_radix_requires_non_lazy_extra_buffer(self):
        for strategy in ("no_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                args = _Args()
                args.mamba_radix_cache_strategy = strategy
                with patch(
                    "sglang.srt.speculative.dvr_server_args."
                    "_is_dvr_gated_linear_state_model",
                    return_value=True,
                ):
                    with self.assertRaisesRegex(ValueError, "requires.*extra_buffer"):
                        handle_dvr_defaults(args)

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

    def test_dvr_rejects_unsupported_full_attention_backend(self):
        args = _Args()
        args.attention_backend = "flashinfer"

        with self.assertRaisesRegex(ValueError, "only Triton and FA3"):
            handle_dvr_speculative_decoding(args)

    def test_dvr_rejects_pdmux_attention(self):
        args = _Args()
        args.enable_pdmux = True

        with self.assertRaisesRegex(ValueError, "PDMux"):
            handle_dvr_speculative_decoding(args)

    def test_dvr_gdn_rejects_streaming_session(self):
        args = _Args()
        args.enable_streaming_session = True

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "streaming sessions"):
                handle_dvr_speculative_decoding(args)

    def test_dvr_gdn_rejects_int8_recurrent_checkpoints(self):
        args = _Args()
        args.enable_int8_mamba_checkpoint = True

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "exact recurrent checkpoints"):
                handle_dvr_speculative_decoding(args)

    def test_plain_transformer_dvr_ignores_gdn_only_options(self):
        args = _Args()
        args.enable_streaming_session = True
        args.enable_int8_mamba_checkpoint = True

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=False,
        ):
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
