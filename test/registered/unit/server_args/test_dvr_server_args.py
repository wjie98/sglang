import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.mem_cache.common import get_req_to_token_extra_context_len
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dvr_server_args import (
    DVR_EAGLE_SPECULATIVE_ALGORITHM,
    DVR_SPECULATIVE_ALGORITHM,
    _is_dvr_gated_linear_state_model,
    handle_dvr_cuda_graph_config,
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
    max_speculative_num_draft_tokens = 16
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
    flashinfer_allreduce_fusion_backend = None
    enforce_disable_flashinfer_allreduce_fusion = False
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
    enable_page_major_kv_layout = False
    enable_linear_replayssm = False
    enable_unified_memory = False

    def __init__(self):
        self.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.FULL, bs=[1, 2, 4], max_bs=4),
            prefill=SimpleNamespace(backend=Backend.BREAKABLE),
        )
        self._cuda_graph_config_locked = set()

    def get_model_config(self):
        return SimpleNamespace(
            hf_config=SimpleNamespace(get_text_config=lambda: object())
        )


class TestDVRServerArgs(unittest.TestCase):
    def test_gdn_capability_uses_normalized_text_config(self):
        text_config = Qwen3NextConfig()

        self.assertTrue(
            _is_dvr_gated_linear_state_model(
                SimpleNamespace(
                    get_model_config=lambda: SimpleNamespace(
                        hf_config=SimpleNamespace(get_text_config=lambda: text_config)
                    )
                )
            )
        )
        self.assertFalse(
            _is_dvr_gated_linear_state_model(
                SimpleNamespace(
                    get_model_config=lambda: SimpleNamespace(
                        hf_config=SimpleNamespace(get_text_config=lambda: object())
                    )
                )
            )
        )

    def test_gdn_capability_uses_registered_backend(self):
        args = SimpleNamespace(
            get_model_config=lambda: SimpleNamespace(
                hf_config=SimpleNamespace(get_text_config=lambda: object())
            )
        )
        registry_path = (
            "sglang.srt.configs.linear_attn_model_registry.get_linear_attn_config"
        )
        for backend, expected in (
            ("sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend", True),
            ("sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend", False),
        ):
            with (
                self.subTest(backend=backend),
                patch(
                    registry_path,
                    return_value=(
                        SimpleNamespace(backend_class_name=backend),
                        object(),
                    ),
                ),
            ):
                self.assertEqual(_is_dvr_gated_linear_state_model(args), expected)

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

    def test_dvr_self_draft_uses_existing_cuda_graph_coverage(self):
        cases = (
            (False, [1, 2, 4], 4),
            (True, [1, 2, 3, 4], 4),
            (False, None, 4),
        )
        for padding_disabled, bs, max_bs in cases:
            with self.subTest(padding_disabled=padding_disabled, bs=bs):
                args = _Args()
                args.disable_cuda_graph_padding = padding_disabled
                args.cuda_graph_config.decode.bs = bs
                args.cuda_graph_config.decode.max_bs = max_bs
                handle_dvr_cuda_graph_config(args)
                self.assertEqual(args.cuda_graph_config.decode.bs, bs)
                self.assertEqual(args.cuda_graph_config.decode.max_bs, max_bs)

    def test_dvr_self_draft_caps_auto_concurrency_to_graph_capacity(self):
        args = _Args()
        args.max_running_requests = None
        args.cuda_graph_config.decode.bs = [1, 2, 8]
        args.cuda_graph_config.decode.max_bs = 8

        handle_dvr_cuda_graph_config(args)

        self.assertEqual(args.max_running_requests, 8)

    def test_dvr_chain_request_row_covers_overlap_reservation(self):
        args = _Args()
        self.assertEqual(get_req_to_token_extra_context_len(args), 32)

    def test_dvr_self_draft_rejects_insufficient_cuda_graph_coverage(self):
        for padding_disabled, bs, max_bs in (
            (False, [1, 2], 2),
            (True, [1, 4], 4),
            (False, None, 2),
        ):
            with self.subTest(padding_disabled=padding_disabled, bs=bs):
                args = _Args()
                args.disable_cuda_graph_padding = padding_disabled
                args.cuda_graph_config.decode.bs = bs
                args.cuda_graph_config.decode.max_bs = max_bs
                with self.assertRaisesRegex(
                    ValueError, "does not cover DVR self-draft"
                ):
                    handle_dvr_cuda_graph_config(args)

    def test_dvr_self_draft_requires_cuda_graph(self):
        for disable_draft, backend in (
            (True, Backend.FULL),
            (False, Backend.DISABLED),
        ):
            with self.subTest(disable_draft=disable_draft, backend=backend):
                args = _Args()
                args.disable_draft_cuda_graph = disable_draft
                args.cuda_graph_config.decode.backend = backend
                with self.assertRaisesRegex(ValueError, "requires draft CUDA graphs"):
                    handle_dvr_cuda_graph_config(args)

    def test_dvr_eagle_keeps_radix_cache(self):
        args = _Args()
        args.speculative_algorithm = DVR_EAGLE_SPECULATIVE_ALGORITHM
        args.speculative_draft_model_path = "draft"

        handle_dvr_speculative_decoding(args)

        self.assertFalse(args.disable_radix_cache)
        self.assertFalse(args.enable_mixed_chunk)

    def test_dvr_gdn_disables_default_prefill_cuda_graph(self):
        args = _Args()
        with patch(
            "sglang.srt.speculative.dvr_server_args._is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            handle_dvr_cuda_graph_config(args)

        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)

    def test_dvr_gdn_rejects_explicit_prefill_cuda_graph(self):
        args = _Args()
        args._cuda_graph_config_locked = {(Phase.PREFILL, "backend")}
        with (
            patch(
                "sglang.srt.speculative.dvr_server_args."
                "_is_dvr_gated_linear_state_model",
                return_value=True,
            ),
            self.assertRaisesRegex(ValueError, "prefill CUDA graphs"),
        ):
            handle_dvr_cuda_graph_config(args)

    def test_plain_transformer_dvr_keeps_prefill_cuda_graph(self):
        args = _Args()
        with patch(
            "sglang.srt.speculative.dvr_server_args._is_dvr_gated_linear_state_model",
            return_value=False,
        ):
            handle_dvr_cuda_graph_config(args)

        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.BREAKABLE)

    def test_dvr_gdn_rejects_unified_memory(self):
        args = _Args()
        args.enable_unified_memory = True
        with (
            patch(
                "sglang.srt.speculative.dvr_server_args."
                "_is_dvr_gated_linear_state_model",
                return_value=True,
            ),
            self.assertRaisesRegex(ValueError, "unified-memory"),
        ):
            handle_dvr_defaults(args)

    def test_dvr_eagle_defaults_to_one_mtp_draft_step(self):
        args = _Args()
        args.speculative_algorithm = DVR_EAGLE_SPECULATIVE_ALGORITHM
        args.speculative_draft_model_path = "draft"
        args.speculative_num_draft_tokens = None

        handle_dvr_speculative_decoding(args)

        self.assertEqual(args.speculative_num_draft_tokens, 2)
        self.assertEqual(args.speculative_num_steps, 1)

    def test_dvr_rejection_sampling_is_always_enabled(self):
        args = _Args()
        handle_dvr_speculative_decoding(args)
        self.assertTrue(args.speculative_use_rejection_sampling)

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
        args = _Args()
        args.sampling_backend = "custom"
        with self.assertRaisesRegex(ValueError, "built-in flashinfer and pytorch"):
            handle_dvr_speculative_decoding(args)

    def test_dvr_preserves_disabled_radix_cache_without_enabling_radix_tracking(self):
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
                self.assertFalse(ServerArgs.enable_mamba_extra_buffer(args))

    def test_dvr_gdn_defaults_to_fla_page_size(self):
        args = _Args()
        args.page_size = None
        with patch(
            "sglang.srt.speculative.dvr_server_args._is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            handle_dvr_defaults(args)
        self.assertEqual(args.page_size, 64)

    def test_dvr_gdn_requires_fla_page_size(self):
        for page_size in (0, 1, 8, 16, 32, 48, 128):
            with self.subTest(page_size=page_size):
                args = _Args()
                args.page_size = page_size
                with patch(
                    "sglang.srt.speculative.dvr_server_args."
                    "_is_dvr_gated_linear_state_model",
                    return_value=True,
                ):
                    with self.assertRaisesRegex(ValueError, "page_size =="):
                        handle_dvr_defaults(args)

    def test_dvr_gdn_rejects_incompatible_state_layouts(self):
        for field, message in (
            ("enable_page_major_kv_layout", "page-major"),
            ("enable_linear_replayssm", "linear-replayssm"),
        ):
            with self.subTest(field=field):
                args = _Args()
                setattr(args, field, True)
                with patch(
                    "sglang.srt.speculative.dvr_server_args."
                    "_is_dvr_gated_linear_state_model",
                    return_value=True,
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        handle_dvr_speculative_decoding(args)

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
            "sglang.srt.speculative.dvr_server_args._is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "linear-attn-prefill-backend"):
                handle_dvr_speculative_decoding(args)

    def test_dvr_rejects_unsupported_full_attention_backend(self):
        args = _Args()
        args.attention_backend = "flashinfer"

        with self.assertRaisesRegex(ValueError, "only Triton and FA3"):
            handle_dvr_speculative_decoding(args)

    def test_dvr_defaults_unspecified_attention_backend_to_triton(self):
        args = _Args()
        args.attention_backend = None

        handle_dvr_defaults(args)

        self.assertEqual(args.attention_backend, "triton")

    def test_dvr_validates_effective_phase_attention_backends(self):
        args = _Args()
        args.attention_backend = "flashinfer"
        args.prefill_attention_backend = "triton"
        args.decode_attention_backend = "fa3"

        handle_dvr_speculative_decoding(args)

        args.decode_attention_backend = "flashinfer"
        with self.assertRaisesRegex(ValueError, "effective decode backend"):
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
            "sglang.srt.speculative.dvr_server_args._is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "streaming sessions"):
                handle_dvr_speculative_decoding(args)

    def test_dvr_gdn_rejects_int8_recurrent_checkpoints(self):
        args = _Args()
        args.enable_int8_mamba_checkpoint = True

        with patch(
            "sglang.srt.speculative.dvr_server_args._is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "exact recurrent checkpoints"):
                handle_dvr_speculative_decoding(args)

    def test_plain_transformer_dvr_ignores_gdn_only_options(self):
        args = _Args()
        args.enable_streaming_session = True
        args.enable_int8_mamba_checkpoint = True
        args.enable_page_major_kv_layout = True
        args.enable_linear_replayssm = True

        with patch(
            "sglang.srt.speculative.dvr_server_args._is_dvr_gated_linear_state_model",
            return_value=False,
        ):
            handle_dvr_speculative_decoding(args)

        self.assertTrue(args.enable_mixed_chunk)

    def test_dvr_gdn_disables_mixed_chunk(self):
        args = _Args()
        with patch(
            "sglang.srt.speculative.dvr_server_args._is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            handle_dvr_speculative_decoding(args)

        self.assertFalse(args.enable_mixed_chunk)

    def test_dvr_enables_deterministic_target_execution(self):
        args = _Args()
        with patch(
            "sglang.srt.speculative.dvr_server_args._is_dvr_gated_linear_state_model",
            return_value=False,
        ):
            handle_dvr_defaults(args)

        self.assertTrue(args.enable_deterministic_inference)

    def test_dvr_preserves_draft_allreduce_fusion_request(self):
        for backend, force_disabled in (("trtllm", False), (None, True)):
            with self.subTest(backend=backend, force_disabled=force_disabled):
                args = _Args()
                args.flashinfer_allreduce_fusion_backend = backend
                args.enforce_disable_flashinfer_allreduce_fusion = force_disabled
                with patch(
                    "sglang.srt.speculative.dvr_server_args."
                    "_is_dvr_gated_linear_state_model",
                    return_value=False,
                ):
                    handle_dvr_defaults(args)

                self.assertEqual(
                    args._dvr_draft_flashinfer_allreduce_fusion,
                    (backend, force_disabled),
                )

    def test_dvr_defaults_normalize_algorithm_before_generic_spec_handling(self):
        args = _Args()
        args.speculative_algorithm = DVR_SPECULATIVE_ALGORITHM.lower()

        with patch(
            "sglang.srt.speculative.dvr_server_args._is_dvr_gated_linear_state_model",
            return_value=False,
        ):
            handle_dvr_defaults(args)

        self.assertEqual(args.speculative_algorithm, DVR_SPECULATIVE_ALGORITHM)

    def test_non_gdn_dvr_does_not_inherit_fla_limits(self):
        args = _Args()
        args.page_size = 32
        args.speculative_num_draft_tokens = 65
        args.speculative_num_steps = 64

        with patch(
            "sglang.srt.speculative.dvr_server_args._is_dvr_gated_linear_state_model",
            return_value=False,
        ):
            handle_dvr_defaults(args)
            handle_dvr_speculative_decoding(args)

        self.assertEqual(args.page_size, 32)
        self.assertEqual(args.speculative_num_draft_tokens, 65)


if __name__ == "__main__":
    unittest.main()
