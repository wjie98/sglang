import unittest
from unittest.mock import patch

from sglang.srt.speculative.dvr_server_args import (
    DVR_EAGLE_SPECULATIVE_ALGORITHM,
    DVR_SPECULATIVE_ALGORITHM,
    handle_dvr_speculative_decoding,
)


class _Args:
    speculative_algorithm = DVR_SPECULATIVE_ALGORITHM
    device = "cuda"
    enable_dp_attention = False
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
    cuda_graph_bs = [1, 2]
    cuda_graph_max_bs = 2
    disable_radix_cache = False


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
            "_is_dvr_gated_linear_state_model_safe",
            return_value=True,
        ):
            handle_dvr_speculative_decoding(args)

        self.assertEqual(args.cuda_graph_bs, [1, 2, 4])
        self.assertEqual(args.cuda_graph_max_bs, 4)

    def test_dvr_self_draft_extends_exact_cuda_graph_bs_without_padding(self):
        args = _Args()
        args.disable_cuda_graph_padding = True
        args.cuda_graph_bs = [1, 4]
        args.cuda_graph_max_bs = 4

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model_safe",
            return_value=True,
        ):
            handle_dvr_speculative_decoding(args)

        self.assertEqual(args.cuda_graph_bs, [1, 2, 3, 4])
        self.assertEqual(args.cuda_graph_max_bs, 4)

    def test_dvr_self_draft_extends_cuda_graph_max_before_default_bs(self):
        args = _Args()
        args.cuda_graph_bs = None
        args.cuda_graph_max_bs = 2

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model_safe",
            return_value=True,
        ):
            handle_dvr_speculative_decoding(args)

        self.assertIsNone(args.cuda_graph_bs)
        self.assertEqual(args.cuda_graph_max_bs, 4)

    def test_dvr_self_draft_gdn_requires_draft_cuda_graph(self):
        args = _Args()
        args.disable_draft_cuda_graph = True

        with patch(
            "sglang.srt.speculative.dvr_server_args."
            "_is_dvr_gated_linear_state_model_safe",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "requires draft CUDA graphs"):
                handle_dvr_speculative_decoding(args)

    def test_dvr_eagle_disables_radix_cache(self):
        args = _Args()
        args.speculative_algorithm = DVR_EAGLE_SPECULATIVE_ALGORITHM
        args.speculative_draft_model_path = "draft"

        handle_dvr_speculative_decoding(args)

        self.assertTrue(args.disable_radix_cache)


if __name__ == "__main__":
    unittest.main()
