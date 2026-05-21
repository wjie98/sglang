import unittest

from sglang.srt.speculative.dvr_server_args import (
    DVR_SPECULATIVE_ALGORITHM,
    handle_dvr_speculative_decoding,
)


class _Args:
    speculative_algorithm = DVR_SPECULATIVE_ALGORITHM
    device = "cuda"
    enable_dp_attention = False
    disaggregation_mode = "null"
    speculative_draft_model_path = None
    speculative_num_draft_tokens = 1
    speculative_num_steps = None
    page_size = 64
    speculative_eagle_topk = None
    max_running_requests = 4
    disable_overlap_schedule = False
    enable_mixed_chunk = True


class TestDVRServerArgs(unittest.TestCase):
    def test_dvr_rejects_zero_step_chain_mode(self):
        args = _Args()
        with self.assertRaisesRegex(ValueError, "speculative_num_draft_tokens >= 2"):
            handle_dvr_speculative_decoding(args)


if __name__ == "__main__":
    unittest.main()
