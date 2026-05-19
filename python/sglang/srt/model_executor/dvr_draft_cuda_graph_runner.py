from __future__ import annotations

from contextlib import contextmanager

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


@contextmanager
def _dvr_draft_decode_graph_capture(model_runner):
    original_spec_algorithm = model_runner.spec_algorithm
    original_init_cuda_graph_state = model_runner.attn_backend.init_cuda_graph_state

    def skip_init_cuda_graph_state(*args, **kwargs):
        return None

    # The target model's main runner has already initialized attention backend
    # graph buffers for the larger TARGET_VERIFY graph. Capture the DVR self
    # draft graph as an ordinary decode graph without reinitializing those
    # shared backend buffers.
    model_runner.spec_algorithm = SpeculativeAlgorithm.NONE
    model_runner.attn_backend.init_cuda_graph_state = skip_init_cuda_graph_state
    try:
        yield
    finally:
        model_runner.attn_backend.init_cuda_graph_state = original_init_cuda_graph_state
        model_runner.spec_algorithm = original_spec_algorithm


class DVRDraftDecodeCudaGraphRunner:
    """CUDA graph runner for DVR self-draft decode.

    DVR's normal target-model graph runner captures TARGET_VERIFY graphs. The
    self-draft path still runs ordinary one-token DECODE steps, so it needs a
    separate decode graph runner instead of reusing the target-verify graph.
    """

    def __init__(self, dvr_worker):
        self.dvr_worker = dvr_worker
        model_runner = dvr_worker.model_runner
        with _dvr_draft_decode_graph_capture(model_runner):
            self.runner = CudaGraphRunner(model_runner)

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        return self.runner.can_run(forward_batch)

    def replay(self, forward_batch: ForwardBatch):
        return self.runner.replay(forward_batch)
