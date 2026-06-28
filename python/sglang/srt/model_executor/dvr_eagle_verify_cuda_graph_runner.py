from __future__ import annotations

from contextlib import contextmanager

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@contextmanager
def dvr_eagle_target_verify_cuda_graph_context(model_runner):
    """Capture/replay DVR-EAGLE target verify with DVR target semantics.

    EAGLE still owns the draft model and sampling metadata, while DVR needs the
    target-verify graph to honor its GDN state-input window and padded-row
    rules.  The flag below lets DVR metadata helpers keep the captured verify
    graph causal and make padded rows respect num_token_non_padded/state-input
    dummy slots without changing the shared cuda graph runner.  Strict
    return-logprob and graph-disabled fallback paths may still rebuild a suffix
    EXTEND oracle in the DVR worker; the no-logprob hot path consumes this graph
    output directly.
    """

    saved_flag = getattr(model_runner, "enable_dvr_target_verify_cuda_graph", None)
    model_runner.enable_dvr_target_verify_cuda_graph = True
    try:
        yield
    finally:
        if saved_flag is None:
            try:
                delattr(model_runner, "enable_dvr_target_verify_cuda_graph")
            except AttributeError:
                pass
        else:
            model_runner.enable_dvr_target_verify_cuda_graph = saved_flag


class DVREagleTargetVerifyCudaGraphRunner:
    """CUDA graph runner for DVR-EAGLE target verify.

    The target worker's default graph is captured with generic EAGLE verifier
    metadata.  DVR-EAGLE needs a target verify graph whose padded rows and GDN
    state-input windows follow DVR verifier rules; draft and draft-extend graphs
    remain owned by the normal EAGLE draft worker.
    """

    def __init__(self, dvr_eagle_worker):
        self.dvr_eagle_worker = dvr_eagle_worker
        model_runner = dvr_eagle_worker.target_worker.model_runner
        with dvr_eagle_target_verify_cuda_graph_context(model_runner):
            self.runner = _DVREagleTargetVerifyCudaGraphRunner(model_runner)

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        with dvr_eagle_target_verify_cuda_graph_context(
            self.dvr_eagle_worker.target_worker.model_runner
        ):
            return self.runner.can_run(forward_batch)

    def replay_prepare(self, forward_batch: ForwardBatch, pp_proxy_tensors=None):
        with dvr_eagle_target_verify_cuda_graph_context(
            self.dvr_eagle_worker.target_worker.model_runner
        ):
            return self.runner.replay_prepare(
                forward_batch, pp_proxy_tensors=pp_proxy_tensors
            )

    def replay(self, forward_batch: ForwardBatch, pp_proxy_tensors=None):
        with dvr_eagle_target_verify_cuda_graph_context(
            self.dvr_eagle_worker.target_worker.model_runner
        ):
            return self.runner.replay(forward_batch, pp_proxy_tensors=pp_proxy_tensors)

    def __getattr__(self, name):
        return getattr(self.runner, name)


class _DVREagleTargetVerifyCudaGraphRunner(CudaGraphRunner):
    def get_spec_info(self, num_tokens: int):
        spec_info = super().get_spec_info(num_tokens)
        if spec_info is None:
            return None

        from sglang.srt.speculative.dvr_worker import DVREagleVerifyInput

        return DVREagleVerifyInput.from_eagle_verify_input(spec_info)
