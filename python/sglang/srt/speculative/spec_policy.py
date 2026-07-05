from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


_DVR_SELF_DRAFT = "DECODE_VERIFY_ROLLBACK"
_DVR_EAGLE = "DECODE_VERIFY_ROLLBACK_EAGLE"
_DVR_ALGOS = {_DVR_SELF_DRAFT, _DVR_EAGLE}
_EAGLE_ALGOS = {"EAGLE", "EAGLE3", _DVR_EAGLE, "FROZEN_KV_MTP"}


def _algo_name(algorithm: Any) -> str:
    return str(getattr(algorithm, "name", algorithm) or "NONE").upper()


def _is_builtin_algorithm(algorithm: Any) -> bool:
    return isinstance(algorithm, Enum)


def _custom_bool(algorithm: Any, method_name: str, default: bool = False) -> bool:
    if _is_builtin_algorithm(algorithm):
        return default
    method = getattr(algorithm, method_name, None)
    if method is None:
        return default
    return bool(method())


@dataclass(frozen=True)
class SpecAlgorithmPolicy:
    """Centralized behavior traits for speculative algorithms.

    SGLang's scheduler, model runner, CUDA graph runner, and cache builder need
    a few cross-cutting algorithm traits.  Keep those traits here so generic
    subsystems do not grow DVR-specific branches and new algorithms can be
    integrated by adding a policy instead of editing every call site.
    """

    algorithm: Any

    @property
    def name(self) -> str:
        return _algo_name(self.algorithm)

    def is_dvr(self) -> bool:
        return self.name in _DVR_ALGOS

    def is_dvr_self_draft(self) -> bool:
        return self.name == _DVR_SELF_DRAFT

    def is_dvr_eagle(self) -> bool:
        return self.name == _DVR_EAGLE

    def is_eagle(self) -> bool:
        return self.name in _EAGLE_ALGOS or _custom_bool(
            self.algorithm, "is_eagle"
        )

    def is_frozen_kv_mtp(self) -> bool:
        return self.name == "FROZEN_KV_MTP" or _custom_bool(
            self.algorithm, "is_frozen_kv_mtp"
        )

    def is_standalone(self) -> bool:
        return self.name == "STANDALONE" or _custom_bool(
            self.algorithm, "is_standalone"
        )

    def is_dflash(self) -> bool:
        return self.name == "DFLASH" or _custom_bool(self.algorithm, "is_dflash")

    def supports_spec_v2(self) -> bool:
        if not _is_builtin_algorithm(self.algorithm):
            method = getattr(self.algorithm, "supports_spec_v2", None)
            return bool(method()) if method is not None else False
        return (
            (self.is_eagle() and not self.is_frozen_kv_mtp())
            or self.is_standalone()
            or self.is_dvr()
        )

    def uses_spec_v2(self, enable_overlap: bool) -> bool:
        """Return whether this algorithm uses the spec-v2 worker/schema.

        DVR self-draft keeps v5's spec-v1 path when overlap is disabled. DVR-EAGLE
        is implemented on EAGLE's v2 worker only; disabling overlap selects the
        synchronous v2 path, not a separate EAGLE-v1 implementation.
        """

        if not _is_builtin_algorithm(self.algorithm):
            method = getattr(self.algorithm, "uses_spec_v2", None)
            return (
                bool(method(enable_overlap))
                if method is not None
                else self.supports_spec_v2()
            )
        if self.is_dvr_eagle():
            return True
        if self.is_dvr_self_draft():
            return enable_overlap
        return self.supports_spec_v2()

    def prepare_cuda_graph_verify_input(self, verify_input: Any) -> Any:
        if not _is_builtin_algorithm(self.algorithm):
            method = getattr(self.algorithm, "prepare_cuda_graph_verify_input", None)
            return method(verify_input) if method is not None else verify_input

        if self.is_dvr_eagle():
            from sglang.srt.speculative.dvr_worker import DVREagleVerifyInput

            return DVREagleVerifyInput.from_eagle_verify_input(verify_input)
        if self.is_dvr_self_draft():
            from sglang.srt.speculative.dvr_worker import DVRSelfDraftVerifyInput

            return DVRSelfDraftVerifyInput.from_eagle_verify_input(verify_input)
        return verify_input

    def target_verify_capture_hidden_mode(
        self,
        default_mode: Any,
        *,
        null_for_standalone: bool = False,
    ) -> Any:
        if not _is_builtin_algorithm(self.algorithm):
            method = getattr(self.algorithm, "target_verify_capture_hidden_mode", None)
            return (
                method(default_mode, null_for_standalone=null_for_standalone)
                if method is not None
                else default_mode
            )

        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        if self.is_dvr_self_draft():
            return CaptureHiddenMode.NULL
        if null_for_standalone and self.is_standalone():
            return CaptureHiddenMode.NULL
        return default_mode

    def uses_eagle_style_target_verify_input(self) -> bool:
        if not _is_builtin_algorithm(self.algorithm):
            method = getattr(
                self.algorithm, "uses_eagle_style_target_verify_input", None
            )
            return bool(method()) if method is not None else False
        return self.is_eagle() or self.is_standalone() or self.is_dvr()

    def target_verify_graph_bs_uses_token_count(self) -> bool:
        if not _is_builtin_algorithm(self.algorithm):
            method = getattr(
                self.algorithm, "target_verify_graph_bs_uses_token_count", None
            )
            return bool(method()) if method is not None else False
        return (
            self.is_eagle()
            or self.is_standalone()
            or self.is_dflash()
            or self.is_dvr()
        )

    def create_target_verify_cuda_graph_input(
        self,
        *,
        custom_mask: Any,
        spec_steps: int,
        topk: Optional[int],
        draft_token_num: int,
        default_capture_hidden_mode: Any,
        null_for_standalone: bool = False,
    ) -> Optional[Any]:
        if not _is_builtin_algorithm(self.algorithm):
            method = getattr(
                self.algorithm, "create_target_verify_cuda_graph_input", None
            )
            if method is not None:
                return method(
                    custom_mask=custom_mask,
                    spec_steps=spec_steps,
                    topk=topk,
                    draft_token_num=draft_token_num,
                    default_capture_hidden_mode=default_capture_hidden_mode,
                    null_for_standalone=null_for_standalone,
                )
            return None

        if not self.uses_eagle_style_target_verify_input():
            return None

        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        spec_info = EagleVerifyInput(
            draft_token=None,
            custom_mask=custom_mask,
            positions=None,
            retrieve_index=None,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_cum_len=None,
            spec_steps=spec_steps,
            topk=topk,
            draft_token_num=draft_token_num,
            capture_hidden_mode=self.target_verify_capture_hidden_mode(
                default_capture_hidden_mode,
                null_for_standalone=null_for_standalone,
            ),
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )
        return self.prepare_cuda_graph_verify_input(spec_info)

    def uses_draft_decode_custom_all_reduce(self) -> bool:
        if not _is_builtin_algorithm(self.algorithm):
            method = getattr(self.algorithm, "uses_draft_decode_custom_all_reduce", None)
            return bool(method()) if method is not None else False
        return self.is_dvr()

    def needs_mamba_radix_snapshot_for_spec_v2(self) -> bool:
        if not _is_builtin_algorithm(self.algorithm):
            method = getattr(
                self.algorithm, "needs_mamba_radix_snapshot_for_spec_v2", None
            )
            return bool(method()) if method is not None else False
        return self.is_dvr()

    def linear_speculative_state_extension_factory(self, model_runner: Any):
        if not _is_builtin_algorithm(self.algorithm):
            method = getattr(
                self.algorithm, "linear_speculative_state_extension_factory", None
            )
            return method(model_runner) if method is not None else None
        if not self.is_dvr():
            return None
        if model_runner.hybrid_gdn_config is None:
            return None

        from sglang.srt.layers.attention.linear.dvr_gdn_state import (
            create_dvr_gdn_speculative_state_extension,
        )

        return create_dvr_gdn_speculative_state_extension

    def uses_target_kv_pool_for_draft(self) -> bool:
        if not _is_builtin_algorithm(self.algorithm):
            method = getattr(self.algorithm, "uses_target_kv_pool_for_draft", None)
            return bool(method()) if method is not None else False
        return self.is_dvr_self_draft()

    def proposed_draft_tokens_per_verify(
        self,
        *,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
    ) -> int:
        """Return strict proposed-draft count per verify step.

        ``accept_rate`` should exclude the target bonus token.  Chain-style
        verifiers propose ``speculative_num_steps`` drafts per verify even when
        their internal tree/window tensor has ``speculative_num_draft_tokens``
        rows.  Keep this in the algorithm policy so tokenizer-side metrics do
        not need to understand each spec worker's layout.
        """

        if not _is_builtin_algorithm(self.algorithm):
            method = getattr(
                self.algorithm, "proposed_draft_tokens_per_verify", None
            )
            if method is not None:
                return int(
                    method(
                        speculative_num_steps=speculative_num_steps,
                        speculative_num_draft_tokens=speculative_num_draft_tokens,
                    )
                )

        if self.is_eagle() or self.is_dvr() or self.is_standalone():
            return max(0, int(speculative_num_steps))
        return max(0, int(speculative_num_draft_tokens) - 1)


def get_spec_algorithm_policy(algorithm: Any) -> SpecAlgorithmPolicy:
    return SpecAlgorithmPolicy(algorithm)


def create_target_verify_cuda_graph_input_for_runner(
    model_runner: Any,
    *,
    custom_mask: Any,
    spec_steps: Optional[int] = None,
    topk: Optional[int] = None,
    draft_token_num: Optional[int] = None,
    default_capture_hidden_mode: Optional[Any] = None,
    null_for_standalone: bool = False,
) -> Optional[Any]:
    """Create target-verify graph metadata from a model runner.

    CUDA graph capture and replay_prepare both need the same EAGLE-style verify
    metadata.  Keep the argument normalization here so model_runner and
    cuda_graph_runner do not duplicate algorithm-specific field construction.
    """

    if default_capture_hidden_mode is None:
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        default_capture_hidden_mode = CaptureHiddenMode.FULL

    server_args = model_runner.server_args
    return get_spec_algorithm_policy(
        model_runner.spec_algorithm
    ).create_target_verify_cuda_graph_input(
        custom_mask=custom_mask,
        spec_steps=(
            spec_steps if spec_steps is not None else server_args.speculative_num_steps
        ),
        topk=topk if topk is not None else server_args.speculative_eagle_topk,
        draft_token_num=(
            draft_token_num
            if draft_token_num is not None
            else server_args.speculative_num_draft_tokens
        ),
        default_capture_hidden_mode=default_capture_hidden_mode,
        null_for_standalone=null_for_standalone,
    )
