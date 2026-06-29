from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type, Union

import torch

from sglang.srt.speculative.spec_registry import (
    CustomSpecAlgo,
    ServerArgsValidator,
    WorkerFactory,
)
from sglang.srt.speculative.spec_registry import get_spec as _get_registered_spec
from sglang.srt.speculative.spec_registry import (
    register_algorithm as _register_algorithm,
)

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.ngram_worker import NGRAMWorker


class SpeculativeAlgorithm(Enum):
    """Builtin speculative decoding algorithms. Plugin-registered ones are
    ``CustomSpecAlgo`` instances; ``from_string`` returns either type, and
    both expose the same ``is_*()`` / ``create_worker`` interface so callers
    dispatch uniformly without isinstance checks.
    """

    DFLASH = auto()
    DECODE_VERIFY_ROLLBACK = auto()
    DECODE_VERIFY_ROLLBACK_EAGLE = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    FROZEN_KV_MTP = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()

    @classmethod
    def from_string(
        cls, name: Optional[str]
    ) -> Union[SpeculativeAlgorithm, CustomSpecAlgo]:
        if name is None:
            return cls.NONE
        upper = name.upper()
        try:
            return cls[upper]
        except KeyError:
            pass
        spec = _get_registered_spec(upper)
        if spec is not None:
            return spec
        raise ValueError(f"Unknown speculative algorithm name: {name}")

    @classmethod
    def register(
        cls,
        name: str,
        *,
        supports_overlap: bool = False,
        validate_server_args: Optional[ServerArgsValidator] = None,
        spec_class: Type[CustomSpecAlgo] = CustomSpecAlgo,
    ) -> Callable[[WorkerFactory], WorkerFactory]:
        """Decorator to register a plugin speculative algorithm. The factory
        takes ``server_args`` and returns the worker class. Pass a
        ``CustomSpecAlgo`` subclass via ``spec_class`` to override any
        ``is_*()`` / ``create_worker`` method.

        Example:
            @SpeculativeAlgorithm.register("MY_SPEC", supports_overlap=False)
            def _factory(server_args):
                return MySpecWorker
        """
        return _register_algorithm(
            name,
            supports_overlap=supports_overlap,
            validate_server_args=validate_server_args,
            spec_class=spec_class,
        )

    def is_some(self) -> bool:
        return self != SpeculativeAlgorithm.NONE

    def is_none(self) -> bool:
        return self == SpeculativeAlgorithm.NONE

    def is_speculative(self) -> bool:
        return self != SpeculativeAlgorithm.NONE

    def is_eagle(self) -> bool:
        # FIXME(kpham_sgl): Remove FROZEN_KV_MTP here once we
        # have established support for it in the scheduler.
        return self in (
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE,
            SpeculativeAlgorithm.FROZEN_KV_MTP,
        )

    def is_eagle3(self) -> bool:
        return self == SpeculativeAlgorithm.EAGLE3

    def is_frozen_kv_mtp(self) -> bool:
        return self == SpeculativeAlgorithm.FROZEN_KV_MTP

    def is_dflash(self) -> bool:
        return self == SpeculativeAlgorithm.DFLASH

    def is_decode_verify_rollback(self) -> bool:
        return self in (
            SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
            SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE,
        )

    def is_decode_verify_rollback_self_draft(self) -> bool:
        return self == SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK

    def is_decode_verify_rollback_eagle(self) -> bool:
        return self == SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE

    def is_standalone(self) -> bool:
        return self == SpeculativeAlgorithm.STANDALONE

    def is_ngram(self) -> bool:
        return self == SpeculativeAlgorithm.NGRAM

    def supports_target_verify_for_draft(self) -> bool:
        return self.is_dflash()

    def create_future_map(
        self,
        device: torch.device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
    ) -> FutureMap:
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(device, self, req_to_token_pool, needs_cpu_seq_lens)

    def build_disagg_draft_input(
        self,
        batch: ScheduleBatch,
        server_args: ServerArgs,
        last_tokens_tensor: torch.Tensor,
        future_map: FutureMap,
    ) -> Optional[SpecInput]:
        if self.is_eagle():
            from sglang.srt.speculative.eagle_disaggregation import (
                build_eagle_disagg_draft_input,
            )

            return build_eagle_disagg_draft_input(
                batch, server_args, last_tokens_tensor, future_map
            )
        return None

    def supports_spec_v2(self) -> bool:
        return (
            (self.is_eagle() and not self.is_frozen_kv_mtp())
            or self.is_standalone()
            or self.is_decode_verify_rollback()
        )

    def uses_spec_v2(self, enable_overlap: bool) -> bool:
        """Return whether the scheduler should use the v2 worker/schema."""

        if self.is_decode_verify_rollback_eagle():
            return True
        if self.is_decode_verify_rollback_self_draft():
            return enable_overlap
        return self.supports_spec_v2()

    def prepare_cuda_graph_verify_input(self, verify_input: SpecInput) -> SpecInput:
        """Let algorithms specialize the generic EAGLE verify graph input."""

        if self.is_decode_verify_rollback_eagle():
            from sglang.srt.speculative.dvr_worker import DVREagleVerifyInput

            return DVREagleVerifyInput.from_eagle_verify_input(verify_input)
        if self.is_decode_verify_rollback_self_draft():
            from sglang.srt.speculative.dvr_worker import DVRSelfDraftVerifyInput

            return DVRSelfDraftVerifyInput.from_eagle_verify_input(verify_input)
        return verify_input

    def target_verify_capture_hidden_mode(
        self,
        default_mode,
        *,
        null_for_standalone: bool = False,
    ):
        """Return the hidden-state capture mode for target-verify graph inputs.

        Most EAGLE-style algorithms need target hidden states for draft extend.
        DVR self-draft consumes only verified tokens, while standalone graph
        capture can use a cheaper NULL mode in the cuda graph runner path.
        """

        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        if self.is_decode_verify_rollback_self_draft():
            return CaptureHiddenMode.NULL
        if null_for_standalone and self.is_standalone():
            return CaptureHiddenMode.NULL
        return default_mode

    def uses_eagle_style_target_verify_input(self) -> bool:
        return (
            self.is_eagle()
            or self.is_standalone()
            or self.is_decode_verify_rollback()
        )

    def target_verify_graph_bs_uses_token_count(self) -> bool:
        return (
            self.is_eagle()
            or self.is_standalone()
            or self.is_dflash()
            or self.is_decode_verify_rollback()
        )

    def create_target_verify_cuda_graph_input(
        self,
        *,
        custom_mask,
        spec_steps: int,
        topk: Optional[int],
        draft_token_num: int,
        default_capture_hidden_mode,
        null_for_standalone: bool = False,
    ) -> Optional[SpecInput]:
        """Build verify graph metadata for algorithms sharing EAGLE layout."""

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
        return self.is_decode_verify_rollback()

    def needs_mamba_radix_snapshot_for_spec_v2(self) -> bool:
        """Whether spec-v2 scheduling must preserve mamba radix checkpoints."""

        return self.is_decode_verify_rollback()

    def linear_speculative_state_extension_factory(self, model_runner):
        """Return an optional linear-state cache extension factory."""

        if not self.is_decode_verify_rollback():
            return None
        if model_runner.hybrid_gdn_config is None:
            return None

        from sglang.srt.layers.attention.linear.dvr_gdn_state import (
            create_dvr_gdn_speculative_state_extension,
        )

        return create_dvr_gdn_speculative_state_extension

    def uses_target_kv_pool_for_draft(self) -> bool:
        """Whether the draft path reuses the target KV pool."""

        return self.is_decode_verify_rollback_self_draft()

    def get_num_tokens_per_bs_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        # FIXME: Remove this after the forward mode refactor. Target verify is
        # essentially a fixed sequence length prefill/extend with full cuda
        # graph support. We can use it for target verify, or we can use it for
        # other cases which is not target verify but fixed length prefill.
        # Here, we expose this interface to allow the other use cases.
        return num_draft_tokens

    def create_worker(
        self, server_args: ServerArgs
    ) -> Optional[Union[Type[BaseSpecWorker], Type[TpModelWorker], Type[NGRAMWorker]]]:
        assert (
            not self.is_none()
        ), "Cannot create worker for NONE speculative algorithm."

        enable_overlap = not server_args.disable_overlap_schedule

        if self.is_dflash():
            if enable_overlap:
                raise ValueError(
                    "DFLASH does not support overlap scheduling (spec v2)."
                )
            from sglang.srt.speculative.dflash_worker import DFlashWorker

            return DFlashWorker

        if self.is_frozen_kv_mtp():
            if enable_overlap:
                raise ValueError(
                    "FROZEN_KV_MTP does not support spec v2. Disable overlap "
                    "scheduling to use FrozenKVMTPWorker."
                )

            from sglang.srt.speculative.frozen_kv_mtp_worker import (
                FrozenKVMTPWorker,
            )

            return FrozenKVMTPWorker

        if self.is_decode_verify_rollback_eagle():
            from sglang.srt.speculative.dvr_eagle_worker import (
                DecodeVerifyRollbackEagleWorkerV2,
            )

            return DecodeVerifyRollbackEagleWorkerV2

        if self.is_decode_verify_rollback_self_draft():
            if enable_overlap:
                from sglang.srt.speculative.dvr_worker_v2 import (
                    DecodeVerifyRollbackWorkerV2,
                )

                return DecodeVerifyRollbackWorkerV2

            from sglang.srt.speculative.dvr_worker import DecodeVerifyRollbackWorker

            return DecodeVerifyRollbackWorker

        # EAGLE / EAGLE3 / STANDALONE / MULTI_LAYER always use the V2 worker,
        # even with overlap disabled (scheduler drives it synchronously).
        if self.is_eagle() and server_args.enable_multi_layer_eagle:
            from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                MultiLayerEagleWorkerV2,
            )

            return MultiLayerEagleWorkerV2

        elif self.is_eagle():
            from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

            return EAGLEWorkerV2
        elif self.is_standalone():
            from sglang.srt.speculative.standalone_worker_v2 import (
                StandaloneWorkerV2,
            )

            return StandaloneWorkerV2
        elif self.is_ngram():
            if enable_overlap:
                raise ValueError(
                    f"Speculative algorithm {self.name} does not support overlap worker creation."
                )

            from sglang.srt.speculative.ngram_worker import NGRAMWorker

            return NGRAMWorker

        raise ValueError("Unreachable code path in create_worker.")


class SpecInputType(IntEnum):
    # NOTE: introduce this to distinguish the SpecInput types of multiple algorithms when asserting in attention backends.
    # If all algorithms can share the same datastrucutre of draft_input and verify_input, consider simplify it
    EAGLE_DRAFT = auto()
    EAGLE_DRAFT_EXTEND = auto()
    EAGLE_VERIFY = auto()
    FROZEN_KV_MTP_DRAFT = auto()
    FROZEN_KV_MTP_DRAFT_EXTEND = auto()
    FROZEN_KV_MTP_VERIFY = auto()
    DFLASH_DRAFT = auto()
    DFLASH_VERIFY = auto()
    NGRAM_VERIFY = auto()


class SpecInput(ABC):
    def __init__(self, spec_input_type: SpecInputType):
        self.spec_input_type = spec_input_type

    # Cross-algorithm phase guards. Used by attention backends and
    # ForwardBatch padding logic to dispatch on phase without hardcoding the
    # specific algo class (EAGLE / FROZEN_KV_MTP / DFLASH / NGRAM each have
    # their own draft / verify SpecInput subclasses).
    def is_draft_input(self) -> bool:
        return self.spec_input_type in {
            SpecInputType.EAGLE_DRAFT,
            SpecInputType.EAGLE_DRAFT_EXTEND,
            SpecInputType.FROZEN_KV_MTP_DRAFT,
            SpecInputType.FROZEN_KV_MTP_DRAFT_EXTEND,
            SpecInputType.DFLASH_DRAFT,
        }

    def is_verify_input(self) -> bool:
        return self.spec_input_type in {
            SpecInputType.EAGLE_VERIFY,
            SpecInputType.FROZEN_KV_MTP_VERIFY,
            SpecInputType.DFLASH_VERIFY,
            SpecInputType.NGRAM_VERIFY,
        }

    @abstractmethod
    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        pass

    def get_spec_adjusted_global_num_tokens(
        self, batch: ScheduleBatch
    ) -> Tuple[List[int], List[int]]:
        c1, c2 = self.get_spec_adjust_token_coefficient()
        global_num_tokens = [x * c1 for x in batch.global_num_tokens]
        global_num_tokens_for_logprob = [
            x * c2 for x in batch.global_num_tokens_for_logprob
        ]
        return global_num_tokens, global_num_tokens_for_logprob

    def prepare_cuda_graph_replay_buffers(self, graph_runner, raw_num_token: int):
        """Optionally adjust graph-resident replay buffers before replay."""

        return None

    def cuda_graph_metadata_context(
        self,
        *,
        model_runner,
        attn_backend,
        forward_mode,
        fallback_custom_mask=None,
    ):
        """Scoped metadata patching for algorithms with graph-only invariants."""

        return nullcontext()
