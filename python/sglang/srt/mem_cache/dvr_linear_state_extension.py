from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


@dataclass(frozen=True)
class LinearSpeculativeStateExtensionConfig:
    """Inputs for optional linear-state speculative cache extensions.

    MambaPool owns generic recurrent-state allocation. Model families that need
    extra speculative state-input storage can provide a factory instead of
    making memory_pool import model-specific or DVR-specific code.
    """

    num_layers: int
    spec_state_size: int
    num_draft_tokens: int
    state_shape: Any
    conv_dtype: torch.dtype
    ssm_dtype: torch.dtype
    device: str


@dataclass(frozen=True)
class LinearSpeculativeStateExtension:
    """Optional extra cache layout for speculative linear-state verification."""

    intermediate_ssm_tokens: int
    intermediate_conv_tokens: int
    state_input_cache: Optional[Any] = None
    log_label: str = "linear_state_input_cache"

    def mem_usage_bytes(self) -> int:
        if self.state_input_cache is None:
            return 0
        if hasattr(self.state_input_cache, "mem_usage_bytes"):
            return self.state_input_cache.mem_usage_bytes()
        if isinstance(self.state_input_cache, torch.Tensor):
            return (
                self.state_input_cache.numel() * self.state_input_cache.element_size()
            )
        return 0


LinearSpeculativeStateExtensionFactory = Callable[
    [LinearSpeculativeStateExtensionConfig],
    Optional[LinearSpeculativeStateExtension],
]
