"""Server-argument rules for decode-verify-rollback speculative decoding."""

from __future__ import annotations

import logging
from typing import Optional

from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE

logger = logging.getLogger(__name__)

DVR_SPECULATIVE_ALGORITHM = "DECODE_VERIFY_ROLLBACK"
DVR_EAGLE_SPECULATIVE_ALGORITHM = "DECODE_VERIFY_ROLLBACK_EAGLE"
DVR_SPECULATIVE_ALGORITHMS = {
    DVR_SPECULATIVE_ALGORITHM,
    DVR_EAGLE_SPECULATIVE_ALGORITHM,
}


def is_dvr_enabled(server_args) -> bool:
    return server_args.speculative_algorithm in DVR_SPECULATIVE_ALGORITHMS


def is_dvr_self_draft_enabled(server_args) -> bool:
    return server_args.speculative_algorithm == DVR_SPECULATIVE_ALGORITHM


def is_dvr_eagle_enabled(server_args) -> bool:
    return server_args.speculative_algorithm == DVR_EAGLE_SPECULATIVE_ALGORITHM


def get_dvr_linear_speculative_state_extension_factory(model_runner):
    """Return DVR's optional linear-state cache factory for hybrid GDN models."""

    if not is_dvr_enabled(model_runner.server_args):
        return None
    if model_runner.hybrid_gdn_config is None:
        return None

    from sglang.srt.layers.attention.linear.dvr_gdn_state import (
        create_dvr_gdn_speculative_state_extension,
    )

    return create_dvr_gdn_speculative_state_extension


def handle_dvr_defaults(server_args):
    if not is_dvr_enabled(server_args):
        return

    normalized_page_size = _normalize_dvr_chunk_page_size(server_args.page_size)
    if normalized_page_size != server_args.page_size:
        logger.warning(
            "DVR requires page_size to be no larger "
            "than and divide FLA_CHUNK_SIZE=%s. Setting --page-size %s "
            "instead of %s.",
            FLA_CHUNK_SIZE,
            normalized_page_size,
            server_args.page_size,
        )
        server_args.page_size = normalized_page_size

    if not _is_dvr_gated_linear_state_model(server_args):
        return

    if server_args.page_size != FLA_CHUNK_SIZE:
        logger.warning(
            "DVR for gated linear-state models requires page_size to match "
            "FLA_CHUNK_SIZE=%s so radix prefixes and chunkwise verify "
            "checkpoints share the same boundary. Setting --page-size %s.",
            FLA_CHUNK_SIZE,
            FLA_CHUNK_SIZE,
        )
        server_args.page_size = FLA_CHUNK_SIZE

    if server_args.mamba_scheduler_strategy != "extra_buffer":
        logger.warning(
            "DVR for gated linear-state models requires mamba extra_buffer "
            "state tracking. Setting --mamba-scheduler-strategy extra_buffer."
        )
        server_args.mamba_scheduler_strategy = "extra_buffer"

    if server_args.mamba_track_interval != FLA_CHUNK_SIZE:
        logger.warning(
            "DVR for gated linear-state models requires mamba_track_interval "
            "to match FLA_CHUNK_SIZE=%s. Larger intervals may still be "
            "chunk-size multiples, but the current extra_buffer prefill path "
            "keeps only one tracked checkpoint and can miss the first "
            "prefill's last chunk boundary. Setting --mamba-track-interval %s.",
            FLA_CHUNK_SIZE,
            FLA_CHUNK_SIZE,
        )
        server_args.mamba_track_interval = FLA_CHUNK_SIZE

    if server_args.mamba_ssm_dtype != "float32":
        logger.warning(
            "DVR for gated linear-state models requires fp32 recurrent "
            "states. Setting --mamba-ssm-dtype float32."
        )
        server_args.mamba_ssm_dtype = "float32"


def handle_dvr_speculative_decoding(server_args):
    if not is_dvr_enabled(server_args):
        return

    if not server_args.device.startswith("cuda"):
        raise ValueError("DVR currently only supports CUDA device.")
    if server_args.enable_dp_attention:
        raise ValueError("DVR currently does not support DP attention.")
    if server_args.disaggregation_mode != "null":
        raise ValueError("DVR currently does not support disaggregation mode.")
    if (
        is_dvr_self_draft_enabled(server_args)
        and server_args.speculative_draft_model_path is not None
    ):
        raise ValueError("DVR self draft does not use a draft model path.")
    if (
        is_dvr_eagle_enabled(server_args)
        and server_args.speculative_draft_model_path is None
    ):
        raise ValueError(
            "DVR EAGLE requires setting --speculative-draft-model-path."
        )

    if server_args.speculative_num_draft_tokens is None:
        server_args.speculative_num_draft_tokens = 16
        logger.warning(
            "speculative_num_draft_tokens is set to 16 by default for DVR. "
            "You can override this by explicitly setting "
            "--speculative-num-draft-tokens."
        )

    _validate_dvr_page_size(server_args)

    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = server_args.speculative_num_draft_tokens - 1
    elif server_args.speculative_num_draft_tokens != server_args.speculative_num_steps + 1:
        logger.warning(
            "speculative_num_draft_tokens is adjusted to "
            "speculative_num_steps + 1 for DVR chain mode."
        )
        server_args.speculative_num_draft_tokens = server_args.speculative_num_steps + 1

    if server_args.speculative_num_draft_tokens < 2:
        raise ValueError(
            "DVR requires speculative_num_draft_tokens >= 2 because chain mode "
            "needs at least one draft step."
        )
    if server_args.speculative_num_draft_tokens > FLA_CHUNK_SIZE:
        raise ValueError(
            "DVR currently commits at most one FLA chunk boundary per verify. "
            f"Please set --speculative-num-draft-tokens <= {FLA_CHUNK_SIZE}."
        )

    if server_args.speculative_eagle_topk is None:
        server_args.speculative_eagle_topk = 1
    elif server_args.speculative_eagle_topk != 1:
        raise ValueError("DVR currently supports only chain mode with topk == 1.")

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for DVR. You can override "
            "this by explicitly setting --max-running-requests."
        )

    if is_dvr_eagle_enabled(server_args):
        if not envs.SGLANG_ENABLE_SPEC_V2.get():
            server_args.disable_overlap_schedule = True
            logger.warning(
                "Non-overlap synchronous spec v2 is used for DVR EAGLE. Set "
                "SGLANG_ENABLE_SPEC_V2=True to use overlap scheduling."
            )
        else:
            server_args.disable_overlap_schedule = False
            logger.warning("Overlap spec v2 is enabled for DVR EAGLE.")
    elif envs.SGLANG_ENABLE_SPEC_V2.get():
        server_args.disable_overlap_schedule = False
        logger.warning(
            "Spec v2 is enabled for DVR and overlap schedule is turned on."
        )
    else:
        server_args.disable_overlap_schedule = True
        logger.warning(
            "Overlap scheduler is disabled for DVR. Set "
            "SGLANG_ENABLE_SPEC_V2=True to use the experimental DVR spec v2 path."
        )
    server_args.enable_mixed_chunk = False
    logger.warning("Mixed chunked prefill is disabled for DVR.")


def _is_dvr_gated_linear_state_model(server_args):
    from sglang.srt.configs import (
        JetNemotronConfig,
        JetVLMConfig,
        Qwen3_5Config,
        Qwen3_5MoeConfig,
        Qwen3NextConfig,
    )

    config = server_args.get_model_config().hf_config.get_text_config()
    return isinstance(
        config,
        Qwen3NextConfig
        | Qwen3_5Config
        | Qwen3_5MoeConfig
        | JetNemotronConfig
        | JetVLMConfig,
    )


def _normalize_dvr_chunk_page_size(page_size: Optional[int]) -> Optional[int]:
    if page_size in (None, 1):
        return page_size
    if page_size <= 0:
        return page_size

    aligned = 1 << (min(page_size, FLA_CHUNK_SIZE).bit_length() - 1)
    while FLA_CHUNK_SIZE % aligned != 0:
        aligned //= 2
    return aligned


def _validate_dvr_page_size(server_args):
    if server_args.page_size == 1:
        return
    if (
        server_args.page_size <= FLA_CHUNK_SIZE
        and FLA_CHUNK_SIZE % server_args.page_size == 0
    ):
        return
    raise ValueError(
        "DVR page_size > 1 requires page_size to be no larger than and divide "
        f"FLA_CHUNK_SIZE={FLA_CHUNK_SIZE}."
    )
