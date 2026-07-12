"""Server-argument rules for decode-verify-rollback speculative decoding."""

from __future__ import annotations

import logging

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


def handle_dvr_defaults(server_args):
    if not is_dvr_enabled(server_args):
        return

    # Deterministic target prefill/verify disables custom all-reduce later in
    # ServerArgs. Preserve the user's original choice for provisional draft
    # graphs without exposing another CLI option.
    server_args._dvr_enable_draft_custom_all_reduce = (
        not server_args.disable_custom_all_reduce
    )

    normalized_page_size = server_args.page_size
    if normalized_page_size not in (None, 1) and normalized_page_size > 0:
        normalized_page_size = 1 << (
            min(normalized_page_size, FLA_CHUNK_SIZE).bit_length() - 1
        )
        while FLA_CHUNK_SIZE % normalized_page_size != 0:
            normalized_page_size //= 2
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
            "FLA_CHUNK_SIZE=%s so attention pages and chunkwise verify "
            "checkpoints share the same boundary. Setting --page-size %s.",
            FLA_CHUNK_SIZE,
            FLA_CHUNK_SIZE,
        )
        server_args.page_size = FLA_CHUNK_SIZE

    if server_args.mamba_radix_cache_strategy != "extra_buffer":
        logger.warning(
            "DVR for gated linear-state models requires mamba extra_buffer "
            "state tracking. Setting --mamba-radix-cache-strategy extra_buffer."
        )
        server_args.mamba_radix_cache_strategy = "extra_buffer"

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
    if server_args.enable_pdmux:
        raise ValueError("DVR currently does not support PDMux attention backends.")
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

    if server_args.page_size != 1 and not (
        server_args.page_size <= FLA_CHUNK_SIZE
        and FLA_CHUNK_SIZE % server_args.page_size == 0
    ):
        raise ValueError(
            "DVR page_size > 1 requires page_size to be no larger than and divide "
            f"FLA_CHUNK_SIZE={FLA_CHUNK_SIZE}."
        )

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

    _ensure_dvr_self_draft_cuda_graph_coverage(server_args)

    if is_dvr_eagle_enabled(server_args):
        if server_args.disable_overlap_schedule:
            logger.warning("Synchronous spec v2 is used for DVR EAGLE.")
        else:
            logger.warning("Overlap spec v2 is enabled for DVR EAGLE.")
    elif not server_args.disable_overlap_schedule:
        logger.warning("Spec v2 is enabled for DVR and overlap schedule is on.")
    else:
        logger.warning(
            "Overlap scheduler is disabled for DVR, so self-draft uses the "
            "synchronous DVR worker path. Omit --disable-overlap-schedule to "
            "use overlap scheduling."
        )
    server_args.enable_mixed_chunk = False
    logger.warning("Mixed chunked prefill is disabled for DVR.")


def _ensure_dvr_self_draft_cuda_graph_coverage(server_args):
    """Keep GDN self-draft on the existing CUDA graph decode contract.

    DVR self-draft for gated linear-state models must use the dedicated draft
    decode graph for normal prompts.  With CUDA graph padding enabled, SGLang
    only needs a captured max batch size; with padding disabled, every exact
    batch size must be captured.  Reuse those existing semantics instead of
    adding a DVR-specific server argument.
    """

    if not is_dvr_self_draft_enabled(server_args):
        return
    if not hasattr(server_args, "get_model_config") or not (
        _is_dvr_gated_linear_state_model(server_args)
    ):
        return

    if getattr(server_args, "disable_cuda_graph", False) or getattr(
        server_args, "disable_draft_cuda_graph", False
    ):
        raise ValueError(
            "DVR self-draft for gated linear-state models requires draft CUDA "
            "graphs. Remove --disable-cuda-graph/--disable-draft-cuda-graph "
            "or use a non-self-draft DVR mode."
        )

    cuda_graph_bs = getattr(server_args, "cuda_graph_bs", None)
    max_running_requests = getattr(server_args, "max_running_requests", None)
    if max_running_requests is None:
        return

    max_running_requests = int(max_running_requests)
    if max_running_requests <= 0:
        return

    if cuda_graph_bs is None:
        cuda_graph_max_bs = getattr(server_args, "cuda_graph_max_bs", None)
        if (
            cuda_graph_max_bs is not None
            and int(cuda_graph_max_bs) < max_running_requests
        ):
            server_args.cuda_graph_max_bs = max_running_requests
        return

    graph_bs = {int(bs) for bs in cuda_graph_bs if int(bs) > 0}
    if not graph_bs:
        return

    if getattr(server_args, "disable_cuda_graph_padding", False):
        graph_bs.update(range(1, max_running_requests + 1))
    elif max(graph_bs) < max_running_requests:
        graph_bs.add(max_running_requests)
    else:
        return

    server_args.cuda_graph_bs = sorted(graph_bs)
    server_args.cuda_graph_max_bs = max(server_args.cuda_graph_bs)


def _is_dvr_gated_linear_state_model(server_args):
    from sglang.srt.configs import (
        InternS2PreviewConfig,
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
        | InternS2PreviewConfig
        | JetNemotronConfig
        | JetVLMConfig,
    )
