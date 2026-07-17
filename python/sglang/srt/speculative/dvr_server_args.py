"""Server-argument rules for decode-verify-rollback speculative decoding."""

from __future__ import annotations

import logging

from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.model_executor.cuda_graph_config import Backend

logger = logging.getLogger(__name__)

DVR_SPECULATIVE_ALGORITHM = "DECODE_VERIFY_ROLLBACK"
DVR_EAGLE_SPECULATIVE_ALGORITHM = "DECODE_VERIFY_ROLLBACK_EAGLE"
DVR_SPECULATIVE_ALGORITHMS = {
    DVR_SPECULATIVE_ALGORITHM,
    DVR_EAGLE_SPECULATIVE_ALGORITHM,
}
_DVR_FULL_ATTENTION_BACKENDS = {"triton", "fa3"}


def handle_dvr_defaults(server_args):
    algorithm = server_args.speculative_algorithm
    if algorithm is None:
        return
    algorithm = algorithm.upper()
    if algorithm not in DVR_SPECULATIVE_ALGORITHMS:
        return
    server_args.speculative_algorithm = algorithm

    # Deterministic target prefill/verify disables custom all-reduce later in
    # ServerArgs. Preserve the user's original choice for provisional draft
    # graphs without exposing another CLI option.
    server_args._dvr_enable_draft_custom_all_reduce = (
        not server_args.disable_custom_all_reduce
    )

    # DVR only permits provisional draft decode to be non-deterministic. Target
    # prefill and verify define the output contract and must use the ordinary
    # deterministic-inference configuration from server initialization onward.
    if not server_args.enable_deterministic_inference:
        logger.warning("Deterministic inference is enabled for DVR target execution.")
        server_args.enable_deterministic_inference = True

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
    if server_args.speculative_algorithm not in DVR_SPECULATIVE_ALGORITHMS:
        return

    if not server_args.device.startswith("cuda"):
        raise ValueError("DVR currently only supports CUDA device.")
    if server_args.enable_dp_attention:
        raise ValueError("DVR currently does not support DP attention.")
    if server_args.enable_pdmux:
        raise ValueError("DVR currently does not support PDMux attention backends.")
    if server_args.disaggregation_mode != "null":
        raise ValueError("DVR currently does not support disaggregation mode.")
    if server_args.pp_size != 1:
        raise ValueError("DVR currently supports only pp_size == 1.")
    if server_args.speculative_adaptive:
        raise ValueError("DVR does not support adaptive speculative decoding.")
    if (
        server_args.speculative_algorithm == DVR_SPECULATIVE_ALGORITHM
        and server_args.speculative_draft_model_path is not None
    ):
        raise ValueError("DVR self draft does not use a draft model path.")
    if (
        server_args.speculative_algorithm == DVR_EAGLE_SPECULATIVE_ALGORITHM
        and server_args.speculative_draft_model_path is None
    ):
        raise ValueError(
            "DVR EAGLE requires setting --speculative-draft-model-path."
        )
    if server_args.speculative_num_draft_tokens is None:
        server_args.speculative_num_draft_tokens = (
            2
            if server_args.speculative_algorithm == DVR_EAGLE_SPECULATIVE_ALGORITHM
            else 16
        )
        logger.warning(
            "speculative_num_draft_tokens is set to %s by default for DVR. "
            "You can override this by explicitly setting "
            "--speculative-num-draft-tokens.",
            server_args.speculative_num_draft_tokens,
        )

    uses_gated_linear_state = _is_dvr_gated_linear_state_model(server_args)
    if uses_gated_linear_state and server_args.enable_streaming_session:
        raise ValueError(
            "DVR gated linear-state models do not yet support streaming sessions."
        )
    if uses_gated_linear_state and server_args.enable_int8_mamba_checkpoint:
        raise ValueError(
            "DVR requires exact recurrent checkpoints and does not support "
            "--enable-int8-mamba-checkpoint."
        )
    for field in (
        "attention_backend",
        "prefill_attention_backend",
        "decode_attention_backend",
    ):
        backend = getattr(server_args, field, None)
        if backend is not None and backend not in _DVR_FULL_ATTENTION_BACKENDS:
            raise ValueError(
                "DVR currently supports only Triton and FA3 full-attention "
                f"backends, got --{field.replace('_', '-')} {backend}."
            )
    linear_prefill_backend = (
        server_args.linear_attn_prefill_backend or server_args.linear_attn_backend
    )
    if uses_gated_linear_state and linear_prefill_backend != "triton":
        raise ValueError(
            "DVR GDN verify requires --linear-attn-prefill-backend triton "
            "because the selected backend must export exact chunk-boundary states."
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
    if (
        server_args.speculative_num_draft_tokens > FLA_CHUNK_SIZE
        and uses_gated_linear_state
    ):
        raise ValueError(
            "DVR currently commits at most one FLA chunk boundary per verify. "
            f"Please set --speculative-num-draft-tokens <= {FLA_CHUNK_SIZE}."
        )

    if server_args.speculative_eagle_topk is None:
        server_args.speculative_eagle_topk = 1
    elif server_args.speculative_eagle_topk != 1:
        raise ValueError("DVR currently supports only chain mode with topk == 1.")
    if server_args.sampling_backend not in ("flashinfer", "pytorch"):
        raise ValueError(
            "DVR supports only the built-in flashinfer and pytorch sampling "
            "backends because target verification must reproduce the sampler's "
            "request distribution."
        )

    # Exact rejection sampling is the DVR default because it preserves more
    # useful MTP proposals under stochastic target sampling. Target-only EAGLE
    # sampling remains available as an explicit throughput/acceptance A/B.
    server_args.speculative_use_rejection_sampling = (
        envs.SGLANG_DVR_USE_REJECTION_SAMPLING.get()
    )
    if server_args.speculative_use_rejection_sampling:
        if (
            server_args.speculative_accept_threshold_single != 1.0
            or server_args.speculative_accept_threshold_acc != 1.0
        ):
            raise ValueError(
                "DVR rejection sampling does not use speculative acceptance "
                "thresholds; both thresholds must remain 1.0."
            )
        if (
            server_args.speculative_algorithm == DVR_EAGLE_SPECULATIVE_ALGORITHM
            and server_args.enable_multi_layer_eagle
        ):
            raise NotImplementedError(
                "DVR EAGLE rejection sampling does not support multi-layer EAGLE."
            )
    else:
        logger.warning(
            "DVR target-only sampling is enabled by "
            "SGLANG_DVR_USE_REJECTION_SAMPLING=0."
        )

    _ensure_dvr_self_draft_cuda_graph_coverage(server_args)

    if server_args.speculative_algorithm == DVR_EAGLE_SPECULATIVE_ALGORITHM:
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
    """Keep self-draft on the existing CUDA graph decode contract.

    DVR self-draft must use the dedicated fast decode graph. With CUDA graph
    padding enabled, SGLang only needs a captured max batch size; with padding
    disabled, every exact batch size must be captured. Reuse those existing
    semantics instead of adding a DVR-specific server argument.
    """

    if server_args.speculative_algorithm != DVR_SPECULATIVE_ALGORITHM:
        return
    if (
        server_args.cuda_graph_config.decode.backend == Backend.DISABLED
        or server_args.disable_draft_cuda_graph
    ):
        raise ValueError(
            "DVR self-draft requires draft CUDA "
            "graphs. Remove --disable-cuda-graph/--disable-draft-cuda-graph "
            "or use a non-self-draft DVR mode."
        )

    decode_graph = server_args.cuda_graph_config.decode
    cuda_graph_bs = decode_graph.bs
    max_running_requests = server_args.max_running_requests
    if max_running_requests is None:
        return

    max_running_requests = int(max_running_requests)
    if max_running_requests <= 0:
        return

    if cuda_graph_bs is None:
        cuda_graph_max_bs = decode_graph.max_bs
        if (
            cuda_graph_max_bs is not None
            and int(cuda_graph_max_bs) < max_running_requests
        ):
            decode_graph.max_bs = max_running_requests
        return

    graph_bs = {int(bs) for bs in cuda_graph_bs if int(bs) > 0}
    if not graph_bs:
        return

    if server_args.disable_cuda_graph_padding:
        graph_bs.update(range(1, max_running_requests + 1))
    elif max(graph_bs) < max_running_requests:
        graph_bs.add(max_running_requests)
    else:
        return

    decode_graph.bs = sorted(graph_bs)
    decode_graph.max_bs = max(decode_graph.bs)


def _is_dvr_gated_linear_state_model(server_args):
    from sglang.srt.configs import JetNemotronConfig, Qwen3NextConfig

    config = server_args.get_model_config().hf_config.get_text_config()
    # Qwen3.5/InternS2 text configs inherit Qwen3NextConfig; JetVLM unwraps
    # to JetNemotronConfig. Check the normalized text families, not wrappers.
    return isinstance(config, Qwen3NextConfig | JetNemotronConfig)
