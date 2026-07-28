#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH is required." >&2
  exit 2
fi

SERVER_MODE="${SERVER_MODE:-${DVR_MODE:-self}}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
PAGE_SIZE="${PAGE_SIZE:-64}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
LINEAR_ATTN_BACKEND="${LINEAR_ATTN_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
RANDOM_SEED="${RANDOM_SEED:-2026}"

case "${ATTENTION_BACKEND}" in
  triton|fa3|flashinfer) ;;
  *)
    echo "ATTENTION_BACKEND must be triton, fa3, or flashinfer." >&2
    exit 2
    ;;
esac

spec_args=()
case "${SERVER_MODE}" in
  normal)
    ;;
  deterministic)
    spec_args+=(--enable-deterministic-inference)
    ;;
  self)
    SPECULATIVE_ALGORITHM="DECODE_VERIFY_ROLLBACK"
    DRAFT_TOKENS="${DRAFT_TOKENS:-16}"
    spec_args+=(
      --enable-deterministic-inference
      --speculative-algorithm "${SPECULATIVE_ALGORITHM}"
      --speculative-num-draft-tokens "${DRAFT_TOKENS}"
    )
    ;;
  eagle)
    SPECULATIVE_ALGORITHM="DECODE_VERIFY_ROLLBACK_EAGLE"
    DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${MODEL_PATH}}"
    DRAFT_TOKENS="${DRAFT_TOKENS:-2}"
    spec_args+=(
      --enable-deterministic-inference
      --speculative-algorithm "${SPECULATIVE_ALGORITHM}"
      --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
      --speculative-num-draft-tokens "${DRAFT_TOKENS}"
    )
    ;;
  *)
    echo "SERVER_MODE must be normal, deterministic, self, or eagle." >&2
    exit 2
    ;;
esac

if [[ "${ATTENTION_BACKEND}" == "flashinfer" && "${SERVER_MODE}" != "normal" ]]; then
  if [[ "${DISABLE_RADIX_CACHE:-0}" != "1" ]]; then
    echo "Disabling Radix cache for deterministic FlashInfer attention." >&2
    DISABLE_RADIX_CACHE=1
  fi
fi

if [[ "${SERVER_MODE}" == "self" || "${SERVER_MODE}" == "eagle" ]]; then
  if ((DRAFT_TOKENS < 2)); then
    echo "DRAFT_TOKENS must be at least 2." >&2
    exit 2
  fi
  DRAFT_STEPS="${DRAFT_STEPS:-$((DRAFT_TOKENS - 1))}"
  spec_args+=(
    --speculative-num-steps "${DRAFT_STEPS}"
    --speculative-eagle-topk 1
  )
fi

args=(
  --model-path "${MODEL_PATH}"
  --host "${SERVER_HOST}"
  --port "${PORT}"
  --tp-size "${TP_SIZE}"
  --page-size "${PAGE_SIZE}"
  --attention-backend "${ATTENTION_BACKEND}"
  --linear-attn-backend "${LINEAR_ATTN_BACKEND}"
  --sampling-backend "${SAMPLING_BACKEND}"
  --random-seed "${RANDOM_SEED}"
  "${spec_args[@]}"
)

if [[ -n "${MAX_RUNNING_REQUESTS:-}" ]]; then
  args+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")
fi
if [[ -n "${MAX_MAMBA_CACHE_SIZE:-}" ]]; then
  args+=(--max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}")
fi
if [[ -n "${MEM_FRACTION_STATIC:-}" ]]; then
  args+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
fi
if [[ -n "${CONTEXT_LENGTH:-}" ]]; then
  args+=(--context-length "${CONTEXT_LENGTH}")
fi
if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
  args+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi
if [[ -n "${CUDA_GRAPH_BS:-}" ]]; then
  read -r -a cuda_graph_bs <<<"${CUDA_GRAPH_BS//,/ }"
  args+=(--cuda-graph-bs-decode "${cuda_graph_bs[@]}")
fi
if [[ -n "${CUDA_GRAPH_MAX_BS_DECODE:-}" ]]; then
  args+=(--cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}")
fi
if [[ "${DISABLE_OVERLAP:-0}" == "1" ]]; then
  args+=(--disable-overlap-schedule)
fi
if [[ "${DISABLE_RADIX_CACHE:-0}" == "1" ]]; then
  args+=(--disable-radix-cache)
fi
if [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]]; then
  args+=(--disable-custom-all-reduce)
fi

cmd=(python -m sglang.launch_server "${args[@]}" "$@")
overlap_status="enabled"
radix_status="enabled"
if [[ "${DISABLE_OVERLAP:-0}" == "1" ]]; then
  overlap_status="disabled"
fi
if [[ "${DISABLE_RADIX_CACHE:-0}" == "1" ]]; then
  radix_status="disabled"
fi
printf 'Starting mode=%s model=%s tp=%s backend=%s overlap=%s radix=%s\n' \
  "${SERVER_MODE}" "${MODEL_PATH}" "${TP_SIZE}" "${ATTENTION_BACKEND}" \
  "${overlap_status}" "${radix_status}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
