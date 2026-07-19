#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH is required." >&2
  exit 2
fi

DVR_MODE="${DVR_MODE:-self}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
PAGE_SIZE="${PAGE_SIZE:-64}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
LINEAR_ATTN_BACKEND="${LINEAR_ATTN_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
RANDOM_SEED="${RANDOM_SEED:-2026}"

case "${ATTENTION_BACKEND}" in
  triton|fa3) ;;
  *)
    echo "ATTENTION_BACKEND must be triton or fa3." >&2
    exit 2
    ;;
esac

case "${DVR_MODE}" in
  self)
    SPECULATIVE_ALGORITHM="DECODE_VERIFY_ROLLBACK"
    DRAFT_TOKENS="${DRAFT_TOKENS:-16}"
    ;;
  eagle)
    SPECULATIVE_ALGORITHM="DECODE_VERIFY_ROLLBACK_EAGLE"
    DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${MODEL_PATH}}"
    DRAFT_TOKENS="${DRAFT_TOKENS:-2}"
    ;;
  *)
    echo "DVR_MODE must be self or eagle." >&2
    exit 2
    ;;
esac

if ((DRAFT_TOKENS < 2)); then
  echo "DRAFT_TOKENS must be at least 2." >&2
  exit 2
fi
DRAFT_STEPS="${DRAFT_STEPS:-$((DRAFT_TOKENS - 1))}"

args=(
  --model-path "${MODEL_PATH}"
  --host "${SERVER_HOST}"
  --port "${PORT}"
  --tp-size "${TP_SIZE}"
  --speculative-algorithm "${SPECULATIVE_ALGORITHM}"
  --speculative-num-draft-tokens "${DRAFT_TOKENS}"
  --speculative-num-steps "${DRAFT_STEPS}"
  --speculative-eagle-topk 1
  --page-size "${PAGE_SIZE}"
  --attention-backend "${ATTENTION_BACKEND}"
  --linear-attn-backend "${LINEAR_ATTN_BACKEND}"
  --sampling-backend "${SAMPLING_BACKEND}"
  --random-seed "${RANDOM_SEED}"
)

if [[ "${DVR_MODE}" == "eagle" ]]; then
  args+=(--speculative-draft-model-path "${DRAFT_MODEL_PATH}")
fi
if [[ -n "${MAX_RUNNING_REQUESTS:-}" ]]; then
  args+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")
fi
if [[ -n "${MAX_MAMBA_CACHE_SIZE:-}" ]]; then
  args+=(--max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}")
fi
if [[ -n "${MEM_FRACTION_STATIC:-}" ]]; then
  args+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
fi
if [[ "${DISABLE_OVERLAP:-0}" == "1" ]]; then
  args+=(--disable-overlap-schedule)
fi
if [[ "${DISABLE_RADIX_CACHE:-0}" == "1" ]]; then
  args+=(--disable-radix-cache)
fi

printf 'Starting DVR mode=%s algorithm=%s model=%s tp=%s backend=%s\n' \
  "${DVR_MODE}" "${SPECULATIVE_ALGORITHM}" "${MODEL_PATH}" "${TP_SIZE}" \
  "${ATTENTION_BACKEND}"
exec python -m sglang.launch_server "${args[@]}" "$@"
