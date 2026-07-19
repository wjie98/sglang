#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${MODEL_PATH:-/mnt/data/hwj/Qwen3.5-0.8B}"
PORT="${PORT:-30124}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.50}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
PAGE_SIZE="${PAGE_SIZE:-64}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
LINEAR_ATTN_BACKEND="${LINEAR_ATTN_BACKEND:-triton}"
DISABLE_RADIX_CACHE="$(resolve_radix_setting "${ATTENTION_BACKEND}" "${DISABLE_RADIX_CACHE:-auto}")"
RUN_V1="${RUN_V1:-1}"
RUN_V2="${RUN_V2:-1}"
DRAFT_TOKENS="${DRAFT_TOKENS:-16}"
DRAFT_STEPS="${DRAFT_STEPS:-$((DRAFT_TOKENS - 1))}"
RANDOM_SEED="${RANDOM_SEED:-2026}"
BASE_URL="http://127.0.0.1:${PORT}"
RESULT_ROOT="${RESULT_ROOT:-${DVR_REPO_ROOT}/../dvr-fixed-validation/latest-run/0p8b-self-dvr-kl}"
SERVER_PID=""

require_precompiled_deep_gemm
if ((DRAFT_TOKENS < 2)) || ((DRAFT_STEPS + 1 != DRAFT_TOKENS)); then
  echo "DVR chain mode requires DRAFT_TOKENS >= 2 and DRAFT_STEPS + 1 == DRAFT_TOKENS." >&2
  exit 1
fi

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/results"
write_run_metadata "${RESULT_ROOT}"
append_run_config "${RESULT_ROOT}" \
  "script=$(basename "$0")" "model=${MODEL_PATH}" "tp=${TP_SIZE}" \
  "context_length=${CONTEXT_LENGTH}" \
  "page_size=${PAGE_SIZE}" "attention_backend=${ATTENTION_BACKEND}" \
  "linear_attn_backend=${LINEAR_ATTN_BACKEND}" \
  "disable_radix_cache=${DISABLE_RADIX_CACHE}" \
  "draft_tokens=${DRAFT_TOKENS}" "draft_steps=${DRAFT_STEPS}" \
  "random_seed=${RANDOM_SEED}"

cleanup() {
  stop_process_group "${SERVER_PID}"
}
trap cleanup EXIT

run_one_mode() {
  local label="$1"
  local spec_v2="$2"
  local server_log="${RESULT_ROOT}/logs/${label}_server.log"
  local client_log="${RESULT_ROOT}/results/${label}_kl.log"
  local overlap_args=()
  local radix_args=()
  if [[ "${spec_v2}" == "0" ]]; then
    overlap_args=(--disable-overlap-schedule)
  fi
  if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
    radix_args=(--disable-radix-cache)
  fi

  echo "==> Starting ${label} server on ${BASE_URL}"
  setsid env \
    SGLANG_RETURN_ORIGINAL_LOGPROB=True \
    PYTHONPATH="${PYTHONPATH}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --tp-size "${TP_SIZE}" \
      --context-length "${CONTEXT_LENGTH}" \
      --speculative-algorithm DECODE_VERIFY_ROLLBACK \
      --speculative-num-draft-tokens "${DRAFT_TOKENS}" \
      --speculative-num-steps "${DRAFT_STEPS}" \
      --page-size "${PAGE_SIZE}" \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --attention-backend "${ATTENTION_BACKEND}" \
      --linear-attn-backend "${LINEAR_ATTN_BACKEND}" \
      --sampling-backend pytorch \
      --random-seed "${RANDOM_SEED}" \
      --enable-deterministic-inference \
      --cuda-graph-bs 1 2 4 \
      --cuda-graph-max-bs-decode 4 \
      --max-running-requests 4 \
      "${radix_args[@]}" \
      "${overlap_args[@]}" \
      --skip-server-warmup \
      >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  wait_for_server "${BASE_URL}" 300 "${SERVER_PID}" "${server_log}"
  assert_server_capacity "${server_log}" 4
  assert_server_config \
    "${server_log}" "${ATTENTION_BACKEND}" "${PAGE_SIZE}" "${spec_v2}" "${DISABLE_RADIX_CACHE}"
  assert_dvr_graphs "${server_log}" self "${DRAFT_TOKENS}" 4

  echo "==> Running ${label} KL and boundary smoke"
  conda_python test/manual/dvr/local/clients/dvr_batch_kl.py \
    --base-url "${BASE_URL}" \
    --request-modes concurrent,batch \
    --prompt-token-lengths 1 \
    --max-new 2,8 \
    --limit-cases 4 \
    --concurrent-workers 2 \
    --ignore-eos \
    2>&1 | tee "${client_log}"

  conda_python test/manual/dvr/local/clients/dvr_batch_kl.py \
    --base-url "${BASE_URL}" \
    --request-modes concurrent,batch \
    --prompt-token-lengths 2,63,64,65 \
    --max-new 1,8,16,17,63,64,65 \
    --limit-cases 12 \
    --concurrent-workers 4 \
    --ignore-eos \
    2>&1 | tee -a "${client_log}"

  echo "==> Running ${label} cross-chunk KL cases"
  conda_python test/manual/dvr/local/clients/dvr_batch_kl.py \
    --base-url "${BASE_URL}" \
    --request-modes concurrent,batch \
    --prompt-token-lengths 63,64,65 \
    --max-new 63,64,65,128 \
    --limit-cases 24 \
    --concurrent-workers 3 \
    --ignore-eos \
    2>&1 | tee -a "${client_log}"

  echo "==> Running ${label} interleaved boundary-ownership KL case"
  conda_python test/manual/dvr/local/clients/dvr_batch_kl.py \
    --base-url "${BASE_URL}" \
    --request-modes concurrent \
    --prompt-token-lengths 65,129 \
    --max-new 128 \
    --limit-cases 2 \
    --concurrent-workers 2 \
    --concurrent-stagger-ms 50 \
    --ignore-eos \
    2>&1 | tee -a "${client_log}"

  if [[ "${DISABLE_RADIX_CACHE}" == "0" ]]; then
    echo "==> Running ${label} nearest-radix-checkpoint replay KL case"
    conda_python test/manual/dvr/local/clients/dvr_batch_kl.py \
      --base-url "${BASE_URL}" \
      --request-modes concurrent \
      --prompt-token-lengths 65 \
      --reuse-generated-prefix-tokens 128 \
      --min-cached-tokens 64 \
      --max-new 128 \
      --limit-cases 1 \
      --concurrent-workers 1 \
      --ignore-eos \
      2>&1 | tee -a "${client_log}"

    echo "==> Running ${label} radix ownership and boundary lifecycle cases"
    local lifecycle_args=()
    if [[ "${spec_v2}" == "1" ]]; then
      # Upstream does not support overlap + speculative decoding + grammar.
      lifecycle_args=(--skip-grammar)
    fi
    conda_python test/manual/dvr/local/clients/dvr_radix_lifecycle.py \
      --base-url "${BASE_URL}" \
      "${lifecycle_args[@]}" \
      2>&1 | tee -a "${client_log}"
  fi

  echo "==> Running ${label} 512-token KL cases"
  conda_python test/manual/dvr/local/clients/dvr_batch_kl.py \
    --base-url "${BASE_URL}" \
    --request-modes concurrent,batch \
    --prompt-token-lengths 65 \
    --max-new 512 \
    --limit-cases 2 \
    --concurrent-workers 2 \
    --ignore-eos \
    2>&1 | tee -a "${client_log}"

  grep -q "ALL_OK True" "${client_log}"
  stop_process_group "${SERVER_PID}"
  SERVER_PID=""
}

if [[ "${RUN_V1}" == "1" ]]; then
  run_one_mode "spec_v1" "0"
fi
if [[ "${RUN_V2}" == "1" ]]; then
  run_one_mode "spec_v2" "1"
fi
