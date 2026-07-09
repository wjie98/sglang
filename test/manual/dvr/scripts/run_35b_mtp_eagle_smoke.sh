#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${MODEL_PATH:-/mnt/data/hwj/Qwen3.5-35B-A3B}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${MODEL_PATH}}"
PORT="${PORT:-30135}"
SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS:-2}"
SPEC_STEPS="${SPEC_STEPS:-1}"
ACCEPT_TEMPERATURE="${ACCEPT_TEMPERATURE:-0.0}"
ACCEPT_TOP_P="${ACCEPT_TOP_P:-1.0}"
BASE_URL="http://127.0.0.1:${PORT}"
RESULT_ROOT="${RESULT_ROOT:-${DVR_REPO_ROOT}/../dvr-fixed-validation/latest-run/35b-mtp-eagle-smoke}"
SERVER_PID=""

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/results"

cleanup() {
  stop_process_group "${SERVER_PID}"
}
trap cleanup EXIT

run_one_mode() {
  local label="$1"
  local spec_v2="$2"
  local server_log="${RESULT_ROOT}/logs/${label}_server.log"
  local kl_log="${RESULT_ROOT}/results/${label}_return_logprob_true.log"
  local no_logprob_log="${RESULT_ROOT}/results/${label}_return_logprob_false.log"
  local prefix_cache_kl_log="${RESULT_ROOT}/results/${label}_prefix_cache_safety_return_logprob_true.log"
  local prefix_cache_no_logprob_log="${RESULT_ROOT}/results/${label}_prefix_cache_safety_return_logprob_false.log"
  local overlap_args=()
  if [[ "${spec_v2}" == "0" ]]; then
    overlap_args=(--disable-overlap-schedule)
  fi

  echo "==> Starting ${label} DVR-EAGLE server on ${BASE_URL}"
  setsid env \
    SGLANG_RETURN_ORIGINAL_LOGPROB=True \
    PYTHONPATH="${PYTHONPATH}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --tp-size 4 \
      --speculative-algorithm DECODE_VERIFY_ROLLBACK_EAGLE \
      --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
      --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}" \
      --speculative-num-steps "${SPEC_STEPS}" \
      --speculative-eagle-topk 1 \
      --page-size 1 \
      --context-length 4096 \
      --max-total-tokens 8192 \
      --mem-fraction-static 0.72 \
      --attention-backend triton \
      --linear-attn-backend triton \
      --sampling-backend pytorch \
      --enable-deterministic-inference \
      --cuda-graph-bs 1 2 4 \
      --cuda-graph-max-bs 4 \
      --max-running-requests 4 \
      "${overlap_args[@]}" \
      --skip-server-warmup \
      >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  wait_for_server "${BASE_URL}" 600 "${SERVER_PID}" "${server_log}"

  echo "==> Running ${label} returned-logprob KL smoke"
  # Acceptance is used here as a hidden/state consistency oracle.  Keep it
  # greedy; stochastic sampling can legitimately reject a correct top-1 MTP
  # draft when the target sample draws a different token.
  conda_python test/manual/dvr/test_dvr_eagle_acceptance.py \
    --base-url "${BASE_URL}" \
    --prompt-token-lengths 63,64,65 \
    --max-new 4,16,65 \
    --cache-mode flush-each \
    --check-kl \
    --min-accept-rate 0.99 \
    --temperature "${ACCEPT_TEMPERATURE}" \
    --top-p "${ACCEPT_TOP_P}" \
    --ignore-eos \
    --seed 2032 \
    2>&1 | tee "${kl_log}"
  grep -q '"kl_failed": 0' "${kl_log}"
  grep -q '"accept_failed": 0' "${kl_log}"

  echo "==> Running ${label} no-return-logprob smoke"
  conda_python test/manual/dvr/test_dvr_eagle_acceptance.py \
    --base-url "${BASE_URL}" \
    --prompt-token-lengths 63,64,65 \
    --max-new 4,16,65 \
    --cache-mode flush-each \
    --no-return-logprob \
    --min-accept-rate 0.99 \
    --temperature "${ACCEPT_TEMPERATURE}" \
    --top-p "${ACCEPT_TOP_P}" \
    --ignore-eos \
    --seed 2032 \
    2>&1 | tee "${no_logprob_log}"
  grep -q '"accept_failed": 0' "${no_logprob_log}"

  echo "==> Running ${label} prefix-cache safety boundary returned-logprob KL smoke"
  conda_python test/manual/dvr/test_dvr_eagle_acceptance.py \
    --base-url "${BASE_URL}" \
    --prompt-token-lengths 65 \
    --max-new 65 \
    --cache-mode warm-all \
    --check-kl \
    --min-accept-rate 0.99 \
    --temperature "${ACCEPT_TEMPERATURE}" \
    --top-p "${ACCEPT_TOP_P}" \
    --ignore-eos \
    --seed 3032 \
    2>&1 | tee "${prefix_cache_kl_log}"
  grep -q '"kl_failed": 0' "${prefix_cache_kl_log}"
  grep -q '"accept_failed": 0' "${prefix_cache_kl_log}"

  echo "==> Running ${label} prefix-cache safety boundary no-return-logprob smoke"
  conda_python test/manual/dvr/test_dvr_eagle_acceptance.py \
    --base-url "${BASE_URL}" \
    --prompt-token-lengths 65 \
    --max-new 65 \
    --cache-mode warm-all \
    --no-return-logprob \
    --min-accept-rate 0.99 \
    --temperature "${ACCEPT_TEMPERATURE}" \
    --top-p "${ACCEPT_TOP_P}" \
    --ignore-eos \
    --seed 3032 \
    2>&1 | tee "${prefix_cache_no_logprob_log}"
  grep -q '"accept_failed": 0' "${prefix_cache_no_logprob_log}"

  stop_process_group "${SERVER_PID}"
  SERVER_PID=""
}

# DVR-EAGLE uses v2 semantics in both modes: sync when disabled, overlap when enabled.
run_one_mode "sync_v2" "0"
run_one_mode "overlap_v2" "1"
