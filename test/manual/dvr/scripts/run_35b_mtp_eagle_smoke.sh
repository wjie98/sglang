#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${MODEL_PATH:-/mnt/data/hwj/Qwen3.5-35B-A3B}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${MODEL_PATH}}"
PORT="${PORT:-30135}"
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

  echo "==> Starting ${label} DVR-EAGLE server on ${BASE_URL}"
  setsid env \
    SGLANG_ENABLE_SPEC_V2="${spec_v2}" \
    SGLANG_RETURN_ORIGINAL_LOGPROB=True \
    PYTHONPATH="${PYTHONPATH}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --tp-size 4 \
      --speculative-algorithm DECODE_VERIFY_ROLLBACK_EAGLE \
      --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
      --speculative-num-draft-tokens 4 \
      --speculative-num-steps 3 \
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
      --skip-server-warmup \
      >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  wait_for_server "${BASE_URL}" 600 "${SERVER_PID}" "${server_log}"

  echo "==> Running ${label} returned-logprob KL smoke"
  conda_python test/manual/dvr/test_dvr_eagle_acceptance.py \
    --base-url "${BASE_URL}" \
    --prompt-token-lengths 63,64,65 \
    --max-new 4,16,65 \
    --cache-mode flush-each \
    --check-kl \
    --ignore-eos \
    --seed 2032 \
    2>&1 | tee "${kl_log}"
  grep -q '"kl_failed": 0' "${kl_log}"

  echo "==> Running ${label} no-return-logprob smoke"
  conda_python test/manual/dvr/test_dvr_eagle_acceptance.py \
    --base-url "${BASE_URL}" \
    --prompt-token-lengths 63,64,65 \
    --max-new 4,16,65 \
    --cache-mode flush-each \
    --no-return-logprob \
    --ignore-eos \
    --seed 2032 \
    2>&1 | tee "${no_logprob_log}"

  stop_process_group "${SERVER_PID}"
  SERVER_PID=""
}

# DVR-EAGLE uses v2 semantics in both modes: sync when disabled, overlap when enabled.
run_one_mode "sync_v2" "0"
run_one_mode "overlap_v2" "1"
