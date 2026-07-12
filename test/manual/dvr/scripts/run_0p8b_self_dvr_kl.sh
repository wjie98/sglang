#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${MODEL_PATH:-/mnt/data/hwj/Qwen3.5-0.8B}"
PORT="${PORT:-30124}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.50}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
BASE_URL="http://127.0.0.1:${PORT}"
RESULT_ROOT="${RESULT_ROOT:-${DVR_REPO_ROOT}/../dvr-fixed-validation/latest-run/0p8b-self-dvr-kl}"
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
  local client_log="${RESULT_ROOT}/results/${label}_kl.log"
  local overlap_args=()
  if [[ "${spec_v2}" == "0" ]]; then
    overlap_args=(--disable-overlap-schedule)
  fi

  echo "==> Starting ${label} server on ${BASE_URL}"
  setsid env \
    SGLANG_RETURN_ORIGINAL_LOGPROB=True \
    PYTHONPATH="${PYTHONPATH}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --speculative-algorithm DECODE_VERIFY_ROLLBACK \
      --speculative-num-draft-tokens 16 \
      --page-size 1 \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --attention-backend "${ATTENTION_BACKEND}" \
      --linear-attn-backend triton \
      --sampling-backend pytorch \
      --enable-deterministic-inference \
      --cuda-graph-bs 1 2 4 \
      --cuda-graph-max-bs 4 \
      --max-running-requests 8 \
      "${overlap_args[@]}" \
      --skip-server-warmup \
      >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  wait_for_server "${BASE_URL}" 300 "${SERVER_PID}" "${server_log}"

  echo "==> Running ${label} KL and boundary smoke"
  conda_python test/manual/dvr/test_dvr_batch_kl.py \
    --base-url "${BASE_URL}" \
    --request-modes concurrent,batch \
    --prompt-token-lengths 2,63,64,65 \
    --max-new 1,8,16,17,63,64,65 \
    --limit-cases 12 \
    --concurrent-workers 4 \
    --ignore-eos \
    2>&1 | tee "${client_log}"

  grep -q "ALL_OK True" "${client_log}"
  stop_process_group "${SERVER_PID}"
  SERVER_PID=""
}

run_one_mode "spec_v1" "0"
run_one_mode "spec_v2" "1"
