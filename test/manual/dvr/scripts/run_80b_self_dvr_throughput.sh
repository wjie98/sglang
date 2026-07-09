#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${MODEL_PATH:-/mnt/data/hwj/Qwen3-Next-80B-A3B-Instruct}"
SHAREGPT_DATASET="${SHAREGPT_DATASET:-/mnt/data/hwj/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json}"
LONGBENCH_CUSTOM_DATASET="${LONGBENCH_CUSTOM_DATASET:-${DVR_REPO_ROOT}/../dvr-v6-batch5-validation/longbench_16x1400_custom.jsonl}"
PORT="${PORT:-30180}"
BASE_URL="http://127.0.0.1:${PORT}"
RESULT_ROOT="${RESULT_ROOT:-${DVR_REPO_ROOT}/../dvr-fixed-validation/latest-run/80b-self-dvr-throughput}"
SERVER_PID=""

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/results"

cleanup() {
  stop_process_group "${SERVER_PID}"
}
trap cleanup EXIT

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file is missing: ${path}" >&2
    exit 1
  fi
}

run_bench() {
  local label="$1"
  local dataset="$2"
  local dataset_name="$3"
  local max_concurrency="$4"
  local return_logprob="$5"
  local output_file="${RESULT_ROOT}/results/${label}.jsonl"
  local output_log="${RESULT_ROOT}/results/${label}.log"
  local return_arg=()

  if [[ "${return_logprob}" == "true" ]]; then
    return_arg=(--return-logprob)
  fi

  echo "==> Running ${label}"
  conda_python -m sglang.bench_serving \
    --backend sglang \
    --base-url "${BASE_URL}" \
    --dataset-name "${dataset_name}" \
    --dataset-path "${dataset}" \
    --tokenizer "${MODEL_PATH}" \
    --num-prompts 16 \
    --sharegpt-output-len 1024 \
    --request-rate inf \
    --max-concurrency "${max_concurrency}" \
    --disable-tqdm \
    --disable-stream \
    --seed 2026 \
    --output-file "${output_file}" \
    "${return_arg[@]}" \
    2>&1 | tee "${output_log}"
}

run_one_mode() {
  local label="$1"
  local spec_v2="$2"
  shift 2
  local server_log="${RESULT_ROOT}/logs/80b_self_${label}_server.log"
  local overlap_args=()
  if [[ "${spec_v2}" == "0" ]]; then
    overlap_args=(--disable-overlap-schedule)
  fi

  echo "==> Starting 80B self-DVR ${label} server on ${BASE_URL}"
  setsid env \
    SGLANG_RETURN_ORIGINAL_LOGPROB=True \
    PYTHONPATH="${PYTHONPATH}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --tp-size 4 \
      --context-length 8192 \
      --max-total-tokens 6144 \
      --mem-fraction-static 0.9 \
      --max-running-requests 4 \
      --max-mamba-cache-size 16 \
      --page-size 64 \
      --attention-backend triton \
      --linear-attn-backend triton \
      --sampling-backend pytorch \
      --enable-deterministic-inference \
      --speculative-algorithm DECODE_VERIFY_ROLLBACK \
      --speculative-num-draft-tokens 16 \
      --speculative-num-steps 15 \
      --cuda-graph-bs 1 2 3 4 \
      --cuda-graph-max-bs 4 \
      "${overlap_args[@]}" \
      "$@" \
      --skip-server-warmup \
      >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  wait_for_server "${BASE_URL}" 600 "${SERVER_PID}" "${server_log}"

  run_bench "80b_${label}_sharegpt_logprob_false" "${SHAREGPT_DATASET}" "sharegpt" 3 false
  run_bench "80b_${label}_sharegpt_logprob_true" "${SHAREGPT_DATASET}" "sharegpt" 3 true
  run_bench "80b_${label}_longbench_logprob_false" "${LONGBENCH_CUSTOM_DATASET}" "custom" 2 false
  run_bench "80b_${label}_longbench_logprob_true" "${LONGBENCH_CUSTOM_DATASET}" "custom" 2 true

  stop_process_group "${SERVER_PID}"
  SERVER_PID=""
}

summarize_results() {
  RESULT_ROOT="${RESULT_ROOT}" conda_python - <<'PY'
import glob
import json
import os

base = os.environ["RESULT_ROOT"]
for path in sorted(glob.glob(os.path.join(base, "results", "80b_*.jsonl"))):
    with open(path) as f:
        rows = [line for line in f if line.strip()]
    if not rows:
        continue
    row = json.loads(rows[-1])
    print(
        "{} out={:.2f} accept={:.2f} completed={} duration={:.2f}".format(
            os.path.basename(path),
            row.get("output_throughput"),
            row.get("accept_length"),
            row.get("completed"),
            row.get("duration"),
        )
    )
PY
}

require_file "${MODEL_PATH}/config.json"
require_file "${SHAREGPT_DATASET}"
require_file "${LONGBENCH_CUSTOM_DATASET}"

# Fixed reproduced口径:
# - v1: compatibility scheduler, overlap disabled.
# - v2: spec-v2 worker with overlap enabled.
# - max_mamba_cache_size must stay 16; omitting it lowers effective concurrency.
run_one_mode "v1" "0"
run_one_mode "v2" "1"
summarize_results
