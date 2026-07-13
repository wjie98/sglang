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
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_DVR="${RUN_DVR:-1}"
SHAREGPT_MAX_CONCURRENCY="${SHAREGPT_MAX_CONCURRENCY:-3}"
LONGBENCH_MAX_CONCURRENCY="${LONGBENCH_MAX_CONCURRENCY:-2}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-16}"
SERVER_PID=""
SERVER_MAX_CONCURRENCY="${SHAREGPT_MAX_CONCURRENCY}"
CUDA_GRAPH_BS=()

if ((LONGBENCH_MAX_CONCURRENCY > SERVER_MAX_CONCURRENCY)); then
  SERVER_MAX_CONCURRENCY="${LONGBENCH_MAX_CONCURRENCY}"
fi
for ((bs = 1; bs <= SERVER_MAX_CONCURRENCY; bs++)); do
  CUDA_GRAPH_BS+=("${bs}")
done

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

run_benchmark_matrix() {
  local label="$1"
  run_bench "80b_${label}_sharegpt_logprob_false" "${SHAREGPT_DATASET}" "sharegpt" "${SHAREGPT_MAX_CONCURRENCY}" false
  run_bench "80b_${label}_sharegpt_logprob_true" "${SHAREGPT_DATASET}" "sharegpt" "${SHAREGPT_MAX_CONCURRENCY}" true
  run_bench "80b_${label}_longbench_logprob_false" "${LONGBENCH_CUSTOM_DATASET}" "custom" "${LONGBENCH_MAX_CONCURRENCY}" false
  run_bench "80b_${label}_longbench_logprob_true" "${LONGBENCH_CUSTOM_DATASET}" "custom" "${LONGBENCH_MAX_CONCURRENCY}" true
}

run_baseline() {
  local server_log="${RESULT_ROOT}/logs/80b_baseline_server.log"

  echo "==> Starting 80B no-DVR baseline server on ${BASE_URL}"
  setsid env \
    PYTHONPATH="${PYTHONPATH}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --tp-size 4 \
      --context-length 8192 \
      --max-total-tokens 6144 \
      --mem-fraction-static 0.9 \
      --max-running-requests "${SERVER_MAX_CONCURRENCY}" \
      --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}" \
      --page-size 1 \
      --attention-backend triton \
      --linear-attn-backend triton \
      --sampling-backend pytorch \
      --cuda-graph-bs "${CUDA_GRAPH_BS[@]}" \
      --cuda-graph-max-bs "${SERVER_MAX_CONCURRENCY}" \
      --disable-overlap-schedule \
      --skip-server-warmup \
      >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  wait_for_server "${BASE_URL}" 600 "${SERVER_PID}" "${server_log}"
  assert_server_capacity "${server_log}" "${SERVER_MAX_CONCURRENCY}"
  run_benchmark_matrix "baseline"
  stop_process_group "${SERVER_PID}"
  SERVER_PID=""
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
      --max-running-requests "${SERVER_MAX_CONCURRENCY}" \
      --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}" \
      --page-size 64 \
      --attention-backend triton \
      --linear-attn-backend triton \
      --sampling-backend pytorch \
      --enable-deterministic-inference \
      --speculative-algorithm DECODE_VERIFY_ROLLBACK \
      --speculative-num-draft-tokens 16 \
      --speculative-num-steps 15 \
      --cuda-graph-bs "${CUDA_GRAPH_BS[@]}" \
      --cuda-graph-max-bs "${SERVER_MAX_CONCURRENCY}" \
      "${overlap_args[@]}" \
      "$@" \
      --skip-server-warmup \
      >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  wait_for_server "${BASE_URL}" 600 "${SERVER_PID}" "${server_log}"
  assert_server_capacity "${server_log}" "${SERVER_MAX_CONCURRENCY}"

  run_benchmark_matrix "${label}"

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
    accept_length = row.get("accept_length")
    accept_text = "n/a" if accept_length is None else f"{accept_length:.2f}"
    print(
        "{} out={:.2f} accept={} completed={} duration={:.2f}".format(
            os.path.basename(path),
            row.get("output_throughput"),
            accept_text,
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
# - baseline: normal no-DVR decode, overlap disabled.
# - v1: compatibility scheduler, overlap disabled.
# - v2: spec-v2 worker with overlap enabled.
# - server/graph concurrency is the maximum requested by either dataset.
# - max_mamba_cache_size must cover that concurrency without a runtime reduction.
if [[ "${RUN_BASELINE}" == "1" ]]; then
  run_baseline
fi
if [[ "${RUN_DVR}" == "1" ]]; then
  run_one_mode "v1" "0"
  run_one_mode "v2" "1"
fi
summarize_results
