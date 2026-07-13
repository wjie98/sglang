#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${MODEL_PATH:-/mnt/data/hwj/Qwen3.5-35B-A3B}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${MODEL_PATH}}"
SHAREGPT_DATASET="${SHAREGPT_DATASET:-/mnt/data/hwj/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json}"
PORT="${PORT:-30136}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_SELF="${RUN_SELF:-1}"
RUN_EAGLE="${RUN_EAGLE:-1}"
BASE_URL="http://127.0.0.1:${PORT}"
RESULT_ROOT="${RESULT_ROOT:-${DVR_REPO_ROOT}/../dvr-fixed-validation/latest-run/35b-dvr-throughput}"
SERVER_PID=""

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/results"

cleanup() {
  stop_process_group "${SERVER_PID}"
}
trap cleanup EXIT

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

start_server() {
  local label="$1"
  shift
  local server_log="${RESULT_ROOT}/logs/${label}_server.log"

  echo "==> Starting ${label} server on ${BASE_URL}"
  setsid env \
    SGLANG_RETURN_ORIGINAL_LOGPROB=True \
    PYTHONPATH="${PYTHONPATH}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --tp-size 4 \
      --context-length 4096 \
      --max-total-tokens 8192 \
      --mem-fraction-static 0.72 \
      --max-running-requests 4 \
      --max-mamba-cache-size 16 \
      --attention-backend triton \
      --linear-attn-backend triton \
      --sampling-backend pytorch \
      --cuda-graph-bs 1 2 3 4 \
      --cuda-graph-max-bs 4 \
      "$@" \
      --skip-server-warmup \
      >"${server_log}" 2>&1 &
  SERVER_PID="$!"
  wait_for_server "${BASE_URL}" 600 "${SERVER_PID}" "${server_log}"
}

stop_server() {
  stop_process_group "${SERVER_PID}"
  SERVER_PID=""
}

run_bench() {
  local label="$1"
  local return_logprob="$2"
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
    --dataset-name sharegpt \
    --dataset-path "${SHAREGPT_DATASET}" \
    --tokenizer "${MODEL_PATH}" \
    --num-prompts "${NUM_PROMPTS}" \
    --sharegpt-output-len "${OUTPUT_LEN}" \
    --request-rate inf \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --disable-tqdm \
    --disable-stream \
    --seed 2026 \
    --output-file "${output_file}" \
    "${return_arg[@]}" \
    2>&1 | tee "${output_log}"
}

run_benchmark_pair() {
  local label="$1"
  run_bench "35b_${label}_sharegpt_logprob_false" false
  run_bench "35b_${label}_sharegpt_logprob_true" true
}

run_self_kl() {
  local label="$1"
  local output_log="${RESULT_ROOT}/results/${label}_kl.log"
  conda_python test/manual/dvr/test_dvr_batch_kl.py \
    --base-url "${BASE_URL}" \
    --request-modes concurrent,batch \
    --prompt-token-lengths 63,64,65 \
    --max-new 16,65 \
    --limit-cases 6 \
    --concurrent-workers 3 \
    --ignore-eos \
    2>&1 | tee "${output_log}"
  grep -q "ALL_OK True" "${output_log}"
}

run_baseline() {
  start_server "35b_baseline" \
    --page-size 1 \
    --disable-overlap-schedule
  run_benchmark_pair baseline
  stop_server
}

run_self_mode() {
  local label="$1"
  local overlap_arg=()
  if [[ "${label}" == "self_v1" ]]; then
    overlap_arg=(--disable-overlap-schedule)
  fi
  start_server "35b_${label}" \
    --page-size 64 \
    --enable-deterministic-inference \
    --speculative-algorithm DECODE_VERIFY_ROLLBACK \
    --speculative-num-draft-tokens 16 \
    --speculative-num-steps 15 \
    "${overlap_arg[@]}"
  run_self_kl "35b_${label}"
  run_benchmark_pair "${label}"
  stop_server
}

run_eagle_mode() {
  local label="$1"
  local overlap_arg=()
  if [[ "${label}" == "eagle_sync" ]]; then
    overlap_arg=(--disable-overlap-schedule)
  fi
  start_server "35b_${label}" \
    --page-size 64 \
    --enable-deterministic-inference \
    --speculative-algorithm DECODE_VERIFY_ROLLBACK_EAGLE \
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
    --speculative-num-draft-tokens 2 \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    "${overlap_arg[@]}"
  run_benchmark_pair "${label}"
  stop_server
}

summarize_results() {
  RESULT_ROOT="${RESULT_ROOT}" conda_python - <<'PY'
import glob
import json
import os

base = os.environ["RESULT_ROOT"]
rows = {}
for path in sorted(glob.glob(os.path.join(base, "results", "35b_*.jsonl"))):
    with open(path) as f:
        records = [line for line in f if line.strip()]
    if records:
        rows[os.path.basename(path)] = json.loads(records[-1])

for name, row in rows.items():
    accept_length = row.get("accept_length")
    accept_text = "n/a" if accept_length is None else f"{accept_length:.3f}"
    print(
        f"{name} out={row['output_throughput']:.2f} "
        f"accept_len={accept_text} completed={row['completed']} "
        f"duration={row['duration']:.2f}"
    )

for suffix in ("false.jsonl", "true.jsonl"):
    baseline = rows.get(f"35b_baseline_sharegpt_logprob_{suffix}")
    if baseline is None:
        continue
    baseline_tps = baseline["output_throughput"]
    for mode, draft_tokens in (
        ("self_v1", 16),
        ("self_v2", 16),
        ("eagle_sync", 2),
        ("eagle_overlap", 2),
    ):
        row = rows.get(f"35b_{mode}_sharegpt_logprob_{suffix}")
        if row is None:
            continue
        acceptance = min(1.0, (row.get("accept_length") or 0.0) / draft_tokens)
        target = baseline_tps * acceptance
        print(
            f"TARGET {mode} logprob={suffix.removesuffix('.jsonl')} "
            f"actual={row['output_throughput']:.2f} "
            f"acceptance_x_baseline={target:.2f} "
            f"ratio={row['output_throughput'] / baseline_tps:.3f}"
        )
PY
}

require_file "${MODEL_PATH}/config.json"
require_file "${SHAREGPT_DATASET}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  run_baseline
fi
if [[ "${RUN_SELF}" == "1" ]]; then
  run_self_mode self_v1
  run_self_mode self_v2
fi
if [[ "${RUN_EAGLE}" == "1" ]]; then
  run_eagle_mode eagle_sync
  run_eagle_mode eagle_overlap
fi
summarize_results
