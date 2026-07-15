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
MAX_CONCURRENCY="${MAX_CONCURRENCY:-3}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_BASELINE_SYNC="${RUN_BASELINE_SYNC:-${RUN_BASELINE}}"
RUN_BASELINE_OVERLAP="${RUN_BASELINE_OVERLAP:-${RUN_BASELINE}}"
RUN_DETERMINISTIC_BASELINE="${RUN_DETERMINISTIC_BASELINE:-${RUN_BASELINE}}"
RUN_DET_SYNC="${RUN_DET_SYNC:-${RUN_DETERMINISTIC_BASELINE}}"
RUN_DET_OVERLAP="${RUN_DET_OVERLAP:-${RUN_DETERMINISTIC_BASELINE}}"
RUN_SELF="${RUN_SELF:-1}"
RUN_EAGLE="${RUN_EAGLE:-1}"
TP_SIZE="${TP_SIZE:-4}"
PAGE_SIZE="${PAGE_SIZE:-64}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
LINEAR_ATTN_BACKEND="${LINEAR_ATTN_BACKEND:-triton}"
DISABLE_RADIX_CACHE="$(resolve_radix_setting "${ATTENTION_BACKEND}" "${DISABLE_RADIX_CACHE:-auto}")"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
BASE_URL="http://127.0.0.1:${PORT}"
RESULT_ROOT="${RESULT_ROOT:-${DVR_REPO_ROOT}/../dvr-fixed-validation/latest-run/35b-dvr-throughput}"
SERVER_PID=""
CUDA_GRAPH_BS=()
RADIX_ARGS=()
CUSTOM_AR_ARGS=()

require_precompiled_deep_gemm

for ((bs = 1; bs <= MAX_CONCURRENCY; bs++)); do
  CUDA_GRAPH_BS+=("${bs}")
done
if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  RADIX_ARGS=(--disable-radix-cache)
fi
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  CUSTOM_AR_ARGS=(--disable-custom-all-reduce)
fi

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/results"
write_run_metadata "${RESULT_ROOT}"
append_run_config "${RESULT_ROOT}" \
  "script=$(basename "$0")" "model=${MODEL_PATH}" "draft_model=${DRAFT_MODEL_PATH}" \
  "dataset=${SHAREGPT_DATASET}" "tp=${TP_SIZE}" "page_size=${PAGE_SIZE}" \
  "attention_backend=${ATTENTION_BACKEND}" \
  "linear_attn_backend=${LINEAR_ATTN_BACKEND}" \
  "disable_radix_cache=${DISABLE_RADIX_CACHE}" \
  "disable_custom_all_reduce=${DISABLE_CUSTOM_ALL_REDUCE}" \
  "num_prompts=${NUM_PROMPTS}" "output_len=${OUTPUT_LEN}" \
  "max_concurrency=${MAX_CONCURRENCY}" \
  "run_baseline_sync=${RUN_BASELINE_SYNC}" \
  "run_baseline_overlap=${RUN_BASELINE_OVERLAP}" \
  "run_det_sync=${RUN_DET_SYNC}" "run_det_overlap=${RUN_DET_OVERLAP}" \
  "run_self=${RUN_SELF}" "run_eagle=${RUN_EAGLE}"

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
  local overlap_enabled="$2"
  shift 2
  local server_log="${RESULT_ROOT}/logs/${label}_server.log"

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
      --max-total-tokens "${MAX_TOTAL_TOKENS}" \
      --mem-fraction-static 0.72 \
      --max-running-requests "${MAX_CONCURRENCY}" \
      --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}" \
      --page-size "${PAGE_SIZE}" \
      --attention-backend "${ATTENTION_BACKEND}" \
      --linear-attn-backend "${LINEAR_ATTN_BACKEND}" \
      --sampling-backend pytorch \
      --cuda-graph-bs "${CUDA_GRAPH_BS[@]}" \
      --cuda-graph-max-bs "${MAX_CONCURRENCY}" \
      "${RADIX_ARGS[@]}" \
      "${CUSTOM_AR_ARGS[@]}" \
      "$@" \
      --skip-server-warmup \
      >"${server_log}" 2>&1 &
  SERVER_PID="$!"
  wait_for_server "${BASE_URL}" 600 "${SERVER_PID}" "${server_log}"
  assert_server_capacity "${server_log}" "${MAX_CONCURRENCY}"
  assert_server_config \
    "${server_log}" "${ATTENTION_BACKEND}" "${PAGE_SIZE}" "${overlap_enabled}" "${DISABLE_RADIX_CACHE}"
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
  printf 'true\n' >"${RESULT_ROOT}/results/${label}_gdn_kl_zero.txt"
}

run_baseline_mode() {
  local label="$1"
  local overlap_enabled="$2"
  local deterministic="$3"
  local overlap_arg=()
  local deterministic_arg=()
  if [[ "${overlap_enabled}" == "0" ]]; then
    overlap_arg=(--disable-overlap-schedule)
  fi
  if [[ "${deterministic}" == "1" ]]; then
    deterministic_arg=(--enable-deterministic-inference)
  fi
  start_server "35b_${label}" "${overlap_enabled}" \
    "${deterministic_arg[@]}" "${overlap_arg[@]}"
  local server_log="${RESULT_ROOT}/logs/35b_${label}_server.log"
  assert_baseline_server_config "${server_log}" "${deterministic}" "${TP_SIZE}"

  # One ordinary deterministic probe is enough to document whether this GDN
  # model happens to match a full-prefill oracle. A mismatch is expected to be
  # reportable; a client/server failure remains fatal.
  if [[ "${label}" == "det_overlap" ]]; then
    local kl_log="${RESULT_ROOT}/results/35b_${label}_kl.log"
    conda_python test/manual/dvr/test_dvr_batch_kl.py \
      --base-url "${BASE_URL}" \
      --request-modes concurrent \
      --prompt-token-lengths 65 \
      --max-new 65 \
      --limit-cases 1 \
      --concurrent-workers 1 \
      --ignore-eos \
      --allow-mismatch \
      2>&1 | tee "${kl_log}"
    if grep -q "ALL_OK True" "${kl_log}"; then
      printf 'true\n' >"${RESULT_ROOT}/results/35b_${label}_gdn_kl_zero.txt"
    elif grep -q "ALL_OK False" "${kl_log}"; then
      printf 'false\n' >"${RESULT_ROOT}/results/35b_${label}_gdn_kl_zero.txt"
    else
      echo "Ordinary deterministic GDN oracle probe did not complete." >&2
      return 1
    fi
  fi
  run_benchmark_pair "${label}"
  stop_server
}

run_self_mode() {
  local label="$1"
  local overlap_enabled=1
  local overlap_arg=()
  if [[ "${label}" == "self_v1" ]]; then
    overlap_enabled=0
    overlap_arg=(--disable-overlap-schedule)
  fi
  start_server "35b_${label}" "${overlap_enabled}" \
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
  local overlap_enabled=1
  local overlap_arg=()
  if [[ "${label}" == "eagle_sync" ]]; then
    overlap_enabled=0
    overlap_arg=(--disable-overlap-schedule)
  fi
  start_server "35b_${label}" "${overlap_enabled}" \
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

for mode in ("det_overlap", "self_v1", "self_v2"):
    path = os.path.join(base, "results", f"35b_{mode}_gdn_kl_zero.txt")
    if os.path.exists(path):
        with open(path) as f:
            exact = f.read().strip()
        print(f"GDN_KL mode={mode} exact_kl_zero={exact}")

product_speedups = []
for logprob in ("false", "true"):
    for mode, draft_tokens, baseline_mode in (
        ("self_v1", 16, "baseline_sync"),
        ("self_v2", 16, "baseline_overlap"),
        ("eagle_sync", 2, "baseline_sync"),
        ("eagle_overlap", 2, "baseline_overlap"),
    ):
        baseline = rows.get(f"35b_{baseline_mode}_sharegpt_logprob_{logprob}.jsonl")
        row = rows.get(f"35b_{mode}_sharegpt_logprob_{logprob}.jsonl")
        if baseline is None or row is None:
            continue
        baseline_tps = baseline["output_throughput"]
        acceptance = min(1.0, (row.get("accept_length") or 0.0) / draft_tokens)
        target = baseline_tps * acceptance
        target_efficiency = row["output_throughput"] / target if target else float("nan")
        print(
            f"TARGET {mode} logprob={logprob} "
            f"actual={row['output_throughput']:.2f} "
            f"acceptance_x_baseline={target:.2f} "
            f"dvr_ratio={row['output_throughput'] / baseline_tps:.3f} "
            f"target_efficiency={target_efficiency:.3f}"
        )

    for mode, det_mode in (("self_v1", "det_sync"), ("self_v2", "det_overlap")):
        det = rows.get(f"35b_{det_mode}_sharegpt_logprob_{logprob}.jsonl")
        row = rows.get(f"35b_{mode}_sharegpt_logprob_{logprob}.jsonl")
        if det is None or row is None:
            continue
        speedup = row["output_throughput"] / det["output_throughput"]
        if mode == "self_v2":
            product_speedups.append(speedup)
        print(
            f"DET_SPEEDUP {mode} logprob={logprob} "
            f"dvr={row['output_throughput']:.2f} "
            f"ordinary_det={det['output_throughput']:.2f} "
            f"speedup={speedup:.3f} faster={str(speedup > 1.0).lower()}"
        )

if product_speedups:
    print(
        "PRODUCT_GATE mode=self_v2 "
        f"comparisons={len(product_speedups)} "
        f"min_speedup={min(product_speedups):.3f} "
        f"faster_than_ordinary_det={str(min(product_speedups) > 1.0).lower()}"
    )
PY
}

require_file "${MODEL_PATH}/config.json"
require_file "${SHAREGPT_DATASET}"

if [[ "${RUN_BASELINE_SYNC}" == "1" ]]; then
  run_baseline_mode baseline_sync 0 0
fi
if [[ "${RUN_BASELINE_OVERLAP}" == "1" ]]; then
  run_baseline_mode baseline_overlap 1 0
fi
if [[ "${RUN_DET_SYNC}" == "1" ]]; then
  run_baseline_mode det_sync 0 1
fi
if [[ "${RUN_DET_OVERLAP}" == "1" ]]; then
  run_baseline_mode det_overlap 1 1
fi
if [[ "${RUN_SELF}" == "1" ]]; then
  run_self_mode self_v1
  run_self_mode self_v2
fi
if [[ "${RUN_EAGLE}" == "1" ]]; then
  run_eagle_mode eagle_sync
  run_eagle_mode eagle_overlap
fi
summarize_results | tee "${RESULT_ROOT}/summary.txt"
