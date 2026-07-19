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
RUN_BASELINE_SYNC="${RUN_BASELINE_SYNC:-${RUN_BASELINE}}"
RUN_BASELINE_OVERLAP="${RUN_BASELINE_OVERLAP:-${RUN_BASELINE}}"
RUN_DETERMINISTIC_BASELINE="${RUN_DETERMINISTIC_BASELINE:-${RUN_BASELINE}}"
RUN_DET_SYNC="${RUN_DET_SYNC:-${RUN_DETERMINISTIC_BASELINE}}"
RUN_DET_OVERLAP="${RUN_DET_OVERLAP:-${RUN_DETERMINISTIC_BASELINE}}"
RUN_DVR="${RUN_DVR:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
RANDOM_SEED="${RANDOM_SEED:-2026}"
FLUSH_CACHE_EACH_RUN="${FLUSH_CACHE_EACH_RUN:-1}"
ALLOW_RESULT_REUSE="${ALLOW_RESULT_REUSE:-0}"
DRAFT_TOKENS="${DRAFT_TOKENS:-16}"
DRAFT_STEPS="${DRAFT_STEPS:-$((DRAFT_TOKENS - 1))}"
SHAREGPT_MAX_CONCURRENCY="${SHAREGPT_MAX_CONCURRENCY:-3}"
LONGBENCH_MAX_CONCURRENCY="${LONGBENCH_MAX_CONCURRENCY:-2}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-16}"
TP_SIZE="${TP_SIZE:-4}"
PAGE_SIZE="${PAGE_SIZE:-64}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
LINEAR_ATTN_BACKEND="${LINEAR_ATTN_BACKEND:-triton}"
DISABLE_RADIX_CACHE="$(resolve_radix_setting "${ATTENTION_BACKEND}" "${DISABLE_RADIX_CACHE:-auto}")"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
SERVER_PID=""
SERVER_MAX_CONCURRENCY="${SHAREGPT_MAX_CONCURRENCY}"
CUDA_GRAPH_BS=()
RADIX_ARGS=()
CUSTOM_AR_ARGS=()

if ((DRAFT_TOKENS < 2)) || ((DRAFT_STEPS + 1 != DRAFT_TOKENS)); then
  echo "DVR chain mode requires DRAFT_TOKENS >= 2 and DRAFT_STEPS + 1 == DRAFT_TOKENS." >&2
  exit 1
fi
if [[ "${FLUSH_CACHE_EACH_RUN}" != "0" && "${FLUSH_CACHE_EACH_RUN}" != "1" ]]; then
  echo "FLUSH_CACHE_EACH_RUN must be 0 or 1." >&2
  exit 1
fi
require_precompiled_deep_gemm

if ((LONGBENCH_MAX_CONCURRENCY > SERVER_MAX_CONCURRENCY)); then
  SERVER_MAX_CONCURRENCY="${LONGBENCH_MAX_CONCURRENCY}"
fi
for ((bs = 1; bs <= SERVER_MAX_CONCURRENCY; bs++)); do
  CUDA_GRAPH_BS+=("${bs}")
done
if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  RADIX_ARGS=(--disable-radix-cache)
fi
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  CUSTOM_AR_ARGS=(--disable-custom-all-reduce)
fi

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/results"
if [[ "${ALLOW_RESULT_REUSE}" != "1" ]] && \
   compgen -G "${RESULT_ROOT}/results/*.jsonl" >/dev/null; then
  echo "RESULT_ROOT already contains benchmark JSONL files: ${RESULT_ROOT}" >&2
  echo "Use a new directory, or set ALLOW_RESULT_REUSE=1 only for an intentional resume." >&2
  exit 1
fi
write_run_metadata "${RESULT_ROOT}"
append_run_config "${RESULT_ROOT}" \
  "script=$(basename "$0")" "model=${MODEL_PATH}" \
  "sharegpt_dataset=${SHAREGPT_DATASET}" \
  "longbench_dataset=${LONGBENCH_CUSTOM_DATASET}" \
  "tp=${TP_SIZE}" "page_size=${PAGE_SIZE}" \
  "attention_backend=${ATTENTION_BACKEND}" \
  "linear_attn_backend=${LINEAR_ATTN_BACKEND}" \
  "disable_radix_cache=${DISABLE_RADIX_CACHE}" \
  "disable_custom_all_reduce=${DISABLE_CUSTOM_ALL_REDUCE}" \
  "num_prompts=${NUM_PROMPTS}" "output_len=${OUTPUT_LEN}" \
  "warmup_requests=${WARMUP_REQUESTS}" "random_seed=${RANDOM_SEED}" \
  "flush_cache_each_run=${FLUSH_CACHE_EACH_RUN}" \
  "allow_result_reuse=${ALLOW_RESULT_REUSE}" \
  "draft_tokens=${DRAFT_TOKENS}" "draft_steps=${DRAFT_STEPS}" \
  "sharegpt_concurrency=${SHAREGPT_MAX_CONCURRENCY}" \
  "longbench_concurrency=${LONGBENCH_MAX_CONCURRENCY}" \
  "run_baseline_sync=${RUN_BASELINE_SYNC}" \
  "run_baseline_overlap=${RUN_BASELINE_OVERLAP}" \
  "run_det_sync=${RUN_DET_SYNC}" "run_det_overlap=${RUN_DET_OVERLAP}" \
  "run_dvr=${RUN_DVR}"

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
  local cache_args=(--cache-report)

  if [[ "${return_logprob}" == "true" ]]; then
    return_arg=(--return-logprob)
  fi
  if [[ "${FLUSH_CACHE_EACH_RUN}" == "1" ]]; then
    # bench_serving flushes after warmup, preserving kernel warmup while making
    # every measured sub-run start from the same empty prefix-cache state.
    cache_args+=(--flush-cache)
  fi

  echo "==> Running ${label}"
  conda_python -m sglang.benchmark.serving \
    --backend sglang \
    --base-url "${BASE_URL}" \
    --dataset-name "${dataset_name}" \
    --dataset-path "${dataset}" \
    --tokenizer "${MODEL_PATH}" \
    --num-prompts "${NUM_PROMPTS}" \
    --sharegpt-output-len "${OUTPUT_LEN}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --request-rate inf \
    --max-concurrency "${max_concurrency}" \
    --disable-tqdm \
    --disable-stream \
    --seed "${RANDOM_SEED}" \
    --output-file "${output_file}" \
    "${cache_args[@]}" \
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

run_baseline_mode() {
  local label="$1"
  local overlap_enabled="$2"
  local deterministic="$3"
  local server_log="${RESULT_ROOT}/logs/80b_${label}_server.log"
  local overlap_args=()
  local deterministic_args=()
  if [[ "${overlap_enabled}" == "0" ]]; then
    overlap_args=(--disable-overlap-schedule)
  fi
  if [[ "${deterministic}" == "1" ]]; then
    deterministic_args=(--enable-deterministic-inference)
  fi

  echo "==> Starting 80B ${label} server on ${BASE_URL}"
  setsid env \
    PYTHONPATH="${PYTHONPATH}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --tp-size "${TP_SIZE}" \
      --context-length 8192 \
      --max-total-tokens 6144 \
      --mem-fraction-static 0.9 \
      --max-running-requests "${SERVER_MAX_CONCURRENCY}" \
      --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}" \
      --page-size "${PAGE_SIZE}" \
      --attention-backend "${ATTENTION_BACKEND}" \
      --linear-attn-backend "${LINEAR_ATTN_BACKEND}" \
      --sampling-backend pytorch \
      --random-seed "${RANDOM_SEED}" \
      --cuda-graph-bs "${CUDA_GRAPH_BS[@]}" \
      --cuda-graph-max-bs-decode "${SERVER_MAX_CONCURRENCY}" \
      "${RADIX_ARGS[@]}" \
      "${CUSTOM_AR_ARGS[@]}" \
      "${deterministic_args[@]}" \
      "${overlap_args[@]}" \
      --skip-server-warmup \
      >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  wait_for_server "${BASE_URL}" 600 "${SERVER_PID}" "${server_log}"
  assert_server_capacity "${server_log}" "${SERVER_MAX_CONCURRENCY}"
  assert_server_config \
    "${server_log}" "${ATTENTION_BACKEND}" "${PAGE_SIZE}" "${overlap_enabled}" "${DISABLE_RADIX_CACHE}"
  assert_baseline_server_config "${server_log}" "${deterministic}" "${TP_SIZE}"
  run_benchmark_matrix "${label}"
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
      --tp-size "${TP_SIZE}" \
      --context-length 8192 \
      --max-total-tokens 6144 \
      --mem-fraction-static 0.9 \
      --max-running-requests "${SERVER_MAX_CONCURRENCY}" \
      --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}" \
      --page-size "${PAGE_SIZE}" \
      --attention-backend "${ATTENTION_BACKEND}" \
      --linear-attn-backend "${LINEAR_ATTN_BACKEND}" \
      --sampling-backend pytorch \
      --random-seed "${RANDOM_SEED}" \
      --enable-deterministic-inference \
      --speculative-algorithm DECODE_VERIFY_ROLLBACK \
      --speculative-num-draft-tokens "${DRAFT_TOKENS}" \
      --speculative-num-steps "${DRAFT_STEPS}" \
      --cuda-graph-bs "${CUDA_GRAPH_BS[@]}" \
      --cuda-graph-max-bs-decode "${SERVER_MAX_CONCURRENCY}" \
      "${RADIX_ARGS[@]}" \
      "${CUSTOM_AR_ARGS[@]}" \
      "${overlap_args[@]}" \
      "$@" \
      --skip-server-warmup \
      >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  wait_for_server "${BASE_URL}" 600 "${SERVER_PID}" "${server_log}"
  assert_server_capacity "${server_log}" "${SERVER_MAX_CONCURRENCY}"
  assert_server_config \
    "${server_log}" "${ATTENTION_BACKEND}" "${PAGE_SIZE}" "${spec_v2}" "${DISABLE_RADIX_CACHE}"
  assert_dvr_graphs "${server_log}" self "${DRAFT_TOKENS}" "${SERVER_MAX_CONCURRENCY}"

  run_benchmark_matrix "${label}"

  stop_process_group "${SERVER_PID}"
  SERVER_PID=""
}

summarize_results() {
  RESULT_ROOT="${RESULT_ROOT}" DRAFT_TOKENS="${DRAFT_TOKENS}" conda_python - <<'PY'
import glob
import json
import os

base = os.environ["RESULT_ROOT"]
draft_tokens = int(os.environ["DRAFT_TOKENS"])
for path in sorted(glob.glob(os.path.join(base, "results", "80b_*.jsonl"))):
    with open(path) as f:
        rows = [line for line in f if line.strip()]
    if not rows:
        continue
    row = json.loads(rows[-1])
    accept_length = row.get("accept_length")
    accept_text = "n/a" if accept_length is None else f"{accept_length:.2f}"
    cache = row.get("cache_report") or {}
    print(
        "{} out={:.2f} accept={} completed={} duration={:.2f} "
        "cached={} hit_rate={:.2f}%".format(
            os.path.basename(path),
            row.get("output_throughput"),
            accept_text,
            row.get("completed"),
            row.get("duration"),
            cache.get("total_cached_tokens", "n/a"),
            cache.get("cache_hit_rate_pct", float("nan")),
        )
    )

rows = {}
for path in glob.glob(os.path.join(base, "results", "80b_*.jsonl")):
    with open(path) as f:
        records = [line for line in f if line.strip()]
    if records:
        rows[os.path.basename(path)] = json.loads(records[-1])

product_speedups = []
for dataset in ("sharegpt", "longbench"):
    for logprob in ("false", "true"):
        v1 = rows.get(f"80b_v1_{dataset}_logprob_{logprob}.jsonl")
        v2 = rows.get(f"80b_v2_{dataset}_logprob_{logprob}.jsonl")
        if v1 is not None and v2 is not None:
            v1_accept = min(1.0, (v1.get("accept_length") or 0.0) / draft_tokens)
            v2_accept = min(1.0, (v2.get("accept_length") or 0.0) / draft_tokens)
            normalized = (
                (v2["output_throughput"] / v2_accept)
                / (v1["output_throughput"] / v1_accept)
                if v1_accept and v2_accept
                else float("nan")
            )
            print(
                f"V2_V1 dataset={dataset} logprob={logprob} "
                f"throughput_ratio={v2['output_throughput'] / v1['output_throughput']:.3f} "
                f"acceptance_ratio={v2_accept / v1_accept:.3f} "
                f"acceptance_normalized_ratio={normalized:.3f}"
            )
        for mode, baseline_mode in (("v1", "baseline_sync"), ("v2", "baseline_overlap")):
            baseline = rows.get(f"80b_{baseline_mode}_{dataset}_logprob_{logprob}.jsonl")
            row = rows.get(f"80b_{mode}_{dataset}_logprob_{logprob}.jsonl")
            if baseline is None or row is None:
                continue
            acceptance = min(
                1.0, (row.get("accept_length") or 0.0) / draft_tokens
            )
            target = baseline["output_throughput"] * acceptance
            efficiency = row["output_throughput"] / target if target else float("nan")
            print(
                f"TARGET {mode} dataset={dataset} logprob={logprob} "
                f"actual={row['output_throughput']:.2f} "
                f"acceptance_x_baseline={target:.2f} "
                f"dvr_ratio={row['output_throughput'] / baseline['output_throughput']:.3f} "
                f"target_efficiency={efficiency:.3f} "
                f"mean_tpot_ms={row.get('mean_tpot_ms', float('nan')):.3f}"
            )

        for mode, det_mode in (("v1", "det_sync"), ("v2", "det_overlap")):
            det = rows.get(f"80b_{det_mode}_{dataset}_logprob_{logprob}.jsonl")
            row = rows.get(f"80b_{mode}_{dataset}_logprob_{logprob}.jsonl")
            if det is None or row is None:
                continue
            speedup = row["output_throughput"] / det["output_throughput"]
            if mode == "v2":
                product_speedups.append(speedup)
            print(
                f"DET_SPEEDUP {mode} dataset={dataset} logprob={logprob} "
                f"dvr={row['output_throughput']:.2f} "
                f"ordinary_det={det['output_throughput']:.2f} "
                f"speedup={speedup:.3f} faster={str(speedup > 1.0).lower()}"
            )

if product_speedups:
    print(
        "PRODUCT_GATE mode=v2 "
        f"comparisons={len(product_speedups)} "
        f"min_speedup={min(product_speedups):.3f} "
        f"faster_than_ordinary_det={str(min(product_speedups) > 1.0).lower()}"
    )
PY
}

require_file "${MODEL_PATH}/config.json"
require_file "${SHAREGPT_DATASET}"
require_file "${LONGBENCH_CUSTOM_DATASET}"

# Fixed reproduced口径:
# - normal and ordinary-deterministic baselines match each DVR scheduler mode.
# - normal baselines measure acceptance-weighted implementation efficiency.
# - deterministic baselines measure the user-facing deterministic speedup.
# - v1: compatibility scheduler, overlap disabled.
# - v2: spec-v2 worker with overlap enabled.
# - server/graph concurrency is the maximum requested by either dataset.
# - max_mamba_cache_size must cover that concurrency without a runtime reduction.
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
if [[ "${RUN_DVR}" == "1" ]]; then
  run_one_mode "v1" "0"
  run_one_mode "v2" "1"
fi
summarize_results | tee "${RESULT_ROOT}/summary.txt"
