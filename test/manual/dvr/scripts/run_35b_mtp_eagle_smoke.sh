#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${MODEL_PATH:-/mnt/data/hwj/Qwen3.5-35B-A3B}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${MODEL_PATH}}"
SHAREGPT_DATASET="${SHAREGPT_DATASET:-/mnt/data/hwj/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json}"
PORT="${PORT:-30135}"
SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS:-2}"
SPEC_STEPS="${SPEC_STEPS:-1}"
ACCEPT_TEMPERATURE="${ACCEPT_TEMPERATURE:-0.7}"
ACCEPT_TOP_P="${ACCEPT_TOP_P:-1.0}"
MTP_MIN_ACCEPT_RATE="${MTP_MIN_ACCEPT_RATE:-0.75}"
MTP_REALDATA_MIN_ACCEPT_RATE="${MTP_REALDATA_MIN_ACCEPT_RATE:-0.70}"
MTP_REALDATA_NUM_PROMPTS="${MTP_REALDATA_NUM_PROMPTS:-8}"
MTP_REALDATA_MAX_NEW="${MTP_REALDATA_MAX_NEW:-64}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-}"
TP_SIZE="${TP_SIZE:-4}"
PAGE_SIZE="${PAGE_SIZE:-64}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
LINEAR_ATTN_BACKEND="${LINEAR_ATTN_BACKEND:-triton}"
DISABLE_RADIX_CACHE="$(resolve_radix_setting "${ATTENTION_BACKEND}" "${DISABLE_RADIX_CACHE:-auto}")"
RUN_SYNC="${RUN_SYNC:-1}"
RUN_OVERLAP="${RUN_OVERLAP:-1}"
BASE_URL="http://127.0.0.1:${PORT}"
RESULT_ROOT="${RESULT_ROOT:-${DVR_REPO_ROOT}/../dvr-fixed-validation/latest-run/35b-mtp-eagle-smoke}"
SERVER_PID=""

require_precompiled_deep_gemm

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/results"
write_run_metadata "${RESULT_ROOT}"
append_run_config "${RESULT_ROOT}" \
  "script=$(basename "$0")" "model=${MODEL_PATH}" "draft_model=${DRAFT_MODEL_PATH}" \
  "tp=${TP_SIZE}" "page_size=${PAGE_SIZE}" \
  "attention_backend=${ATTENTION_BACKEND}" \
  "linear_attn_backend=${LINEAR_ATTN_BACKEND}" \
  "disable_radix_cache=${DISABLE_RADIX_CACHE}" \
  "draft_tokens=${SPEC_DRAFT_TOKENS}" "draft_steps=${SPEC_STEPS}"

cleanup() {
  stop_process_group "${SERVER_PID}"
}
trap cleanup EXIT

run_one_mode() {
  local label="$1"
  local spec_v2="$2"
  local server_log="${RESULT_ROOT}/logs/${label}_server.log"
  local short_prompt_log="${RESULT_ROOT}/results/${label}_short_prompt.log"
  local kl_log="${RESULT_ROOT}/results/${label}_return_logprob_true.log"
  local no_logprob_log="${RESULT_ROOT}/results/${label}_return_logprob_false.log"
  local prefix_cache_kl_log="${RESULT_ROOT}/results/${label}_prefix_cache_safety_return_logprob_true.log"
  local prefix_cache_no_logprob_log="${RESULT_ROOT}/results/${label}_prefix_cache_safety_return_logprob_false.log"
  local interleaved_kl_log="${RESULT_ROOT}/results/${label}_interleaved_radix_kl.log"
  local radix_lifecycle_log="${RESULT_ROOT}/results/${label}_radix_lifecycle.log"
  local realdata_kl_log="${RESULT_ROOT}/results/${label}_sharegpt_return_logprob_true.log"
  local realdata_no_logprob_log="${RESULT_ROOT}/results/${label}_sharegpt_return_logprob_false.log"
  local overlap_args=()
  local radix_args=()
  local mamba_cache_args=()
  if [[ "${spec_v2}" == "0" ]]; then
    overlap_args=(--disable-overlap-schedule)
  fi
  if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
    radix_args=(--disable-radix-cache)
  fi
  if [[ -n "${MAX_MAMBA_CACHE_SIZE}" ]]; then
    mamba_cache_args=(--max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}")
  fi

  echo "==> Starting ${label} DVR-EAGLE server on ${BASE_URL}"
  setsid env \
    SGLANG_RETURN_ORIGINAL_LOGPROB=True \
    PYTHONPATH="${PYTHONPATH}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --tp-size "${TP_SIZE}" \
      --speculative-algorithm DECODE_VERIFY_ROLLBACK_EAGLE \
      --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
      --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}" \
      --speculative-num-steps "${SPEC_STEPS}" \
      --speculative-eagle-topk 1 \
      --page-size "${PAGE_SIZE}" \
      --context-length 4096 \
      --max-total-tokens 8192 \
      --mem-fraction-static 0.72 \
      --attention-backend "${ATTENTION_BACKEND}" \
      --linear-attn-backend "${LINEAR_ATTN_BACKEND}" \
      --sampling-backend pytorch \
      --enable-deterministic-inference \
      --cuda-graph-bs 1 2 4 \
      --cuda-graph-max-bs 4 \
      --max-running-requests 4 \
      "${radix_args[@]}" \
      "${mamba_cache_args[@]}" \
      "${overlap_args[@]}" \
      --skip-server-warmup \
      >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  wait_for_server "${BASE_URL}" 600 "${SERVER_PID}" "${server_log}"
  assert_server_capacity "${server_log}" 4
  assert_server_config \
    "${server_log}" "${ATTENTION_BACKEND}" "${PAGE_SIZE}" "${spec_v2}" "${DISABLE_RADIX_CACHE}"

  echo "==> Running ${label} one-token verify-sentinel KL smoke"
  conda_python test/manual/dvr/test_dvr_batch_kl.py \
    --base-url "${BASE_URL}" \
    --request-modes concurrent,batch \
    --prompt-token-lengths 1 \
    --max-new 2,8 \
    --limit-cases 4 \
    --concurrent-workers 2 \
    --ignore-eos \
    2>&1 | tee "${short_prompt_log}"
  grep -q "ALL_OK True" "${short_prompt_log}"

  echo "==> Running ${label} returned-logprob KL smoke"
  # The default rejection path is meaningful only under stochastic sampling.
  # Proposal and verify coins are keyed by request seed plus token position, so
  # sync/overlap and return-logprob variants must report identical histograms.
  conda_python test/manual/dvr/test_dvr_eagle_acceptance.py \
    --base-url "${BASE_URL}" \
    --prompt-token-lengths 63,64,65 \
    --max-new 4,16,65 \
    --cache-mode flush-each \
    --check-kl \
    --min-accept-rate "${MTP_MIN_ACCEPT_RATE}" \
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
    --min-accept-rate "${MTP_MIN_ACCEPT_RATE}" \
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
    --min-accept-rate "${MTP_MIN_ACCEPT_RATE}" \
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
    --min-accept-rate "${MTP_MIN_ACCEPT_RATE}" \
    --temperature "${ACCEPT_TEMPERATURE}" \
    --top-p "${ACCEPT_TOP_P}" \
    --ignore-eos \
    --seed 3032 \
    2>&1 | tee "${prefix_cache_no_logprob_log}"
  grep -q '"accept_failed": 0' "${prefix_cache_no_logprob_log}"

  echo "==> Running ${label} interleaved boundary-ownership KL smoke"
  conda_python test/manual/dvr/test_dvr_batch_kl.py \
    --base-url "${BASE_URL}" \
    --request-modes concurrent \
    --prompt-token-lengths 65,129 \
    --max-new 128 \
    --limit-cases 2 \
    --concurrent-workers 2 \
    --concurrent-stagger-ms 50 \
    --ignore-eos \
    2>&1 | tee "${interleaved_kl_log}"
  grep -q "ALL_OK True" "${interleaved_kl_log}"

  if [[ "${DISABLE_RADIX_CACHE}" == "0" ]]; then
    echo "==> Running ${label} nearest-radix-checkpoint replay KL smoke"
    conda_python test/manual/dvr/test_dvr_batch_kl.py \
      --base-url "${BASE_URL}" \
      --request-modes concurrent \
      --prompt-token-lengths 65 \
      --reuse-generated-prefix-tokens 128 \
      --min-cached-tokens 64 \
      --max-new 128 \
      --limit-cases 1 \
      --concurrent-workers 1 \
      --ignore-eos \
      2>&1 | tee -a "${interleaved_kl_log}"
    grep -q "ALL_OK True" "${interleaved_kl_log}"

    echo "==> Running ${label} generated-prefix radix lifecycle smoke"
    conda_python test/manual/dvr/test_dvr_radix_lifecycle.py \
      --base-url "${BASE_URL}" \
      --slot-cycles 4 \
      --prompt-lengths 63,64,65 \
      --generated-lengths 1,63,64,65,127,128,129 \
      2>&1 | tee "${radix_lifecycle_log}"
    grep -q "ALL_OK True" "${radix_lifecycle_log}"
  fi

  echo "==> Running ${label} ShareGPT real-data returned-logprob acceptance/KL smoke"
  conda_python test/manual/dvr/test_dvr_eagle_acceptance.py \
    --base-url "${BASE_URL}" \
    --dataset-path "${SHAREGPT_DATASET}" \
    --num-prompts "${MTP_REALDATA_NUM_PROMPTS}" \
    --max-prompt-tokens 1536 \
    --max-new "${MTP_REALDATA_MAX_NEW}" \
    --cache-mode flush-each \
    --check-kl \
    --min-accept-rate "${MTP_REALDATA_MIN_ACCEPT_RATE}" \
    --temperature "${ACCEPT_TEMPERATURE}" \
    --top-p "${ACCEPT_TOP_P}" \
    --ignore-eos \
    --seed 4032 \
    2>&1 | tee "${realdata_kl_log}"
  grep -q '"kl_failed": 0' "${realdata_kl_log}"
  grep -q '"accept_failed": 0' "${realdata_kl_log}"

  echo "==> Running ${label} ShareGPT real-data no-return-logprob acceptance smoke"
  conda_python test/manual/dvr/test_dvr_eagle_acceptance.py \
    --base-url "${BASE_URL}" \
    --dataset-path "${SHAREGPT_DATASET}" \
    --num-prompts "${MTP_REALDATA_NUM_PROMPTS}" \
    --max-prompt-tokens 1536 \
    --max-new "${MTP_REALDATA_MAX_NEW}" \
    --cache-mode flush-each \
    --no-return-logprob \
    --min-accept-rate "${MTP_REALDATA_MIN_ACCEPT_RATE}" \
    --temperature "${ACCEPT_TEMPERATURE}" \
    --top-p "${ACCEPT_TOP_P}" \
    --ignore-eos \
    --seed 4032 \
    2>&1 | tee "${realdata_no_logprob_log}"
  grep -q '"accept_failed": 0' "${realdata_no_logprob_log}"

  stop_process_group "${SERVER_PID}"
  SERVER_PID=""
}

# DVR-EAGLE uses v2 semantics in both modes: sync when disabled, overlap when enabled.
if [[ "${RUN_SYNC}" == "1" ]]; then
  run_one_mode "sync_v2" "0"
fi
if [[ "${RUN_OVERLAP}" == "1" ]]; then
  run_one_mode "overlap_v2" "1"
fi
