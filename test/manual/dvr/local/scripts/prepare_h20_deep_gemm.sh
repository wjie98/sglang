#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model used by the H20 run.}"
TP_SIZE="${TP_SIZE:-4}"
MAX_GRAPH_BS="${MAX_GRAPH_BS:-3}"
DRAFT_TOKEN_COUNTS="${DRAFT_TOKEN_COUNTS:-${DRAFT_TOKENS:-16}}"
PORT="${PORT:-30280}"
BASE_URL="http://127.0.0.1:${PORT}"
PAGE_SIZE="${PAGE_SIZE:-64}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
LINEAR_ATTN_BACKEND="${LINEAR_ATTN_BACKEND:-triton}"
RESULT_ROOT="${RESULT_ROOT:-${DVR_REPO_ROOT}/../dvr-h20-deepgemm-prewarm}"
SGLANG_DG_CACHE_DIR="${SGLANG_DG_CACHE_DIR:?Set a dedicated SGLANG_DG_CACHE_DIR.}"
SERVER_PID=""

export SGLANG_DG_CACHE_DIR
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=1

mkdir -p "${RESULT_ROOT}" "${SGLANG_DG_CACHE_DIR}"
write_run_metadata "${RESULT_ROOT}"
append_run_config "${RESULT_ROOT}" \
  "script=$(basename "$0")" "model=${MODEL_PATH}" "tp=${TP_SIZE}" \
  "max_graph_bs=${MAX_GRAPH_BS}" "draft_token_counts=${DRAFT_TOKEN_COUNTS}" \
  "attention_backend=${ATTENTION_BACKEND}" \
  "linear_attn_backend=${LINEAR_ATTN_BACKEND}" "page_size=${PAGE_SIZE}"

cleanup() {
  stop_process_group "${SERVER_PID}"
}
trap cleanup EXIT

# Warm the deterministic target's exact linear/MoE token counts without CUDA
# graph capture. A normal EXTEND with M=graph_bs*draft_tokens exercises the
# same batch-invariant GEMM shapes as DVR target verify while avoiding a second
# speculative implementation solely for compilation.
server_log="${RESULT_ROOT}/prewarm_server.log"
setsid env \
  PYTHONPATH="${PYTHONPATH}" \
  conda run --no-capture-output -n "${CONDA_ENV}" python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --tp-size "${TP_SIZE}" \
    --context-length 4096 \
    --max-total-tokens 4096 \
    --mem-fraction-static 0.9 \
    --max-running-requests 1 \
    --max-mamba-cache-size 1 \
    --page-size "${PAGE_SIZE}" \
    --attention-backend "${ATTENTION_BACKEND}" \
    --linear-attn-backend "${LINEAR_ATTN_BACKEND}" \
    --sampling-backend pytorch \
    --enable-deterministic-inference \
    --disable-custom-all-reduce \
    --cuda-graph-config '{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}' \
    --skip-server-warmup \
    >"${server_log}" 2>&1 &
SERVER_PID="$!"

wait_for_server "${BASE_URL}" 1800 "${SERVER_PID}" "${server_log}"

BASE_URL="${BASE_URL}" MAX_GRAPH_BS="${MAX_GRAPH_BS}" \
DRAFT_TOKEN_COUNTS="${DRAFT_TOKEN_COUNTS}" conda_python - <<'PY'
import os

import requests

base_url = os.environ["BASE_URL"]
max_graph_bs = int(os.environ["MAX_GRAPH_BS"])
draft_token_counts = {
    int(value)
    for value in os.environ["DRAFT_TOKEN_COUNTS"].replace(",", " ").split()
}
for draft_tokens in sorted(draft_token_counts):
    for graph_bs in range(1, max_graph_bs + 1):
        token_count = graph_bs * draft_tokens
        response = requests.post(
            f"{base_url}/generate",
            json={
                "input_ids": [1] * token_count,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
            },
            timeout=3600,
        )
        response.raise_for_status()
        print(f"warmed target GEMM shape M={token_count}", flush=True)
PY

stop_process_group "${SERVER_PID}"
SERVER_PID=""

if grep -Eqi "Traceback|illegal memory access|device-side assert" "${server_log}"; then
  echo "DeepGEMM prewarm failed; inspect ${server_log}." >&2
  exit 1
fi
if [[ -z "$(find "${SGLANG_DG_CACHE_DIR}" -type f -print -quit)" ]]; then
  echo "DeepGEMM prewarm produced an empty cache; inspect ${server_log}." >&2
  exit 1
fi
find "${SGLANG_DG_CACHE_DIR}" -type f -printf '%P %s\n' | sort \
  >"${RESULT_ROOT}/deepgemm-cache-manifest.txt"
touch "${RESULT_ROOT}/PREWARM_COMPLETE"
echo "DeepGEMM cache is ready at ${SGLANG_DG_CACHE_DIR}."
echo "Set SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 before CUDA graph serving."
