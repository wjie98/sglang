#!/usr/bin/env bash
set -euo pipefail

DVR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DVR_REPO_ROOT="$(cd "${DVR_SCRIPT_DIR}/../../../.." && pwd)"

cd "${DVR_REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-dvr_dev}"
export PYTHONPATH="${PYTHONPATH:-python}"
export SGLANG_RETURN_ORIGINAL_LOGPROB="${SGLANG_RETURN_ORIGINAL_LOGPROB:-True}"

if [[ -z "${CUDAHOSTCXX:-}" ]]; then
  DVR_CONDA_CXX="$(
    conda run --no-capture-output -n "${CONDA_ENV}" bash -lc \
      'command -v x86_64-conda-linux-gnu-g++ || command -v g++'
  )"
  export CUDAHOSTCXX="${DVR_CONDA_CXX}"
fi
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin ${CUDAHOSTCXX}}"

conda_python() {
  conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

wait_for_server() {
  local base_url="$1"
  local timeout_sec="$2"
  local server_pid="$3"
  local server_log="$4"

  for ((i = 0; i < timeout_sec; i++)); do
    if curl -fsS "${base_url}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "Server process exited before becoming ready. Last log lines:" >&2
      tail -120 "${server_log}" >&2 || true
      return 1
    fi
    sleep 1
  done

  echo "Timed out waiting for ${base_url}. Last log lines:" >&2
  tail -120 "${server_log}" >&2 || true
  return 1
}

stop_process_group() {
  local server_pid="${1:-}"
  if [[ -z "${server_pid}" ]]; then
    return 0
  fi

  if kill -0 "${server_pid}" >/dev/null 2>&1; then
    kill -INT "-${server_pid}" >/dev/null 2>&1 || true
    for _ in {1..30}; do
      if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if kill -0 "${server_pid}" >/dev/null 2>&1; then
      kill -TERM "-${server_pid}" >/dev/null 2>&1 || true
    fi
  fi

  wait "${server_pid}" >/dev/null 2>&1 || true
}
