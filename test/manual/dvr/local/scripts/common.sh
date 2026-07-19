#!/usr/bin/env bash
set -euo pipefail

DVR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DVR_REPO_ROOT="$(cd "${DVR_SCRIPT_DIR}/../../../../.." && pwd)"

cd "${DVR_REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-dvr_dev}"
# Put this worktree first even when the conda environment contains an editable
# SGLang install from another worktree.
export PYTHONPATH="${DVR_REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
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

resolve_radix_setting() {
  local attention_backend="$1"
  local requested="${2:-auto}"

  case "${requested}" in
    0|1)
      printf '%s\n' "${requested}"
      ;;
    auto)
      printf '0\n'
      ;;
    *)
      echo "DISABLE_RADIX_CACHE must be 0, 1, or auto; got ${requested}." >&2
      return 1
      ;;
  esac
}

write_run_metadata() {
  local result_root="$1"
  local metadata_file="${result_root}/run_metadata.txt"

  {
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'commit=%s\n' "$(git rev-parse HEAD)"
    printf 'branch=%s\n' "$(git branch --show-current)"
    printf 'conda_env=%s\n' "${CONDA_ENV}"
    printf 'pythonpath=%s\n' "${PYTHONPATH}"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-all}"
    printf 'sglang_enable_jit_deepgemm=%s\n' "${SGLANG_ENABLE_JIT_DEEPGEMM:-default}"
    printf 'sglang_jit_deepgemm_precompile=%s\n' "${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-default}"
    printf 'sglang_batch_invariant_deepgemm=%s\n' "${SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM:-default}"
    printf 'sglang_batch_invariant_deepgemm_fallback=%s\n' "${SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT:-default}"
    printf 'sglang_dg_cache_dir=%s\n' "${SGLANG_DG_CACHE_DIR:-default}"
    printf '\n[gpus]\n'
    nvidia-smi \
      --query-gpu=index,name,memory.total,driver_version,pci.bus_id \
      --format=csv,noheader 2>/dev/null || true
    printf '\n[topology]\n'
    nvidia-smi topo -m 2>/dev/null || true
  } >"${metadata_file}"
}

require_precompiled_deep_gemm() {
  if [[ "${REQUIRE_PRECOMPILED_DEEPGEMM:-0}" != "1" ]]; then
    return 0
  fi
  if [[ "${SGLANG_ENABLE_JIT_DEEPGEMM:-}" != "1" ]] || \
     [[ "${SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM:-}" != "1" ]] || \
     [[ "${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-}" != "0" ]]; then
    echo "H20 release runs require DeepGEMM enabled and runtime precompile disabled." >&2
    echo "Run prepare_h20_deep_gemm.sh first, then export the environment from the guide." >&2
    return 1
  fi
  if [[ "${SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT:-0}" != "0" ]]; then
    echo "H20 release runs must not allow batch-variant GEMM fallback." >&2
    return 1
  fi
  if [[ -z "${SGLANG_DG_CACHE_DIR:-}" ]] || \
     [[ ! -d "${SGLANG_DG_CACHE_DIR}" ]] || \
     [[ -z "$(find "${SGLANG_DG_CACHE_DIR}" -mindepth 1 -print -quit)" ]]; then
    echo "SGLANG_DG_CACHE_DIR is missing or empty: ${SGLANG_DG_CACHE_DIR:-unset}" >&2
    return 1
  fi
  if ! conda_python -c \
    'from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM; assert ENABLE_JIT_DEEPGEMM'; then
    echo "DeepGEMM is unavailable in the selected environment or GPU architecture." >&2
    return 1
  fi
}

append_run_config() {
  local result_root="$1"
  shift
  {
    printf '\n[run_config]\n'
    printf '%s\n' "$@"
  } >>"${result_root}/run_metadata.txt"
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

assert_server_capacity() {
  local server_log="$1"
  local expected_graph_bs="$2"

  if grep -q "max_running_requests was reduced" "${server_log}"; then
    echo "Server reduced the requested concurrency; benchmark results would not be comparable." >&2
    grep "max_running_requests was reduced" "${server_log}" >&2 || true
    return 1
  fi
  if ! grep -q "Capturing batches (bs=${expected_graph_bs}" "${server_log}"; then
    echo "CUDA graph batch size ${expected_graph_bs} was not captured." >&2
    tail -120 "${server_log}" >&2 || true
    return 1
  fi
  if grep -Eqi "CUDA graph.*(failed|fallback)|failed.*CUDA graph|illegal memory access|device-side assert" "${server_log}"; then
    echo "CUDA graph capture/replay failed; benchmark results are invalid." >&2
    grep -Ei "CUDA graph.*(failed|fallback)|failed.*CUDA graph|illegal memory access|device-side assert" "${server_log}" >&2 || true
    return 1
  fi
  if [[ "${REQUIRE_PRECOMPILED_DEEPGEMM:-0}" == "1" ]] && \
     grep -q "Entering DeepGEMM JIT Pre-Compile session" "${server_log}"; then
    echo "DeepGEMM re-entered exhaustive JIT during graph startup; prewarm cache is incomplete." >&2
    return 1
  fi
  if [[ "${REQUIRE_PRECOMPILED_DEEPGEMM:-0}" == "1" ]] && \
     grep -q "Disable DeepGEMM in batch invariant ops" "${server_log}"; then
    echo "Batch-invariant DeepGEMM was disabled at runtime; benchmark is invalid." >&2
    return 1
  fi
}

assert_dvr_graphs() {
  local server_log="$1"
  local draft_mode="$2"
  local draft_tokens="$3"
  local expected_graph_bs="$4"

  if ! grep -Eq \
    "Capture target verify CUDA graph begin\..*num_tokens_per_bs=${draft_tokens}.*bs=\[[^]]*${expected_graph_bs}[^]]*\]" \
    "${server_log}"; then
    echo "DVR target-verify graph did not capture tokens=${draft_tokens}, bs=${expected_graph_bs}." >&2
    tail -160 "${server_log}" >&2 || true
    return 1
  fi
  if [[ "${draft_mode}" == "self" ]] && \
     ! grep -q "Capture DVR self-draft CUDA graph end" "${server_log}"; then
    echo "DVR self-draft graph capture did not complete." >&2
    tail -160 "${server_log}" >&2 || true
    return 1
  fi
}

assert_server_config() {
  local server_log="$1"
  local attention_backend="$2"
  local page_size="$3"
  local overlap_enabled="$4"
  local radix_disabled="$5"

  local overlap_value="True"
  if [[ "${overlap_enabled}" == "1" ]]; then
    overlap_value="False"
  fi
  local radix_value="False"
  if [[ "${radix_disabled}" == "1" ]]; then
    radix_value="True"
  fi

  for expected in \
    "attention_backend='${attention_backend}'" \
    "page_size=${page_size}" \
    "disable_overlap_schedule=${overlap_value}" \
    "disable_radix_cache=${radix_value}"; do
    if ! grep -q "${expected}" "${server_log}"; then
      echo "Server did not resolve expected config: ${expected}" >&2
      tail -120 "${server_log}" >&2 || true
      return 1
    fi
  done
}

assert_baseline_server_config() {
  local server_log="$1"
  local deterministic="$2"
  local tp_size="$3"
  local deterministic_value="False"

  if [[ "${deterministic}" == "1" ]]; then
    deterministic_value="True"
  fi

  for expected in \
    "speculative_algorithm=None" \
    "enable_deterministic_inference=${deterministic_value}"; do
    if ! grep -q "${expected}" "${server_log}"; then
      echo "Baseline server did not resolve expected config: ${expected}" >&2
      tail -120 "${server_log}" >&2 || true
      return 1
    fi
  done

  # Ordinary TP deterministic inference uses the upstream deterministic
  # collective policy. DVR may override collectives only inside draft graphs.
  if [[ "${deterministic}" == "1" ]] && ((tp_size > 1)) && \
     ! grep -q "disable_custom_all_reduce=True" "${server_log}"; then
    echo "Ordinary deterministic TP baseline did not disable custom all-reduce." >&2
    tail -120 "${server_log}" >&2 || true
    return 1
  fi
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
