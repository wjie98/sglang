#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# Keep this list explicit: adding a DVR touchpoint in a shared module must also
# extend the static guard instead of being hidden by a broad directory scan.
DVR_PY_FILES=(
  python/sglang/srt/layers/attention/fla/chunk_delta_h.py
  python/sglang/srt/layers/attention/fla/chunk_o.py
  python/sglang/srt/layers/attention/linear/dvr_gdn.py
  python/sglang/srt/layers/attention/linear/dvr_state.py
  python/sglang/srt/layers/attention/linear/gdn_backend.py
  python/sglang/srt/managers/scheduler_components/batch_result_processor.py
  python/sglang/srt/managers/tokenizer_manager.py
  python/sglang/srt/managers/utils.py
  python/sglang/srt/mem_cache/kv_cache_builder.py
  python/sglang/srt/mem_cache/memory_pool.py
  python/sglang/srt/model_executor/dvr_draft_cuda_graph_runner.py
  python/sglang/srt/model_executor/model_runner.py
  python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
  python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
  python/sglang/srt/server_args.py
  python/sglang/srt/speculative/dvr_server_args.py
  python/sglang/srt/speculative/dvr_state_flow.py
  python/sglang/srt/speculative/dvr_worker.py
  python/sglang/srt/speculative/eagle_utils.py
  python/sglang/srt/speculative/spec_info.py
  python/sglang/srt/speculative/spec_registry.py
  test/manual/dvr/test_dvr_batch_kl.py
  test/manual/dvr/test_dvr_eagle_acceptance.py
)

git diff --check
bash -n test/manual/dvr/scripts/*.sh
conda_python -m py_compile "${DVR_PY_FILES[@]}"
conda_python -m pytest \
  test/registered/unit/speculative/test_dvr_contracts.py \
  test/registered/unit/server_args/test_dvr_server_args.py \
  test/registered/unit/layers/test_dvr_gdn_state_input_cache.py
