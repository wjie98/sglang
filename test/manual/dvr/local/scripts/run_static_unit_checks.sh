#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# Keep this list explicit: adding a DVR touchpoint in a shared module must also
# extend the static guard instead of being hidden by a broad directory scan.
DVR_PY_FILES=(
  python/sglang/srt/distributed/parallel_state.py
  python/sglang/srt/layers/attention/fla/chunk_delta_h.py
  python/sglang/srt/layers/attention/fla/chunk.py
  python/sglang/srt/layers/attention/fla/chunk_o.py
  python/sglang/srt/layers/attention/linear/dvr_gdn.py
  python/sglang/srt/layers/attention/linear/dvr_state.py
  python/sglang/srt/layers/attention/linear/gdn_backend.py
  python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py
  python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py
  python/sglang/srt/layers/elementwise.py
  python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py
  python/sglang/srt/managers/scheduler.py
  python/sglang/srt/managers/tokenizer_manager.py
  python/sglang/srt/mem_cache/kv_cache_builder.py
  python/sglang/srt/mem_cache/memory_pool.py
  python/sglang/srt/model_executor/dvr_draft_cuda_graph_runner.py
  python/sglang/srt/model_executor/model_runner.py
  python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
  python/sglang/srt/model_executor/pool_configurator.py
  python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
  python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py
  python/sglang/srt/server_args.py
  python/sglang/srt/speculative/dvr_server_args.py
  python/sglang/srt/speculative/dvr_state_flow.py
  python/sglang/srt/speculative/dvr_worker.py
  python/sglang/srt/speculative/eagle_utils.py
  python/sglang/srt/speculative/spec_info.py
  python/sglang/srt/speculative/spec_registry.py
  test/manual/dvr/local/clients/dvr_batch_kl.py
  test/manual/dvr/local/clients/dvr_eagle_acceptance.py
  test/manual/dvr/local/clients/dvr_radix_lifecycle.py
  test/registered/unit/layers/test_dvr_gdn.py
  test/manual/layers/test_fused_gate_sigmoid_mul_add.py
  test/registered/unit/model_executor/test_dvr_cuda_graph_runner.py
  test/registered/unit/server_args/test_dvr_server_args.py
  test/registered/unit/speculative/test_dvr_state_flow.py
  test/registered/unit/speculative/test_dvr_worker.py
)

DVR_DIFF_BASE="${DVR_DIFF_BASE:-upstream/sglang-miles}"
git rev-parse --verify "${DVR_DIFF_BASE}^{commit}" >/dev/null
git diff --check "${DVR_DIFF_BASE}...HEAD"
git diff --check
bash -n test/manual/dvr/launch_server.sh test/manual/dvr/local/scripts/*.sh
conda_python -m py_compile "${DVR_PY_FILES[@]}"
conda_python -m pytest \
  test/registered/unit/speculative/test_dvr_worker.py \
  test/registered/unit/speculative/test_dvr_state_flow.py \
  test/registered/unit/model_executor/test_dvr_cuda_graph_runner.py \
  test/registered/unit/server_args/test_dvr_server_args.py \
  test/registered/unit/layers/test_dvr_gdn.py \
  test/manual/layers/test_fused_gate_sigmoid_mul_add.py::test_large_batch_warp_policy_follows_deterministic_mode \
  test/registered/unit/layers/test_mamba_state_scatter_triton.py::TestMambaStateScatterCorrectness::test_fused_supports_explicit_source_rows \
  test/registered/attention/test_chunk_gated_delta_rule.py::TestChunkGatedDeltaRule::test_boundary_state_preserves_initial_state_dtype \
  test/registered/attention/test_chunk_gated_delta_rule.py::TestChunkGatedDeltaRule::test_non_inplace_verify_preserves_initial_state \
  test/registered/unit/model_executor/test_pool_configurator.py::TestEagleConfigurator::test_dvr_eagle_does_not_exceed_budget
