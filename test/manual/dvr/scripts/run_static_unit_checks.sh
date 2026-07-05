#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

mapfile -t DVR_PY_FILES < <(
  find \
    python/sglang/srt/speculative \
    python/sglang/srt/model_executor \
    python/sglang/srt/layers/attention/linear \
    -maxdepth 1 -name 'dvr_*.py' -print | sort
)
DVR_PY_FILES+=(python/sglang/srt/speculative/eagle_info_v2.py)

git diff --check
conda_python -m py_compile "${DVR_PY_FILES[@]}"
conda_python -m pytest \
  test/registered/unit/speculative/test_dvr_spec_policy.py \
  test/registered/unit/server_args/test_dvr_server_args.py \
  test/registered/unit/layers/test_dvr_gdn_state_input_cache.py
