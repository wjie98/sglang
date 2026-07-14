#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

BASE_URL="${BASE_URL:-http://127.0.0.1:30124}"
RESULT_ROOT="${RESULT_ROOT:-${DVR_REPO_ROOT}/../dvr-fixed-validation/latest-run/profile}"
PROFILE_NUM_STEPS="${PROFILE_NUM_STEPS:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SERVER_LOG="${SERVER_LOG:-}"
PROFILE_DIR="${RESULT_ROOT}/torch-profile"

mkdir -p "${RESULT_ROOT}"

request_payload="$(printf '{"text":"Explain why deterministic verification can validate provisional decoding.","sampling_params":{"temperature":0,"max_new_tokens":%s,"ignore_eos":true}}' "${MAX_NEW_TOKENS}")"
profile_payload="$(printf '{"output_dir":"%s","num_steps":%s,"activities":["CPU","GPU"],"record_shapes":false,"with_stack":false,"profile_prefix":"dvr-decode"}' "${PROFILE_DIR}" "${PROFILE_NUM_STEPS}")"

# Warm kernels before collecting a decode-only trace.
curl -fsS -H 'Content-Type: application/json' \
  -d '{"text":"Warm up DVR profiling.","sampling_params":{"temperature":0,"max_new_tokens":32,"ignore_eos":true}}' \
  "${BASE_URL}/generate" >"${RESULT_ROOT}/warmup_response.json"

curl -fsS -H 'Content-Type: application/json' \
  -d "${profile_payload}" "${BASE_URL}/start_profile" \
  >"${RESULT_ROOT}/start_profile_response.json"
curl -fsS -H 'Content-Type: application/json' \
  -d "${request_payload}" "${BASE_URL}/generate" \
  >"${RESULT_ROOT}/profile_response.json"

if [[ -n "${SERVER_LOG}" ]]; then
  grep -E 'Capture (target verify |DVR self-draft )?CUDA graph|Capturing batches|max_total_num_tokens=' \
    "${SERVER_LOG}" >"${RESULT_ROOT}/graph_memory_summary.log" || true
fi

PROFILE_DIR="${PROFILE_DIR}" python3 - <<'PY' >"${RESULT_ROOT}/profile_summary.txt"
import collections
import glob
import gzip
import json
import os
import statistics

paths = glob.glob(os.path.join(os.environ["PROFILE_DIR"], "*.trace.json.gz"))
if not paths:
    raise SystemExit("No torch profiler trace was produced.")
with gzip.open(paths[0], "rt") as f:
    events = json.load(f)["traceEvents"]

stage_names = {
    "draft", "dvr_prepare", "verify_prepare", "dvr_state_restore", "verify",
    "verify_sample", "verify_logprob", "dvr_rollback", "dvr_checkpoint",
    "draft_extend",
}
durations = collections.defaultdict(list)
graph_launches = 0
for event in events:
    if event.get("ph") != "X":
        continue
    name = event.get("name")
    if name == "cudaGraphLaunch":
        graph_launches += 1
    if name in stage_names:
        durations[name].append(event["dur"] / 1000.0)

print(f"cuda_graph_launches={graph_launches}")
for name, values in sorted(durations.items()):
    print(
        f"{name}: count={len(values)} mean_ms={statistics.mean(values):.3f} "
        f"median_ms={statistics.median(values):.3f} total_ms={sum(values):.3f}"
    )
PY

echo "Profile written under ${PROFILE_DIR}"
