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
    if name in stage_names and event.get("cat") == "user_annotation":
        durations[name].append(event["dur"] / 1000.0)

print(f"cuda_graph_launches={graph_launches}")
for name, values in sorted(durations.items()):
    print(
        f"{name}: count={len(values)} mean_ms={statistics.mean(values):.3f} "
        f"median_ms={statistics.median(values):.3f} total_ms={sum(values):.3f}"
    )

cpu_stage_events = collections.defaultdict(list)
for event in events:
    if (
        event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and event.get("name") in {"dvr_prepare", "draft", "verify_prepare"}
    ):
        cpu_stage_events[event["name"]].append(event)
for group in cpu_stage_events.values():
    group.sort(key=lambda event: event["ts"])
if all(cpu_stage_events.get(name) for name in ("dvr_prepare", "draft", "verify_prepare")):
    context_entry_us = []
    context_exit_and_glue_us = []
    for prepare, draft, verify_prepare in zip(
        cpu_stage_events["dvr_prepare"],
        cpu_stage_events["draft"],
        cpu_stage_events["verify_prepare"],
    ):
        context_entry_us.append(
            draft["ts"] - (prepare["ts"] + prepare["dur"])
        )
        context_exit_and_glue_us.append(
            verify_prepare["ts"] - (draft["ts"] + draft["dur"])
        )
    print(
        "draft_context_gate: "
        f"entry_upper_mean_us={statistics.mean(context_entry_us):.1f} "
        "exit_and_verify_glue_upper_mean_us="
        f"{statistics.mean(context_exit_and_glue_us):.1f}"
    )

# Estimate the best possible gain from replacing the per-step self-draft graphs
# with one chain graph. Pick the GPU annotation stream that covers the widest
# draft span, then union all kernels in each span. The uncovered fraction is a
# conservative ceiling: a chain graph cannot remove model compute and will not
# eliminate every dependency gap counted here.
gpu_drafts_by_tid = collections.defaultdict(list)
for event in events:
    if (
        event.get("ph") == "X"
        and event.get("name") == "draft"
        and event.get("cat") == "gpu_user_annotation"
    ):
        gpu_drafts_by_tid[event.get("tid")].append(event)

if gpu_drafts_by_tid:
    draft_spans = max(
        gpu_drafts_by_tid.values(),
        key=lambda group: statistics.mean(event["dur"] for event in group),
    )
    kernels = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") == "kernel"
    ]
    total_span_us = 0.0
    total_busy_us = 0.0
    for draft in draft_spans:
        start = draft["ts"]
        end = start + draft["dur"]
        intervals = sorted(
            (max(start, event["ts"]), min(end, event["ts"] + event["dur"]))
            for event in kernels
            if event["ts"] < end and event["ts"] + event["dur"] > start
        )
        merged = []
        for interval_start, interval_end in intervals:
            if merged and interval_start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], interval_end))
            else:
                merged.append((interval_start, interval_end))
        total_span_us += draft["dur"]
        total_busy_us += sum(end - start for start, end in merged)

    cpu_drafts = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("name") == "draft"
        and event.get("cat") == "user_annotation"
    ]
    draft_graph_launches = []
    for draft in cpu_drafts:
        start = draft["ts"]
        end = start + draft["dur"]
        draft_graph_launches.append(
            sum(
                event.get("ph") == "X"
                and event.get("name") == "cudaGraphLaunch"
                and start <= event.get("ts", 0) < end
                for event in events
            )
        )

    utilization = total_busy_us / total_span_us if total_span_us else 0.0
    speedup_ceiling = total_span_us / total_busy_us if total_busy_us else 0.0
    print(
        "self_draft_chain_gate: "
        f"iterations={len(draft_spans)} "
        f"graph_launches_per_iteration={statistics.mean(draft_graph_launches):.1f} "
        f"gpu_kernel_busy_fraction={utilization:.4f} "
        f"perfect_chain_speedup_ceiling={speedup_ceiling:.4f}x"
    )
PY

echo "Profile written under ${PROFILE_DIR}"
