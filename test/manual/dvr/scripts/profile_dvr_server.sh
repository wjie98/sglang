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

profile_input_ids='[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]'
request_payload="$(printf '{"input_ids":%s,"sampling_params":{"temperature":0,"max_new_tokens":%s,"ignore_eos":true}}' "${profile_input_ids}" "${MAX_NEW_TOKENS}")"
profile_payload="$(printf '{"output_dir":"%s","num_steps":%s,"activities":["CPU","GPU"],"record_shapes":false,"with_stack":false,"profile_prefix":"dvr-decode"}' "${PROFILE_DIR}" "${PROFILE_NUM_STEPS}")"

# Warm kernels before collecting a decode-only trace.
curl -fsS -H 'Content-Type: application/json' \
  -d "{\"input_ids\":${profile_input_ids},\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":32,\"ignore_eos\":true}}" \
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
trace_path = max(paths, key=os.path.getmtime)
with gzip.open(trace_path, "rt") as f:
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

print(f"trace={os.path.basename(trace_path)}")
print(f"cuda_graph_launches={graph_launches}")
for name, values in sorted(durations.items()):
    print(
        f"host_{name}: count={len(values)} "
        f"mean_ms={statistics.mean(values):.3f} "
        f"median_ms={statistics.median(values):.3f} total_ms={sum(values):.3f}"
    )

# Host annotations measure enqueue time. GPU annotations carry the actual stage
# duration and may appear on more than one stream when a graph uses child
# streams. Select the stream with the largest total span for each stage.
gpu_stage_names = stage_names | {
    "scheduler.run_batch",
    "scheduler.get_next_batch_to_run",
}
gpu_stage_groups = collections.defaultdict(lambda: collections.defaultdict(list))
for event in events:
    if (
        event.get("ph") == "X"
        and event.get("cat") == "gpu_user_annotation"
        and (
            event.get("name") in gpu_stage_names
            or event.get("name", "").startswith("step[DECODE ")
            or event.get("name", "").startswith("step[TARGET_VERIFY ")
        )
    ):
        gpu_stage_groups[event["name"]][event.get("tid")].append(event)

gpu_stages = {}
for name, by_stream in gpu_stage_groups.items():
    stream, selected = max(
        by_stream.items(), key=lambda item: sum(event["dur"] for event in item[1])
    )
    selected.sort(key=lambda event: event["ts"])
    gpu_stages[name] = selected
    values = [event["dur"] / 1000.0 for event in selected]
    print(
        f"gpu_{name}: stream={stream} count={len(values)} "
        f"mean_ms={statistics.mean(values):.3f} "
        f"median_ms={statistics.median(values):.3f} total_ms={sum(values):.3f}"
    )


def start_period_ms(stage_events):
    return [
        (cur["ts"] - prev["ts"]) / 1000.0
        for prev, cur in zip(stage_events, stage_events[1:])
    ]


decode_name = next(
    (name for name in gpu_stages if name.startswith("step[DECODE ")), None
)
if decode_name is not None and len(gpu_stages[decode_name]) > 1:
    decode_events = gpu_stages[decode_name]
    periods = start_period_ms(decode_events)
    decode_ms = statistics.mean(event["dur"] for event in decode_events) / 1000.0
    period_ms = statistics.median(periods)
    print(
        "decode_timeline: "
        f"gpu_decode_mean_ms={decode_ms:.3f} "
        f"start_period_median_ms={period_ms:.3f} "
        f"inter_iteration_gap_ms={max(0.0, period_ms - decode_ms):.3f}"
    )

draft_events = gpu_stages.get("draft")
verify_name = next(
    (name for name in gpu_stages if name.startswith("step[TARGET_VERIFY ")), None
)
if draft_events and len(draft_events) > 1 and verify_name is not None:
    periods = start_period_ms(draft_events)
    period_ms = statistics.median(periods)
    draft_ms = statistics.mean(event["dur"] for event in draft_events) / 1000.0
    verify_ms = (
        statistics.mean(event["dur"] for event in gpu_stages[verify_name]) / 1000.0
    )
    maintenance_names = (
        "verify_prepare",
        "dvr_state_restore",
        "verify_sample",
        "verify_logprob",
        "dvr_rollback",
        "dvr_checkpoint",
        "draft_extend",
    )
    maintenance_ms = sum(
        statistics.mean(event["dur"] for event in gpu_stages[name]) / 1000.0
        for name in maintenance_names
        if name in gpu_stages
    )
    accounted_ms = draft_ms + verify_ms + maintenance_ms
    print(
        "dvr_iteration_timeline: "
        f"start_period_median_ms={period_ms:.3f} "
        f"draft_ms={draft_ms:.3f} target_verify_ms={verify_ms:.3f} "
        f"maintenance_ms={maintenance_ms:.3f} "
        f"unaccounted_gap_ms={max(0.0, period_ms - accounted_ms):.3f} "
        f"target_verify_fraction={verify_ms / period_ms:.4f}"
    )

# CUDA graph children are not reliably nested under the host annotation that
# launched them. Coarse kernel totals still show whether GDN output or DVR state
# maintenance is worth a more invasive backend change on this machine.
kernel_groups = {
    "gdn_verify_recurrence": (
        "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "recompute_w_u_fwd_kernel",
    ),
    "gdn_verify_output": ("chunk_fwd_kernel_o",),
    "dvr_state_rebuild": ("_dvr_gdn_rebuild_live_state_kernel",),
    "dvr_state_scatter": (
        "_fused_mamba_state_scatter_with_mask_kernel",
        "_fused_conv_window_scatter_with_mask_kernel",
    ),
}
kernels = [
    event
    for event in events
    if event.get("ph") == "X" and event.get("cat") == "kernel"
]
for group, names in kernel_groups.items():
    matched = [event for event in kernels if event.get("name") in names]
    print(
        f"{group}: launches={len(matched)} "
        f"total_ms={sum(event['dur'] for event in matched) / 1000.0:.3f}"
    )

memcpy = collections.defaultdict(lambda: [0, 0, 0.0])
for event in events:
    if event.get("ph") != "X" or event.get("cat") != "gpu_memcpy":
        continue
    name = event.get("name", "unknown")
    memcpy[name][0] += 1
    memcpy[name][1] += int(event.get("args", {}).get("bytes", 0))
    memcpy[name][2] += event["dur"] / 1000.0
for name, (count, num_bytes, total_ms) in sorted(memcpy.items()):
    print(f"{name}: count={count} bytes={num_bytes} total_ms={total_ms:.3f}")

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
