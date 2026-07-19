# DVR H20/NVLink release qualification

This guide drives the checked-in local scripts. It deliberately keeps machine
paths and release thresholds outside the upstream-facing DVR documentation.

## 1. Environment

Run from the repository root. Override paths for the machine under test:

```bash
export CONDA_ENV=dvr_dev
export MODEL_0P8B=/path/to/Qwen3.5-0.8B
export MODEL_35B=/path/to/Qwen3.5-35B-A3B
export MODEL_80B=/path/to/Qwen3-Next-80B-A3B-Instruct
export SHAREGPT_DATASET=/path/to/ShareGPT.json
export LONGBENCH_DATASET=/path/to/LongBench
export RESULT_ROOT=/path/to/dvr-results
```

Each script sets this worktree's `PYTHONPATH`, writes commit and GPU topology
metadata, and checks that the requested server configuration was actually
resolved. Keep `CUDA_VISIBLE_DEVICES`, TP size, backend, and result directory in
the final report. The scripts use `RANDOM_SEED=2026` for server startup; keep it
fixed together with the benchmark client seed.

## 2. Static gate

```bash
bash test/manual/dvr/local/scripts/run_static_unit_checks.sh
```

This must pass before allocating a multi-GPU model.

## 3. DeepGEMM preparation

On H20, compile the exact target-verify GEMM shapes before CUDA graph capture.
Use a cache dedicated to the source revision and driver/toolchain combination:

```bash
export SGLANG_DG_CACHE_DIR=/path/to/cache
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=1
DRAFT_TOKEN_COUNTS="2 16 32" \
bash test/manual/dvr/local/scripts/prepare_h20_deep_gemm.sh
```

For the formal run, retain the cache and set:

```bash
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export REQUIRE_PRECOMPILED_DEEPGEMM=1
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=0
```

Do not use a globally DeepGEMM-disabled run as the release performance number.
If formal serving enters another exhaustive precompile session, the cache is
incomplete and the result is invalid.

## 4. Correctness matrix

Run self-DVR first:

```bash
MODEL_PATH="${MODEL_0P8B}" \
RESULT_ROOT="${RESULT_ROOT}/0p8b-self" \
bash test/manual/dvr/local/scripts/run_0p8b_self_dvr_kl.sh

MODEL_PATH="${MODEL_0P8B}" \
DRAFT_TOKENS=32 \
RESULT_ROOT="${RESULT_ROOT}/0p8b-self-dvr32" \
bash test/manual/dvr/local/scripts/run_0p8b_self_dvr_kl.sh
```

Required coverage:

- synchronous and overlap workers;
- `return_logprob=True` and `False`;
- prompt and output lengths around 63/64/65;
- one-token prompt sentinel;
- concurrent shared-prefix requests and request-slot reuse;
- at least one 512-token generation;
- Radix enabled, plus a separately labelled Radix-disabled run.

Then run the Qwen3.5 MTP path:

```bash
MODEL_PATH="${MODEL_35B}" \
DRAFT_MODEL_PATH="${MODEL_35B}" \
RESULT_ROOT="${RESULT_ROOT}/35b-eagle" \
bash test/manual/dvr/local/scripts/run_35b_mtp_eagle_smoke.sh
```

Both sync and overlap must satisfy the strict full-prefill KL oracle. Compare
aggregate real-data acceptance instead of requiring identical stochastic token
trajectories across batch shapes or schedules.

## 5. Throughput matrix

Run matched normal, ordinary deterministic, and DVR configurations:

```bash
MODEL_PATH="${MODEL_35B}" \
RESULT_ROOT="${RESULT_ROOT}/35b-throughput" \
bash test/manual/dvr/local/scripts/run_35b_dvr_throughput.sh

MODEL_PATH="${MODEL_80B}" \
RESULT_ROOT="${RESULT_ROOT}/80b-throughput" \
bash test/manual/dvr/local/scripts/run_80b_self_dvr_throughput.sh
```

For each backend and logprob mode, compare only configurations with identical
TP, scheduling, Radix, page size, request count, output length, effective
concurrency, and cache policy. The scripts default to
`FLUSH_CACHE_EACH_RUN=1`: warmup compiles kernels, then the measured requests
start from an empty prefix cache. Cover Triton and FA3 where the model and
machine support them.

Run self-DVR with both chain widths. Keep separate result directories because
the target-verify graph and DeepGEMM shapes differ:

```bash
for draft_tokens in 16 32; do
  MODEL_PATH="${MODEL_80B}" \
  DRAFT_TOKENS="${draft_tokens}" \
  RESULT_ROOT="${RESULT_ROOT}/80b-dvr${draft_tokens}" \
  bash test/manual/dvr/local/scripts/run_80b_self_dvr_throughput.sh
done
```

`dvr16` remains the latency-oriented release configuration. `dvr32` is also a
diagnostic: if it recovers target efficiency while draft decode remains
unchanged, the remaining gap is fixed verify/state cost rather than an overlap
scheduler bubble. Report its acceptance and absolute throughput as well: a
longer draft chain can improve implementation efficiency while reducing both.

For a production-warm-cache measurement, run a second, clearly labelled matrix
with `FLUSH_CACHE_EACH_RUN=0`. Do not compare one policy's DVR result with the
other policy's baseline. Every JSON row contains `cache_report`; reject a cold
comparison if prompt/cache totals differ unexpectedly between modes.

Report:

```text
acceptance_fraction = accept_length / draft_tokens
target_tps = matching_normal_tps * acceptance_fraction
target_efficiency = dvr_tps / target_tps
det_speedup = dvr_tps / matching_ordinary_deterministic_tps
```

The NVLink target is `target_efficiency >= 0.95` after warmup. DVR must also be
faster than the matched ordinary deterministic baseline. Record lower results
rather than changing benchmark knobs in place.

## 6. Log audit

Before accepting a result, confirm from the server log that:

- the expected Triton or FA3 backend was selected;
- GDN linear-attention prefill uses Triton;
- requested CUDA graph batch sizes were captured without fallback;
- target prefill/verify remain deterministic;
- provisional draft capture uses the fast decode settings;
- custom all-reduce is confined to provisional draft graphs;
- the server did not reduce `max_running_requests`;
- no illegal memory access, device assertion, or graph replay fallback occurred;
- formal serving did not enter an unplanned DeepGEMM precompile session.

Attach `run_metadata.txt`, `summary.txt`, server logs, script overrides, and the
exact commit to the qualification report.

## 7. Decode-only profiler

Profile already-running normal-sync, normal-overlap, DVR-sync, and DVR-overlap
servers independently. Use a unique output directory for every trace:

```bash
BASE_URL=http://127.0.0.1:30000 \
SERVER_LOG=/path/to/server.log \
PROFILE_BATCH_SIZE=2 \
RESULT_ROOT="${RESULT_ROOT}/profiles/dvr16-triton-overlap" \
bash test/manual/dvr/local/scripts/profile_dvr_server.sh
```

Repeat for Triton and FA3. The decisive checks are:

- self-draft time per draft step is close to normal non-deterministic overlap
  decode on the same backend;
- `unaccounted_gap_ms` is near zero;
- FA3 D2H copies and host `verify_prepare` do not dominate the iteration;
- reducing target-verify and state-maintenance time, not graph launch count,
  explains any dvr32 gain.

Use `PROFILE_BATCH_SIZE` values that match the throughput matrix. The helper
waits for asynchronous trace export and rejects stale traces from an earlier
run. For finite workloads, compare steady-state iteration periods separately
from pipeline fill/drain time.

Do not implement a whole-chain self-draft graph solely to remove launches. The
existing profiler reports `perfect_chain_speedup_ceiling`; prior A40 traces put
that ceiling near 1.015x.
