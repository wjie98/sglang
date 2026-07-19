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
the final report.

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
bash test/manual/dvr/local/scripts/prepare_h20_deep_gemm.sh
```

For the formal run, retain the cache and set:

```bash
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export REQUIRE_PRECOMPILED_DEEPGEMM=1
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
TP, scheduling, Radix, page size, request count, output length, and effective
concurrency. Cover Triton and FA3 where the model and machine support them.

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
