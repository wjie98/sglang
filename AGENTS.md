# DVR agent notes

## Fixed validation scripts

DVR validation must use the checked-in scripts under
`test/manual/dvr/scripts/` instead of hand-composed one-off commands.  The goal
is to keep KL, acceptance, and throughput results comparable across code
changes and machines.

Use `test/manual/dvr/H20_RELEASE_VALIDATION.md` for the final H20/NVLink
qualification and release gates. It defines the backend matrix, upstream A/B,
custom all-reduce ownership check, result layout, and pass/fail thresholds.

Use these entry points:

- `test/manual/dvr/scripts/run_static_unit_checks.sh`
  - `git diff --check`
  - DVR-focused `py_compile`
  - DVR unit tests
- `test/manual/dvr/scripts/prepare_h20_deep_gemm.sh`
  - graph-disabled deterministic forwards that precompile the exact
    `batch_size * draft_tokens` target-verify GEMM shapes
  - run once per H20 model/cache before enabling CUDA graph serving; do not use
    a globally disabled DeepGEMM run as a release throughput result
- `test/manual/dvr/scripts/run_0p8b_self_dvr_kl.sh`
  - 0.8B self-DVR spec v1/v2 KL=0 and boundary smoke
  - includes one-token prompts in concurrent and batch modes; their first DVR
    iteration uses a zero-step, one-root target-verify sentinel and must pass
    the same strict full-prefill KL oracle
  - includes an interleaved shared-prefix request pair that exercises prefill
    result processing while another request owns the worker, plus concurrent
    and batch 512-token strict KL checks
  - includes a completed-generation replay case: a 65-token prompt generates
    128 tokens, the next request must reuse the prompt's 64-token checkpoint,
    rebuild the generated suffix with ordinary EXTEND, and still match the
    flushed full-prefill oracle exactly
  - with radix enabled, runs `test_dvr_radix_lifecycle.py` to cover same-rid
    request-slot reuse, nearest-checkpoint replay around 64/128 tokens, and
    stop-token truncation; pass `--exhaustive` to that client for the full
    prompt 63/64/65 by generated length 1..129 scan
  - `ATTENTION_BACKEND` may select Triton or FA3; the fixed default remains
    `triton`
  - `RUN_V1=0` or `RUN_V2=0` may resume one half of an interrupted run in the
    same `RESULT_ROOT`; the default remains the complete v1/v2 matrix
- `test/manual/dvr/scripts/run_35b_mtp_eagle_smoke.sh`
  - 35B Qwen3.5 MTP/DVR-EAGLE sync-v2 and overlap-v2 smoke
  - covers `return_logprob=True` KL and `return_logprob=False` output path
  - includes the seeded `prompt_len=65`, `max_new=65` boundary acceptance
    regression; stochastic rejection runs use a `0.75` floor
  - includes a staggered shared-prefix KL case so request-local GDN boundary
    ownership is checked with radix enabled
  - applies the same nearest-checkpoint replay check to both sync and overlap
    DVR-EAGLE
  - ShareGPT cases use a separate `0.70` floor based on the fixed two-token
    MTP baseline; this catches routing regressions without pretending real-data
    acceptance should be nearly one
- `test/manual/dvr/scripts/run_35b_dvr_throughput.sh`
  - matching 35B non-deterministic normal, ordinary deterministic,
    self-v1/v2, and EAGLE sync/overlap ShareGPT runs
  - runs TP=4 self-DVR KL boundary checks before its throughput pair
  - records one ordinary-deterministic GDN oracle probe without treating an
    observed nonzero mismatch as a DVR test failure
  - fixes 8 requests, 512 output tokens, concurrency 3, and covers
    `return_logprob=True/False`
- `test/manual/dvr/scripts/run_80b_self_dvr_throughput.sh`
  - Qwen3-Next 80B non-deterministic normal, ordinary deterministic, and
    self-DVR spec v1/v2 long-output throughput
  - covers ShareGPT and fixed LongBench custom-cache inputs
  - covers `return_logprob=True/False`
- `test/manual/dvr/scripts/profile_dvr_server.sh`
  - profiles an already running DVR server through SGLang's standard profiler
  - distinguishes host enqueue spans (`host_*`) from real GPU spans (`gpu_*`)
  - emits decode/DVR iteration timelines and extracts graph-memory startup logs
  - is development-only and must not be mixed with throughput measurements

The scripts accept the common overrides `CONDA_ENV`, `MODEL_PATH`, `PORT`,
`RESULT_ROOT`, `TP_SIZE`, `PAGE_SIZE`, `ATTENTION_BACKEND`, and
`LINEAR_ATTN_BACKEND`; the 35B/80B scripts also expose dataset/model-specific
variables near the top of each file. Every run writes commit, GPU, and topology
metadata under `RESULT_ROOT`. When reporting a run, include the script path,
key overrides, result directory, and whether the server log confirms the
expected backend, radix mode, CUDA graph, and effective concurrency.

`DISABLE_RADIX_CACHE` accepts `0`, `1`, or `auto`. `auto` keeps radix enabled
for both supported full-attention backends. A separately labeled radix-disabled
run remains part of the state-lifecycle matrix.

The production sampling path always uses exact rejection sampling. Sampling
follows SGLang's ordinary EAGLE RNG contract; do not require token-for-token or
histogram identity across different batch shapes and overlap schedules. Strict
KL replay and real-data aggregate acceptance are the correctness guards.

Do not use the removed `SGLANG_ENABLE_SPEC_V2` environment variable.  The fixed
scripts select the self-DVR v1 compatibility worker with
`--disable-overlap-schedule`; omitting that flag uses the spec-v2 overlap worker.

Do not invoke `pytest` from an activated environment without setting this
worktree's `PYTHONPATH`; the editable install may point at another DVR worktree.
The fixed scripts set `PYTHONPATH` to the repository under test.

Do not silently change benchmark knobs such as request count, output length,
max concurrency, `--max-mamba-cache-size`, overlap mode, backend, or returned
logprob handling.  If a new experiment needs different knobs, create a separate
script or clearly mark it as a new baseline and keep the fixed scripts intact.

The throughput scripts pin `--max-mamba-cache-size 16` and fail if the runtime
reduces the requested concurrency or omits its largest CUDA graph batch.
Their concurrency knobs must be overridden together with enough Mamba cache
capacity; otherwise the run is not a valid comparison. The throughput scripts
launch two no-DVR baseline classes in matching sync and overlap modes.
`baseline_sync/baseline_overlap` are non-deterministic normal serving and are
used only for acceptance-weighted implementation efficiency.
`det_sync/det_overlap` are ordinary no-DVR serving with
`--enable-deterministic-inference` and are used only for the user-facing DVR
speedup. Compare v1 only with sync and v2 only with overlap; all sides use the
same page size, backend, radix mode, TP, and returned-logprob setting. Never use
ordinary det as the denominator of `acceptance_x_baseline`, and never use
non-deterministic normal to claim DVR is faster than deterministic serving.
`RUN_BASELINE=0` disables both baseline classes by default;
`RUN_DETERMINISTIC_BASELINE`, `RUN_DET_SYNC`, and `RUN_DET_OVERLAP` may override
that behavior only when resuming an interrupted matrix in the same
`RESULT_ROOT`.

Throughput reported by `bench_serving` already includes the speculative
acceptance benefit. Compute `acceptance_fraction = accept_length / draft_tokens`,
`target_tps = matching_baseline_tps * acceptance_fraction`, and
`target_efficiency = dvr_tps / target_tps`. Never multiply DVR throughput by
acceptance a second time. Separately compute
`det_speedup = dvr_tps / matching_ordinary_det_tps`. The fixed scripts emit
both metrics and write them to `summary.txt`.

The 35B script exposes `CONTEXT_LENGTH` and `MAX_TOTAL_TOKENS` for separately
reported large-batch EAGLE runs. Keep its defaults for the fixed BS=3 matrix;
when increasing `MAX_CONCURRENCY`, set both capacity overrides explicitly and
retain the script's server-capacity check so a silently reduced batch is never
reported as the requested large-batch result.
