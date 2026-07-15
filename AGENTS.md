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
  - includes concurrent and batch 512-token strict KL checks
  - `ATTENTION_BACKEND` may select a separately reported Triton/FlashInfer/FA3
    compatibility run; the fixed default remains `triton`
  - `RUN_V1=0` or `RUN_V2=0` may resume one half of an interrupted run in the
    same `RESULT_ROOT`; the default remains the complete v1/v2 matrix
- `test/manual/dvr/scripts/run_35b_mtp_eagle_smoke.sh`
  - 35B Qwen3.5 MTP/DVR-EAGLE sync-v2 and overlap-v2 smoke
  - covers `return_logprob=True` KL and `return_logprob=False` output path
  - includes the seeded `prompt_len=65`, `max_new=65` boundary acceptance
    regression and fails if reported accept rate drops below `0.96`
  - ShareGPT cases use a separate `0.70` floor based on the fixed two-token
    MTP baseline; this catches routing regressions without pretending real-data
    acceptance should be nearly one
- `test/manual/dvr/scripts/run_35b_dvr_throughput.sh`
  - matching 35B no-DVR, self-v1/v2, and EAGLE sync/overlap ShareGPT runs
  - runs TP=4 self-DVR KL boundary checks before its throughput pair
  - fixes 8 requests, 512 output tokens, concurrency 3, and covers
    `return_logprob=True/False`
- `test/manual/dvr/scripts/run_80b_self_dvr_throughput.sh`
  - Qwen3-Next 80B no-DVR baseline and self-DVR spec v1/v2 long-output throughput
  - covers ShareGPT and fixed LongBench custom-cache inputs
  - covers `return_logprob=True/False`
- `test/manual/dvr/scripts/profile_dvr_server.sh`
  - profiles an already running DVR server through SGLang's standard profiler
  - emits the coarse DVR stage spans and extracts graph-memory startup logs
  - is development-only and must not be mixed with throughput measurements

The scripts accept the common overrides `CONDA_ENV`, `MODEL_PATH`, `PORT`,
`RESULT_ROOT`, `TP_SIZE`, `PAGE_SIZE`, `ATTENTION_BACKEND`, and
`LINEAR_ATTN_BACKEND`; the 35B/80B scripts also expose dataset/model-specific
variables near the top of each file. Every run writes commit, GPU, and topology
metadata under `RESULT_ROOT`. When reporting a run, include the script path,
key overrides, result directory, and whether the server log confirms the
expected backend, radix mode, CUDA graph, and effective concurrency.

`DISABLE_RADIX_CACHE` accepts `0`, `1`, or `auto`. `auto` disables radix for
FlashInfer and keeps it for Triton/FA3. DVR must preserve request-local GDN
checkpoints in both cases; do not add a FlashInfer-specific correctness path.

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
launch separate no-DVR sync and overlap baselines. Compare v1/sync only with
`baseline_sync`, and v2/overlap only with `baseline_overlap`; all sides use the
same page size, backend, radix mode, TP, and returned-logprob setting. The 80B
default run includes both baselines, v1, and v2;
`RUN_BASELINE=0` or `RUN_DVR=0` may be used only to resume an interrupted
matrix in the same `RESULT_ROOT`.

Throughput reported by `bench_serving` already includes the speculative
acceptance benefit. Compute `acceptance_fraction = accept_length / draft_tokens`,
`target_tps = matching_baseline_tps * acceptance_fraction`, and
`target_efficiency = dvr_tps / target_tps`. Never multiply DVR throughput by
acceptance a second time.

The 35B script exposes `CONTEXT_LENGTH` and `MAX_TOTAL_TOKENS` for separately
reported large-batch EAGLE runs. Keep its defaults for the fixed BS=3 matrix;
when increasing `MAX_CONCURRENCY`, set both capacity overrides explicitly and
retain the script's server-capacity check so a silently reduced batch is never
reported as the requested large-batch result.
