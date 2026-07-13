# DVR agent notes

## Fixed validation scripts

DVR validation must use the checked-in scripts under
`test/manual/dvr/scripts/` instead of hand-composed one-off commands.  The goal
is to keep KL, acceptance, and throughput results comparable across code
changes and machines.

Use these entry points:

- `test/manual/dvr/scripts/run_static_unit_checks.sh`
  - `git diff --check`
  - DVR-focused `py_compile`
  - DVR unit tests
- `test/manual/dvr/scripts/run_0p8b_self_dvr_kl.sh`
  - 0.8B self-DVR spec v1/v2 KL=0 and boundary smoke
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

The scripts accept the common overrides `CONDA_ENV`, `MODEL_PATH`, `PORT`, and
`RESULT_ROOT`; the 35B/80B scripts also expose dataset/model-specific override
variables near the top of each file.  When reporting a run, include the script
path, commit, key override variables, result directory, and whether the server
log confirms the expected CUDA graph and effective concurrency.

The 0.8B and 35B scripts also accept `DISABLE_RADIX_CACHE=1`. Use this fixed
variant to validate deterministic attention backends such as FlashInfer that
disable prefix matching; DVR must still preserve request-local GDN checkpoints.

Do not use the removed `SGLANG_ENABLE_SPEC_V2` environment variable.  The fixed
scripts select the self-DVR v1 compatibility worker with
`--disable-overlap-schedule`; omitting that flag uses the spec-v2 overlap worker.

Do not silently change benchmark knobs such as request count, output length,
max concurrency, `--max-mamba-cache-size`, overlap mode, backend, or returned
logprob handling.  If a new experiment needs different knobs, create a separate
script or clearly mark it as a new baseline and keep the fixed scripts intact.

The throughput scripts pin `--max-mamba-cache-size 16` and fail if the runtime
reduces the requested concurrency or omits its largest CUDA graph batch.
Their concurrency knobs must be overridden together with enough Mamba cache
capacity; otherwise the run is not a valid comparison.  The 80B default run
includes baseline, v1, and v2;
`RUN_BASELINE=0` or `RUN_DVR=0` may be used only to resume an interrupted
matrix in the same `RESULT_ROOT`.

The 35B script exposes `CONTEXT_LENGTH` and `MAX_TOTAL_TOKENS` for separately
reported large-batch EAGLE runs. Keep its defaults for the fixed BS=3 matrix;
when increasing `MAX_CONCURRENCY`, set both capacity overrides explicitly and
retain the script's server-capacity check so a silently reduced batch is never
reported as the requested large-batch result.
