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
  - `ATTENTION_BACKEND` may select a separately reported Triton/FlashInfer/FA3
    compatibility run; the fixed default remains `triton`
- `test/manual/dvr/scripts/run_35b_mtp_eagle_smoke.sh`
  - 35B Qwen3.5 MTP/DVR-EAGLE sync-v2 and overlap-v2 smoke
  - covers `return_logprob=True` KL and `return_logprob=False` output path
  - includes the seeded `prompt_len=65`, `max_new=65` boundary acceptance
    regression and fails if reported accept rate drops below `0.99`
- `test/manual/dvr/scripts/run_80b_self_dvr_throughput.sh`
  - Qwen3-Next 80B no-DVR baseline and self-DVR spec v1/v2 long-output throughput
  - covers ShareGPT and fixed LongBench custom-cache inputs
  - covers `return_logprob=True/False`

The scripts accept the common overrides `CONDA_ENV`, `MODEL_PATH`, `PORT`, and
`RESULT_ROOT`; the 35B/80B scripts also expose dataset/model-specific override
variables near the top of each file.  When reporting a run, include the script
path, commit, key override variables, result directory, and whether the server
log confirms the expected CUDA graph and effective concurrency.

Do not use the removed `SGLANG_ENABLE_SPEC_V2` environment variable.  The fixed
scripts select the self-DVR v1 compatibility worker with
`--disable-overlap-schedule`; omitting that flag uses the spec-v2 overlap worker.

Do not silently change benchmark knobs such as request count, output length,
max concurrency, `--max-mamba-cache-size`, overlap mode, backend, or returned
logprob handling.  If a new experiment needs different knobs, create a separate
script or clearly mark it as a new baseline and keep the fixed scripts intact.

The 80B throughput script intentionally pins `--max-mamba-cache-size 16`.
Omitting it can lower effective request concurrency and produce misleadingly
low self-DVR throughput.  Its default run includes baseline, v1, and v2;
`RUN_BASELINE=0` or `RUN_DVR=0` may be used only to resume an interrupted
matrix in the same `RESULT_ROOT`.
