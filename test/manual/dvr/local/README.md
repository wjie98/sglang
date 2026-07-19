# DVR local qualification assets

This directory contains the fixed development and release-lab matrix. It is
tracked on the DVR development branch so results remain comparable across
machines, but it should be omitted when preparing a minimal upstream PR.

Do not replace these entry points with hand-written commands when comparing
correctness, acceptance, or throughput:

- `scripts/run_static_unit_checks.sh`: diff, syntax, import, and DVR unit tests.
- `scripts/run_0p8b_self_dvr_kl.sh`: self-DVR synchronous/overlap KL and Radix
  boundary cases, including long output and one-token prompts.
- `scripts/run_35b_mtp_eagle_smoke.sh`: Qwen3.5 MTP DVR-EAGLE acceptance and KL
  in synchronous and overlap modes.
- `scripts/run_35b_dvr_throughput.sh`: matched 35B normal, deterministic,
  self-DVR, and DVR-EAGLE throughput.
- `scripts/run_80b_self_dvr_throughput.sh`: ShareGPT and LongBench long-output
  self-DVR qualification.
- `scripts/prepare_h20_deep_gemm.sh`: precompile verify GEMM shapes before graph
  capture on H20.
- `scripts/profile_dvr_server.sh`: development-only server profiling.

The clients under `clients/` are intentionally not named `test_*.py`; they are
command-line experiment drivers, not pytest collection targets.

## Fixed-result discipline

- Record the commit, topology, model, backend, TP size, page size, overlap mode,
  Radix mode, logprob mode, request count, output length, and concurrency.
- Compare self v1 with a synchronous baseline and self v2 with an overlap
  baseline. Do not compare unlike scheduling modes.
- `bench_serving` throughput already includes accepted draft tokens. Compute
  `acceptance_fraction = accept_length / draft_tokens` and
  `target_efficiency = dvr_tps / (baseline_tps * acceptance_fraction)`.
- Use ordinary deterministic serving only for the user-facing DVR speedup. Use
  matched non-deterministic serving for acceptance-weighted implementation
  efficiency.
- Cover `return_logprob=True` and `False`; strict KL replay uses returned
  logprobs and a full target prefill oracle.
- Do not report a DeepGEMM run if graph startup triggers exhaustive JIT. Prewarm
  first and retain the cache for the formal run.

See `H20_RELEASE_VALIDATION.md` for the release sequence and pass criteria.
