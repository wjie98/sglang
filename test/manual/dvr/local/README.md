# DVR local qualification assets

This directory contains the fixed development and release-lab matrix. It is
tracked on the DVR development branch so results remain comparable across
machines, but it should be omitted when preparing a minimal upstream PR.

Do not replace these entry points with hand-written commands when comparing
correctness, acceptance, or throughput:

- `scripts/run_static_unit_checks.sh`: upstream-range and worktree diff checks,
  syntax/import checks, and DVR unit tests. Set `DVR_DIFF_BASE` when validating
  against a baseline other than `upstream/sglang-miles`.
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
- `tools/export_qwen3_5_smoke_model.py`: exports a deliberately inaccurate
  two-layer Qwen3.5 checkpoint containing one GDN layer, one full-attention
  layer, and the original MTP layer for fast lifecycle and CUDA graph checks.

`STATE_LIFECYCLE_REDESIGN.md` is the implementation contract for the isolated
self-draft state lifecycle. It defines target/Radix checkpoint ownership,
self-draft private state, the common self/EAGLE target transaction, and the
state-level and H20 gates required for release qualification.

The clients under `clients/` are intentionally not named `test_*.py`; they are
command-line experiment drivers, not pytest collection targets.

## Fixed-result discipline

- Record the commit, topology, model, backend, TP size, page size, overlap mode,
  Radix mode, logprob mode, request count, output length, and concurrency.
- The fixed scripts pass `--random-seed 2026` by default. Keep the same server
  seed as well as the same client seed when comparing independently launched
  v1/v2 servers.
- Compare self v1 with a synchronous baseline and self v2 with an overlap
  baseline. Do not compare unlike scheduling modes.
- The throughput scripts flush Radix after each warmup by default and include a
  cache report in every JSON row. Set `FLUSH_CACHE_EACH_RUN=0` only for a
  separately labelled production-warm-cache experiment; never mix the two
  policies in one summary.
- A result directory containing JSONL rows is rejected by default so a partial
  rerun cannot silently mix commits. Prefer a new `RESULT_ROOT`; use
  `ALLOW_RESULT_REUSE=1` only to resume the exact same matrix deliberately.
- Read the `V2_V1` line before comparing target efficiencies. It separates the
  absolute scheduler-path ratio, acceptance drift, and the residual ratio per
  accepted token.
- A finite overlap run includes pipeline fill and drain. For short matrices with
  only a few verify rounds per request, report a decode-only profile as well as
  end-to-end throughput; do not attribute a drain bubble to the DVR GPU core.
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
