# DVR Cleanup Phase 6 Plan

This phase keeps the v3 branch as the working baseline and focuses on
minimum-intrusion cleanup that preserves the already validated DVR behavior.

## Rules

- Touch only DVR-related code paths.
- Keep DVR worker control flow close to EAGLE: draft, target verify,
  postprocess, state commit.
- Keep GDN backend generic. It may expose data and kernel hooks for DVR, but
  it must not own speculative accept/reject logic.
- Prefer small utilities over framework-wide abstractions.
- Preserve reject sampling and strict KL=0 validation.
- Do not add piecewise CUDA graph support in this branch.

## Planned Changes

1. Normalize DVR page size.
   - Clamp values larger than `FLA_CHUNK_SIZE` down to `FLA_CHUNK_SIZE`.
   - If the requested value does not divide `FLA_CHUNK_SIZE`, lower it to the
     nearest power-of-two divisor not greater than the requested value.
   - Do not require `speculative_num_draft_tokens` to be page-aligned.

2. Centralize piecewise CUDA graph handling.
   - DVR chunk-boundary verify disables piecewise CUDA graph in server-arg
     postprocessing.
   - Avoid extra worker/backend hooks for piecewise graph behavior.

3. Add a small Mamba DVR utility layer.
   - Provide a `MambaDVRFlaOps` container for the recurrent and chunk-scan FLA
     operators used by DVR.
   - Provide cache helpers for q/k/v/g/beta export and rolling-window shifts.
   - Provide helpers for packed chunk-boundary state writes and fixed-window
     conv-window export.

4. Refactor GDN backend onto the utility layer.
   - Keep `GDNKernelDispatcher` as the source of backend-specific kernels.
   - Set DVR FLA ops from decode/prefill backend state before target verify.
   - Use the utility layer for q/k/v/g/beta cache writes, recurrent state
     rebuild, boundary state writes, and conv-window export.

5. Keep worker checks narrow.
   - Worker validates only launch invariants that are required at runtime.
   - Page ownership remains page-size aware, but verify window semantics remain
     `verified_token + draft_token + padding_token`.

## Validation Matrix

- Static:
  - `python -m py_compile` on touched Python files.
  - `git diff --check`.
- Qwen3 attention-only:
  - batch size 1 and larger batches.
  - short generations, cross-64, cross-128, and long irregular lengths.
  - CUDA graph enabled and disabled.
  - strict KL=0 against full prefill scoring.
- Qwen3.5/GDN:
  - batch size 1 and larger batches.
  - page sizes `1, 2, 4, 8, 16, 32, 64`, values larger than 64, and
    non-divisor values.
  - generations crossing 64 and 128.
  - strict KL=0 for the currently supported paths; any failure must be tied to
    a concrete state/conv/qkvg path.

## Validation Results

- Static checks passed:
  - `python3 -m py_compile` on touched files.
  - `conda run -n dvr_dev python -m py_compile` on touched files.
  - `git diff --check`.
- `ruff` was not run because it is not installed in the local `dvr_dev`
  environment.
- Qwen3 attention-only, no CUDA graph:
  - launch used `--page-size 48`; server normalized it to `32`.
  - batch test `max_new=17,65,129` passed strict KL=0.
- Qwen3 attention-only, CUDA graph:
  - launch used `--page-size 7`; server normalized it to `4`.
  - batch test `max_new=17,65,129` passed strict KL=0.
- Qwen3.5/GDN, no CUDA graph:
  - launch used `--page-size 96`; server normalized it to `64`.
  - with `max_running_requests=4`, `max_total_tokens=8192`, and
    `mem_fraction_static=0.35`, batch test `max_new=17,65,129,257`
    passed strict KL=0.
  - the same batch test was run twice on one server to cover radix/mamba state
    reuse; both runs passed strict KL=0.
- Qwen3.5/GDN, CUDA graph:
  - launch used `--cuda-graph-max-bs 4`.
  - batch test `max_new=17,65,129` passed strict KL=0.

## Notes From This Phase

- GDN DVR state caches are large because fp32 `temporal`,
  `intermediate_ssm`, `intermediate_conv_window`, and q/k/v/g/beta caches are
  all preallocated. For local 5090 tests, avoid the default
  `max_running_requests=48`; use a smaller request count and an explicit
  `max_total_tokens` unless testing capacity behavior.
- `MambaDVRQKVGBetaCache.shift_suffix` must decide whether a cache has a
  leading layer dimension from `dvr_qkvg_beta_pos`, not from `cache.dim()`.
  q/k/v and g/beta have different ranks in the all-layer cache, but all of them
  share the same `[layers, slots, window, ...]` ownership model.

## Phase 6.1: Readability and Naming Cleanup

This pass keeps the validated behavior unchanged and focuses on naming,
comments, and local helper structure. It follows the same cleanup rules as the
previous phases:

- Keep code style close to SGLang EAGLE speculative decoding.
- Use DVR reference-branch terms for `draft_token`, `verify_window`,
  q/k/v/g/beta, chunk-boundary state, and accepted-token state.
- Touch only DVR-related code paths.
- Prefer small local helpers over new framework abstractions.
- Keep comments only where state/cache ownership is easy to misuse.

Planned changes:

1. Rename the q/k/v/g/beta helper from a "view" to a "cache" because it writes
   and shifts owned DVR cache tensors, not just views them.
2. Replace single-method writer/builder classes with small module-level helper
   functions for chunk-boundary state write and conv-window construction.
3. Rename GDN backend helpers so the stage/action is clear:
   - cache extend tail q/k/v/g/beta,
   - cache verify-window q/k/v/g/beta,
   - run target-verify conv,
   - run target-verify chunk scan,
   - commit state after verify.
4. Construct the q/k/v/g/beta cache helper once in GDN state commit instead of
   recreating it in nested branches.
5. Move runtime invariant checks into small private helpers so the commit flow
   reads as state lifecycle rather than error handling.
6. Shorten DVR page-size warning text in `server_args.py`; keep the detailed
   rationale in `DVR_KL_TESTING_GUIDE.md`.

Validation:

- `python -m py_compile` on touched Python files.
- `git diff --check`.
- Qwen3 attention-only strict KL=0 with CUDA graph and without CUDA graph.
- Qwen3.5/GDN strict KL=0 with CUDA graph and without CUDA graph.

Execution results:

- Renamed the q/k/v/g/beta helper to `MambaDVRQKVGBetaCache`.
- Replaced single-method helper classes with `write_dvr_chunk_boundary_state`
  and `build_dvr_conv_windows`.
- Renamed GDN DVR helper methods around cache, verify, and state commit
  stages.
- Built the q/k/v/g/beta cache helper once in GDN state commit and reused it
  in nested rebuild/shift paths.
- Moved GDN DVR runtime position checks into private helper methods.
- Shortened the DVR page-size normalization warning.
- Static checks passed:
  - `python3 -m py_compile` on touched Python files.
  - `conda run -n dvr_dev python -m py_compile` on touched Python files.
  - `git diff --check`.
- Runtime checks passed strict KL=0:
  - Qwen3 no CUDA graph, launch `--page-size 48` normalized to `32`,
    `max_new=17,65,129`.
  - Qwen3 CUDA graph, launch `--page-size 7` normalized to `4`,
    `max_new=17,65,129`.
  - Qwen3.5/GDN no CUDA graph, launch `--page-size 96` normalized to `64`,
    `max_new=17,65,129,257`.
  - Qwen3.5/GDN CUDA graph, `--cuda-graph-max-bs 4`,
    `max_new=17,65,129`.
