# DVR GDN Adapter Interface Plan

This round keeps the working v4 DVR behavior unchanged and focuses on making
the GDN integration easier to review and upstream.

## Goals

1. Keep the GDN backend as a thin wiring layer.
   - The backend should still own normal GDN forward math and the existing
     non-DVR speculative fallback.
   - DVR rolling-window details, verify shape, q/k/v/g/beta export, and state
     commit mechanics should live in the DVR adapter or DVR ops layer.
2. Keep the adapter finite and practical.
   - Do not introduce a registry or an abstract framework for every possible
     linear-state layer.
   - Continue using small presets such as `DVRStateOps.for_gdn(...)`.
3. Make control flow explicit.
   - Avoid opaque `maybe_*` names for target verify paths once the backend has
     already checked DVR is active.
   - Keep fallback target-verify behavior visible in `gdn_backend.py` so it is
     easy to compare with upstream SGLang.
4. Preserve current correctness.
   - Qwen3 pure-attention DVR must keep strict KL=0.
   - Qwen3.5 GDN DVR must keep strict KL=0 in no-graph and cuda-graph modes.

## Phase 1: Backend-visible DVR shape helpers

- Move target-verify `batch_size`, `draft_token_num`, and
  `has_initial_states` derivation into `DVRGatedStateAdapter`.
- This removes the backend's direct dependence on the DVR physical window shape
  formula except where fallback target verify already needs it.

Validation:

- `py_compile` for GDN backend, adapter, ops, and the manual recurrent test.
- `git diff --check`.

## Phase 2: Explicit adapter methods for active DVR verify

- Rename `maybe_process_target_verify_conv` to `process_target_verify_conv`.
- Rename `maybe_process_target_verify_state` to `process_target_verify_state`.
- The backend calls these only after `is_verify_enabled(...)` is true.
- This keeps the branch condition in one place and makes the adapter methods
  read as active DVR operations.

Validation:

- Same static checks as Phase 1.
- `test/manual/dvr/test_dvr_gdn_recurrent_state.py`.

## Phase 3: GDN backend thinning

- Introduce a local `dvr_verify` boolean in `forward_extend`.
- Use adapter helpers for DVR has-initial-state and DVR target-verify conv/state.
- Keep upstream-like fallback target verify blocks in place for non-DVR
  speculative paths.

Validation:

- Qwen3.5 GDN no-graph strict KL test for `max_new=17,65,129`.
- Qwen3 pure-attention no-graph strict KL test for `max_new=17,65,129`.

## Phase 4: CUDA graph regression

- Run Qwen3.5 GDN cuda-graph strict KL test for `max_new=17,65,129,257`.
- Confirm no new cuda-graph runner or model-runner changes are needed.

## Phase 5: Patch view and report

- Commit the refactor if all tests pass.
- Refresh `/home/hwj/dvr_qwen3_5/sglang-dvr-v4-patch-view`.
- Report changed files, test results, and remaining review notes.

## Execution Results

- Phase 1/2 completed:
  - Added `target_verify_shape(...)` and
    `target_verify_has_initial_states(...)` to `DVRGatedStateAdapter`.
  - Replaced target-verify `maybe_process_*` methods with explicit
    `process_target_verify_*` methods. The GDN backend now calls them only when
    `dvr_verify` is true.
  - Kept non-DVR target-verify fallback blocks in `gdn_backend.py` for easier
    upstream comparison.
- Static validation:
  - `py_compile` passed for DVR state ops, adapter, GDN backend, and manual
    recurrent-state test.
  - `git diff --check` passed.
- Kernel validation:
  - `test/manual/dvr/test_dvr_gdn_recurrent_state.py`:
    `max_diff=0.0 triton_max_diff=0.0`.
- Runtime validation:
  - Qwen3.5-0.8B no-graph, `max_new=17,65,129`: all `maxdiff=0.0`, `kl=0.0`.
  - Qwen3-0.6B no-graph, `max_new=17,65,129`: all `maxdiff=0.0`, `kl=0.0`.
  - Qwen3.5-0.8B cuda graph, `max_new=17,65,129,257`: all `maxdiff=0.0`,
    `kl=0.0`.
