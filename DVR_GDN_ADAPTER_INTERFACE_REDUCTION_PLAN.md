# DVR GDN Adapter Interface Reduction Plan

This round continues from `f209ece9b` and reduces the DVR-specific surface left
in `gdn_backend.py`.

## Goals

1. Keep `gdn_backend.py` as close as possible to upstream GDN wiring.
2. Keep DVR request/window/state lifecycle inside the DVR adapter.
3. Avoid a generic registry or broad framework. The current finite preset style
   (`DVRStateOps.for_gdn(...)`) is enough for GDN/KDA/Mamba-style follow-up work.
4. Preserve the existing strict KL=0 behavior for Qwen3 and Qwen3.5.

## Phase 1: Hide target-verify initial-state details

- Change `DVRGatedStateAdapter.process_target_verify_conv(...)` so it computes
  target-verify `has_initial_states` internally.
- Remove `target_verify_has_initial_states(...)` calls from `gdn_backend.py`.
- Keep normal extend `has_initial_states = forward_batch.extend_prefix_lens > 0`
  in the backend because that is upstream GDN behavior, not DVR-specific.

Validation:

- `py_compile` for adapter/backend/ops.
- `git diff --check`.

## Phase 2: Remove GDN public commit wrapper

- Remove `GDNAttnBackend.commit_dvr_state_after_verify(...)`.
- In `DVRWorker`, locate `linear_backend.dvr_state_adapter` directly and call
  `commit_after_verify(...)` with the speculative all-layer Mamba state cache.
- This makes the commit path explicitly owned by DVR worker + DVR adapter,
  rather than adding a DVR-specific public API to GDN.

Validation:

- `py_compile` for worker/backend/adapter.
- Manual GDN recurrent state test.

## Phase 3: Rename extend export hook

- Rename `maybe_cache_extend_state_inputs(...)` to `cache_extend_state_inputs(...)`.
- Keep the internal early returns because the adapter should no-op when DVR
  state windows are absent.
- This makes the backend call read as a hook point, while the adapter decides
  whether it has DVR work to do.

Validation:

- Static checks and recurrent state test.

## Phase 4: Runtime regression

- Qwen3.5 GDN no-graph strict KL test: `max_new=17,65,129`.
- Qwen3 pure-attention no-graph strict KL test: `max_new=17,65,129`.
- Qwen3.5 GDN cuda graph strict KL test: `max_new=17,65,129,257`.

## Phase 5: Finish

- Record results in this file.
- Commit with validation details.
- Refresh `/home/hwj/dvr_qwen3_5/sglang-dvr-v4-patch-view`.

## Execution Results

- Phase 1 completed:
  - `process_target_verify_conv(...)` now computes DVR target-verify
    `has_initial_states` inside `DVRGatedStateAdapter`.
  - `gdn_backend.py` no longer passes this DVR-specific mask.
- Phase 2 completed:
  - Removed `GDNAttnBackend.commit_dvr_state_after_verify(...)`.
  - `DVRWorker` now gets `linear_backend.dvr_state_adapter` and calls
    `commit_after_verify(...)` directly with the all-layer speculative Mamba
    state cache.
  - This keeps DVR state commit owned by DVR worker + DVR adapter instead of
    adding a public DVR method to the GDN backend.
- Phase 3 completed:
  - Renamed `maybe_cache_extend_state_inputs(...)` to
    `cache_extend_state_inputs(...)`.
  - The adapter still no-ops internally when DVR state windows are absent.
- Static validation:
  - `py_compile` passed for `dvr_worker.py`, `dvr_state_ops.py`,
    `dvr_state_adapter.py`, `gdn_backend.py`, and
    `test_dvr_gdn_recurrent_state.py`.
  - `git diff --check` passed.
- Kernel validation:
  - `test/manual/dvr/test_dvr_gdn_recurrent_state.py`:
    `max_diff=0.0 triton_max_diff=0.0`.
- Runtime validation:
  - Qwen3.5-0.8B no-graph, `max_new=17,65,129`: all `maxdiff=0.0`, `kl=0.0`.
  - Qwen3-0.6B no-graph, `max_new=17,65,129`: all `maxdiff=0.0`, `kl=0.0`.
  - Qwen3.5-0.8B cuda graph, `max_new=17,65,129,257`: all `maxdiff=0.0`,
    `kl=0.0`.
