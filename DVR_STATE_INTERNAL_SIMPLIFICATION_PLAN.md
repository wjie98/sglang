# DVR State Internal Simplification Plan

This note tracks the v5 code-review pass for the four `dvr_state_*` modules.
The goal is to keep the public DVR state interface small while making the
internal flow easier to read and merge upstream.

## Constraints

- Do not change DVR runtime semantics.
- Do not expand the public API.
- Do not move DVR details back into `gdn_backend.py`.
- Keep the GDN backend integration incremental and easy to review.
- Keep patch-view code synced after each completed stage.

## Current Public Surface

- `dvr_state_adapter.py`
  - `DVRGatedStateAdapter`
- `dvr_state_cache.py`
  - `DVRRecurrentStateBackup`
  - `DVRStateInputWindow`
  - `allocate_dvr_state_input_cache`
- `dvr_state_ops.py`
  - `DVRStateOps`
- `dvr_state_verify.py`
  - no public export; adapter-internal helpers only

## Stage 1: Simplify Verify Helpers

- Remove the `DVRStateCommitPlan` dataclass and compute commit tensors directly
  in `DVRGatedStateAdapter.commit_after_verify`.
- Remove `build_dvr_state_commit_plan`.
- Fold `build_dvr_conv_windows` into `write_dvr_conv_windows`, because it has a
  single caller and does not define an independent lifecycle boundary.
- Keep `run_dvr_chunkwise_verify` and `rebuild_dvr_live_state_grouped`; these
  remain useful because they represent separate DVR stages.

Validation:

- Compile the four `dvr_state_*` modules.
- Run `git diff --check`.

## Stage 2: Simplify Adapter Thin Wrappers

- Inline `verify_shape`, `target_verify_shape`, and
  `target_verify_has_initial_states` into the target-verify methods.
- Keep external adapter methods unchanged.

Validation:

- Compile the four `dvr_state_*` modules.
- Run `git diff --check`.

## Stage 3: Sync Patch View

- Apply the same code changes to `sglang-dvr-v5-patch-view`.
- Run `git diff --check` in the patch view.

## Stage 4: Report

- Summarize removed abstractions.
- Summarize preserved public interfaces.
- Summarize validation results.
