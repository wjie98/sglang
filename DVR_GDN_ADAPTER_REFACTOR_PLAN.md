# DVR GDN Adapter Refactor Plan

This file tracks the staged refactor that makes the GDN DVR code use a clearer
adapter boundary. The goal is to keep the runtime behavior unchanged while
making it easier for future KDA/Mamba-like linear-state backends to plug into
DVR without knowing DVR's internal rolling-window details.

## Principles

- Keep DVR-specific behavior in DVR files or clearly marked DVR adapter helpers.
- Keep GDN backend responsible for producing GDN tensors, not for owning DVR
  rolling-window and commit semantics.
- Avoid names tied to `mamba` or `fla` in adapter-level classes. Backend-specific
  kernel function names may still mention q/k/v/g/beta because that is the GDN
  state-input representation.
- Do not rename memory-pool storage fields in this round. They are implementation
  details and renaming them would create unnecessary merge noise.
- Test after each meaningful stage. GDN/Qwen3.5 must keep strict KL=0.

## Stage 1: Adapter Naming And API Boundary

Planned changes:

- Rename `DVRLinearStateOps` to `DVRStateKernels`.
- Rename `DVRGatedStateInputCache` to `DVRStateInputWindow`.
- Rename `has_dvr_state_input_cache()` to `has_dvr_state_window()`.
- Rename local variables from `state_input_cache` to `state_window`.
- Keep old memory-pool fields unchanged.

Validation:

- `py_compile` for `dvr_state_adapter.py` and `gdn_backend.py`.
- `git diff --check`.
- Qwen3.5 no-graph strict KL test for representative lengths.

Status: done.

Result:

- Renamed the adapter-level classes/functions to `DVRStateKernels`,
  `DVRStateInputWindow`, and `has_dvr_state_window`.
- Updated GDN local naming from `state_input_cache` to `state_window`.
- Validation passed:
  - `py_compile dvr_state_adapter.py gdn_backend.py`
  - `git diff --check`
  - Qwen3.5 no-graph KL cases `max_new=17,65,129`: all
    `maxdiff=0.0`, `kl=0.0`, acceptance `1.0`.

## Stage 2: Move Chunkwise Verify Window Logic Into Adapter

Planned changes:

- Add an adapter helper that appends draft rows to the rolling window, runs the
  chunkwise state kernel, stores the chunk-boundary state, and returns the draft
  suffix output.
- Keep GDN backend responsible for generating `q/k/v/g/beta` and shape metadata.
- Keep the physical verify window as `CHUNK_SIZE + draft_token_num`.

Validation:

- Static checks.
- Qwen3.5 no-graph strict KL test.
- Qwen3.5 cuda graph strict KL test.

Status: done.

Result:

- Added `run_dvr_chunkwise_verify()` to the adapter. The adapter now owns:
  - appending draft rows to the rolling window,
  - reading the fixed `chunk_size + draft_token_num` physical window,
  - invoking the chunkwise state kernel,
  - exporting the chunk-boundary state,
  - returning the draft suffix used by target verify.
- GDN backend now only provides GDN-produced tensors and layer shape metadata.
- Validation passed:
  - `py_compile dvr_state_adapter.py gdn_backend.py`
  - `git diff --check`
  - Qwen3.5 no-graph KL cases `max_new=17,65,129`: all
    `maxdiff=0.0`, `kl=0.0`.
  - Qwen3.5 cuda graph KL cases `max_new=17,65,129,257`: all
    `maxdiff=0.0`, `kl=0.0`.

## Stage 3: Move Extend Tail Seeding Into Adapter

Planned changes:

- Add an adapter helper for writing prompt/extend tail rows after the latest
  chunk boundary into the rolling state-input window.
- GDN backend passes reshaped `q/k/v/g/beta`, cache indices, query starts, and
  extend lens; the adapter handles boundary/tail offsets.

Validation:

- Static checks.
- Qwen3.5 no-graph strict KL test.
- Qwen3.5 cuda graph strict KL test.

Status: done.

Result:

- Added `DVRStateInputWindow.write_extend_tail()`.
- GDN backend now passes reshaped GDN state-input tensors and forward metadata;
  the adapter owns chunk-boundary offset calculation and tail-window writes.
- Validation passed:
  - `py_compile dvr_state_adapter.py gdn_backend.py`
  - `git diff --check`
  - Qwen3.5 no-graph KL cases `max_new=17,65,129`: all
    `maxdiff=0.0`, `kl=0.0`, acceptance `1.0`.

## Stage 4: Simplify Commit Logic Boundary

Planned changes:

- Keep `commit_dvr_state_after_verify()` as the GDN backend entry used by DVR
  worker, but move pure window/checkpoint helpers into the adapter where it
  reduces duplication and clarifies ownership.
- Preserve current recurrent-live-state and chunk-boundary-state semantics.

Validation:

- Static checks.
- Qwen3.5 no-graph and cuda graph strict KL tests.
- Qwen3 pure-attention smoke KL test to ensure generic DVR path still works.

Status: done.

Result:

- Added `DVRStateCommitPlan` and `build_dvr_state_commit_plan()` for pure DVR
  window-position validation and crossing computation.
- Added `DVRStateInputWindow.shift_after_boundary()` for rolling-window suffix
  movement after a committed chunk boundary.
- Kept GDN/MambaPool-specific scatter calls in `gdn_backend.py`, because those
  are concrete storage-layout operations rather than generic adapter behavior.
- Validation passed:
  - `py_compile dvr_state_adapter.py gdn_backend.py`
  - `git diff --check`
  - Qwen3.5 no-graph KL cases `max_new=17,65,129`: all
    `maxdiff=0.0`, `kl=0.0`.

## Stage 5: Final Regression And Patch View

Planned validation:

- `py_compile` on touched files.
- `git diff --check`.
- Qwen3.5 no-graph/cuda graph strict KL tests with lengths crossing chunk
  boundaries.
- Qwen3 smoke strict KL test.
- Commit with validation details.
- Refresh `/home/hwj/dvr_qwen3_5/sglang-dvr-v4-patch-view`.

Status: done.

Result:

- Final validation passed:
  - `py_compile dvr_state_adapter.py gdn_backend.py dvr_worker.py dvr_utils.py`
  - `git diff --check`
  - no old adapter names left in DVR/GDN files.
  - Qwen3.5 no-graph KL cases `max_new=17,65,129`: all
    `maxdiff=0.0`, `kl=0.0`.
  - Qwen3.5 cuda graph KL cases `max_new=17,65,129,257`: all
    `maxdiff=0.0`, `kl=0.0`.
  - Qwen3 no-graph smoke KL cases `max_new=17,65,129`: all
    `maxdiff=0.0`, `kl=0.0`.

Patch-view refresh: pending until commit.
