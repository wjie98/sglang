# DVR Linear State Cache Interface Plan

This round narrows the lower-level DVR state interface after the GDN backend
hook surface was reduced. The goal is to keep `memory_pool.py` and
`dvr_worker.py` from exposing GDN-specific q/k/v/g/beta names while avoiding a
large framework or registry.

## Goals

1. Make memory-pool DVR fields layer-family neutral.
   - The memory pool allocates generic DVR state-input windows.
   - GDN/KDA/Mamba adapters interpret those windows.
2. Keep DVR worker mostly independent from the concrete adapter.
   - The worker should not read `qkvg_beta`-named fields directly.
   - It can still use Mamba/GDN live and boundary state indices for this round.
3. Keep the implementation simple.
   - No registry, base class hierarchy, or dynamic plugin mechanism.
   - Preserve tensor shapes and runtime behavior.

## Phase 1: Neutralize memory-pool DVR field names

- Rename speculative state fields:
  - `dvr_q_state_cache` -> `dvr_input_0_cache`
  - `dvr_k_state_cache` -> `dvr_input_1_cache`
  - `dvr_v_state_cache` -> `dvr_input_2_cache`
  - `dvr_g_state_cache` -> `dvr_input_3_cache`
  - `dvr_beta_state_cache` -> `dvr_input_4_cache`
  - `dvr_qkvg_beta_pos` -> `dvr_input_pos`
- Rename `_alloc_dvr_qkvg_beta_cache(...)` to
  `_alloc_dvr_state_input_cache(...)`.
- Rename logs to `dvr_state_input_cache size`.
- Update skip lists for disaggregation/TP slicing.

Validation:

- `py_compile` for `memory_pool.py` and `dvr_state_adapter.py`.
- `git diff --check`.

## Phase 2: Keep GDN interpretation in the adapter

- Update `DVRStateInputWindow.from_cache(...)` to read the generic
  `dvr_input_*` fields.
- Keep `DVRStateInputWindow` attributes named `q/k/v/g/beta` internally for now;
  this is the GDN adapter's interpretation of the generic inputs.
- Add adapter helpers:
  - `tail_lens(state_cache, live_indices)`
  - `set_tail_lens(state_cache, live_indices, tail_lens)`

Validation:

- `test/manual/dvr/test_dvr_gdn_recurrent_state.py`.

## Phase 3: Stop direct worker access to input-position fields

- Replace direct worker access to `ctx.mamba_cache.dvr_qkvg_beta_pos` with the
  adapter helpers.
- Remove `_set_gdn_verified_tail_lens(...)` if the adapter helper covers its
  only use.
- Keep broader worker method names (`_gdn_state_context`, etc.) unchanged in
  this round to avoid churn.

Validation:

- `py_compile` for `dvr_worker.py`.
- Static search for old field names.

## Phase 4: Runtime regression

- Qwen3.5-0.8B no-graph strict KL test: `max_new=17,65,129`.
- Qwen3-0.6B no-graph strict KL test: `max_new=17,65,129`.
- Qwen3.5-0.8B cuda graph strict KL test: `max_new=17,65,129,257`.

## Phase 5: Finish

- Record results in this file.
- Commit with validation details.
- Refresh `/home/hwj/dvr_qwen3_5/sglang-dvr-v4-patch-view`.

## Execution Results

- Phase 1 completed:
  - `MambaPool.SpeculativeState` now exposes generic `dvr_input_0_cache` ...
    `dvr_input_4_cache` plus `dvr_input_pos`.
  - Allocation/logging/reset/disaggregation skip lists use the generic names.
- Phase 2 completed:
  - `DVRStateInputWindow.from_cache(...)` maps the generic storage back to the
    GDN adapter's q/k/v/g/beta interpretation.
  - Added adapter-level `tail_lens(...)` and `set_tail_lens(...)`.
- Phase 3 completed:
  - `dvr_worker.py` no longer directly references the old qkvg/beta position
    field; it uses the adapter helpers for DVR tail lengths.
  - Static search for old fields returned no matches.
- Runtime issue fixed:
  - The first version of `tail_lens(...)` indexed the two-dimensional
    `[layers, slots]` position tensor as `pos[indices]`, which returned all
    layers for each request and broke batch commit shape checks.
  - It now uses `pos[0, indices]`, matching the previous single-layer read
    semantics for shared per-request tail lengths.

Validation:

- Static:
  - `py_compile` passed for `memory_pool.py`, `dvr_state_adapter.py`,
    `dvr_worker.py`, `gdn_backend.py`, and
    `test/manual/dvr/test_dvr_gdn_recurrent_state.py`.
  - `git diff --check` passed.
  - Old field-name search returned no matches.
- Unit/manual:
  - `test/manual/dvr/test_dvr_gdn_recurrent_state.py`
    returned `max_diff=0.0 triton_max_diff=0.0`.
- Runtime strict KL:
  - Qwen3.5-0.8B, no cuda graph, `max_new=17,65,129`:
    all cases `maxdiff=0.0`, `kl=0.0`, `ALL_OK True`.
  - Qwen3-0.6B, no cuda graph, `max_new=17,65,129`:
    all cases `maxdiff=0.0`, `kl=0.0`, `ALL_OK True`.
  - Qwen3.5-0.8B, cuda graph, `max_new=17,65,129,257`:
    all cases `maxdiff=0.0`, `kl=0.0`, `ALL_OK True`.
