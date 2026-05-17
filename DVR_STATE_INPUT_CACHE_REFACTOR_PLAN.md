# DVR State Input Cache Refactor Plan

This round fixes the previous `dvr_input_0_cache` ... `dvr_input_4_cache`
renaming. Those names removed GDN-specific wording from `memory_pool.py`, but
they also made the storage harder to understand and still left downstream code
aware of low-level fields.

The goal is a small value-object abstraction:

- `memory_pool.py` owns allocation and lifetime of a DVR state-input cache.
- linear-state adapters interpret the cache tensors as model-specific inputs
  such as GDN q/k/v/g/beta.
- `dvr_worker.py` only reads and updates rolling-window tail lengths through a
  uniform interface.

This is intentionally not a registry or plugin system. It is just a named
container around the tensors and tail positions DVR already needs.

## Phase 1: Add a Small Cache Container

- Add `MambaPool.DVRStateInputCache`.
- Fields:
  - `tensors: Tuple[torch.Tensor, ...]`
  - `tail_lens: torch.Tensor`
- Methods:
  - `enabled`
  - `mem_usage_bytes`
  - `capacity`
  - `layer_view(layer_id)`
  - `get_tail_lens(indices)`
  - `set_tail_lens(indices, value)`
  - `reset(indices)`
- Replace `SpeculativeState` fields:
  - remove `dvr_input_0_cache` ... `dvr_input_4_cache`
  - remove `dvr_input_pos`
  - add `dvr_state_inputs: Optional[DVRStateInputCache]`

Validation:

- `py_compile memory_pool.py`
- `git diff --check`

## Phase 2: Keep Model Semantics in the Adapter

- Change `DVRStateInputWindow.from_cache(...)` to read
  `state_cache.dvr_state_inputs`.
- Keep q/k/v/g/beta names inside the GDN adapter only.
- Use `cache.tensors` and `cache.tail_lens` instead of numeric field names.

Validation:

- `py_compile dvr_state_adapter.py`
- search for `dvr_input_[0-4]`

## Phase 3: Tighten Worker Access

- Replace worker tail length reads/writes with direct calls on
  `state_cache.dvr_state_inputs`.
- Rename local helper from GDN-specific position wording to recurrent DVR tail
  wording if this can be done without broad churn.
- Keep existing GDN boundary-state naming for this round because that code still
  owns Mamba/GDN ping-pong checkpoint details.

Validation:

- `py_compile dvr_worker.py`
- search for old names:
  - `dvr_qkvg`
  - `dvr_q_state_cache`
  - `dvr_input_0_cache`

## Phase 4: Runtime Validation

- `test/manual/dvr/test_dvr_gdn_recurrent_state.py`
- Qwen3.5-0.8B no-graph strict KL: `max_new=17,65,129`
- Qwen3-0.6B no-graph strict KL: `max_new=17,65,129`
- Qwen3.5-0.8B cuda graph strict KL: `max_new=17,65,129,257`

All runtime tests must use:

- `SGLANG_RETURN_ORIGINAL_LOGPROB=True`
- `SGLANG_ENABLE_JIT_DEEPGEMM=0`
- `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`
- `SGLANG_ENABLE_SPEC_V2=1`

## Phase 5: Finish

- Record exact validation results below.
- Commit the refactor.
- Refresh `/home/hwj/dvr_qwen3_5/sglang-dvr-v4-patch-view`.

## Execution Results

Completed.

- Phase 1:
  - Added `MambaPool.DVRStateInputCache`.
  - Replaced `dvr_input_0_cache` ... `dvr_input_4_cache` and
    `dvr_input_pos` with `dvr_state_inputs`.
  - `MambaPool.State.at_layer_idx(...)`, `mem_usage_bytes()`, allocation reset,
    and disaggregation skip lists now treat the DVR state-input window as one
    named cache object.
- Phase 2:
  - `DVRStateInputWindow.from_cache(...)` now reads
    `state_cache.dvr_state_inputs`.
  - GDN-specific q/k/v/g/beta naming remains inside the adapter interpretation,
    not in memory-pool field names.
- Phase 3:
  - `dvr_worker.py` reads and writes rolling tail lengths through
    `state_cache.dvr_state_inputs`.
  - Removed adapter-level tail-lens forwarding helpers to avoid multiple public
    access paths.

Validation:

- Static:
  - `py_compile` passed for `memory_pool.py`, `dvr_state_adapter.py`,
    `dvr_worker.py`, `gdn_backend.py`, and
    `test/manual/dvr/test_dvr_gdn_recurrent_state.py`.
  - `git diff --check` passed.
  - Static search found no remaining `dvr_input_[0-4]`, `dvr_input_pos`,
    `dvr_qkvg`, or old q/k/v/g/beta memory-pool field names.
- Unit/manual:
  - `test/manual/dvr/test_dvr_gdn_recurrent_state.py` returned
    `max_diff=0.0 triton_max_diff=0.0`.
- Runtime strict KL:
  - Qwen3.5-0.8B no-graph, `max_new=17,65,129`: all cases
    `maxdiff=0.0`, `kl=0.0`, `ALL_OK True`.
  - Qwen3-0.6B no-graph, `max_new=17,65,129`: all cases
    `maxdiff=0.0`, `kl=0.0`, `ALL_OK True`.
  - Qwen3.5-0.8B cuda graph, `max_new=17,65,129,257`: all cases
    `maxdiff=0.0`, `kl=0.0`, `ALL_OK True`. The `257` case ended at
    `gen=236`, which is an EOS-length outcome; the generated sequence still
    matched the full-prefill oracle exactly.
