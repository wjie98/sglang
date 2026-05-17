# DVR Memory Pool Intrusion Reduction Plan

The previous `DVRStateInputCache` object improved naming, but it still put too
much DVR behavior inside `memory_pool.py`. This round keeps memory-pool changes
limited to allocation and minimal lifecycle hooks.

## Target Boundary

- `memory_pool.py`
  - owns tensor allocation;
  - owns layer slicing for `mamba2_layer_cache(...)`;
  - resets tail lengths when slots are allocated;
  - does not expose get/set tail helpers or DVR commit semantics.
- `dvr_state_adapter.py`
  - interprets generic DVR state-input tensors as GDN q/k/v/g/beta;
  - owns tail length accessors used by DVR worker;
  - owns rolling-window and commit semantics.
- `dvr_worker.py`
  - may call adapter helpers directly when it needs DVR state information.

## Steps

1. Replace `MambaPool.DVRStateInputCache` methods with a plain data container.
2. Move tail length get/set helpers back into `DVRGatedStateAdapter`.
3. Change `dvr_worker.py` to use adapter helpers instead of memory-pool methods.
4. Run static checks, recurrent state test, and strict KL runtime tests.

## Execution Results

Completed.

- `memory_pool.py`
  - `DVRStateInputCache` is now a plain data container with only
    `tensors` and `tail_lens`.
  - Removed `enabled`, `capacity`, `mem_usage_bytes`, `layer_view`,
    `get_tail_lens`, `set_tail_lens`, and `reset` methods from memory pool.
  - Kept only allocation, layer slicing, memory accounting, and allocation-time
    tail reset hooks.
- `dvr_state_adapter.py`
  - Added `state_input_tail_lens(...)` and `set_state_input_tail_lens(...)` so
    DVR state access lives with the adapter.
- `dvr_worker.py`
  - Switched tail length reads/writes back to adapter helpers.

Validation:

- `py_compile` passed for `memory_pool.py`, `dvr_state_adapter.py`,
  `dvr_worker.py`, `gdn_backend.py`, and
  `test/manual/dvr/test_dvr_gdn_recurrent_state.py`.
- `git diff --check` passed.
- `test/manual/dvr/test_dvr_gdn_recurrent_state.py` returned
  `max_diff=0.0 triton_max_diff=0.0`.
- Qwen3.5-0.8B no-graph, `max_new=17,65,129`: all cases
  `maxdiff=0.0`, `kl=0.0`, `ALL_OK True`.
- Qwen3.5-0.8B cuda graph, `max_new=17,65,129,257`: all cases
  `maxdiff=0.0`, `kl=0.0`, `ALL_OK True`.
