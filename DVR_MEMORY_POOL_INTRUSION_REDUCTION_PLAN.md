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

## Final Tightening

The follow-up cleanup removes the remaining DVR-specific behavior from
`MambaPool` while keeping the allocation layout unchanged.

- `memory_pool.py`
  - Removed the nested `DVRStateInputCache` wrapper.
  - Removed `_alloc_dvr_state_input_cache`; q/k/v/g/beta tensors are allocated
    inline next to the existing speculative Mamba state tensors.
  - Stores DVR state inputs as `List[torch.Tensor]` plus a separate
    `dvr_state_input_tail_lens` tensor.
  - Removed `backup_state(...)` and `restore_state(...)`; recurrent state
    backup is no longer a memory-pool API.
  - Keeps only the required lifecycle hook: reset DVR tail lengths when slots
    are allocated.
  - Skips `dvr_state_inputs` and `dvr_state_input_tail_lens` in contiguous
    buffer and state-dimension checks. These tensors are DVR scratch inputs,
    not live recurrent states; their window dimension is not the tensor-parallel
    state dimension.
- `dvr_state_adapter.py`
  - Reads the plain q/k/v/g/beta tensor list from the state cache.
  - Owns `backup_recurrent_state(...)` and `restore_recurrent_state(...)` for
    DVR recurrent slot switching.
- `dvr_worker.py`
  - Restores and backs up GDN boundary/live recurrent states through the DVR
    adapter, not through `MambaPool`.

Additional validation after this tightening:

- `py_compile` passed for `memory_pool.py`, `dvr_state_adapter.py`,
  `dvr_worker.py`, `gdn_backend.py`, and
  `test/manual/dvr/test_dvr_gdn_recurrent_state.py`.
- `git diff --check` passed.
- Old symbols/calls are gone:
  `_alloc_dvr_state_input_cache`, `DVRStateInputCache`,
  `mamba_pool.backup_state`, `mamba_pool.restore_state`.
- `test/manual/dvr/test_dvr_gdn_recurrent_state.py` returned
  `max_diff=0.0 triton_max_diff=0.0`.
- Qwen3.5-0.8B no-graph, `max_new=17,65,129`: all cases
  `maxdiff=0.0`, `kl=0.0`, `ALL_OK True`.
- Qwen3.5-0.8B cuda graph, `max_new=17,65,129,257`: all cases
  `maxdiff=0.0`, `kl=0.0`, `ALL_OK True`.
- Qwen3-0.6B cuda graph, `max_new=17,65,129,257`: all cases
  `maxdiff=0.0`, `kl=0.0`, `ALL_OK True`.

## Adapter-Owned Scratch Layout

One remaining leak was that `memory_pool.py` still spelled out the exact
q/k/v/g/beta scratch tensor shapes. That made the memory pool aware of GDN/DVR
storage details.

Follow-up cleanup:

- Moved q/k/v/g/beta scratch allocation into
  `allocate_dvr_state_input_cache(...)` in `dvr_state_adapter.py`.
- `memory_pool.py` now only decides whether the DVR scratch cache is needed and
  stores the returned tensors in `SpeculativeState`.
- `get_global_server_args` is imported at module scope in `memory_pool.py`;
  the local import inside `MambaPool.__init__` was removed.

Validation:

- `git diff --check` passed.
- `py_compile` passed for `memory_pool.py`, `dvr_state_adapter.py`, and
  `dvr_worker.py`.
- `test/manual/dvr/test_dvr_gdn_recurrent_state.py` returned
  `max_diff=0.0 triton_max_diff=0.0`.
