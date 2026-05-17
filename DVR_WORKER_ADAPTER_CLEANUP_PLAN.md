# DVR Worker And Adapter Cleanup Plan

Branch: `dvr-v4-performance-opt`

Goal: keep the working DVR behavior while reducing direct GDN/Mamba knowledge in
`dvr_worker.py` and trimming adapter wrappers that do not carry meaningful
semantics.

## Pre-check

`DVRStateInputCache.__getitem__` is required. It is used indirectly by
`MambaPool.SpeculativeState.at_layer_idx()`:

- `MambaPool.mamba2_layer_cache(layer_id)` calls `self.mamba_cache.at_layer_idx`.
- `SpeculativeState.at_layer_idx()` applies `v[layer]` to non-list fields.
- `dvr_state_inputs` is one of those fields, so it relies on
  `DVRStateInputCache.__getitem__` to produce a layer-local cache view.

Do not remove this method unless `memory_pool.py` is changed to special-case the
field, which would be more invasive.

## Step 1: Worker-Side Linear State Lifecycle

Add `python/sglang/srt/speculative/dvr_linear_state.py` with:

- `DVRLinearStateContext`
- `DVRLinearStateLifecycle`

Move the current GDN boundary state lifecycle from `dvr_worker.py` into this
class:

- active-state detection
- live/boundary cache indices
- ping-pong boundary checkpoint selection
- boundary initialization
- backup before self-draft decode
- restore before target verify
- commit after verify

`dvr_worker.py` should only call:

- `self.linear_state.prepare_for_draft(batch)`
- `linear_state_ctx = self.linear_state.restore_for_verify(batch)`
- `self.linear_state.commit_after_verify(batch, accepted_tokens, accepted_steps, ctx)`

The lifecycle class may still contain Mamba/GDN implementation details for now,
but those details should not leak into the main speculative worker flow.

## Step 2: Adapter API Cleanup

Keep the adapter as the backend-facing interface for gated linear-state DVR, but
remove weak wrappers:

- Keep `DVRStateInputCache.__getitem__`.
- Inline `has_window()` into `is_verify_enabled()`.
- Remove standalone `has_dvr_state_window()` if it has no remaining external
  users.
- Replace `cache_extend_state_inputs_from_forward()` and
  `cache_extend_state_inputs()` with one clearer method:
  `cache_extend_tail_from_forward(...)`.

Do not introduce a registry or broad abstraction. `DVRStateOps.for_gdn(...)`
remains the preset boundary for GDN-specific kernels.

## Step 3: Validation

After each meaningful edit:

- `git diff --check`
- `py_compile`:
  - `python/sglang/srt/speculative/dvr_worker.py`
  - `python/sglang/srt/speculative/dvr_linear_state.py`
  - `python/sglang/srt/layers/attention/linear/dvr_state_adapter.py`
  - `python/sglang/srt/layers/attention/linear/gdn_backend.py`
- `test/manual/dvr/test_dvr_gdn_recurrent_state.py`

Full regression before final report:

- Qwen3-0.6B no graph and cuda graph, `max_new=17,65,129,257`, strict KL=0.
- Qwen3.5-0.8B no graph and cuda graph, `max_new=17,65,129,257`, strict KL=0.

Finally refresh `/home/hwj/dvr_qwen3_5/sglang-dvr-v4-patch-view`.
