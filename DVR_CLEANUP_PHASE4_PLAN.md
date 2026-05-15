# DVR Cleanup Phase 4 Plan

This pass continues after Phase 3. The goal is to further reduce review burden
while keeping DVR's current strict-KL behavior unchanged.

## Optimization Rules

1. Keep the control flow close to SGLang EAGLE speculative decoding.
2. Keep DVR/reference names for draft tokens, verify windows, q/k/v/g/beta,
   GDN boundary state, and accepted-token state.
3. Minimize shared-path changes and keep them guarded by DVR-specific flags.
4. Use loop-invariant reasoning to remove repeated checks and repeated tensor
   allocations where the state is fixed for one forward/verify pass.
5. Preserve reject sampling, chain topk=1, fixed `FLA_CHUNK_SIZE + draft`
   physical verify windows, and strict KL=0.

## Phase 4.1: Name The GDN DVR Target-Verify Branch

Goal: make `GDNAttnBackend.forward_extend` easier to inspect.

Implementation:

- Add a local `is_dvr_target_verify` boolean.
- Use it instead of repeating
  `is_target_verify and dvr_q_state_cache is not None`.
- Keep ordinary EAGLE target verify and ordinary extend unchanged.

Validation:

- `py_compile` for `gdn_backend.py`.
- `git diff --check`.
- Qwen3.5/GDN graph and no-graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 4.2: Extract DVR QKVG/Beta Cache Allocation

Goal: shrink `MambaPool.__init__` without changing memory layout.

Implementation:

- Add a private helper that allocates q/k/v/g/beta/pos DVR caches.
- Keep allocation shapes, dtypes, and log accounting unchanged.
- Keep the feature guarded by `enable_dvr_qkvg_beta_cache`.

Validation:

- `py_compile` for `memory_pool.py`.
- `git diff --check`.
- Qwen3.5/GDN graph and no-graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 4.3: Split GDN Boundary Initialization Per Request

Goal: make DVR worker boundary state initialization smaller and easier to audit.

Implementation:

- Move the per-request branch of `_ensure_gdn_boundary_state` into a helper.
- Keep request dictionaries, ping-pong track index updates, and zero-state
  initialization unchanged.
- Preserve the current fallback error when a nonzero boundary checkpoint is
  missing.

Validation:

- `py_compile` for `dvr_worker.py`.
- `git diff --check`.
- Qwen3 graph strict KL=0, bs=2, lengths `65,129,257`.
- Qwen3.5/GDN graph and no-graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 4.4: Clarify Target-Verify Graph Token Helper Name

Goal: make shared CUDA graph helper naming precise.

Implementation:

- Rename `get_target_verify_num_tokens_per_bs` to
  `get_target_verify_graph_num_tokens_per_bs`.
- Update imports and call sites only.

Validation:

- `py_compile` for `cuda_graph_runner.py` and `model_runner.py`.
- `git diff --check`.
- Qwen3 graph strict KL=0, bs=2, lengths `65,129,257`.
- Qwen3.5/GDN graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 4.5: Documentation Inventory

Goal: reduce future PR noise from development-only DVR notes.

Implementation:

- Do not delete records in this pass.
- Add a short inventory file that marks which DVR root files are candidates for
  final upstream docs and which are development-only experiment records.

Validation:

- `git diff --check`.

## Stop Condition

Stop after these phases unless another change is clearly DVR-local, behavior
neutral, and independently testable.
