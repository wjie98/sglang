# DVR Cleanup Phase 3 Plan

This pass continues from `dvr-v3-controlled-migration` after Phase 2 cleanup.
The goal is to reduce review burden without changing DVR semantics.

## Optimization Rules

1. Keep code style close to SGLang EAGLE speculative decoding.
2. Use DVR reference-branch names for draft tokens, verify windows,
   q/k/v/g/beta, GDN boundary state, and accepted-token state.
3. Minimize invasive changes to shared SGLang paths.
4. Use loop-invariant reasoning to merge repeated logic and boundary branches.
5. Preserve readability and performance together; add comments only where the
   state/cache lifecycle is easy to misuse.
6. Preserve reject sampling and chain topk=1 semantics.
7. Preserve strict KL=0 for Qwen3 and Qwen3.5/GDN.

## Phase 3.1: Vectorize Fixed Verify Row Selection

Goal: remove Python list construction from the fixed-window draft-row selector.

Implementation:

- Replace `_fixed_verify_draft_rows` list/range construction with tensor
  arithmetic over `verified_tail_lens`.
- Keep the returned row order unchanged: per request, draft rows immediately
  after `verified_tail`.

Validation:

- `py_compile` for `dvr_worker.py`.
- `git diff --check`.
- Qwen3 strict KL=0, bs=2, lengths `65,129,257`.
- Qwen3.5/GDN graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 3.2: Split Fixed Verify Request Window Assembly

Goal: make `_prepare_fixed_verify_window` easier to audit.

Implementation:

- Add a small DVR-local helper that builds one request's
  `verified_token + draft_token + padding_token` pieces.
- Keep allocation ownership unchanged: padding slots are still freed in
  `_restore_after_fixed_verify_window`.
- Keep the physical window fixed at `FLA_CHUNK_SIZE + num_draft_tokens`.

Validation:

- `py_compile` for `dvr_worker.py`.
- `git diff --check`.
- Qwen3.5/GDN graph and no-graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 3.3: Add GDN QKVG/Beta Cache Tuple Helper

Goal: reduce repeated five-cache tuple listings in `gdn_backend.py`.

Implementation:

- Add a private helper returning
  `(q_cache, k_cache, v_cache, g_cache, beta_cache)`.
- Use it only inside DVR-specific backend helper paths.
- Do not change memory layout or public backend interfaces.

Validation:

- `py_compile` for `gdn_backend.py`.
- `git diff --check`.
- Qwen3.5/GDN graph and no-graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 3.4: Reuse Per-Verify GDN State Context

Goal: make the per-verify live/boundary state indices explicit.

Implementation:

- Thread `_GDNDVRStateContext` through the DVR worker's restore, fixed-window,
  restore-after-window, and commit helpers where practical.
- Avoid repeated cache/index lookup inside one verify iteration.
- Preserve the current request dictionaries and state lifecycle.

Validation:

- `py_compile` for `dvr_worker.py`.
- `git diff --check`.
- Qwen3 strict KL=0, bs=2, lengths `65,129,257`.
- Qwen3.5/GDN graph and no-graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 3.5: Move DVR CUDA Graph Context Helpers

Goal: avoid shared model-executor code importing helper context managers from
the worker implementation file.

Implementation:

- Move `dvr_causal_verify_cuda_graph_metadata` and
  `dvr_runtime_verify_window` to a small DVR-local utility module.
- Update imports in `dvr_worker.py` and `cuda_graph_runner.py`.
- Keep the behavior and comments unchanged.

Validation:

- `py_compile` for `dvr_worker.py`, `dvr_utils.py`, and
  `cuda_graph_runner.py`.
- `git diff --check`.
- Qwen3 strict KL=0, bs=2, lengths `65,129,257`.
- Qwen3.5/GDN graph strict KL=0, bs=2, lengths `65,129,257`.

## Stop Condition

Stop after these steps unless another redundancy is clearly DVR-local and can
be validated independently. Do not refactor broad shared paths just for style.
