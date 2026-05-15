# DVR Cleanup Phase 2 Plan

This pass continues from `dvr-v3-controlled-migration` after the GDN graph and
RMSNorm determinism fixes are strict-KL clean.

## Guardrails

- Keep DVR close to SGLang EAGLE speculative decoding.
- Keep DVR names aligned with the reference branch: draft tokens, verify window,
  q/k/v/g/beta, GDN boundary state, and accepted-token state.
- Minimize changes to shared SGLang paths.
- Preserve reject sampling and chain topk=1 semantics.
- Preserve strict KL=0 for Qwen3 and Qwen3.5/GDN.
- Keep GDN physical target verify as `FLA_CHUNK_SIZE + speculative_num_draft_tokens`.
- Do not convert DVR into ordinary EAGLE; DVR owns chunk-aligned verify/state
  semantics.

## Phase 2.1: DVR Worker State Context

Goal: reduce repeated GDN lookup boilerplate inside `dvr_worker.py`.

Implementation:

- Add a small local `_GDNDVRStateContext` dataclass.
- Add `_gdn_state_context(batch, require_boundary=False)`.
- Use it in GDN state restore, fixed-window preparation, backup, and commit.
- Keep existing request-level dictionaries and lifecycle semantics unchanged.

Validation:

- `py_compile` for `dvr_worker.py`.
- Qwen3.5/GDN graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 2.2: Verify Window Naming Cleanup

Goal: make the distinction between logical draft tokens and physical fixed
verify window explicit.

Implementation:

- Rename local variables in fixed-window helpers:
  - `physical_verify_len` for `FLA_CHUNK_SIZE + draft`.
  - `logical_draft_len` for `speculative_num_draft_tokens`.
- Keep public args and existing fields unchanged.
- Do not change tensor shapes or cache ownership.

Validation:

- `py_compile` for `dvr_worker.py`.
- Qwen3 graph strict KL=0, bs=2, lengths `65,129,257`.
- Qwen3.5/GDN graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 2.3: Function Order And Comments

Goal: make `dvr_worker.py` easier to review without changing behavior.

Implementation:

- Keep the public flow in EAGLE-like order:
  1. entrypoints,
  2. target extend,
  3. draft,
  4. verify/postprocess,
  5. fixed-window helpers,
  6. GDN state helpers,
  7. accepted/logit helpers.
- Only move methods and tighten comments; no logic changes.

Validation:

- `py_compile` for `dvr_worker.py`.
- Qwen3.5/GDN graph and no-graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 2.4: GDN Backend Helper Extraction

Goal: isolate DVR target-verify conv/core export code while keeping GDN backend
generic.

Implementation:

- Extract only local private helpers inside `GDNAttnBackend`.
- Do not add speculative worker objects or request-level semantics to the GDN
  backend.
- Keep the backend contract data-oriented: q/k/v/g/beta export, intermediate
  conv/state export, and state commit.

Validation:

- `py_compile` for `gdn_backend.py`.
- Qwen3.5/GDN graph strict KL=0, bs=2, lengths `65,129,257`.
- Qwen3.5/GDN no-graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 2.5: Shared Path Review

Goal: stop once remaining changes would require broader architectural review.

Actions:

- Re-scan for stale DVR debug code and unused helper APIs.
- Do not refactor `memory_pool.py` allocation shape logic unless a dead path is
  proven.
- Do not further abstract CUDA graph hooks unless a correctness issue appears.

Validation:

- `git diff --check`.
- Worktree clean after commit.
