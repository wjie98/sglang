# DVR Cleanup Phase 5 Plan

This pass focuses on DVR-local vectorization and launch-command simplification.
The target is to reduce Python-side overhead and clarify the minimum working
runtime settings without changing DVR semantics.

## Optimization Rules

1. Touch only DVR-related code paths.
2. Keep the flow close to SGLang EAGLE speculative decoding.
3. Keep DVR/reference names for draft tokens, fixed verify windows,
   q/k/v/g/beta, GDN boundary state, and accepted-token state.
4. Prefer loop-invariant reasoning over new framework abstractions.
5. Preserve reject sampling, chain topk=1, fixed
   `FLA_CHUNK_SIZE + draft_tokens` physical verify windows, and strict KL=0.
6. Separate minimum working launch parameters from local hardware/debug
   parameters such as DeepGEMM workarounds and logprob oracle settings.

## Phase 5.1: Vectorize GDN Conv Window Export

Goal: remove the per-step Python loop in DVR target-verify conv window export.

Implementation:

- Replace the fixed-window `torch.stack([... for step in range(window)])`
  with a tensor view such as `unfold`.
- Keep the output layout `[bs, verify_window_size, channels, conv_state_len]`.
- Keep the existing intermediate conv cache write unchanged.

Validation:

- `py_compile` for `gdn_backend.py`.
- `git diff --check`.
- Qwen3 pure-attention graph strict KL=0, bs=2, lengths `17,65,129`.
- Qwen3.5/GDN graph and no-graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 5.2: Vectorize QKVG/Beta Rolling Buffer Shift

Goal: remove the per-request crossing loop after a chunk boundary commit.

Implementation:

- For crossing requests, copy the whole draft suffix
  `[FLA_CHUNK_SIZE : FLA_CHUNK_SIZE + num_draft_tokens]` to the beginning of
  the q/k/v/g/beta rolling buffer in one indexed copy per cache tensor.
- Keep `dvr_qkvg_beta_pos` as the source of truth for the effective live tail
  length, so extra copied padding/rejected rows remain ignored.
- Keep recurrent live-state rebuild behavior unchanged.

Validation:

- `py_compile` for `gdn_backend.py`.
- `git diff --check`.
- Qwen3 pure-attention graph strict KL=0, bs=2, lengths `17,65,129`.
- Qwen3.5/GDN graph and no-graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 5.3: Simplify Fixed Verify Window Construction Where Safe

Goal: reduce repeated small tensor creation without making variable-length KV
slot ownership harder to audit.

Implementation:

- Consider vectorizing positions and boundary length tensor construction.
- Keep input id list assembly, KV slot collection, and padding slot ownership
  in the existing request loop unless a simpler equivalent is obvious.
- Skip this phase if the diff is not clearly simpler.

Validation:

- `py_compile` for `dvr_worker.py`.
- `git diff --check`.
- Qwen3 pure-attention graph strict KL=0, bs=2, lengths `17,65,129`.
- Qwen3.5/GDN graph and no-graph strict KL=0, bs=2, lengths `65,129,257`.

## Phase 5.4: Minimum Working Launch Parameters

Goal: separate DVR's true required parameters from local test/debug settings.

Implementation:

- Update DVR testing docs with minimum attention-only and GDN launch commands.
- Mark `SGLANG_RETURN_ORIGINAL_LOGPROB=True` as a KL oracle requirement, not a
  serving requirement.
- Mark DeepGEMM-disabling env vars as local 5090 test protection, not DVR
  semantics.
- Do not silently enable upstream deterministic inference inside DVR; tests and
  launch examples should pass `--enable-deterministic-inference` explicitly.

Validation:

- `git diff --check`.
- Final Qwen3.5/GDN graph stress strict KL=0, bs=2, lengths `257,513`.

## Stop Condition

Stop after these phases unless another optimization is clearly DVR-local,
behavior-neutral, and independently testable.
