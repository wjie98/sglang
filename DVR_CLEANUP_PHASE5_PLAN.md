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

## Execution Log

- `d9fc0f88e dvr: add cleanup phase 5 plan`
  - `git diff --check`.
- `23991870b dvr: vectorize gdn conv window export`
  - `py_compile` for `gdn_backend.py`.
  - `git diff --check`.
  - Qwen3 pure-attention graph bs=2 lengths `17,65,129`: strict KL=0.
  - Qwen3.5/GDN graph bs=2 lengths `65,129,257`: strict KL=0.
  - Qwen3.5/GDN no-graph bs=2 lengths `65,129,257`: strict KL=0.
- `f3fc05cb1 dvr: vectorize gdn qkvg beta rolling shift`
  - `py_compile` for `gdn_backend.py`.
  - `git diff --check`.
  - Qwen3 pure-attention graph bs=2 lengths `17,65,129`: strict KL=0.
  - Qwen3.5/GDN graph bs=2 lengths `65,129,257`: strict KL=0.
  - Qwen3.5/GDN no-graph bs=2 lengths `65,129,257`: strict KL=0.
- `af8ae01ce dvr: vectorize fixed verify positions`
  - `py_compile` for `dvr_worker.py`.
  - `git diff --check`.
  - Qwen3 pure-attention graph bs=2 lengths `17,65,129`: strict KL=0.
  - Qwen3.5/GDN graph bs=2 lengths `65,129,257`: strict KL=0.
  - Qwen3.5/GDN no-graph bs=2 lengths `65,129,257`: strict KL=0.
- Documentation cleanup
  - Added minimum attention-only and GDN DVR launch commands to
    `DVR_KL_TESTING_GUIDE.md`.
  - Marked `SGLANG_RETURN_ORIGINAL_LOGPROB=True` as a strict-KL oracle setting.
  - Marked the DeepGEMM-disabling env vars as local 5090 test protection.
  - Final Qwen3.5/GDN graph stress bs=2 lengths `257,513`: strict KL=0.
- Parameter postprocessing follow-up
  - DVR chunk-boundary verify now auto-fixes GDN/Mamba state settings to
    `extra_buffer`, `mamba_track_interval=64`, and `mamba_ssm_dtype=float32`.
  - Tested `page_size=16` on Qwen3 attention-only DVR chunk verify. It still
    fails with a fixed-window row mismatch (`73` actual rows vs `80` graph/KV
    rows) in both CUDA graph and no-graph paths, so DVR continues to force
    `page_size=1` for now.
  - Validation after forcing `page_size=1` in server-args postprocessing:
    - `py_compile` for `server_args.py` and `dvr_worker.py`.
    - `git diff --check`.
    - Qwen3 attention-only graph, launched with `--page-size 16` and auto-fixed
      to `1`, bs=2 lengths `17,65,129`: strict KL=0.
    - Qwen3.5/GDN graph, launched without explicit mamba state flags and with
      `--page-size 16`, auto-fixed to required settings, bs=2 lengths
      `65,129,257`: strict KL=0.
    - Qwen3.5/GDN no-graph with the same auto-fixed launch settings, bs=2
      lengths `65,129,257`: strict KL=0.
