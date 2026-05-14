# DVR Cleanup Phase 1 Plan

This pass applies only to the v3 worktree:

`/home/hwj/dvr_qwen3_5/sglang-dvr-clean-v3`

The reference DVR branch remains in:

`/home/hwj/dvr_qwen3_5/sglang`

## Requirements

- Keep the structure close to SGLang EAGLE speculative decoding.
- Keep DVR-specific naming compatible with the reference DVR branch where it
  describes draft tokens, verify windows, q/k/v/g/beta caches, and GDN state.
- Minimize invasive changes to shared SGLang paths.
- Remove redundant code, debug leftovers, and overly broad checks where the
  same behavior can be expressed with a smaller DVR-specific condition.
- Preserve reject sampling support.
- Preserve the current Qwen3 attention-only KL=0 behavior.

## Phase 1 Scope

1. Clean only DVR-related code in the v3 branch.
2. Keep `chain_rejection.py` and the non-greedy reject-sampling path.
3. Keep the EAGLE-like `draft -> verify -> postprocess` control flow.
4. Keep the GDN fixed verify window shape:
   `FLA_CHUNK_SIZE + speculative_num_draft_tokens`.
5. Do not attempt to fix the known Qwen3.5/GDN divergence in this pass.

## Validation Result

- Syntax check passed with `conda run -n dvr_dev python -m py_compile` for
  DVR-touched modules.
- Qwen3-0.6B DVR server was launched with:
  `--speculative-algorithm DECODE_VERIFY_ROLLBACK`,
  `--speculative-num-draft-tokens 16`, `--page-size 1`,
  `--attention-backend triton`, `--sampling-backend pytorch`,
  `--enable-deterministic-inference`, and DeepGEMM disabled.
- Strict returned-logprob vs full-prefill scoring passed:
  `cases=28`, lengths `1, 8, 16, 33, 65, 97, 129, 257`,
  `max_abs=0.0`, `max_kl=0.0`.
- Non-greedy sampling was used (`temperature=0.7`, `top_p=0.9`, `top_k=20`).
  Observed accept rates ranged from `0.4666666666666667` to `1.0`.

## Notes

- Any failure caused by environment or unrelated import compatibility should be
  reported without modifying unrelated files.
