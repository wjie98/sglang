# DVR GDN Adapter And Recurrent Kernel Plan

Created: 2026-05-16 23:42:21 +0800
Branch: `dvr-v4-performance-opt`
Worktree: `/home/hwj/dvr_qwen3_5/sglang-dvr-clean-v4`

## Goal

Continue optimizing the DVR GDN integration while keeping the patch easy to
merge into `sglang-miles`.

The adapter is the boundary between model backends and DVR:

- Upstream callers such as GDN, KDA, or Mamba-like models should opt in with a
  small number of calls and should not need to understand DVR's rolling
  `verified + draft + padding` window.
- Downstream details such as chunkwise scan, recurrent state rebuild, q/k/v/g
  beta state-input windows, boundary state, and conv windows should stay inside
  DVR adapter utilities.
- Kernel hooks must stay generic. The adapter should not be named after FLA or
  Mamba because future linear-state backends may use different kernels.
- Keep the current correctness baseline: Qwen3 and Qwen3.5 DVR must keep strict
  full-prefill oracle equality (`KL=0`, exact logprob equality) in the tested
  modes.

## Reference DVR Notes

The original DVR branch uses self decode as draft and target verify for
acceptance. For GDN-like layers it needs target verify to use chunkwise-scan
semantics rather than decode recurrent semantics. Since chunkwise scan does not
directly provide every accepted token's final live state, the implementation
keeps q/k/v/g/beta and later rebuilds the live recurrent state for the accepted
suffix.

The current v4 code already has the same high-level semantics:

- target verify runs a fixed physical window of `CHUNK_SIZE + num_draft_tokens`;
- q/k/v/g/beta are cached in a rolling state-input window;
- chunk boundary state and conv window are exported during verify;
- live recurrent state is rebuilt after verify from boundary state plus accepted
  q/k/v/g/beta suffix;
- boundary state is committed when the accepted window crosses `CHUNK_SIZE`.

The remaining performance gap is mainly in `recurrent_state_from_qkvg_beta`.
The old v4 path grouped by accepted length and called FLA recurrent once per
distinct length. The optimized path packs variable accepted lengths into one
varlen update-kernel launch, computes only final states, and keeps the previous
grouped path semantics as the validation reference.

## Phase 1: Adapter Boundary Cleanup

Status: completed.

Changes:

- Added `DVRGatedForwardContext` for one gated linear-state forward.
- Moved repeated GDN backend arguments into the context:
  `layer`, `forward_batch`, `state_cache`, `cache_indices`, `query_start_loc`,
  `conv_states`, `ssm_states`, and `seq_len`.
- Updated adapter hook signatures to take that context:
  `maybe_cache_extend_state_inputs`, `maybe_process_target_verify_conv`, and
  `maybe_process_target_verify_state`.
- Kept GDN backend fallback logic local, while removing repeated DVR-specific
  parameter plumbing from call sites.

Validation:

- `py_compile` passed for DVR adapter and GDN backend.
- `git diff --check` passed.

## Phase 2: Dedicated Recurrent State From QKVG Beta

Status: completed.

Changes:

- Replaced Python-side per-length grouping in `recurrent_state_from_qkvg_beta`
  with a DVR-specific varlen rebuild path.
- The new path consumes `q, k, v, g, beta, initial_state, token_count`.
- It computes only final states using `fused_recurrent_gated_delta_rule_update`
  with `disable_output_calculation=True`.
- It supports per-row `token_count`, including zero-token rows.
- Uniform accepted lengths still use the original direct FLA recurrent path.
- Added `test/manual/dvr/test_dvr_gdn_recurrent_state.py` to compare the varlen
  rebuild path with the old grouped recurrent reference.

Validation:

- `test/manual/dvr/test_dvr_gdn_recurrent_state.py`: `max_diff=0.0`.

## Phase 3: End-To-End KL Regression

Status: completed.

Environment:

- `SGLANG_RETURN_ORIGINAL_LOGPROB=True`
- `SGLANG_ENABLE_JIT_DEEPGEMM=0`
- `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`
- `SGLANG_ENABLE_SPEC_V2=1`

Results:

- Qwen3.5 no-graph, `max_new=17,65,129`: all exact, `maxdiff=0.0`, `kl=0.0`.
- Qwen3 attention no-graph, `max_new=17,65,129`: all exact, `maxdiff=0.0`,
  `kl=0.0`.
- Qwen3.5 cuda graph, `max_new=17,65,129,257`: all exact, `maxdiff=0.0`,
  `kl=0.0`; the longest case stopped at 236 generated tokens due to EOS.

Acceptance rates observed:

- Qwen3.5 no-graph: `1.0`, `0.8933333333333333`, `1.0`.
- Qwen3 attention no-graph: `1.0`, `0.9466666666666667`, `1.0`.
- Qwen3.5 cuda graph: `1.0`, `1.0`, `1.0`, `0.7633333333333333`.

## Deferred Items

- A lower-level Triton kernel dedicated to DVR recurrent rebuild can still be
  added later. The current varlen update-kernel wrapper already removes the
  multi-launch Python grouping in the non-uniform accepted-length case while
  keeping code reuse high.
- A more fused GDN path that recomputes q/k/v/g/beta and recurrent state inside
  one kernel is a later performance patch.
- Piecewise cuda graph remains intentionally unsupported for this DVR branch.
- Wider KDA/Mamba integration should use the same adapter context and kernel
  hook shape after GDN is stable.
