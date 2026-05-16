# DVR v4 Optimization Plan

This branch starts from the v3 correctness baseline and optimizes it toward the
original DVR reference design. The goal is to keep the KL=0 behavior validated in
v3 while reducing verify compute, GDN state memory, and upstream merge surface.

## Target Design

The final flow should be:

1. Use the target model's normal decode path as the self draft provider.
2. Run EAGLE-style target verify with only `num_draft_tokens` physical rows.
3. Keep DVR chain verify on causal attention by using `custom_mask=None`.
4. For GDN/Mamba-like layers only, maintain a rolling
   `FLA_CHUNK_SIZE + num_draft_tokens` q/k/v/g/beta window.
5. Run GDN target verify with chunkwise scan over the internal rolling window
   and return only the draft suffix to the normal model forward.
6. Commit live state, chunk-boundary state, conv state, and q/k/v/g/beta rolling
   state after EAGLE-compatible acceptance.

## Phase 0: Baseline

- Compile DVR touched Python files.
- Run `git diff --check`.
- Record the current branch state before functional changes.

## Phase 1: Remove Full-Model 64+Draft Verify

- Remove or bypass worker-side fixed physical verify window construction.
- Keep target verify input as exactly `num_draft_tokens` rows.
- Keep `custom_mask=None` for DVR chain verify.
- Verify Qwen3 attention-only KL=0 before touching GDN internals.

## Phase 2: Shrink CUDA Graph Verify Shape

- Capture DVR target-verify graph with `num_draft_tokens` rows, not
  `FLA_CHUNK_SIZE + num_draft_tokens`.
- Remove the graph token helper path that adds `FLA_CHUNK_SIZE` for DVR.
- Keep only the minimal metadata patch needed to preserve causal attention.

## Phase 3: Move 64+Draft Window Inside GDN

- In GDN target verify, write the draft q/k/v/g/beta rows into the existing DVR
  rolling cache.
- Run chunkwise scan over the rolling `FLA_CHUNK_SIZE + draft` cache.
- Return only the draft suffix rows to the rest of the model.
- Export the boundary state needed for post-verify state commit.

## Phase 4: Compress GDN State Memory

- Replace 80-token `intermediate_ssm` usage for DVR with compact chunk-boundary
  state storage.
- Keep q/k/v/g/beta rolling cache at `FLA_CHUNK_SIZE + draft`.
- Keep conv windows at draft-token granularity whenever possible.

## Phase 5: State Commit Cleanup

- Commit non-crossing requests by rebuilding live recurrent state from boundary
  state plus accepted q/k/v/g/beta tail.
- Commit crossing requests by advancing the chunk boundary, shifting the rolling
  q/k/v/g/beta window, and rebuilding the live state from the new boundary.
- Keep batch requests independent so unequal acceptance lengths are supported.

## Phase 6: Server Args Cleanup

- Remove public chunk-boundary verify flag if DVR always uses that behavior.
- Centralize DVR defaults and validation in one server-args helper.
- Keep required GDN settings fixed: `mamba_track_interval=FLA_CHUNK_SIZE`,
  fp32 SSM states, page size not larger than `FLA_CHUNK_SIZE`.
- Disable piecewise CUDA graph for DVR until explicitly supported.

## Validation Gates

After each code phase:

- `python3 -m py_compile` on touched files.
- `git diff --check`.
- Qwen3 attention-only KL=0 for graph and no-graph when the phase affects verify
  or graph shape.

After GDN phases:

- Qwen3.5 GDN KL=0 for no-graph and graph.
- Long generations crossing 64, 128, and 256 tokens.
- Batch tests with unequal accepted lengths.
- Radix shared-prefix stress where applicable.

All 5090 runtime tests should disable DeepGEMM:

```bash
SGLANG_ENABLE_JIT_DEEPGEMM=0
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
```
