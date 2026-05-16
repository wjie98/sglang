# DVR v4 Review Cleanup Plan

This pass keeps the validated DVR behavior but reduces review noise before a new
v5 patch view is generated.

## Goals

- Remove the public `--speculative-dvr-chunk-boundary-verify` switch. DVR always
  uses chunk-boundary verify semantics.
- Keep `server_args.py` close to upstream flow by appending DVR-specific
  normalization at the end of speculative decoding handling instead of creating
  a separate early-return path.
- Merge small DVR sampling files into `dvr_utils.py`.
- Remove explicit `enable_dvr_qkvg_beta_cache` argument plumbing. Mamba cache
  allocation should detect DVR mode from `ServerArgs`.
- Remove the disaggregation decode signature change. DVR disaggregation remains
  unsupported until explicitly validated.
- Narrow and document DVR's skip of the generic Mamba prefix-cache update.
- Keep CUDA graph/model runner changes only where DVR target-verify graph shape
  requires them, with comments.
- Make GDN's DVR adaptation look like a small data interface rather than
  request-level speculative logic inside the backend.

## Validation

After each structural cleanup: static compile and `git diff --check`.

Final regression:

- Qwen3-0.6B graph/no-graph strict KL=0.
- Qwen3.5-0.8B graph/no-graph strict KL=0.
- `page_size > FLA_CHUNK_SIZE` downgrade smoke test.

## Execution Notes

- Removed the public chunk-boundary verify switch. `DECODE_VERIFY_ROLLBACK`
  always enables the chunk-boundary DVR path.
- Kept `server_args.py` flow close to upstream by appending DVR normalization at
  the end of speculative decoding handling instead of returning early.
- Merged DVR chain sampling helpers into `dvr_utils.py`; no separate
  `dvr_sampling.py` or `chain_rejection.py` remains.
- Removed explicit q/k/v/g/beta cache allocation arguments from model runner and
  memory-pool plumbing. The Mamba pool now infers DVR cache allocation from the
  global speculative algorithm.
- Removed the disaggregation decode signature change. DVR still rejects
  disaggregation mode at argument normalization time.
- Restricted the generic Mamba prefix-cache skip to the ping-pong checkpoint
  update block and documented why DVR owns chunk-aligned checkpoint commits.
- Kept cuda graph/model runner changes limited to target-verify graph shape
  handling and comments.
- Moved DVR GDN cache predicates and validation helpers into
  `mamba_dvr_utils.py`, leaving `gdn_backend.py` focused on forward and commit
  sequencing.

## Regression Results

Environment for all runtime tests:

- `SGLANG_RETURN_ORIGINAL_LOGPROB=True`
- `SGLANG_ENABLE_JIT_DEEPGEMM=0`
- `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`
- `SGLANG_ENABLE_SPEC_V2=1`
- Conda env: `dvr_dev`

Results:

- Qwen3-0.6B no graph, max_new `17,65,129`: strict KL=0, all cases passed.
- Qwen3-0.6B cuda graph, max_new `17,65,129`: strict KL=0, all cases passed.
- Qwen3.5-0.8B no graph, max_new `17,65,129,257`: strict KL=0, all cases
  passed.
- Qwen3.5-0.8B cuda graph, max_new `17,65,129,257`: strict KL=0, all generated
  cases passed. The 257-token case stopped at 181 generated tokens due to EOS.
- Qwen3.5-0.8B cuda graph with `--page-size 128`: startup downgraded page size
  to 64 and max_new `17,65` passed strict KL=0.

Observed but not addressed in this cleanup:

- Some graph-mode batch cases show acceptance below 1.0 while strict KL remains
  zero. This pass preserves the current sampling/verification behavior and only
  reduces merge noise.
