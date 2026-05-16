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
