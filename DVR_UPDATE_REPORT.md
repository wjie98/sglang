# DVR v3 Update Report Against `upstream/sglang-miles`

This report summarizes the current `dvr-v3-controlled-migration` branch after
archiving development notes into `/home/hwj/dvr_qwen3_5/dvr-records-archive`.
It lists every file that remains changed relative to `upstream/sglang-miles`.

## Scope

- Branch base: `upstream/sglang-miles`
- Main algorithm: `DECODE_VERIFY_ROLLBACK`
- Draft provider: target/self decode only
- Verify mode: EAGLE-compatible target verify plus DVR fixed physical window
- GDN state mode: chunk-boundary verify with fp32 Mamba/GDN state checkpoints
- Unsupported in this branch: piecewise CUDA graph for DVR, DP attention for DVR,
  non-chain topk, external draft model integration

## High-Level Flow

1. Normal prefill/extend runs first and returns the first verified token.
2. DVR draft uses the target model's normal decode path to produce
   `speculative_num_draft_tokens` draft tokens.
3. DVR target verify runs in `ForwardMode.TARGET_VERIFY`, matching EAGLE's
   scheduler and postprocess contract.
4. Attention-only models verify a chain with causal attention and no custom tree
   mask.
5. GDN models use a fixed physical verify window:
   `verified_token + draft_token + padding_token`, with physical size
   `FLA_CHUNK_SIZE + speculative_num_draft_tokens`.
6. Verify logits are sliced back to the logical draft-token rows before SPS
   acceptance.
7. Accepted tokens are committed through EAGLE-style postprocess. For GDN,
   temporal state, conv state, q/k/v/g/beta rolling windows, and chunk-boundary
   checkpoints are committed together.

## Current File Inventory

The current net diff contains 22 files: 2 root-level documents, 19 runtime code
files, and 1 test script.

1. `DVR_KL_TESTING_GUIDE.md`
   - Strict KL=0 testing guide.
   - Documents launch flags, DeepGEMM disable flags, page-size rules, full
     prefill scoring oracle, and local validation commands.

2. `DVR_UPDATE_REPORT.md`
   - This branch-level report.
   - Kept as a concise review index after archiving raw experiment logs.

3. `python/sglang/srt/disaggregation/decode.py`
   - Adds `enable_dvr_qkvg_beta_cache` plumbing into
     `HybridMambaDecodeReqToTokenPool`.
   - Ensures decode-disaggregation Mamba pools can allocate DVR q/k/v/g/beta
     cache when DVR is enabled.

4. `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`
   - Small FLA wrapper adjustment for deterministic GDN/DVR behavior.
   - Keeps chunkwise scan behavior aligned with DVR's chunk-boundary verify
     assumptions.

5. `python/sglang/srt/layers/attention/fla/chunk_o.py`
   - Companion FLA chunk-output adjustment.
   - Part of the local deterministic FLA/GDN compatibility changes.

6. `python/sglang/srt/layers/attention/fla/fused_norm_gate.py`
   - Sets gated norm Triton block rows to deterministic policy
     `MAX_ROWS_PER_BLOCK = 1`.
   - Prevents row-block fusion from introducing GDN logit drift.

7. `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
   - Applies the same deterministic gated RMSNorm row-block policy.
   - Keeps Qwen3.5/GDN prefill, decode, and verify numerically aligned.

8. `python/sglang/srt/layers/attention/linear/gdn_backend.py`
   - Adds DVR-specific data interfaces to GDN backend while keeping
     accept/reject logic in the worker.
   - Caches prompt-tail and verify-window q/k/v/g/beta.
   - Runs DVR target verify through chunkwise scan.
   - Exports conv windows and chunk-boundary state.
   - Commits live recurrent state, boundary state, conv state, and q/k/v/g/beta
     rolling positions after verify.

9. `python/sglang/srt/layers/attention/linear/mamba_dvr_utils.py`
   - New Mamba/DVR helper module.
   - Provides `MambaDVRFlaOps`, `MambaDVRQKVGBetaCache`,
     `write_dvr_chunk_boundary_state`, and `build_dvr_conv_windows`.
   - Keeps reusable Mamba-like DVR state operations out of the generic GDN
     backend body.

10. `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
    - Skips ordinary Mamba checkpoint postprocessing for DVR requests.
    - DVR worker/backend owns the GDN state commit path after target verify.

11. `python/sglang/srt/mem_cache/memory_pool.py`
    - Extends `MambaPool.SpeculativeState` with DVR q/k/v/g/beta caches and
      rolling position tracking.
    - Allocates `FLA_CHUNK_SIZE + num_draft_tokens` rows for DVR fixed windows.
    - Logs q/k/v/g/beta cache memory usage for capacity debugging.

12. `python/sglang/srt/model_executor/cuda_graph_runner.py`
    - Adds `get_target_verify_graph_num_tokens_per_bs`.
    - Captures DVR target-verify CUDA graph with physical window size
      `FLA_CHUNK_SIZE + draft`.
    - Copies Mamba track metadata into replay buffers.
    - Supports ordinary CUDA graph for DVR while piecewise graph remains
      disabled.

13. `python/sglang/srt/model_executor/model_runner.py`
    - Treats DVR as a target-verify speculative algorithm for CUDA graph
      capture.
    - Builds `EagleVerifyInput` metadata for DVR graph capture using physical
      verify token count.

14. `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
    - Computes `enable_dvr_qkvg_beta_cache` from the speculative algorithm.
    - Passes the flag into Mamba/hybrid request-token pool construction.

15. `python/sglang/srt/models/qwen3_5.py`
    - Formatting-only import cleanup around `RMSNormGated`.
    - No behavioral model change in this file.

16. `python/sglang/srt/server_args.py`
    - Adds `speculative_dvr_chunk_boundary_verify`.
    - Adds DVR defaulting/validation for CUDA-only, topk=1, self-draft, default
      draft count 16, and `num_steps=draft-1`.
    - For GDN chunk-boundary verify, forces `extra_buffer`,
      `mamba_track_interval=64`, and fp32 SSM state.
    - Normalizes page size to a divisor of `FLA_CHUNK_SIZE`.
    - Disables piecewise CUDA graph for DVR.

17. `python/sglang/srt/speculative/chain_rejection.py`
    - New chain-mode reject sampling helper.
    - Used by DVR when self-draft probabilities are available.
    - Keeps EAGLE-compatible output layout.

18. `python/sglang/srt/speculative/dvr_utils.py`
    - New DVR utility context managers.
    - Clears custom tree mask for causal DVR verify in CUDA graph metadata.
    - Temporarily adjusts backend verify-window token count for fixed physical
      windows.

19. `python/sglang/srt/speculative/dvr_worker.py`
    - New DVR worker implementation.
    - Implements self-decode draft, EAGLE-compatible target verify/postprocess,
      fixed GDN verify windows, page-safe padding KV management, GDN state
      backup/restore/commit, and accepted-logit selection.

20. `python/sglang/srt/speculative/eagle_info.py`
    - Extends EAGLE verify metadata with optional `draft_probs`.
    - Adds non-greedy chain rejection sampling path for DVR/self-draft.
    - Keeps target-only fallback for ordinary EAGLE-style target verify.

21. `python/sglang/srt/speculative/spec_info.py`
    - Adds `SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK`.
    - Adds helper predicates for DVR and target-verify-forward algorithms.
    - Wires DVR algorithm to `DecodeVerifyRollbackWorker`.

22. `test/srt/test_dvr_batch_kl.py`
    - Batch strict-KL oracle script.
    - Sends concurrent generation requests, rescoring each prompt plus output
      with a full-prefill pass.
    - Reports token-id equality, max logprob difference, KL-like absolute sum,
      and speculative acceptance metrics.

## Runtime Validation

Latest cleanup validation:

- Static:
  - `python3 -m py_compile` on touched Python files.
  - `conda run -n dvr_dev python -m py_compile` on touched Python files.
  - `git diff --check`.
- Qwen3 attention-only:
  - No CUDA graph, launch `--page-size 48` normalized to `32`,
    `max_new=17,65,129`, strict KL=0.
  - CUDA graph, launch `--page-size 7` normalized to `4`,
    `max_new=17,65,129`, strict KL=0.
- Qwen3.5/GDN:
  - No CUDA graph, launch `--page-size 96` normalized to `64`,
    `max_new=17,65,129,257`, strict KL=0.
  - CUDA graph, `--cuda-graph-max-bs 4`, `max_new=17,65,129`, strict KL=0.

## Operational Notes

- Always disable DeepGEMM on the local 5090 validation setup:
  - `SGLANG_ENABLE_JIT_DEEPGEMM=0`
  - `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`
- Use `SGLANG_RETURN_ORIGINAL_LOGPROB=True` for strict KL oracle tests.
- GDN DVR fp32 state caches are large. On the local test machine, use small
  `--max-running-requests` and explicit `--max-total-tokens` unless testing
  capacity behavior.
- `page_size` for DVR chunk-boundary verify must be no larger than
  `FLA_CHUNK_SIZE` and divide `FLA_CHUNK_SIZE`; launch normalization lowers
  incompatible values automatically.
- DVR in this branch intentionally disables piecewise CUDA graph. Ordinary CUDA
  graph is supported and validated.
