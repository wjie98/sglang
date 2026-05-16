# DVR v3 Update Report Against `upstream/sglang-miles`

This report summarizes the current `dvr-v3-controlled-migration` branch
relative to `upstream/sglang-miles`. It focuses on code paths that matter for
reviewing and continuing the DVR merge.

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

## File-Level Summary

## Complete File Inventory

This section lists every file changed relative to `upstream/sglang-miles`.

1. `DVR_CLEANUP_PHASE1_PLAN.md` (+49)
   - Records the first cleanup pass for the v3 migration.
   - Focus: keep DVR close to EAGLE control flow and reduce early scaffolding.

2. `DVR_CLEANUP_PHASE2_PLAN.md` (+129)
   - Documents the self-decode draft provider cleanup.
   - Records the decision to use target/self decode as the draft path instead
     of an extend-based draft path.

3. `DVR_CLEANUP_PHASE3_PLAN.md` (+134)
   - Tracks readability and structure cleanup around fixed verify windows.
   - Contains the optimization rules used later: EAGLE-style control flow,
     minimal intrusion, loop-invariant simplification, and strict KL=0.

4. `DVR_CLEANUP_PHASE4_PLAN.md` (+127)
   - Records the GDN deterministic-verify cleanup direction.
   - Focus: separate ordinary EAGLE verify from DVR's chunk-aligned GDN verify.

5. `DVR_CLEANUP_PHASE5_PLAN.md` (+143)
   - Documents page-size and radix-cache cleanup.
   - Records the page-safe allocation/free model for DVR fixed windows.

6. `DVR_CLEANUP_PHASE6_PLAN.md` (+165)
   - Latest cleanup plan and execution log.
   - Records the `MambaDVRQKVGBetaCache` helper, page-size normalization,
     piecewise CUDA graph disabling, and final KL=0 validation matrix.

7. `DVR_DOCUMENTATION_INVENTORY.md` (+46)
   - Index of root-level DVR notes.
   - Separates upstream-facing candidates from development-only records.

8. `DVR_EAGLE_POSTPROCESS_EXPERIMENT.md` (+50)
   - Records the experiment that aligned DVR verify/postprocess with EAGLE's
     anchor/bonus and acceptance contract.

9. `DVR_GDN_EXPERIMENT_NOTES.md` (+580)
   - Detailed GDN debugging notes.
   - Covers conv state, q/k/v/g/beta, temporal state, chunkwise/recurrent
     differences, and known failure modes encountered during migration.

10. `DVR_GDN_MERGE_PLAN.md` (+168)
    - GDN merge plan.
    - Explains target verify, post-verify, q/k/v/g/beta caching, and state
      commit requirements.

11. `DVR_KL_TESTING_GUIDE.md` (+230)
    - Strict KL=0 testing guide.
    - Documents full-prefill scoring oracle, launch flags, DeepGEMM disable
      flags, page-size rules, and expected validation commands.

12. `DVR_PAGE_SIZE_FIX_PLAN.md` (+163)
    - Page-size and radix-cache design note.
    - Explains why DVR can support page sizes that divide `FLA_CHUNK_SIZE`
      without requiring `num_draft_tokens % page_size == 0`.

13. `DVR_QWEN35_DIVERGENCE_REPORT.md` (+38)
    - Early Qwen3.5/GDN divergence note.
    - Records the observed divergence before later GDN state fixes.

14. `DVR_QWEN35_DIVERGENCE_REPORT.py` (+194)
    - Debug script for Qwen3.5/GDN divergence measurements.
    - Generates comparison data for different generation lengths.

15. `DVR_QWEN35_DIVERGENCE_REPORT_256.md` (+44)
    - Longer Qwen3.5 divergence report at 256-token scale.
    - Kept as historical debugging evidence.

16. `DVR_QWEN35_DIVERGENCE_SUMMARY.md` (+101)
    - Condensed summary of Qwen3.5 divergence experiments.
    - Captures the path from divergence to GDN state-management fixes.

17. `DVR_QWEN3_BATCH_KL_TEST.py` (+144)
    - Batch KL test script used for Qwen3 and Qwen3.5.
    - Compares served generation logprobs against full-prefill scoring and
      reports strict KL/maxdiff plus acceptance stats.

18. `DVR_UPDATE_REPORT.md` (+258 in the previous commit, now expanded)
    - Current implementation report against `upstream/sglang-miles`.
    - This file now includes a complete changed-file inventory.

19. `DVR_V3_BASELINE_RESULTS.md` (+103)
    - Baseline deterministic inference results before DVR-specific changes.
    - Used to distinguish upstream deterministic behavior from DVR behavior.

20. `DVR_V3_CONTROLLED_MIGRATION_PLAN.md` (+398)
    - Main v3 controlled migration plan and execution record.
    - Tracks how v3 moved from scaffold to self-draft, fixed-window verify,
      GDN state management, and cleanup.

21. `DVR_V3_IMPLEMENTATION_PLAN.md` (+470)
    - Early v3 implementation plan.
    - Captures the intended architecture before the controlled migration.

22. `DVR_V3_MERGE_CONTEXT.md` (+193)
    - Context note for the third merge attempt.
    - Summarizes reference-branch intent, assumptions, and upstream target.

23. `python/sglang/srt/disaggregation/decode.py` (+2)
    - Adds `enable_dvr_qkvg_beta_cache` plumbing into
      `HybridMambaDecodeReqToTokenPool`.
    - Ensures decode-disaggregation Mamba pools can allocate DVR q/k/v/g/beta
      cache when DVR is enabled.

24. `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` (+7/-1)
    - Small FLA wrapper adjustment for deterministic GDN/DVR behavior.
    - Keeps chunk size and chunkwise scan behavior aligned with DVR's
      chunk-boundary verify assumptions.

25. `python/sglang/srt/layers/attention/fla/chunk_o.py` (+5/-1)
    - Companion FLA chunk-output adjustment.
    - Part of the local deterministic FLA/GDN compatibility changes.

26. `python/sglang/srt/layers/attention/fla/fused_norm_gate.py` (+1/-1)
    - Sets gated norm Triton block rows to deterministic policy
      `MAX_ROWS_PER_BLOCK = 1`.
    - This matches the GDN deterministic-inference requirement discussed with
      upstream/core developers.

27. `python/sglang/srt/layers/attention/fla/layernorm_gated.py` (+1/-1)
    - Applies the same deterministic gated RMSNorm row-block policy.
    - Prevents row-block fusion from introducing GDN logit drift.

28. `python/sglang/srt/layers/attention/linear/gdn_backend.py` (+560/-? overall)
    - Adds DVR-specific data interfaces to GDN backend while keeping
      accept/reject logic in the worker.
    - Caches prompt-tail and verify-window q/k/v/g/beta.
    - Runs DVR target verify through chunkwise scan.
    - Exports conv windows and chunk-boundary state.
    - Commits live recurrent state, boundary state, conv state, and q/k/v/g/beta
      rolling positions after verify.

29. `python/sglang/srt/layers/attention/linear/mamba_dvr_utils.py` (+206)
    - New Mamba/DVR helper module.
    - Provides `MambaDVRFlaOps`, `MambaDVRQKVGBetaCache`,
      `write_dvr_chunk_boundary_state`, and `build_dvr_conv_windows`.
    - Keeps reusable Mamba-like DVR state operations out of the generic GDN
      backend body.

30. `python/sglang/srt/managers/scheduler_output_processor_mixin.py` (+2)
    - Skips ordinary Mamba checkpoint postprocessing for DVR requests.
    - DVR worker/backend owns the GDN state commit path after target verify.

31. `python/sglang/srt/mem_cache/memory_pool.py` (+167/-?)
    - Extends `MambaPool.SpeculativeState` with DVR q/k/v/g/beta caches and
      rolling position tracking.
    - Allocates `FLA_CHUNK_SIZE + num_draft_tokens` rows for DVR fixed windows.
    - Logs q/k/v/g/beta cache memory usage for capacity debugging.

32. `python/sglang/srt/model_executor/cuda_graph_runner.py` (+95/-?)
    - Adds `get_target_verify_graph_num_tokens_per_bs`.
    - Captures DVR target-verify CUDA graph with physical window size
      `FLA_CHUNK_SIZE + draft`.
    - Copies Mamba track metadata into replay buffers.
    - Supports ordinary CUDA graph for DVR while piecewise graph remains
      disabled.

33. `python/sglang/srt/model_executor/model_runner.py` (+17/-?)
    - Treats DVR as a target-verify speculative algorithm for CUDA graph
      capture.
    - Builds `EagleVerifyInput` metadata for DVR graph capture using physical
      verify token count.

34. `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` (+5)
    - Computes `enable_dvr_qkvg_beta_cache` from the speculative algorithm.
    - Passes the flag into Mamba/hybrid request-token pool construction.

35. `python/sglang/srt/models/qwen3_5.py` (+4/-1)
    - Formatting-only import cleanup around `RMSNormGated`.
    - No behavioral model change in this file.

36. `python/sglang/srt/server_args.py` (+148/-?)
    - Adds `speculative_dvr_chunk_boundary_verify`.
    - Adds DVR defaulting/validation for CUDA-only, topk=1, self-draft, default
      draft count 16, and `num_steps=draft-1`.
    - For GDN chunk-boundary verify, forces `extra_buffer`,
      `mamba_track_interval=64`, and fp32 SSM state.
    - Normalizes page size to a divisor of `FLA_CHUNK_SIZE`.
    - Disables piecewise CUDA graph for DVR.

37. `python/sglang/srt/speculative/chain_rejection.py` (+69)
    - New chain-mode reject sampling helper.
    - Used by DVR when self-draft probabilities are available.
    - Keeps EAGLE-compatible output layout.

38. `python/sglang/srt/speculative/dvr_utils.py` (+78)
    - New DVR utility context managers.
    - Clears custom tree mask for causal DVR verify in CUDA graph metadata.
    - Temporarily adjusts backend verify-window token count for fixed physical
      windows.

39. `python/sglang/srt/speculative/dvr_worker.py` (+1138)
    - New DVR worker implementation.
    - Implements self-decode draft, EAGLE-compatible target verify/postprocess,
      fixed GDN verify windows, page-safe padding KV management, GDN state
      backup/restore/commit, and accepted-logit selection.

40. `python/sglang/srt/speculative/eagle_info.py` (+17/-?)
    - Extends EAGLE verify metadata with optional `draft_probs`.
    - Adds non-greedy chain rejection sampling path for DVR/self-draft.
    - Keeps target-only fallback for ordinary EAGLE-style target verify.

41. `python/sglang/srt/speculative/spec_info.py` (+21)
    - Adds `SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK`.
    - Adds helper predicates for DVR and target-verify-forward algorithms.
    - Wires DVR algorithm to `DecodeVerifyRollbackWorker`.

### `python/sglang/srt/speculative/dvr_worker.py`

- Lines 50-108: fixed-window state dataclasses.
  - `_GDNFixedVerifyWindowState` snapshots request/spec tensors before GDN
    fixed-window verify rewrites them.
  - `_GDNDVRStateContext` carries live and boundary Mamba slot indices.
  - `_GDNFixedVerifyRequestWindow` describes one request's
    `verified + draft + padding` physical window.
- Lines 110-193: `DecodeVerifyRollbackWorker` initialization.
  - Enforces chain topk=1.
  - Validates page size for chunk-boundary verify.
  - Requires GDN `mamba_track_interval == FLA_CHUNK_SIZE` and fp32 SSM state.
  - Sets `verify_window_size = FLA_CHUNK_SIZE + num_draft_tokens`.
- Lines 204-250: public generation entrypoint.
  - Extend/prefill is delegated to the target worker.
  - Decode generation uses `draft -> verify -> postprocess`.
- Lines 255-324: self-decode draft KV allocation.
  - Uses normal paged extension allocation for `page_size > 1`.
  - Draft writes directly into the KV slots that target verify will read.
- Lines 335-400: self-draft construction.
  - Reuses EAGLE tree metadata for retrieve/order bookkeeping.
  - Forces `custom_mask=None` for DVR chain verify so attention follows the
    causal path.
- Lines 503-760: GDN boundary-state lifecycle.
  - Tracks chunk-aligned boundary state separately from live decode state.
  - Restores boundary state before target verify so draft decode does not
    pollute the deterministic verify start.
  - Reuses prefill ping-pong checkpoint when available; zero-initializes only
    the boundary-0 case.
- Lines 768-848: GDN fixed physical verify window.
  - Rewrites batch/spec tensors to a fixed `64 + draft` physical window.
  - Treats prefill tail tokens as already verified tokens.
  - Resets `dvr_qkvg_beta_pos` for target verify export.
- Lines 850-907: per-request window builder.
  - Builds `verified_token + draft_token + padding_token`.
  - Allocates padding KV slots separately from draft-owned slots.
- Lines 909-937: restore after fixed verify.
  - Keeps only logical draft rows in logits/hidden states.
  - Frees padding-only KV slots.
  - Restores original batch/spec metadata.
- Lines 953-994: GDN state commit after verify.
  - Computes accepted token metadata.
  - Calls GDN backend state commit.
  - Advances per-request chunk boundary when a request crosses `FLA_CHUNK_SIZE`.
- Lines 999-1133: accepted-token and target-verify postprocess.
  - Keeps output/logprob handling close to EAGLE.
  - Calls `add_output_logprobs_for_spec_v1` after accepted rows are selected.

### `python/sglang/srt/layers/attention/linear/gdn_backend.py`

- Lines 333-343: DVR FLA op hooks.
  - Stores the current chunkwise scan and recurrent state rebuild callables.
  - Refreshes hooks in decode/extend before verify.
- Lines 345-412: prefill/extend q/k/v/g/beta tail cache.
  - Seeds the rolling DVR q/k/v/g/beta window with the prompt tail after the
    latest chunk boundary.
- Lines 414-455: target-verify q/k/v/g/beta window cache.
  - Saves the fixed physical verify window inputs for later recurrent state
    rebuild.
- Lines 457-499: DVR verify conv path.
  - Runs linear fixed-window conv.
  - Exports per-token conv windows into `intermediate_conv_window`.
- Lines 501-551: DVR chunkwise target verify.
  - Runs the prefill-like chunkwise GDN scan for target verify.
  - Writes the first chunk-boundary state from FLA output `h`.
  - Caches q/k/v/g/beta inputs for post-accept state commit.
- Lines 553-581: runtime invariant checks.
  - Guards q/k/v/g/beta tail positions and accepted state steps.
- Lines 583-691: GDN state commit.
  - Rebuilds live temporal state with the recurrent kernel for accepted suffixes.
  - Scatters accepted conv windows.
  - Commits chunk-boundary temporal/conv state when a request crosses 64 tokens.
  - Shifts the q/k/v/g/beta suffix after boundary commit.
- Lines 769-930: forward integration.
  - Splits ordinary target verify from DVR target verify.
  - Keeps speculative accept/reject logic out of the generic GDN backend.

### `python/sglang/srt/layers/attention/linear/mamba_dvr_utils.py`

- Lines 9-67: `MambaDVRFlaOps`.
  - Holds chunk-scan and recurrent-state FLA callables.
- Lines 70-166: `MambaDVRQKVGBetaCache`.
  - Wraps q/k/v/g/beta DVR caches and the rolling position tensor.
  - Writes prefill tail rows and fixed verify-window rows.
  - Shifts suffix rows after a chunk-boundary commit.
  - Important detail: layer-dimension handling is inferred from
    `dvr_qkvg_beta_pos`, not from individual cache rank, because q/k/v and
    g/beta cache ranks differ.
- Lines 169-193: `write_dvr_chunk_boundary_state`.
  - Extracts the first chunk boundary from packed FLA `h`.
  - Handles batched fixed windows where each request contributes multiple FLA
    chunks.
- Lines 196-206: `build_dvr_conv_windows`.
  - Builds per-token conv windows from the boundary conv state plus fixed verify
    input.

### `python/sglang/srt/server_args.py`

- Lines 496-505: new DVR CLI/state fields.
  - Adds `speculative_dvr_chunk_boundary_verify`.
- Lines 1059-1117: DVR chunk-boundary launch defaults.
  - Normalizes page size for chunk-boundary verify.
  - For GDN, forces `extra_buffer`, `mamba_track_interval=FLA_CHUNK_SIZE`, and
    fp32 SSM state.
- Lines 1119-1129: page-size normalization.
  - Values larger than `FLA_CHUNK_SIZE` are lowered to `FLA_CHUNK_SIZE`.
  - Non-divisor values are lowered to the nearest power-of-two divisor of
    `FLA_CHUNK_SIZE`.
- Lines 1168-1178: DVR disables piecewise CUDA graph.
  - This branch supports ordinary CUDA graph, not piecewise CUDA graph.
- Lines 3107-3150: DVR speculative argument validation.
  - Requires CUDA, no DP attention, no draft model path, topk=1.
  - Defaults `speculative_num_draft_tokens` to 16.
  - Sets `speculative_num_steps = num_draft_tokens - 1`.
  - Disables overlap schedule and mixed chunked prefill.

### `python/sglang/srt/speculative/eagle_info.py`

- Lines 55-72: `EagleVerifyInput` carries optional `draft_probs`.
- Lines 329-387: non-greedy verify path.
  - Computes target probabilities with temperature/top-k/top-p.
  - Uses DVR's chain rejection sampler when draft probabilities are present.
  - Keeps target-only fallback for paths without draft probabilities.
- Lines 430-442: speculative stats.
  - Updates committed KV length and acceptance counters.

### `python/sglang/srt/speculative/chain_rejection.py`

- Lines 84-147: chain speculative sampling.
  - Implements top-1 chain reject sampling for self-draft.
  - Anchor token is already target-sampled.
  - Accepted draft tokens follow `min(1, p/q)`.
  - Final slot is filled with target residual distribution or target argmax
    fallback if residual mass is zero.

### `python/sglang/srt/speculative/dvr_utils.py`

- Lines 4-50: causal verify metadata helper.
  - Temporarily provides a custom-mask buffer only for CUDA graph metadata shape
    construction.
  - Clears captured metadata so DVR verify uses causal attention, not tree mask.
- Lines 53-79: runtime verify window helper.
  - Temporarily overrides backend verify-token count for GDN fixed physical
    windows without changing generic backend code.

### `python/sglang/srt/model_executor/cuda_graph_runner.py`

- Lines 100-112: target-verify graph token count.
  - Adds `FLA_CHUNK_SIZE` to graph capture token count for DVR chunk-boundary
    verify.
  - Ensures CUDA graph buffers are captured as physical `64 + draft`, not
    logical draft-only.
- Lines 317-329 and 363-370: replay input copying.
  - Copies mamba track indices/masks into graph buffers when present.
  - Uses grouped GPU copies where possible, then CPU `seq_lens_cpu` copy.

### `python/sglang/srt/mem_cache/memory_pool.py`

- Lines 191-220: Mamba state dataclasses extended for speculative state access.
- Lines 350-420: speculative Mamba cache allocation.
  - Allocates `FLA_CHUNK_SIZE + num_draft_tokens` intermediate state/cache rows
    when DVR q/k/v/g/beta cache is enabled.
  - Adds q/k/v/g/beta caches and q/k/v/g/beta position tracking.
- Lines 421-428: logs DVR q/k/v/g/beta cache memory usage.

### FLA deterministic fixes

- `python/sglang/srt/layers/attention/fla/fused_norm_gate.py` lines 21-22:
  `MAX_ROWS_PER_BLOCK = 1`.
- `python/sglang/srt/layers/attention/fla/layernorm_gated.py`: same
  deterministic gated RMSNorm block-size policy.
- `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` and
  `python/sglang/srt/layers/attention/fla/chunk_o.py`: small DVR/GDN
  determinism-related FLA adjustments kept local to the vendored FLA wrappers.

### Other integration points

- `python/sglang/srt/model_executor/model_runner.py`: wires DVR worker creation
  and forwards DVR-specific CUDA graph metadata context.
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`: passes
  DVR q/k/v/g/beta cache allocation flags into Mamba pool setup.
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`: honors DVR
  committed-token handling from the worker.
- `python/sglang/srt/disaggregation/decode.py`: carries the new speculative
  mode through decode-side setup.
- `python/sglang/srt/models/qwen3_5.py`: keeps Qwen3.5/GDN model path aligned
  with the new GDN state handling.
- `python/sglang/srt/speculative/spec_info.py`: adds DVR/spec metadata needed
  by verify and CUDA graph paths.

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
