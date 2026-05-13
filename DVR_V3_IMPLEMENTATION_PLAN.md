# DVR V3 Implementation Plan

This is the executable merge plan for adding DVR to the current
`dvr-clean-v3` branch. The base is latest `upstream/sglang-miles`.

The plan deliberately avoids copying v2/ref implementation details. Use
`origin/dvr_qwen3.5` only to understand DVR semantics and GDN state handling.

## Global Principles

- Keep changes small and phase-separated.
- Preserve upstream deterministic inference behavior; tests pass
  `--enable-deterministic-inference` explicitly.
- Start with attention-only correctness before touching GDN state.
- Returned tokens and returned logprobs must come from target verify, not draft
  decode.
- Chain speculative layout only: `topk == 1`.
- `num_draft_tokens` is internal DVR round size; user `max_new_tokens` remains
  normal API generation length.
- GDN/Mamba/KDA persistent recurrent state may only advance at verified
  `CHUNK_SIZE` boundaries.

## Phase 0: Metadata And Baseline

Goal: keep the branch traceable and verify the upstream baseline before code
changes.

Files:

- `DVR_V3_MERGE_CONTEXT.md`
- `DVR_KL_TESTING_GUIDE.md`
- `DVR_V3_IMPLEMENTATION_PLAN.md`

Actions:

- Confirm branch is based on latest `upstream/sglang-miles`.
- Confirm `origin/dvr_qwen3.5` is available as the only reference branch.
- Run a non-DVR deterministic inference smoke test when needed.

Completion criteria:

- Branch has only planning/docs changes.
- We know the exact upstream commit used as merge base.

Commit boundary:

- `docs: seed dvr v3 merge context`
- `docs: add dvr v3 implementation plan`

## Phase 1: Add DVR Algorithm Shell

Goal: make DVR selectable without changing normal inference behavior.

Primary files:

- `python/sglang/srt/speculative/spec_info.py`
- `python/sglang/srt/speculative/dvr_worker.py`
- `python/sglang/srt/server_args.py`
- scheduler worker creation path, if current upstream requires a separate hook

Implementation:

- Add `SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK`.
- Make `create_worker()` return `DecodeVerifyRollbackWorker`.
- Reject overlap scheduling for DVR initially.
- Require:
  - `page_size == 1`
  - `speculative_eagle_topk == 1`
  - self-model draft only
- Do not add a separate deterministic inference override. DVR launch examples
  pass `--enable-deterministic-inference`.
- Add a minimal `DecodeVerifyRollbackWorker` that delegates to the target worker
  until Phase 2 logic is added.

Design notes:

- Prefer integrating through existing speculative algorithm plumbing rather than
  introducing a parallel `--enable-dvr` path.
- Use explicit errors for unsupported combinations.

Tests:

- Import smoke:
  - `PYTHONPATH=python python -c "from sglang.srt.speculative.spec_info import SpeculativeAlgorithm; print(SpeculativeAlgorithm.from_string('DECODE_VERIFY_ROLLBACK'))"`
- Server argument parse smoke.
- Non-DVR path must remain unchanged.

Commit boundary:

- `dvr: add speculative algorithm shell`

## Phase 2: Self Decode Draft Provider

Goal: produce chain draft tokens with the target model's normal decode path.

Primary files:

- `python/sglang/srt/speculative/dvr_worker.py`
- possibly small helpers under `python/sglang/srt/speculative/dvr_*.py`

Implementation:

- On prompt/extend batches, run normal target extend and store the generated
  token as the next DVR anchor.
- On decode batches:
  - use current anchor as the first chain token;
  - run normal decode repeatedly for `num_draft_tokens - 1` additional draft
    steps;
  - sample with the same sampling parameters as normal generation;
  - support non-greedy sampling with deterministic seed/position behavior;
  - build an `EagleVerifyInput`-compatible chain layout with `topk == 1`.
- Restore batch fields after draft forward:
  - `seq_lens`
  - `seq_lens_cpu`
  - `seq_lens_sum`
  - `spec_info`
  - `positions`
  - `out_cache_loc`
  - `forward_mode`
- Draft KV is temporary and must not be returned to the scheduler as committed
  cache.

State semantics:

- Draft decode is allowed to mutate temporary KV/state during the draft phase.
- Phase 2 only guarantees attention-only correctness; recurrent state rollback
  is introduced in Phase 4.

Tests:

- Start Qwen3-0.6B with DVR algorithm.
- Generate with `temperature=0.7`, `top_p=0.9`, `max_new_tokens=8`.
- Print/inspect acceptance metadata if available.
- Server must not leak KV pages or crash after repeated requests.

Commit boundary:

- `dvr: implement self decode draft`

## Phase 3: Attention Target Verify And Commit

Goal: make attention-only DVR semantically correct: returned tokens/logprobs
come from prefill/extend-equivalent verify.

Primary files:

- `python/sglang/srt/speculative/dvr_worker.py`
- optional small state file such as `dvr_state.py`

Core logic:

- Maintain per-request DVR state:
  - current anchor token;
  - whether the anchor has already been returned;
  - pending chunk token IDs after the last aligned start;
  - logprob/top-logprob payload for held anchor if needed.
- Build target verify input:
  - `aligned_start = seq_len // CHUNK_SIZE * CHUNK_SIZE` for attention path;
  - replay tokens from `aligned_start` to current prefix;
  - append chain candidate tokens;
  - run ordinary target `EXTEND`, not tree/custom-mask verify if it causes
    numeric differences.
- Compute accepted visible tokens.
- Commit visible tokens and logprobs in the DVR worker.
- Hold the final bonus token as next round anchor and do not return it early.
- Replace KV mapping for the replay/accepted window:
  - free draft temporary slots;
  - allocate verify slots;
  - update `req_to_token_pool` from `aligned_start`;
  - free old replaced window slots after cloning old locations;
  - keep only committed verify cache locations.

Logprob rule:

- If upstream ordinary `EXTEND` already returns needed input logprobs exactly,
  use them.
- If it returns only final logits, compute token logprobs from captured hidden
  states through the model's logits processor.
- Never return draft decode logits as client output logprobs.

Tests:

- Qwen3-0.6B, bs=1:
  - `max_new_tokens=1,8,16,20,63,64,80,128`
  - non-greedy sampling
  - strict `max_abs_diff == 0.0` against full-prefill scoring oracle
- Qwen3-0.6B, bs=2:
  - mixed prompt lengths
  - mixed output lengths
  - strict KL target remains 0
- Acceptance should be close to 1.0 for self draft.

Commit boundary:

- `dvr: verify attention outputs through extend`

## Phase 4: State Manager And Draft Rollback Hooks

Goal: isolate state lifecycle before adding GDN-specific logic.

Primary files:

- `python/sglang/srt/speculative/dvr_state_manager.py`
- `python/sglang/srt/speculative/dvr_worker.py`

Interface:

```python
class DVRStateHandler:
    def before_draft(self, batch): ...
    def after_draft(self, batch): ...
    def before_target_verify(self, batch, verify_plan): ...
    def after_target_verify(self, batch, verify_result): ...
```

Handlers:

- Attention KV remains mostly managed by `DVRWorker`.
- GDN/Mamba handler is registered only when the model runner has hybrid/GDN
  config.

Initial GDN handler behavior:

- Backup persistent conv state and temporal/SSM state before self draft.
- Restore them after self draft.
- No chunk-boundary commit yet.

Reasoning:

- Self draft decode uses the target model, so recurrent state is mutated before
  target verify starts. Verification must start from the pre-draft state.

Tests:

- Qwen3-0.6B regression from Phase 3.
- Qwen3.5-0.8B short request should run without obvious state corruption, even
  if strict KL is not enabled yet.

Commit boundary:

- `dvr: add state manager and draft rollback hooks`

## Phase 5: GDN Full Correctness Bring-Up

Goal: get a simple, correct GDN DVR path before optimizing memory and replay.

Primary files:

- `python/sglang/srt/speculative/dvr_worker.py`
- `python/sglang/srt/speculative/dvr_state_manager.py`
- possibly `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

Implementation:

- Use a conservative GDN verify path first:
  - restore/zero state to the correct verify start;
  - run prefill-equivalent verify;
  - make returned logits/logprobs match full-prefill oracle.
- This may temporarily use full replay from sequence start if needed to prove
  the worker and token/logprob commit semantics.

Why this phase exists:

- It separates "DVR returned output is correct" from "GDN state update is fast".
- It gives a known-good oracle before adding q/k/v/g/beta buffers.

Tests:

- Qwen3.5-0.8B:
  - bs=1, short and long generations;
  - lengths crossing 64;
  - strict full-prefill logprob equality where upstream deterministic prefill is
    expected to be exact.
- Qwen3-0.6B regression.

Commit boundary:

- `dvr: add conservative gdn correctness path`

## Phase 6: GDN Pending Buffer And Chunk Commit

Goal: remove full replay overhead with the reference branch's memory-saving
state protocol.

Primary files:

- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `python/sglang/srt/speculative/dvr_state_manager.py`
- `python/sglang/srt/speculative/dvr_worker.py`

Data to store:

- For each GDN layer and mamba state slot:
  - pending q
  - pending k
  - pending v
  - pending g
  - pending beta
  - pending position count
  - conv window needed for the same pending boundary

Shape guideline:

- sequence dimension is `CHUNK_SIZE + num_draft_tokens`.
- q/k/v use model dtype where safe.
- g/beta can follow upstream fused-gating output dtype.
- Avoid storing full per-token SSM state because it is much larger.

Verify behavior:

- Target verify starts at the persistent recurrent state corresponding to a
  `CHUNK_SIZE` boundary.
- It appends current verify q/k/v/g/beta into the pending buffer.
- It computes verify logits through the prefill-equivalent chunkwise path.
- It does not permanently advance persistent recurrent state beyond verified
  chunk boundaries.

Commit behavior:

- After acceptance:
  - increment pending position by accepted tokens;
  - clear any rejected/past-tail buffer entries;
  - if pending position >= `CHUNK_SIZE`, recompute exactly the first chunk with
    FLA chunkwise scan and commit the resulting persistent state;
  - shift tail tokens from `[CHUNK_SIZE:CHUNK_SIZE + num_draft_tokens]` to
    `[0:num_draft_tokens]`;
  - decrement pending position by `CHUNK_SIZE`.

State update options:

- Preferred: use a scratch/track mamba state slot for recompute and copy the
  committed result into the request's persistent slot.
- Acceptable: backup persistent state, run FLA chunkwise scan in-place for one
  chunk, and restore if the request did not actually cross the boundary.
- Avoid broad FLA rewrites unless required.

Reference branch details to borrow conceptually:

- q/k/v/g/beta pending buffers.
- `clear_buffers` and `shift_buffers` semantics.
- grouped chunkwise recompute for multiple GDN layers.

Reference branch details to simplify:

- Triton clear/shift kernels can start as simpler torch indexing if they are
  outside the hot path or only used for initial correctness.
- CUDA graph support should wait until correctness is proven.
- Do not copy broad `chunk.py` rewrites unless a minimal `compute_o=False` or
  `inplace_update=False` hook becomes necessary.

Tests:

- Qwen3.5-0.8B, bs=1:
  - random prompt lengths around chunk boundaries;
  - `max_new_tokens=16,64,80,128,160`;
  - strict full-prefill oracle.
- Qwen3.5-0.8B, bs=2+:
  - mixed prompt lengths;
  - mixed output lengths;
  - acceptance mismatch hack to force one request to cross chunk boundary while
    another remains pending.
- Verify that persistent state is updated only for crossed boundaries.

Commit boundary:

- `dvr: commit gdn state at chunk boundaries`

## Phase 7: Batch Robustness

Goal: make DVR reliable when requests in the same batch accept different counts.

Focus areas:

- Per-request pending length.
- Per-request aligned start.
- Per-request KV replacement window.
- Per-request GDN pending buffer shift.
- Request completion before a chunk boundary.
- Prefix/radix cache interaction.

Tests:

- bs=1, bs=2, bs=4.
- mixed `max_new_tokens`.
- one request finishes early.
- different prompt lengths modulo 64.
- radix cache enabled.

Commit boundary:

- `dvr: harden batched request state`

## Phase 8: CUDA Graph And Performance

Goal: optimize only after correctness is stable.

Graph candidates:

- self draft decode graph;
- target verify graph with fixed `CHUNK_SIZE + num_draft_tokens` logical shape;
- GDN post-verify chunk recompute graph with fixed `CHUNK_SIZE`.

Requirements:

- Stable graph buffers for:
  - input IDs;
  - positions;
  - cache locations;
  - req pool indices;
  - seq lens;
  - GDN pending q/k/v/g/beta buffers;
  - accepted step metadata.
- No graph should own private persistent recurrent state. Graph replay must use
  explicit state indices and the state manager must define write ownership.

Tests:

- Repeat Phase 3 and Phase 6 correctness with CUDA graph enabled.
- Compare throughput with graph disabled/enabled.
- Watch for illegal memory access around long generation and batch padding.

Commit boundary:

- `dvr: capture fixed verify graphs`

## Phase 9: Cleanup And Documentation

Goal: make the implementation maintainable.

Actions:

- Remove temporary debug prints.
- Keep short comments at non-obvious state transitions:
  - anchor/bonus holdback;
  - draft rollback;
  - chunk-boundary state commit;
  - pending buffer shift.
- Add a concise runtime warning or error for unsupported modes.
- Keep tests/scripts in a dedicated location if they are useful long term.

Commit boundary:

- `dvr: document deterministic verify flow`

## First Implementation Checklist

Before touching GDN optimization, these must already pass:

- `DECODE_VERIFY_ROLLBACK` can launch.
- self decode draft generates non-greedy draft tokens.
- attention target verify returns strict KL=0.
- bonus anchor is not returned early.
- client output logprobs are verify-derived.
- batch requests do not share or corrupt KV windows.
- disabling DVR leaves upstream behavior unchanged.

## Known Risks

- Ordinary extend may not expose all logprobs needed without capturing hidden
  states.
- Current upstream `MambaPool.SpeculativeState` stores full intermediate SSM,
  while DVR wants q/k/v/g/beta pending buffers.
- FLA chunkwise scan updates state in-place, so post-verify recompute must use a
  scratch state or explicit backup/restore.
- CUDA graphs capture tensor addresses, not logical state ownership. DVR must
  keep state indices and write destinations explicit.
- Radix/prefix cache must stay enabled eventually for RLVR speed; do not solve
  state bugs by permanently disabling radix cache.

