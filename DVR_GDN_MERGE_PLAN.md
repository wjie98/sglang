# DVR GDN Merge Plan

This file tracks the minimal merge path for adding GDN support to DVR on top of
the clean v3 branch.

## Goal

DVR target verify should compute verification logits with a prefill-like
chunkwise FLA path, while still maintaining enough recurrent state to continue
self-draft decode after accepted tokens.

## State Model

DVR GDN state must be split into three roles:

- Verify logits state: computed by the chunkwise scan path so logits match the
  deterministic prefill/extend semantics.
- Live tail state: the current decode-continuation state after the last accepted
  token. If the accepted token is not at a chunk boundary, this can be rebuilt
  by recurrent replay from saved q/k/v/g/beta.
- Chunk boundary checkpoint: only states at `FLA_CHUNK_SIZE` boundaries should be
  committed as deterministic prefix/checkpoint state. These states must come
  from chunkwise scan, not recurrent replay.

## Risks To Fix

1. Target verify kernel path
   - Current upstream target verify for GDN uses recurrent-style verify kernels.
   - DVR needs a chunkwise-scan path for logits, with the start position aligned
     to `FLA_CHUNK_SIZE`.

2. Missing token-wise state from chunkwise scan
   - Chunkwise scan does not expose every token's recurrent state.
   - Save q/k/v/g/beta during target verify so a later recurrent replay can
     reconstruct the state at the exact accepted tail position.

3. Conv state consistency
   - GDN has both temporal/SSM state and conv state.
   - Draft decode mutates live conv/temporal states. Target verify must start
     from the correct boundary state and later commit live tail/boundary states
     consistently.

4. Batch divergence
   - Different requests can accept different token counts.
   - Boundary progress, tail q/k/v/g/beta length, and state commits must be
     tracked per request, not per batch.

5. CUDA graph capture
   - Target verify cuda graphs are captured before the DVR worker is created.
   - Any save/replay buffers that are written inside the graph must exist before
     capture and have stable addresses. Store the low-level q/k/v/g/beta buffers
     in the speculative Mamba/GDN cache, and let DVR worker logic treat those
     buffers as its state backend.

## Phase 1: Save q/k/v/g/beta Inputs

Implement a fixed-address cache for target-verify GDN inputs:

- Add q/k/v/g/beta buffers to `MambaPool.SpeculativeState`.
- Use shape `[num_layers, spec_state_size + 1, speculative_num_draft_tokens, ...]`
  for this first step.
- In `GDNAttnBackend.forward_extend()` target verify, compute `g,beta` from
  `a,b` and save q/k/v/g/beta into the cache using the same per-batch
  `intermediate_state_indices` layout as existing intermediate state caches.
- Validate both non-cuda-graph and cuda-graph launches with Qwen3.5.

Status:

- Implemented a graph-stable q/k/v/g/beta cache behind the
  `DECODE_VERIFY_ROLLBACK` speculative algorithm only. The cache is allocated
  inside the Mamba/GDN speculative state rather than as Python-only DVR worker
  fields because cuda graph capture needs fixed tensor addresses before the DVR
  worker handles runtime batches.
- The backend writes the current target-verify q/k/v/g/beta tensors into that
  cache keyed by `intermediate_state_indices`, matching the existing
  intermediate state layout.
- Static checks passed:
  `py_compile` for the touched files and `git diff --check`.
- Runtime smoke tests passed with `/home/hwj/Qwen3.5-0.8B` in both
  non-cuda-graph and cuda-graph modes. These tests only validate that saving
  q/k/v/g/beta is compatible with forward and graph replay; GDN strict KL=0 is
  intentionally a later phase because target verify still uses the existing
  recurrent-style GDN verify kernel.

## Later Phases

1. Extend the q/k/v/g/beta cache to hold `FLA_CHUNK_SIZE + draft_tokens` tail
   windows per request.
2. Add DVR GDN state handler to restore boundary state before self-draft verify.
3. Replace GDN target verify logits path with chunkwise scan.
4. Add recurrent replay helper for live tail state at accepted token positions.
5. Add post-verify chunk commit: when accepted tail reaches `FLA_CHUNK_SIZE`,
   run chunkwise scan for that chunk, commit boundary state, and slide the
   q/k/v/g/beta window.
6. Add batch tests with different accepted lengths.

## Phase 2: State Backup And Post-Verify Helpers

Scope:

- Add a small Mamba state backup/restore API for conv and temporal states. DVR
  will use this before draft decode and target verify so live recursive decode
  state can be rolled back to the aligned boundary state.
- Add two GDN post-verify helper paths that consume saved q/k/v/g/beta:
  - recurrent helper: compute the state at an accepted tail token, including
    per-request accepted lengths;
  - chunkwise helper: run the prefill-like chunkwise path and return the state
    at the last `FLA_CHUNK_SIZE` boundary.

Status:

- Implemented backup/restore on `MambaPool` plus request-index wrappers on
  `HybridReqToTokenPool`.
- Implemented GDN helper methods in `GDNKernelDispatcher`. They are not yet
  wired into DVR scheduling/state commit, so current generation behavior is
  unchanged.
- Static checks passed, and a direct CUDA tensor smoke test verified both helper
  output shapes and chunkwise state-pool update semantics.

## Phase 3: Boundary State Wiring

Scope:

- Keep one DVR-owned GDN boundary state per request in the existing Mamba extra
  buffer.
- Before self-draft decode, make sure the boundary state exists. Draft decode is
  allowed to advance the live temporal/conv state to the speculative tail.
- Before target verify, restore the live temporal/conv state from the DVR
  boundary state so GDN verify starts from the intended chunk window base.
- During GDN target verify, save q/k/v/g/beta into a per-request rolling window
  with length `FLA_CHUNK_SIZE + speculative_num_draft_tokens`.
- For DVR GDN target verify, run the chunkwise scan path over the fixed rolling
  window and gather the active draft-token logits.
- After acceptance, rebuild the live tail temporal state by recurrent replay
  from saved q/k/v/g/beta. If a request crosses `FLA_CHUNK_SIZE`, compute the new
  boundary temporal state with chunkwise scan, update boundary conv state, slide
  q/k/v/g/beta left by one chunk, and let DVR own the Mamba checkpoint metadata.

Status:

- Implemented the first working version of this wiring in `dvr_worker.py`,
  `gdn_backend.py`, `memory_pool.py`, and the scheduler Mamba checkpoint update.
- Static checks passed: `py_compile` for touched files and `git diff --check`.
- Non-cuda-graph Qwen3.5 smoke tests run without crashes:
  - `max_new_tokens=16`: generated 16, acceptance 1.0, strict oracle
    `max_abs_diff=0.023709863424301147`.
  - `max_new_tokens=32`: generated 19 due EOS, acceptance 0.3333333333333333,
    strict oracle `max_abs_diff=15.424726724624634`.
  - `max_new_tokens=80`: generated 80, acceptance 0.18412698412698414,
    strict oracle `max_abs_diff=9.052518844604492`.
- A 64-token aligned prompt still showed strict-oracle drift:
  `max_abs_diff=0.27963473147246987`, acceptance 1.0. This means Phase 3 is a
  state-management checkpoint, not yet a correct GDN KL=0 implementation.

Known issues:

- The initial DVR boundary currently falls back to the prompt-end state when no
  chunk-aligned prefill checkpoint/qkvg window is available. For prompts whose
  length is not `FLA_CHUNK_SIZE` aligned, strict DVR semantics require treating
  the prompt tail as part of the first verify window or caching its q/k/v/g/beta
  during prefill.
- Even with a 64-token aligned prompt, GDN logits still differ from the full
  prefill oracle, so the next debugging step is to compare the exact GDN
  q/k/v/g/beta, conv state, temporal state, and gather offsets between full
  prefill and DVR target verify.
- The current q/k/v/g/beta cache is indexed by live Mamba pool slot for
  graph-stable backend access. This is simple but memory-heavy; Qwen3.5-0.8B
  logs roughly 4.7GB for this cache at `mem_fraction_static=0.45`.
