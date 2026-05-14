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
