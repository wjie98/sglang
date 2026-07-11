# DVR Unified Worker Refactor Plan

This is the working plan for replacing the current split self-DVR and
DVR-EAGLE workers with one DVR orchestration flow and two draft backends.

## Goal

DVR should be implemented as one target-verify/rollback pipeline:

1. target prefill or decode entry
2. target GDN/radix boundary preparation
3. draft backend proposal
4. target verify
5. optional suffix replay oracle
6. backend-specific accept/reject sampling
7. target GDN state/output/logprob commit
8. backend-specific next-draft state update
9. `GenerationBatchResult` publication

Self draft and EAGLE/MTP should be sibling draft backends under this common
pipeline.  EAGLE/MTP is the superset: it owns an independent draft KV cache,
needs target hidden states for the next draft step, and runs draft-extend after
prefill/verify.  Self draft disables those features and keeps only target-model
decode as the draft proposal mechanism.

## Target Structure

```text
python/sglang/srt/speculative/dvr_worker.py
  DecodeVerifyRollbackWorkerV2
    Owns the DVR target pipeline, GDN state lifecycle, output/logprob repair,
    spec-v2 deferred actions, and final GenerationBatchResult assembly.

  _DVRSelfDraftBackend
    Uses the target model as the draft model.  No independent draft KV, no
    target hidden states, no draft-extend.  Must keep the dedicated self-draft
    CUDA graph and non-deterministic decode performance knobs.

  _DVREagleDraftBackend
    Wraps upstream EAGLE/MTP draft-side logic.  Owns the independent draft KV
    path, draft prefill/extend, and next-draft hidden-state handoff.  It must not
    own DVR target verify or GDN state commit.
```

`DECODE_VERIFY_ROLLBACK` and `DECODE_VERIFY_ROLLBACK_EAGLE` should both dispatch
to `DecodeVerifyRollbackWorkerV2`; the worker chooses the draft backend from the
speculative algorithm.

## Draft Feature Matrix

| Feature | self draft | EAGLE/MTP |
| --- | --- | --- |
| independent draft KV cache | no | yes |
| target hidden states needed | no | yes |
| draft-extend after prefill/verify | no | yes |
| sampler | chain speculative sampler | upstream EAGLE sampler |
| partial suffix replay after rejection | no | yes |
| fast commit from verify state | yes | no |

These feature switches are internal worker state.  They must not become new
server arguments and must not leak into scheduler code.

## Migration Steps

1. Self-draft backendization only.
   Move self proposal, self sampling, compact logprob, and next-draft-input
   construction behind `_DVRSelfDraftBackend`.  Keep the existing self-DVR
   target flow and validate that KL and throughput do not change.

2. Common DVR pipeline.
   Make `DecodeVerifyRollbackWorkerV2` call backend methods for proposal,
   sampling, and next-draft state.  Keep behavior equivalent for self draft.

3. EAGLE backend introduction.
   Add `_DVREagleDraftBackend` by wrapping the upstream EAGLE draft-side worker.
   The old `DecodeVerifyRollbackEagleWorkerV2` reference path has been removed;
   no target-verify logic should be reintroduced outside the unified worker.

4. EAGLE dispatch through the unified worker.
   Route `DECODE_VERIFY_ROLLBACK_EAGLE` to `DecodeVerifyRollbackWorkerV2` and
   run EAGLE/MTP through the same DVR target pipeline.

5. Remove the old DVR-EAGLE worker main flow.
   The temporary `dvr_eagle_worker.py` compatibility alias has been removed;
   both DVR algorithms now dispatch directly to the unified worker.

6. Shrink DVR core parameters.
   Once self and EAGLE share the same caller, replace broad
   `finish_dvr_verify()` parameter lists with a smaller local context/result
   object or direct local variables if that is clearer.

7. Shrink state-flow entry points.
   After the worker is unified, remove replay/boundary branches that existed
   only to support two separate callers.

## Required Validation

After step 1:

- `git diff --check`
- Python `py_compile`
- DVR unit tests
- 0.8B self-DVR v1/v2 KL and boundary smoke
- fixed 80B self-DVR throughput guard, or at least the fixed ShareGPT/LongBench
  subset if a quick iteration is required

After touching EAGLE/MTP:

- all step-1 checks
- 35B MTP/EAGLE smoke for sync/overlap and return_logprob true/false

Final validation:

- 0.8B self-DVR v1/v2 KL and boundary smoke
- 35B MTP/EAGLE smoke
- 80B self-DVR ShareGPT/LongBench long-output throughput with return_logprob
  true/false
- patch-view refresh against the current upstream baseline
