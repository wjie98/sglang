# DVR Production Cleanup Plan

This document is the implementation plan for reducing the current DVR
integration to a production-quality draft -> verify -> rollback pipeline. It
replaces the completed backend-unification plan and describes only work that is
still relevant to the current tree.

## Design Rules

1. Do not add a class, data object, helper, or module unless it owns a distinct
   lifetime or removes a real duplicated algorithm.
2. Preserve SGLang's existing speculative worker, scheduler, CUDA graph, and
   linear-state contracts. DVR-specific behavior stays behind explicit DVR
   branches or DVR-owned objects.
3. Self draft and EAGLE/MTP may differ while proposing tokens and preparing the
   next draft input. Target verify, sampling, rollback, logprob, and result
   publication use one DVR pipeline wherever their data contracts are equal.
4. Keep the model-independent rolling linear-state cache reusable by future
   KDA-like adapters. Do not present GDN-specific q/k/v/g/beta code as a generic
   adapter contract.
5. Fail at initialization for unsupported hybrid linear-state configurations.
   Do not silently continue without deterministic state rollback.
6. Prefill and target verify remain deterministic. Provisional draft decode
   keeps normal performance-first attention and collective settings.

## Batch 1: Correctness And Semantic Boundaries

- Make self-DVR constrained sampling build the same per-node grammar mask as
  the EAGLE verify path.
- Stop using `enable_overlap_schedule=True` as an alias for "allocate two
  request checkpoint slots". Pass the required slot count explicitly inside
  memory-pool construction without adding a server argument.
- Bind and validate the target linear-state adapter after attention backend
  initialization. An unsupported hybrid target must fail before serving.
- Align gated-linear model detection with SGLang's existing model registry,
  including models already recognized by `ModelRunner.hybrid_gdn_config`.
- Audit target-verify plan-stream use with the dedicated DVR graph runner. Keep
  synchronous preparation only when a concrete graph safety constraint exists.
- Add focused coverage for the global fp32 FLA boundary-state behavior.

## Batch 2: Dead State And One-Use Helpers

- Remove unused worker fields copied from EAGLE v2 and the no-op adaptive call.
- Represent self-DVR hidden state as `None`, using EagleDraftInput's supported
  no-hidden-state contract instead of zero-width tensors.
- Merge the two self draft-input builders.
- Inline one-use worker wrappers for EAGLE forwarding, rollback packaging, plan
  stream creation, and self-draft graph initialization when this makes the
  execution order more direct.
- Move rollback action construction onto the linear-state lifecycle and inline
  one-use state copy/index helpers.
- Remove unused optional checkpoint entries and ignored return values.

## Batch 3: GDN State Consolidation

- Keep `dvr_state.py` as the model-independent rolling state window and grouped
  rebuild implementation.
- Consolidate GDN allocation, rebuild kernels, and adapter behavior into one
  clearly named GDN module; future KDA support gets its own adapter.
- Remove forwarding methods that only unpack q/k/v/g/beta into another helper.
- Reuse preallocated verify indices. Retain the request-index offset slot in
  addition to graph padding slot 0; runtime request-pool indices can start at 1.
- Reuse the same conv/QKV split/gating primitives as normal GDN prefill so DVR
  verify does not maintain a slower copy of the model forward path.
- Determine backend boundary-state export capability once instead of mutating
  it as an incidental side effect of every layer forward.
- Construct the oracle adapter only on the target backend; draft-model GDN
  layers must not allocate or write target state-input windows.

## Batch 4: CUDA Graph And Server Arguments

- Store the standard `DecodeCudaGraphRunner` directly for self draft; remove
  the three-method DVR wrapper.
- Validate the short-prompt graph boundary once and remove duplicate block
  reason/min-length helpers.
- Limit composite attention-backend traversal to actual SGLang wrapper shapes.
- Keep the dedicated target-verify runner and the phase-specific deterministic
  context; these encode real execution differences.
- Remove repeated DVR string predicates and duplicated page-size validation.
- Simplify CUDA graph coverage using real ServerArgs invariants rather than
  defensive branches added for unit-test fake objects.
- Do not force an arbitrary `max_running_requests` default as a DVR feature.

## Required Invariants

- Non-DVR output, logprob behavior, memory-pool sizing, and throughput do not
  change.
- Self-DVR spec v1/v2 and DVR-EAGLE sync/overlap keep exact target logits across
  chunk boundaries with `return_logprob=True` and `False`.
- Radix cache may be enabled or disabled without changing DVR correctness.
- Self draft always uses its dedicated CUDA graph for supported serving inputs;
  unsupported graph shapes fail explicitly instead of silently using eager.
- Triton and FA3 full-attention backends remain selectable. Unsupported draft
  decode backends fail during initialization instead of retaining deterministic
  settings silently.
- The H20/NVLink target remains effective throughput close to
  `acceptance_rate * non_DVR_baseline`.

## Validation After Every Batch

```bash
test/manual/dvr/scripts/run_static_unit_checks.sh
test/manual/dvr/scripts/run_0p8b_self_dvr_kl.sh
```

When EAGLE/MTP or shared verify logic changes:

```bash
test/manual/dvr/scripts/run_35b_mtp_eagle_smoke.sh
```

Final validation also runs the fixed 80B ShareGPT/LongBench throughput script
with `return_logprob=True` and `False`, followed by a patch-view refresh against
the current upstream baseline.

## Size Target

The starting production Python delta is 3,692 added lines. The low-risk cleanup
should remove roughly 450-700 lines. Reaching a stable sub-3K implementation
depends on eliminating the duplicated GDN verify forward path, not on replacing
clear code with denser abstractions.
