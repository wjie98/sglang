# DVR isolated-state lifecycle redesign

Status: implemented on the v6 development branch; H20 qualification is pending.

This document records the implemented state ownership and execution boundaries
for DVR GDN. Local state/kernel tests cover the transition rules below; the H20
matrix remains the release gate for long-sequence KL and throughput.

## 1. Goals and decisions

The redesign must:

- make an exact 64-token target checkpoint the only recurrent state published
  to Radix;
- keep provisional self-draft writes outside target and Radix state;
- retain the existing GDN recurrent decode kernel;
- keep target verify graph-static and deterministic;
- share target verify, acceptance, target commit, and output handling between
  self-draft and EAGLE/MTP;
- preserve strict decode-to-prefill KL equality;
- keep exactly one full recurrent workspace per layer and request, independent
  of proposal width, by reusing the one-step buffer already reserved for DVR;
- add no host readback or global synchronization to a steady-state block; and
- leave non-DVR, no-GDN, Triton full-attention, and FA3 full-attention behavior
  unchanged.

The following alternatives are deliberately rejected:

- changing the self-draft recurrent kernel to support separate input and output
  state pointers;
- letting target verify overwrite an exact checkpoint and restoring it later;
- storing one full recurrent state for every proposal token;
- using the final state after all padded `64 + D` rows as an accepted endpoint;
- disabling Radix to avoid checkpoint ownership bugs; and
- capturing all self-draft steps as one large CUDA graph before state
  maintenance and target verify are no longer the measured bottlenecks.

## 2. State model

Use these names only inside DVR. Do not redefine the meaning of upstream
`live state` or expose these concepts through generic scheduler interfaces.

| Name | Meaning | Physical storage | Owner | Update cadence | Radix-visible |
| --- | --- | --- | --- | --- | --- |
| `B_active` | Latest exact target state at a 64-token boundary | Active Mamba ping-pong slot | Target/Radix | On an accepted boundary crossing | Yes |
| `B_staging` | Alternate exact boundary slot | Other ping-pong slot | Target/Radix | On an accepted boundary crossing | Only after publication |
| `draft_state` | Rebuildable seed for the next self-draft | `workspace` plus `draft_conv` | DVR self-draft | Seeded after target EXTEND; rebuilt after each accepted block | No |
| `workspace` | One reusable full recurrent state per layer/request | `intermediate_ssm[..., 0]` | DVR state lifecycle | Draft endpoint, then `h64` staging, then rebuilt endpoint | No |
| `draft_conv` | Private self-draft convolution state | Request-indexed DVR conv allocation | DVR self-draft | Seeded after EXTEND; advanced by draft; finalized from accepted verify state | No |
| `target_conv` | Accepted convolution endpoint used by target verify | Existing target live convolution slot | Target | Updated from accepted verify state | No |
| `W` | Exact transition inputs after `B_active` | Request-indexed `q/k/v/g/beta` window of length `64 + D` | Target verify | Append accepted rows; compact on a crossing | No |
| `V` | Non-recurrent verify outputs | Target logits, candidate conv windows, and EAGLE hidden states | Target verify | One proposal block | No |

`draft_state` is a derived cache, not a second authoritative target state.
Deterministic target verify always starts from `B_active` and the exact accepted
transition history in `W`. After acceptance, self-DVR reconstructs the temporal
part of `draft_state` from that same pair. Missing or stale `draft_state` is recoverable;
missing `B_active` or accepted rows in `W` is a correctness error.

For EAGLE/MTP, target temporal `draft_state` is unused and is not rebuilt.
`target_conv` still advances to the accepted endpoint because the next target
verify needs the correct finite convolution history. The EAGLE worker
owns its draft KV, hidden state, and any draft-model recurrent state through
existing upstream mechanisms.

The physical field keeps its upstream name `intermediate_ssm`; renaming that
shared memory-pool API would create unnecessary integration churn. DVR code
must immediately take the one-step view and call it `recurrent_workspace`.
`speculative_ssm_state_steps` remains exactly one for DVR. One state means one
state per layer and concurrently addressable request, not one global state
shared by a batch.

## 3. Required invariants

For every active GDN request, let `boundary_pos` be the position represented by
`B_active` and let `tail_len` be the accepted rows currently stored at the front
of `W`.

1. `boundary_pos % 64 == 0`.
2. `boundary_pos + tail_len` is the current accepted target-state position.
3. `W[0:tail_len]` and `B_active` come from the same accepted target history.
4. `0 <= tail_len < 64` and `D <= 64`, so one proposal crosses at most one
   boundary.
5. Self-draft may write only `workspace`, `draft_conv`, speculative token KV,
   and draft-backend-owned state. It may not write `target_conv`, `B_active`,
   `B_staging`, or accepted rows in `W`.
6. Target verify reads `B_active`, `target_conv`, and `W`. It overwrites the
   recurrent `workspace` with `h64` only after self-draft has finished.
7. Finalization is the only phase allowed to rebuild `draft_state`, commit
   accepted rows in `W`, change boundary ownership, or finalize EAGLE state.
8. A new boundary becomes publishable only after both temporal and convolution
   checkpoint writes are enqueued before the ownership update on the forward
   stream.
9. Scratch state and candidate rows are never donated, matched, or released as
   Radix state.
10. `return_logprob=True` and `False` execute the same draft, verify, and state
    transaction. Output materialization cannot trigger another target replay.

## 4. Closed execution transaction

### 4.1 Acquire or rebuild target state

Target EXTEND resolves the newest exact checkpoint no later than the matched
prefix. A warm Radix hit restores that boundary through the existing deferred
Mamba COW path. EXTEND computes the unmatched suffix once, stores transition
inputs after the last 64-token boundary in `W`, and seeds request-local
`draft_state` from the resulting accepted endpoint.

If no checkpoint exists, use the zero boundary and ordinary EXTEND. If a cached
physical slot was donated, resolve the new physical mapping before constructing
the DVR state context. If the requested suffix is newer than the available
checkpoint, recompute it instead of manufacturing a state backup or disabling
Radix.

Radix-disabled execution uses the same transaction with a request-local
boundary slot. It omits only cross-request publication and reuse.

### 4.2 Prepare self-draft

Target EXTEND initializes private state once when it establishes a new accepted
endpoint:

```text
copy(target endpoint temporal -> recurrent_workspace)
copy(target endpoint conv     -> draft_conv)
```

This initialization is outside steady-state DVR blocks. It deliberately reuses
the existing PyTorch pool copy shape until profiling shows that cold/warm
EXTEND initialization is material; no dedicated full-state copy kernel is
introduced without measured benefit.

During self-draft, the existing GDN decode path selects `recurrent_workspace`,
`draft_conv`, and request-pool-indexed scratch rows. All proposal steps then run
in place on those tensors; the recurrent decode kernel itself is unchanged. The
dedicated DVR draft CUDA graph remains a normal single-step decode graph
replayed `D - 1` times.

After each accepted block, finalization reconstructs the workspace from
`B_active + W`, so the next proposal starts directly from the accepted endpoint
without another full-state copy. The existing one-step `intermediate_ssm`
allocation is sufficient and does not grow with `D`. Its draft lifetime ends
before target verify, after which the same address stages verify's exported
`h64`. The former conv backup allocation is private draft state, not
backup/restore storage.

### 4.3 Prepare EAGLE/MTP draft

EAGLE/MTP does not initialize the recurrent workspace for draft and does not
route target GDN decode through it. It calls the existing upstream EAGLE draft
worker, which owns:

- draft-model KV and recurrent state;
- the hidden state consumed by the next draft round;
- draft-extend CUDA graphs; and
- accepted/rejected draft-cache finalization.

The DVR wrapper supplies the same target bonus token and consumes the same
proposal token/probability contract as self-draft. It must not duplicate EAGLE
cache allocation or replay target history through the draft model.

### 4.4 Deterministic target verify

The target model computes projections, dense layers, MoE, and full attention
for only the `D` proposal rows. Full attention continues to use Triton or FA3.

GDN verify is different: strict decode-to-prefill equality requires the same
64-aligned chunk partition as target prefill. It therefore reads:

```text
B_active + W[0:tail_len] + D candidate transition rows
```

The physical GDN input remains a fixed `64 + D` window. With `D=16`, `T=80`
belongs to this verify scan and normally covers two FLA chunks. It is not the
length of the post-verify state update. Rows after `tail_len + D` may remain
stale because they are causally after every returned logit. The final state
after all 80 physical rows is never authoritative.

Target verify treats `B_active` as read-only and does not touch self-draft
state. The implementation restores a
default-preserving `inplace_update=False` path in the Triton GDN prefill
dispatcher rather than copying `B_active` into a target live slot first. Verify
produces:

- target logits for the `D` logical rows;
- candidate convolution windows;
- hidden states required by EAGLE/MTP; and
- `h64`, the exact recurrent state after the first 64 rows, written into the
  same one-state recurrent workspace that self-draft has finished using.

The fixed-window data plumbing is DVR-private. At the start of target verify,
the existing GDN metadata hook resolves request slots, boundary slots, tail
lengths, padding, and possible boundary crossings once for the whole block.
Every GDN layer reuses those graph-stable tensors. Three small Triton launches
pack `q+k`, `v`, and `g+beta`: each launch both inserts the current candidate
rows into the persistent transition window and materializes the contiguous
`64 + D` input expected by the unchanged FLA dispatcher. A masked scatter
stages `h64` only for lanes that could cross the boundary, and a direct gather
writes only the `D` logical outputs in their final layout.

This deliberately does not change the generic FLA producer, its public state
index contract, or ordinary GDN execution. It also does not claim that the
fixed `64 + D` materialization is the final performance design. A segmented
verify path or direct producer write is allowed only after the private packing
path is measured on H20 and the replacement proves bitwise equivalence for all
physical chunk positions.

`h64` is publishable only when accepted rows actually cross the boundary. It is
valid in that case because every row before the boundary is accepted target
history. A proposed crossing that is later rejected must not alter checkpoint
ownership.

### 4.5 Accept and finalize

Sampling produces one GPU `accept_lens` tensor. Self-draft, EAGLE, target KV,
target linear state, output tokens, and logprobs must all consume that same
result. No phase may independently recompute acceptance.

For both draft backends:

- select the accepted target convolution endpoint from verify's intermediate
  conv windows and write it to `target_conv` and, for self-draft, `draft_conv`;
- append only accepted candidate transition rows to `W`;
- discard rejected candidate rows logically; and
- let the existing speculative KV rollback retain only accepted target KV.

If accepted history crosses 64, first copy workspace `h64` and the matching
convolution checkpoint into `B_staging`, then make that slot `B_active`. Compact
the accepted remainder in `W` after the boundary decision.

Self-DVR then reconstructs the recurrent workspace from the newest `B_active` and
the accepted `W[0:tail_len]`. This is the only temporal rollback rule, for both
crossing and non-crossing blocks. It processes at most 63 rows; after a crossing
it normally processes at most `D - 1`. The reconstruction is a DVR state-only
kernel with a 64-row compile bound. The fixed `T=64+D` belongs only to target
verify and must not leak into this kernel. The self-draft recurrent decode
kernel remains unchanged.

EAGLE skips target temporal workspace reconstruction. It finalizes its draft
cache and next hidden state through the existing upstream EAGLE worker using the
same accepted indices. Target boundary and `W` finalization remain identical to
self-draft.

On a crossing, compact only the accepted remainder:

```text
W[0:remainder] <- W[64:64+remainder]
tail_len <- remainder
swap(B_active, B_staging)
```

Use a masked direct-copy kernel that exits before loading data for non-crossing
requests. Start with one small parameterized kernel called for each state-input
tensor. Fuse all five tensors only if profiling shows launch overhead remains
material after eliminating the current gather/`torch.where`/scatter traffic.

### 4.6 Publish, release, and re-hit

Radix observes only the newest complete `B_active` not later than the visible
committed prefix. It truncates reusable token KV to the same aligned length.
The next request restores that boundary and recomputes the unclosed suffix via
ordinary target EXTEND, which also reconstructs `draft_state` and `W`.

`draft_state`, `W`, candidate verify state, and EAGLE draft cache
are request-local or worker-local and are never published. This keeps physical
checkpoint donation independent from provisional draft execution.

## 5. Worker structure

Keep one DVR target transaction around two draft backends:

```text
state_ctx = target_state.begin(batch)
proposal  = draft_backend.propose(batch, state_ctx)
verified  = target.verify(batch, proposal, state_ctx)
accepted  = target.sample(batch, proposal, verified)
target_state.finalize(state_ctx, verified, accepted)
draft_backend.finalize(batch, proposal, verified, accepted)
return materialize_output(batch, verified, accepted)
```

Self-draft differs only in `propose`: it runs target decode on the private
workspace left by the preceding EXTEND/finalize transaction. Target-state
finalization reconstructs that workspace from `B_active + W`; the self backend
has no separate model cache.
EAGLE differs only in `propose` and `draft_backend.finalize`: both delegate to
the existing EAGLE worker. Target verify, GDN boundary handling, acceptance,
target KV rollback, logprob handling, and output materialization remain common.

Spec v1 consumes the finalized result synchronously. Spec v2 may defer result
materialization, but it stores one transaction result containing the accepted
indices and verify outputs; it must not defer or repeat target-state ownership
decisions through a second code path.

## 6. Implementation batches

### Batch 1: lock the state contract

- Add state-level tests for the invariants above using the current
  implementation as a token/logit reference.
- Add profiler spans for `draft_state_copy`, `target_verify`,
  `draft_state_rebuild`, `state_window_compact`, and `boundary_publish`.
- Record accepted-length and crossing histograms on GPU without host readback in
  the serving path.

Exit gate: no runtime semantic change; static checks and the existing fixed
correctness matrix pass.

### Batch 2: isolate self-draft state

- Assert DVR allocates exactly one intermediate SSM step and expose
  `intermediate_ssm[..., 0]` locally as `recurrent_workspace`.
- Repurpose the existing request-indexed conv backup as `draft_conv`.
- Seed workspace and draft conv after target EXTEND; retain them across DVR
  blocks by rebuilding from accepted target history during finalize.
- Route only self-draft GDN decode to scratch inside the existing DVR draft
  context and graph runner.
- Keep the recurrent decode kernel and ordinary GDN decode behavior unchanged.

Primary files: `dvr_state_flow.py`, `dvr_gdn.py`, `gdn_backend.py`, and
`dvr_cuda_graph_runner.py`.

Exit gate: workspace and the target EXTEND endpoint match after initialization;
target recurrent/checkpoint state remains unchanged after all draft steps;
allocated recurrent workspace bytes are independent of `D`; self-DVR
acceptance and throughput do not regress beyond run-to-run tolerance.

### Batch 3: make target verify read-only

- Start deterministic GDN verify directly from `B_active`.
- Keep convolution verify based on unchanged accepted `target_conv`.
- Use a default-preserving non-in-place Triton FLA verify call.
- Remove temporal boundary-to-live restore and conv restore.
- Continue exporting `h64` into the recurrent workspace after draft has
  finished.

Exit gate: target logits, `h64`, candidate conv windows, KL, and accepted tokens
match the reference; the `dvr_state_restore` profiler span disappears.

### Batch 4: make live reconstruction canonical

- Commit a crossing `h64` from the one-state workspace before reusing it.
- Compact `W` when crossing, then reconstruct the self-draft workspace only from
  the newest `B_active + W[0:tail_len]`.
- Bound the state-only reconstruction by the 64-token chunk, not `64 + D`.
- Skip temporal self-draft workspace work for EAGLE/MTP.
- Preserve the accepted conv endpoint update for both draft backends.

Exit gate: force accept lengths `1`, `D-1`, and `D`; compare target checkpoint,
draft live state, conv state, and next-round proposal against the reference.

### Batch 5: compact the transition window directly

- Replace advanced-index gather/`torch.where`/put with a masked direct copy.
- Copy only crossing lanes and only the accepted remainder.
- Keep stale rows outside `tail_len` unspecified and unread until overwritten.

Exit gate: mixed crossing/non-crossing batches preserve `W`, `tail_len`, logits,
and KL exactly; `state_window_compact` falls materially below the old shift.

### Batch 6: unify backend finalization and remove old paths

- Keep the existing upstream EAGLE worker as the EAGLE draft backend.
- Make self and EAGLE consume one target verify/accept/finalize transaction.
- Remove draft backup/restore naming and the boundary-to-live restore path.
- Remove the fixed-80 rebuild contract and redundant ownership fields; retain
  one explicit boundary-plus-tail draft-state reconstruction.
- Keep one DVR state context; do not add generic scheduler or `ForwardBatch`
  fields for scratch state.

Exit gate: self v1/v2 and EAGLE sync/overlap pass the complete matrix, and
non-DVR/no-GDN diffs contain no behavior changes.

## 7. Risk checks

- Verify the token-position convention explicitly: `accept_lens` includes the
  bonus token, while the recurrent state represents consumed input tokens.
  State tests must catch an off-by-one at the next draft block.
- Assert that draft and verify use the one-state recurrent workspace
  sequentially on the same forward stream. No overlap may overwrite one
  request's draft state with `h64` before draft ends.
- Verify that CUDA graph padding maps padded requests to a safe scratch slot and
  never copies or commits it.
- The EXTEND-only full-state initialization still uses real bandwidth. Profile
  `draft_state_copy`, but do not add a dedicated copy kernel unless it is
  material outside startup/prefill.
- Do not publish `h64` based only on a proposed crossing. Publication is gated
  by accepted history.
- EAGLE acceptance must be measured on real prompts. An aggregate acceptance of
  exactly 1.0 is a diagnostic failure until proposal and acceptance accounting
  are independently confirmed.
- Keep target full-attention backend coverage for Triton and FA3. GDN target
  verify continues to require the Triton linear-attention prefill backend.

## 8. Expected performance effect

The H20 BS3 profile measured about 2.351 ms per DVR16 block in maintenance:
0.843 ms restore and 1.391 ms rollback dominated by recurrent rebuild and
window movement. The redesign replaces those operations with:

- no steady-state full recurrent restore/copy before verify;
- one canonical private draft-state reconstruction of at most 63 rows;
- one boundary scatter only on a crossing; and
- a compact of at most 15 transition rows only on crossing lanes.

The EXTEND initialization copy remains, but it is not part of each decode block.
The rebuild source and ownership are unambiguous and its compile bound falls
from 80 to 64. The first H20 target is to remove the 0.843 ms restore and
materially reduce the 2.351 ms maintenance total; do not set a sub-millisecond
release gate until the 64-row rebuild and compact kernels are measured. This
optimization cannot remove deterministic target-verify compute; report
`verify_state_pack`, `verify_boundary_stage`, `verify_output_gather`,
`r_verify`, rebuild, compact, boundary publication, and total `r_maint`
separately. Do not modify shared FLA kernels or add graph-time host decisions
until this decomposition identifies a remaining material cost.

## 9. Completion criteria

The implementation is release-qualified only when:

- state-level boundary, draft-state, workspace, window, and conv checks pass;
- self-DVR v1/v2 and EAGLE sync/overlap preserve strict long-sequence KL=0;
- `return_logprob=True` and `False` share identical state transitions;
- prompt/output boundaries around `1/63/64/65`, mixed batches, Radix donation,
  generated-prefix re-hit, and Radix-disabled execution pass;
- 0.8B, 35B MTP, and 80B real-data acceptance remain within the established
  ranges;
- H20 profiles show no target-state restore, no draft reconstruction longer than
  63 rows, exactly one full recurrent workspace per request, no new D2H
  synchronization, and no draft-kernel regression; and
- the matched Triton/FA3 throughput report includes normal non-deterministic,
  ordinary deterministic, self-DVR, and EAGLE results where supported.

The public README describes this implementation. H20 results must still be
attached before treating the branch as release-qualified.
