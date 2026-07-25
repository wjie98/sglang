# DVR recurrent-state lifecycle

Status: implemented on the v6 development branch. The local unit and extracted
model tests cover the contract below; H20 long-sequence and throughput tests
remain the release gate.

## 1. Scope

DVR adds one deterministic target transaction around either self-decode or an
EAGLE/MTP draft backend:

```text
prepare state -> draft -> deterministic target verify -> sample -> commit
```

Full-attention KV stays in SGLang's existing request and Radix pools. DVR owns
only the additional state needed to reproduce GDN prefill arithmetic while
keeping provisional draft execution outside authoritative target state.

The design must:

- keep target prefill and verify deterministic;
- let provisional draft decode use normal performance-first kernels;
- preserve exact decode-to-prefill logits and returned logprobs;
- support Radix enabled and disabled without changing inference semantics;
- add no steady-state host readback or scheduler synchronization; and
- leave non-DVR and no-GDN execution unchanged.

## 2. State and ownership

For an active request, the target Mamba slot intentionally has two timestamps:

- its temporal recurrent state is the latest exact 64-token boundary;
- its convolution state is the current accepted endpoint.

These timestamps differ because verify replays the accepted post-boundary
transition tail from cached post-convolution inputs. It does not convolve that
tail again.

| Name | Storage | Meaning | Owner |
| --- | --- | --- | --- |
| `live_boundary` | target Mamba temporal slot | latest exact 64-token target boundary | target request |
| `accepted_conv` | target Mamba convolution slot | convolution state at the accepted endpoint | target request |
| `transition_window` | request-indexed `k/v/g/beta` rows | accepted inputs after `live_boundary`, plus temporary candidate rows | DVR target |
| `draft_state` | one request-indexed recurrent workspace | private self-draft endpoint; reused as verify boundary staging | DVR self-draft |
| `draft_conv` | request-indexed convolution state | private self-draft convolution endpoint | DVR self-draft |
| `verify_conv` | request-indexed per-candidate windows | candidate convolution endpoints used by acceptance | DVR verify |
| `boundary_seq_lens` | one device value per request row | logical position represented by `live_boundary` | DVR lifecycle |
| `published_boundary_lens` | one device value per request row and Radix lane | exact boundary represented by each ordinary tracking slot | DVR lifecycle |
| `DVRStateCommitPlan` | one block-local object | graph-stable rows, slots, lane, and pre-verify tail length | DVR lifecycle |

`live_boundary` and accepted rows in `transition_window` are authoritative.
`draft_state` is derived and can always be rebuilt from them. EAGLE/MTP does
not use `draft_state`; its KV and hidden-state lifecycle remains owned by the
upstream EAGLE worker.

Radix ping-pong slots are publication buffers, not verify inputs. They retain
their upstream allocation and ownership rules. DVR records only which exact
boundary each lane contains so request release can publish a compatible
aligned prefix.

## 3. Invariants

Let `boundary_len` be the position represented by `live_boundary`, and
`tail_len` be the accepted rows at the front of `transition_window`.

1. `boundary_len % 64 == 0`.
2. `boundary_len + tail_len` equals the accepted target-state position.
3. `0 <= tail_len < 64`.
4. `live_boundary` and `transition_window[:tail_len]` come from the same
   accepted target history.
5. A proposal contains at most 64 tokens, so one verify crosses at most one
   recurrent boundary.
6. Self-draft writes only `draft_state`, `draft_conv`, speculative token KV,
   and draft-backend-owned buffers.
7. Target verify treats `live_boundary` as read-only. A possible `h64` is
   written to `draft_state` only after self-draft has finished using it.
8. Only accepted candidate rows may update `accepted_conv`,
   `transition_window`, `live_boundary`, or a Radix publication lane.
9. `return_logprob=True` and `False` run the same draft, verify, sample, and
   state commit transaction.
10. All accept, tail, and crossing decisions remain on the GPU.

## 4. Target EXTEND

For GDN models, DVR starts target EXTEND at a chunk-aligned matched prefix.
`page_size=64` makes the ordinary Radix prefix boundary compatible with this
requirement. A cache miss or Radix-disabled request performs ordinary target
prefill; it does not enter a separate fallback inference path.

During EXTEND, `DVRGDNAttnBackend`:

1. runs the ordinary GDN projections and convolution;
2. preserves the exact recurrent state at the latest completed 64-token
   boundary in the request's target temporal slot;
3. keeps the target convolution slot at the full accepted endpoint;
4. stores only post-boundary `k/v/g/beta` rows in `transition_window`; and
5. initializes private self-draft recurrent and convolution state.

If EXTEND creates a new exact boundary, the same temporal and convolution
checkpoint is copied to an ordinary Radix tracking lane. No DVR code changes
the request-to-Mamba mapping or the cache's donation/rebinding semantics.

For a model without a registered linear-state adapter, all lifecycle methods
are no-ops and DVR uses only the common speculative target transaction.

## 5. Draft

### Self-decode

Self-decode replays one dedicated single-token CUDA graph for each provisional
edge. GDN decode selects `draft_state` and `draft_conv`, so provisional writes
cannot corrupt target or Radix state. The recurrent decode kernel itself is
unchanged.

The graph is captured with ordinary non-deterministic decode tuning and may
capture custom all-reduce on a supported NVLink topology. Target prefill and
verify retain deterministic collective and attention settings.

### EAGLE/MTP

DVR delegates proposal generation, draft KV, hidden-state input, and draft
cache finalization to the upstream EAGLE worker. The DVR wrapper does not
allocate a second EAGLE KV cache or replay target history through the draft
model.

Both draft backends return the same chain token/probability contract to the
common target verify path.

### One-token prompt sentinel

Some draft backends cannot be seeded from a one-token prefill. Such a request
first runs one fixed-shape root-only target verify transaction. That transaction
accepts only its root token and establishes the ordinary draft input; all later
rounds use the normal backend. This is a protocol sentinel, not a slower
deterministic fallback.

## 6. Deterministic target verify

Full attention computes the proposal rows using Triton or FA3. GDN must retain
the same physical 64-token chunk partition as deterministic prefill, so its
dedicated DVR backend builds:

```text
live_boundary + accepted tail + proposal rows
```

in a fixed `64 + draft_tokens` window. Candidate `q` is verify-local; cached
tail `q` is unnecessary because prefix outputs are discarded. The dedicated
DVR GDN operator:

- reads `live_boundary` without modifying it;
- returns only the logical proposal outputs;
- keeps candidate convolution endpoints in `verify_conv`; and
- exports the exact state at the next physical 64-token boundary into the one
  reusable `draft_state` workspace.

Rows after `tail_len + draft_tokens` are never authoritative. The final state
after every physical row in the fixed window must not be committed.

The DVR-specific recurrent producer and packing/scatter kernels live under
`layers/attention/dvr/`. Ordinary FLA, Mamba, and GDN kernels keep their
upstream signatures and behavior.

## 7. Sample and commit

One GPU acceptance result drives target KV rollback, output materialization,
GDN state commit, and draft-backend finalization.

For every request:

1. select the accepted candidate convolution endpoint and write it to
   `accepted_conv`;
2. append only accepted `k/v/g/beta` rows;
3. if `tail_len + accepted >= 64`, copy staged `h64` into `live_boundary`,
   optionally publish the matching temporal and convolution state to the
   selected Radix lane, and compact the transition window by 64;
4. update `boundary_seq_lens` and the selected
   `published_boundary_lens` entry; and
5. rebuild private self-draft state from the new `live_boundary` plus the
   accepted tail.

The rebuild processes at most 63 rows. EAGLE/MTP skips target recurrent-state
rebuild and finalizes its own draft cache through the upstream worker.

## 8. Overlap and request release

State commit is enqueued as part of the same GPU transaction as target verify.
Spec v2 can therefore return a FutureMap immediately without reading accept or
sequence lengths on the host.

Overlap may enqueue one additional DVR round before CPU result processing sees
that a request has finished. The worker publishes one existing read-done event
after rollback. Cache release waits for that event on the current GPU stream
before selecting a Radix checkpoint; it does not synchronize the host in
steady state.

At release, DVR selects the newest recorded boundary not later than the visible
committed prefix. If no new aligned boundary exists, insertion is skipped and
the next request performs normal prefix recomputation. This can reduce cache
reuse but cannot change generated tokens or logits.

With Radix disabled, publication metadata is not allocated. The target live
boundary, transition window, draft, verify, and commit flow is otherwise
identical.

## 9. Supported integration

- CUDA with pipeline parallel size one.
- Triton or FlashAttention 3 full-attention backend.
- Triton GDN prefill/verify backend.
- Top-k one chain proposals and exact rejection sampling.
- Radix cache enabled or disabled.
- Synchronous spec-v1 compatibility and overlap spec-v2 scheduling.

Unsupported state representations are rejected during argument handling:
page-major recurrent storage, ReplaySSM, streaming sessions, and int8 recurrent
checkpoints.

## 10. Validation

The fixed local gates are:

```bash
test/manual/dvr/local/scripts/run_static_unit_checks.sh

MODEL_PATH=/path/to/Qwen3.5-0.8B-DVR-Smoke-2L \
QUICK=1 \
test/manual/dvr/local/scripts/run_0p8b_self_dvr_kl.sh
```

The extracted checkpoint contains one GDN layer, one full-attention layer, and
an MTP layer. It verifies startup, CUDA graphs, one-token prompts, 63/64/65
boundaries, batch and concurrent requests, strict returned-logprob replay, and
the no-logprob output path. It is deliberately inaccurate and must not be used
for acceptance or throughput conclusions.

H20 qualification additionally covers 0.8B/35B/80B, self/EAGLE, sync/overlap,
Triton/FA3, Radix on/off, 512/1024-token generation, real datasets, TP, and
matched normal/deterministic throughput baselines.
