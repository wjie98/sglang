# Decode-Verify-Rollback

Decode-Verify-Rollback (DVR) is a speculative decoding mode for deterministic
target execution. The target model performs deterministic prefill and verify;
provisional draft decoding uses the normal fast decode configuration. DVR can
therefore reproduce generated tokens with a later target prefill, including on
models with gated recurrent linear-attention layers.

Two draft backends are available:

- `DECODE_VERIFY_ROLLBACK`: the target model drafts from its committed state.
- `DECODE_VERIFY_ROLLBACK_EAGLE`: an EAGLE or MTP model produces draft tokens.

## Supported configuration

- CUDA execution with pipeline parallel size one.
- Triton or FlashAttention 3 for full-attention layers.
- Triton linear-attention prefill for models with GDN layers.
- Chain proposals (`speculative_eagle_topk=1`) and exact rejection sampling.
- Radix cache enabled or disabled. Disabling Radix keeps ordinary full-prefill
  semantics and uses request-local recurrent checkpoints while the request is
  active.
- Overlap scheduling enabled by default. Pass `--disable-overlap-schedule` for
  the synchronous compatibility path.

For GDN models, `page_size` must equal the FLA chunk size (64). This keeps
Radix prefix boundaries identical to the recurrent checkpoints used by verify.

## Start a server

The launcher uses the current Python environment and accepts additional SGLang
arguments after its own environment-based defaults:

```bash
MODEL_PATH=/models/Qwen3.5-35B-A3B \
TP_SIZE=4 \
ATTENTION_BACKEND=fa3 \
test/manual/dvr/launch_server.sh
```

To use the model's MTP layers as an EAGLE draft model:

```bash
MODEL_PATH=/models/Qwen3.5-35B-A3B \
DVR_MODE=eagle \
TP_SIZE=4 \
test/manual/dvr/launch_server.sh
```

`DRAFT_MODEL_PATH` defaults to `MODEL_PATH` in EAGLE mode. Set it explicitly
when using a separate EAGLE checkpoint. Useful environment overrides include
`SERVER_HOST`, `PORT`, `TP_SIZE`, `DRAFT_TOKENS`, `DRAFT_STEPS`, `PAGE_SIZE`,
`ATTENTION_BACKEND`, `LINEAR_ATTN_BACKEND`, `SAMPLING_BACKEND`,
`MAX_RUNNING_REQUESTS`, `MAX_MAMBA_CACHE_SIZE`, `MEM_FRACTION_STATIC`,
`DISABLE_OVERLAP`, `DISABLE_RADIX_CACHE`, and `RANDOM_SEED`.

The launcher does not activate Conda, select GPUs, configure datasets, or pin a
benchmark-specific CUDA graph batch list.

## Request example

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Explain deterministic speculative decoding.",
    "sampling_params": {
      "max_new_tokens": 128,
      "temperature": 0
    },
    "return_logprob": true
  }'
```

## Recurrent-state lifecycle

The isolated self-draft state lifecycle and its H20 qualification gates are
recorded in `local/STATE_LIFECYCLE_REDESIGN.md`. The transition below describes
the implemented lifecycle; release qualification still requires the documented
NVLink correctness and performance matrix.

For gated linear-attention models, the ordinary token KV cache and the
recurrent state have different lifetimes. DVR keeps the full-attention KV in
SGLang's existing request and Radix pools. It treats only a recurrent state at
an exact 64-token boundary as a persistent checkpoint and keeps the unclosed
tail as request-owned `k/v/g/beta` transition inputs. Cached `q` is unnecessary:
verify consumes `q` only for the current candidate outputs.

The state transition is:

1. Target EXTEND establishes the latest exact boundary and caches the inputs
   after that boundary. With Radix disabled, this is a request-local checkpoint
   and a later request performs an ordinary full prefill.
2. Under overlap scheduling, the request's two physical checkpoint slots stay
   fixed. Unfinished-prefill result processing copies the latest `keep`
   checkpoint into an independent Radix slot, so the first DVR decode can be
   scheduled before that CPU result is consumed.
3. Target EXTEND seeds one request-owned self-draft recurrent workspace and
   convolution state. Self-draft mutates only this private state; EAGLE/MTP
   advances its upstream-owned draft cache. Neither backend changes the
   authoritative target boundary.
4. Target verify reads the exact boundary without overwriting it and runs
   deterministic EXTEND over one fixed `64 + draft_tokens` window. The first 64
   rows reproduce the original chunk partition; only logical draft rows are
   returned, and the exported boundary state is staged in the now-idle
   self-draft workspace.
5. Rollback commits only accepted rows. Crossing a 64-token boundary publishes
   the staged boundary state into the alternate checkpoint slot and compacts
   the input window. Self-draft then rebuilds its private endpoint from the
   newest exact boundary plus the accepted tail; EAGLE/MTP skips this target
   recurrent-state reconstruction.
6. Request release publishes the newest exact checkpoint no later than the
   visible committed prefix. Radix stores only that aligned prefix; a later
   request rebuilds the non-aligned suffix through ordinary EXTEND. If no such
   checkpoint is available, insertion is skipped rather than changing the
   inference result. Under overlap, release first orders itself after the
   already-enqueued rollback event; this is a GPU stream dependency on the
   finishing request, not a steady-state host synchronization.

At every draft boundary, `checkpoint_length + tail_length` must equal the
committed target history. The checkpoint and cached state inputs must originate
from that same history. An active request's physical checkpoint mapping is
immutable from target EXTEND through draft, verify, and rollback.

## Compatibility contracts

The registered tests intentionally protect behavior rather than private buffer
layouts. The CUDA graph tests are the exception where a narrow integration
contract is required: upstream backend changes must not cause target
deterministic settings to leak into provisional draft capture.

In particular:

- target prefill and verify remain deterministic;
- draft graph capture restores environment, backend, server-argument, MoE, and
  collective state when it exits;
- custom all-reduce may be captured for provisional draft execution but is not
  enabled for target prefill or verify;
- Triton and FA3 draft decode retain their normal split configuration;
- recurrent checkpoints are request-owned; unfinished Radix insertion copies
  `keep` instead of rebinding those slots, without changing ordinary
  Radix-disabled semantics.

GDN DVR does not currently support page-major recurrent-state storage,
ReplaySSM, streaming sessions, or int8 recurrent checkpoints. These modes
change the representation or lifetime of the exact state that rollback needs;
the server rejects them instead of silently running with an incomplete state.

Fast regression tests live under `test/registered/unit/`. The larger KL,
acceptance, H20, and throughput matrices are development qualification assets
under `test/manual/dvr/local/`; they are not intended to be included in a
minimal upstream change.
