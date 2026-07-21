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

For gated linear-attention models, the ordinary token KV cache and the
recurrent state have different lifetimes. DVR keeps the full-attention KV in
SGLang's existing request and Radix pools. It treats only a recurrent state at
an exact 64-token boundary as a persistent checkpoint and keeps the unclosed
tail as request-owned `q/k/v/g/beta` inputs.

The state transition is:

1. Target EXTEND establishes the latest exact boundary and caches the inputs
   after that boundary. With Radix disabled, this is a request-local checkpoint
   and a later request performs an ordinary full prefill.
2. Under overlap scheduling, unfinished-prefill result processing completes
   any Radix checkpoint donation before the first DVR decode resolves physical
   checkpoint slots.
3. Draft execution is provisional. Self-draft may mutate the target live state
   and therefore saves its convolution state; EAGLE/MTP uses its own draft
   cache. Neither draft backend changes the authoritative target boundary.
4. Target verify restores that boundary and runs deterministic EXTEND over one
   fixed `64 + draft_tokens` window. The first 64 rows reproduce the original
   chunk partition; only the logical draft rows are returned.
5. Rollback commits only accepted rows. Crossing a 64-token boundary publishes
   the exact intermediate recurrent state into the alternate checkpoint slot
   and shifts the input window by one chunk.
6. Request release publishes the newest exact checkpoint no later than the
   visible committed prefix. Radix stores only that aligned prefix; a later
   request rebuilds the non-aligned suffix through ordinary EXTEND. If no such
   checkpoint is available, insertion is skipped rather than changing the
   inference result.

At every draft boundary, `checkpoint_length + tail_length` must equal the
committed target history. The checkpoint and cached state inputs must originate
from that same history, and physical slot replacement must finish before DVR
captures the slot indices used by verify and rollback.

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
- recurrent checkpoints are request-owned and survive Radix donation or slot
  reuse without changing ordinary Radix-disabled semantics.

GDN DVR does not currently support page-major recurrent-state storage,
ReplaySSM, streaming sessions, or int8 recurrent checkpoints. These modes
change the representation or lifetime of the exact state that rollback needs;
the server rejects them instead of silently running with an incomplete state.

Fast regression tests live under `test/registered/unit/`. The larger KL,
acceptance, H20, and throughput matrices are development qualification assets
under `test/manual/dvr/local/`; they are not intended to be included in a
minimal upstream change.
