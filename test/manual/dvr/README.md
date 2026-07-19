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

For GDN models, `page_size` must be a positive divisor of the FLA chunk size
(64). A page size of 64 is the recommended default.

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
`HOST`, `PORT`, `TP_SIZE`, `DRAFT_TOKENS`, `DRAFT_STEPS`, `PAGE_SIZE`,
`ATTENTION_BACKEND`, `LINEAR_ATTN_BACKEND`, `SAMPLING_BACKEND`,
`MAX_RUNNING_REQUESTS`, `MAX_MAMBA_CACHE_SIZE`, `MEM_FRACTION_STATIC`,
`DISABLE_OVERLAP`, and `DISABLE_RADIX_CACHE`.

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

Fast regression tests live under `test/registered/unit/`. The larger KL,
acceptance, H20, and throughput matrices are development qualification assets
under `test/manual/dvr/local/`; they are not intended to be included in a
minimal upstream change.
