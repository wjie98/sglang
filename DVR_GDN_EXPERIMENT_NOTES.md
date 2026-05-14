# DVR GDN Experiment Notes

Date: 2026-05-15

This file records temporary exploration results from the v3 GDN migration. It is
intended as a local note and should not be committed unless explicitly needed.

## Stable Findings

1. Qwen3.5 GDN ordinary prefill can differ for the same prefix when the total
   prefill shape changes.
   - Test: compare a fixed 17-token sequence against the same 17-token prefix
     padded to 80 tokens.
   - `core_attn_out` and `z` before GDN gated RMSNorm matched bitwise.
   - The first mismatch appeared in the GDN gated RMSNorm output.
   - Replacing the Triton gated RMSNorm with `rms_norm_ref` under deterministic
     inference made the prefix rows bitwise equal.
   - Implication: fixed 64+16 DVR verify windows can expose shape-dependent
     dense-kernel differences even before state-management issues.

2. DVR target verify with `causal_conv1d_fn` does not automatically populate
   `intermediate_conv_window`.
   - EAGLE-style verify using `causal_conv1d_update` populates intermediate conv
     windows for accepted-step state scatter.
   - DVR's fixed linear window currently uses ordinary causal conv, so accepted
     token conv state must be cached explicitly or the next draft round starts
     from an incorrect conv state.

3. Live GDN temporal state and chunk-boundary state have different semantics.
   - Live state for the latest accepted token should follow recurrent/decode
     semantics.
   - Chunk-boundary checkpoint state should be produced by chunkwise scan at
     `FLA_CHUNK_SIZE` boundaries.
   - A pure chunkwise live-state update for non-boundary tokens is likely wrong.

4. The first DVR verify round and later DVR verify rounds have different anchor
   ownership.
   - The first generated token's logit comes from prompt prefill.
   - Later verify rounds need a verified-anchor row to predict the first draft
     token in that round.
   - A naive `req.output_ids` based heuristic was unreliable because scheduler
     state is already mutated at the point where fixed-window logits are
     restored.
   - This needs an explicit DVR-owned per-request round/anchor state rather than
     inference from `req.output_ids`.

## Test Results Observed

1. Attention-only Qwen3 remained strict KL=0 in the earlier v3 baseline.

2. Qwen3.5 after the deterministic gated RMSNorm experiment:
   - `max_new_tokens=8` reached returned-logprob vs full-prefill-scoring
     equality in one run (`maxdiff=0`).
   - Longer generations still diverged around the second round, commonly around
     output token 8.

3. After experimenting with anchor-row slicing:
   - Always starting fixed-window logits at `verified_tail_len - 1` broke the
     first round.
   - Always starting at `verified_tail_len` preserved short runs but did not fix
     later rounds.
   - A `req.output_ids` heuristic also broke the first round and was reverted.

## Candidate Fixes Worth Reconsidering

1. Add an explicit DVR per-request anchor/round state.
   - The state should say whether the first logit for the next verify round is
     already owned by ordinary prefill/previous verify, instead of deriving this
     from `req.output_ids`.

2. Reintroduce a minimal deterministic gated RMSNorm path for Qwen3.5/GDN if the
   fixed-window verify path is kept.
   - Scope it to `enable_deterministic_inference` or DVR GDN if possible.
   - The change is correctness-oriented but may be slower.

3. Reintroduce explicit DVR conv-window caching only after anchor semantics are
   made explicit.
   - The idea is sound, but the current experiment did not prove it sufficient.

4. Keep live temporal state recurrent and boundary temporal state chunkwise.
   - This matches the intended state semantics and the author's explanation.

## Migrated From `dvr-eagle-postprocess-exp`

1. EAGLE-style verify/postprocess remains the right compatibility target.
   - Select accepted logits by `accepted_indices`.
   - Commit DVR/GDN state before adding returned logprobs.
   - Prepare the next draft from `verify_output.draft_input`.
   - Preserve the `accept_length + 1` bonus-token contract.

2. Conv state must be treated as part of the verified state.
   - The exp branch showed the largest Qwen3.5 drift came from using the wrong
     conv state at the next verify/draft boundary.
   - DVR's fixed-window verify should still be a linear causal chain, but it
     needs EAGLE-like `intermediate_conv_window` output so accepted live and
     boundary conv states can be scattered after verification.
   - Directly replacing the fixed-window conv logits path with no-tree
     `causal_conv1d_update` ran but made Qwen3.5 acceptance much worse. Keep
     `causal_conv1d_fn` for logits and explicitly materialize the linear
     `intermediate_conv_window` from the raw fixed-window input instead.

3. The useful exp result was not the old branch structure itself.
   - Do not port the exp branch wholesale: it predates the current fixed
     `verified_token + draft_token + padding_token` window.
   - Keep the current v3 fixed-window q/k/v/g/beta cache and only migrate the
     conv-window lifecycle pieces.

## 2026-05-15 Migration Test Results

1. Static checks passed after the conservative conv-window migration:
   - `py_compile` for `gdn_backend.py` and `dvr_worker.py`
   - `git diff --check`

2. Qwen3-0.6B attention-only regression still passes with cuda graph enabled.
   - `max_new_tokens`: `1, 8, 16, 31, 64, 80, 127, 128`
   - every case had `maxdiff=0.0` against full-prefill scoring.

3. Qwen3.5-0.8B needs `SGLANG_ENABLE_SPEC_V2=1` and
   `--mamba-scheduler-strategy extra_buffer`.
   - DVR's default `max_running_requests=48` is too large for this model on the
     5090 with current speculative GDN buffers; it tried to allocate about
     59GiB for intermediate SSM state.
   - `--max-running-requests 1` starts successfully.

4. Direct no-tree `causal_conv1d_update` for DVR fixed-window conv is rejected.
   - It ran, but Qwen3.5 acceptance collapsed and long-generation KL got much
     worse.
   - Keep `causal_conv1d_fn` as the logits path.

5. Conservative explicit conv-window materialization is runtime-safe but not
   sufficient.
   - Qwen3.5, cuda graph disabled, prompt length 9:
     - `max_new_tokens=1`: `maxdiff=0.0`
     - `max_new_tokens=8`: first mismatch 5, `maxdiff=0.0032685399055480957`,
       accept rate `0.4666666666666667`
     - `max_new_tokens=16`: first mismatch 8, `maxdiff=4.419391632080078`,
       accept rate `0.13333333333333333`
     - `max_new_tokens=32`: first mismatch 8, `maxdiff=11.070916652679443`,
       accept rate `0.042105263157894736`
     - `max_new_tokens=64`: first mismatch 8, `maxdiff=11.836380004882812`,
       accept rate `0.01904761904761905`
     - `max_new_tokens=80`: generated 39 before EOS, first mismatch 8,
       `maxdiff=14.399996757507324`, accept rate `0.048484848484848485`
   - Interpretation: target verify logits are already not full-prefill-equivalent
     before post-verify state repair can help. The next fix should inspect
     fixed-window GDN inputs and anchor/verified-tail logits, not only conv
     state commit.

## 2026-05-15 Qwen3 Batch And Stress Baseline

This checkpoint intentionally does not fix the Qwen3.5/GDN divergence after
token 8. It establishes a stable Qwen3 attention-only base for the next repair
round.

Server command shape:

- model: `/home/hwj/Qwen3-0.6B`
- `--speculative-algorithm DECODE_VERIFY_ROLLBACK`
- `--speculative-num-steps 15`
- `--speculative-num-draft-tokens 16`
- `--speculative-eagle-topk 1`
- `--page-size 1`
- `--max-running-requests 8`
- `--attention-backend triton`
- `--sampling-backend pytorch`
- `--enable-deterministic-inference`
- cuda graph enabled
- DeepGEMM disabled through environment variables

Concurrent batch test, strict full-prefill scoring oracle:

- `max_new_tokens`: `17, 65, 129, 257`
- server formed a real batch: log showed `#new-seq: 3` while one request was
  already running
- all cases had `ids=True`, `maxdiff=0.0`, `kl=0.0`
- accept rates: `1.0`, `0.9866666666666667`, `0.9111111111111111`, `1.0`

Longer concurrent stress test:

- `max_new_tokens`: `257, 383, 511, 769`
- server log showed `#running-req: 3`, `cuda graph: True`
- all cases had `ids=True`, `maxdiff=0.0`, `kl=0.0`
- accept rates: `1.0`, `0.9546666666666667`, `0.9509803921568627`,
  `0.9795918367346939`

Eight-client batch test:

- `max_new_tokens`: `33, 65, 97, 129, 161, 193, 225, 257`
- server log showed `#new-seq: 7` and later decode with `cuda graph: True`
- all cases had `ids=True`, `maxdiff=0.0`, `kl=0.0`
- accept rates ranged from `0.7538461538461538` to `1.0`

Operational note:

- Qwen3.5/GDN with current speculative buffers cannot use DVR's default
  `max_running_requests=48` on RTX 5090; it OOMs while allocating speculative
  Mamba/GDN intermediate state. Use low `--max-running-requests` for GDN until
  buffer sizing is optimized.

## Temporary Code Removed

The following debug-only code was removed before returning to a clean baseline:

- `SGLANG_DVR_DEBUG_DUMP_DIR` tensor dumps in `gdn_backend.py`
- `SGLANG_DVR_LAYER_DEBUG_DUMP_DIR` layer dumps in `qwen3_5.py`
- `SGLANG_DVR_DEBUG_ACCEPT` accept metadata logs in `dvr_worker.py`
