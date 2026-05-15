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

## 2026-05-15 Non-DVR GDN Cache-Hit Determinism

Purpose:

- Check whether Qwen3.5/GDN has a fundamental obstacle to deterministic
  prefill/checkpoint reuse.
- Separate ordinary GDN radix/mamba checkpoint correctness from DVR worker state
  handoff bugs.

Server shape:

- model: `/home/hwj/Qwen3.5-0.8B`
- no DVR/speculative algorithm
- `--enable-deterministic-inference`
- `--mamba-scheduler-strategy extra_buffer`
- `--mamba-track-interval 64`
- `--disable-overlap-schedule`
- `--disable-cuda-graph`
- `--disable-piecewise-cuda-graph`
- DeepGEMM disabled through environment variables

Important testing detail:

- Do not use `logprob_start_len=0` when testing prefix-cache hits. In scheduler
  prefill matching, full input-logprob requests cap `max_prefix_len` at
  `logprob_start_len`, so `logprob_start_len=0` intentionally disables prefix
  cache matching.
- To test the cache-hit path, compare the next-token `output_token_logprobs` and
  `output_top_logprobs` after the same prefix with and without cache.

Results:

- Dense checkpoints were created by warming shared prefixes at 64-token
  boundaries.
- The server log confirmed real cache hits:
  - `#new-token: 64, #cached-token: 64`
  - `#new-token: 64, #cached-token: 128`
- `L=128`, with 64 cached tokens:
  - generated next-token id matched
  - output logprob diff: `0.0`
  - output top-5 logprobs matched exactly
- `L=192`, with 128 cached tokens:
  - generated next-token id matched
  - output logprob diff: `0.0`
  - output top-5 logprobs matched exactly

Interpretation:

- Ordinary Qwen3.5/GDN deterministic prefill plus radix/mamba checkpoint reuse is
  strict-equal at 64-token boundaries under the tested path.
- Therefore, implementing deterministic inference for GDN through DVR has no
  observed principle-level blocker. The remaining DVR divergence is an
  implementation issue in how DVR obtains, restores, and commits the chunk
  boundary temporal state and conv state.
- The next debugging target should be the DVR worker/backend boundary-state
  path: compare the state/conv state used by DVR target verify against the state
  used by the ordinary non-DVR cache-hit path above.

Rejected/unstable testing setup:

- The same extra-buffer cache-hit experiment without `--disable-overlap-schedule`
  hung around the mamba radix cache runtime sanity check. This appears to be an
  overlap/cache stability risk and was not used for the correctness conclusion.

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
## 2026-05-15 GDN DVR Acceptance Debug on v3

Worktree: `/home/hwj/dvr_qwen3_5/sglang-dvr-clean-v3`
Branch: `dvr-v3-controlled-migration`

Test setup:

- Model: `/home/hwj/Qwen3.5-0.8B`
- DVR: `DECODE_VERIFY_ROLLBACK`, topk=1, `speculative_num_steps=15`, `speculative_num_draft_tokens=16`
- GDN state mode: `--mamba-scheduler-strategy extra_buffer --mamba-track-interval 64`
- Deterministic inference enabled, overlap/cuda graph disabled, DeepGEMM disabled.

Findings:

- The current fixed-window GDN verify already forwards a `64 + 16` window and slices the logits back to the 16 `draft_token` rows before calling `EagleVerifyInput.verify()`. Passing all 80 rows into SPS verify is not the current bug.
- For chain verify, `draft_token[0]` is the current anchor. The verify logits row selected at `verified_tail_len` predicts `draft_token[1]`, so the existing row offset is semantically correct.
- A temporary debug log showed greedy first-round alignment like:
  - `candidates=[anchor, draft1, draft2, ...]`
  - `target_top[0] == draft1`
  This confirms the first verify row is not off by one.
- I tried changing GDN state commit to advance by `accept_length` only, assuming `accept_length + 1` was committing the bonus token. That was wrong. The `+1` is the previous anchor, not the new bonus. The correct rolling tail after verify should include `old_anchor + accepted_draft_tokens` and exclude the new bonus anchor. The original `accept_length + 1` behavior matches that invariant.
- Low acceptance is therefore not explained by verify row selection or bonus duplication. The remaining issue is that recurrent self-decode draft and chunkwise target verify diverge within a chunk. After the first accepted span, the next round's self-draft is often no longer close to the next chunkwise verify distribution.

Useful invariant:

- `verified_tail` for the next 80-window should contain tokens from the chunk boundary through the token before the next anchor.
- `draft_token[0]` is the next anchor and is processed by the next verify/draft step.
- GDN live recurrent/conv state used by self-decode should correspond to the state before this next anchor.

Next debugging direction:

- Compare the live recurrent/conv state rebuilt in `update_dvr_state_after_verify()` against a normal deterministic cache-hit prefill state at the same prefix, especially after non-greedy accepted spans.
- Check whether the recurrent replay from saved q/k/v/g/beta is expected to match the state needed for high-acceptance self-decode, or whether high acceptance requires a different draft path for GDN within a chunk.

## 2026-05-15 Layerwise DVR GDN Decode-vs-Verify Comparison

Temporary instrumentation compared each GDN layer's self-decode `core_attn_out`
against the corresponding row in the `64 + 16` target verify window for the
same `anchor + draft` sequence.

Setup:

- Model: `/home/hwj/Qwen3.5-0.8B`
- Prompt: `Explain deterministic verification in one short sentence:`
- Sampling: greedy, `max_new_tokens=8`
- DVR: topk=1, draft tokens=16, fixed GDN verify window=80
- CUDA graph disabled, overlap disabled, DeepGEMM disabled

Observed first verify round with `tail=[9]`:

- GDN layer 0: first 8 step max abs diffs around `2e-4 ~ 5e-4`
- GDN layer 1: around `3e-5 ~ 2e-4`
- GDN layer 2: around `1e-4 ~ 5e-4`
- GDN layer 4: first 7 steps still small, but step 7 jumps to about `6.6e-2`
- Later GDN layers inherit/amplify the difference.

Observed second verify round with `tail=[11]`:

- GDN layer 0/1/2 are still small for the first several steps.
- GDN layer 4 jumps around step 5 to about `1.9e-1`; later layers amplify it.

Interpretation:

- This result argues against the hypothesis that recurrent and chunkwise GDN
  kernels produce a huge mismatch immediately in short sequences.
- The first large mismatch is visible at GDN layer 4. In Qwen3.5 the missing
  layer between GDN layer 2 and GDN layer 4 is a full-attention layer, so the
  most likely first source is the attention layer 3 output or the hidden-state
  handoff through that layer, not the early GDN recurrent/chunkwise kernel
  itself.
- Next focused check should dump model-layer outputs around layers 2, 3, and 4:
  layer 2 output, layer 3 attention output/post-MLP output, and layer 4 input.
  That should distinguish an attention KV/mask/position issue from a later GDN
  state issue.

## 2026-05-15 Full-Layer DVR GDN Trace: Root Cause of >=8 Divergence

Temporary trace hooks exported:

- normal deterministic full prefill/scoring for `prompt + greedy_output`;
- DVR self-decode rows;
- DVR target verify rows;
- GDN target-verify q/k/v/g/beta and chunkwise `core_attn_out`.

Setup:

- Model: `/home/hwj/Qwen3.5-0.8B`
- Prompt: `Explain deterministic verification in one short sentence:`
- Greedy output ids: `[271, 248068, 271, 248069, 271, 89454, 4384, 22188]`
- Full ids length: 17, prompt length: 9
- DVR: topk=1, draft tokens=16, GDN fixed verify window=80
- CUDA graph disabled, overlap disabled, DeepGEMM disabled

Important alignment correction:

- The first verify round did not contain the final greedy output after position
  10, because the later draft candidates were rejected.
- The correct baseline comparison must select the verify round whose
  `input_ids` prefix matches the complete `prompt + greedy_output`.
- After aligning this way, positions 9..15 match exactly. The first real
  divergence is position 16.

Model layer output, aligned to the matching verify pass:

| layer | type | verify vs full-prefill max | first pos >1e-2 | decode vs full-prefill max |
|---:|---|---:|---|---:|
| 0 | GDN | 0 | - | 0.00195312 |
| 1 | GDN | 0.000244141 | - | 0.00390625 |
| 2 | GDN | 0.000488281 | - | 0.00390625 |
| 3 | ATTN | 0.100098 | 16 | 0.00830078 |
| 4 | GDN | 0.0722656 | 16 | 0.00585938 |
| 5 | GDN | 0.0869141 | 16 | 0.00390625 |
| 6 | GDN | 0.107666 | 16 | 0.00292969 |
| 7 | ATTN | 0.174683 | 16 | 0.00634766 |
| 8 | GDN | 0.12439 | 16 | 0.00427246 |
| 9 | GDN | 0.12915 | 16 | 0.00439453 |
| 10 | GDN | 0.109863 | 16 | 0.00390625 |
| 11 | ATTN | 0.157715 | 16 | 0.00585938 |
| 12 | GDN | 0.169434 | 16 | 0.0057373 |
| 13 | GDN | 0.300491 | 16 | 0.00585938 |
| 14 | GDN | 0.15918 | 16 | 0.00927734 |
| 15 | ATTN | 0.222656 | 16 | 0.00585938 |
| 16 | GDN | 0.201538 | 16 | 0.00537109 |
| 17 | GDN | 0.189087 | 16 | 0.00537109 |
| 18 | GDN | 0.816406 | 16 | 0.0117188 |
| 19 | ATTN | 0.215332 | 16 | 0.0117188 |
| 20 | GDN | 0.235352 | 16 | 0.0166016 |
| 21 | GDN | 0.246338 | 16 | 0.012085 |
| 22 | GDN | 1.20166 | 16 | 0.015625 |
| 23 | ATTN | 0.887695 | 16 | 0.140625 |

GDN core/qkvg-beta comparison, aligned to the same verify pass:

| gdn layer | verify core max | first pos >1e-2 | q max | k max | v max | g max | beta max |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0 | - | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | - | 0 | 0 | 0 | 0 | 0 |
| 2 | 3.05176e-05 | - | 0.00390625 | 0.00390625 | 0.00195312 | 8.56519e-05 | 0.00195312 |
| 4 | 0.0419922 | 16 | 1.59375 | 0.605469 | 0.759766 | 0.81162 | 0.132812 |
| 5 | 0.00390625 | - | 0.726562 | 0.351562 | 0.1875 | 0.234367 | 0.0664062 |
| 6 | 0.000961304 | - | 1.01367 | 0.230469 | 0.0793457 | 0.201281 | 0.09375 |
| 8 | 0.0800781 | 16 | 1.28223 | 0.724609 | 0.354492 | 0.180371 | 0.234375 |
| 9 | 0.00378418 | - | 1.21973 | 0.242676 | 0.265625 | 0.109712 | 0.368164 |
| 10 | 0.00259399 | - | 1.29688 | 0.722656 | 0.20459 | 0.335996 | 0.195312 |
| 12 | 0.0144615 | 16 | 1.90625 | 0.730652 | 0.880859 | 0.19733 | 0.138672 |
| 13 | 0.00730896 | - | 2.53906 | 1.03223 | 0.392578 | 0.208069 | 0.154297 |
| 14 | 0.00259399 | - | 1.08984 | 0.194824 | 0.548828 | 0.183772 | 0.265625 |
| 16 | 0.03125 | 16 | 4.53125 | 0.640625 | 0.914062 | 0.300087 | 0.304688 |
| 17 | 0.0477295 | 16 | 5.125 | 1.00781 | 1.51562 | 0.144813 | 0.269531 |
| 18 | 0.0410156 | 16 | 0.896484 | 0.9375 | 0.84375 | 0.181114 | 0.144531 |
| 20 | 0.121094 | 16 | 1.75 | 1.7627 | 1.39062 | 0.252073 | 0.359375 |
| 21 | 0.0712891 | 16 | 1.53125 | 2.125 | 0.796875 | 0.265596 | 0.21875 |
| 22 | 0.0288086 | 16 | 1.23047 | 1.35156 | 0.976562 | 0.684968 | 0.171875 |

First divergence by token position:

| pos | token | first layer >1e-2 | max layer diff |
|---:|---:|---|---:|
| 9 | 271 | - | 0 |
| 10 | 248068 | - | 0 |
| 11 | 271 | - | 0 |
| 12 | 248069 | - | 0 |
| 13 | 271 | - | 0 |
| 14 | 89454 | - | 0 |
| 15 | 4384 | - | 0 |
| 16 | 22188 | 3 | 1.20166 |

Root cause:

- Layer 0/1/2 are GDN and match full prefill through position 16.
- The first bad layer is layer 3, which is a full-attention layer.
- This points to target-verify full-attention metadata, not to early GDN
  recurrent/chunkwise state.
- The fixed DVR GDN verify window changes `spec_info.draft_token_num` from 16
  to 80 before forward. GDN backend reads `spec_info.draft_token_num`, so it
  processes all 80 rows.
- The triton full-attention target-verify metadata still uses backend
  `self.num_draft_tokens`, initialized from CLI as 16, to build `qo_indptr`,
  `seq_mask_len`, and `max_extend_len`.
- Therefore the attention target-verify kernel only computes query rows 0..15
  in the 80-row window. Row 16 is the first row outside that range, so the
  repeated ">=8 generated tokens" symptom is actually "the first generated row
  that maps to fixed-window row 16".

Next fix:

- Make target-verify attention metadata use the runtime verify length
  (`spec_info.draft_token_num` / `spec_info.num_tokens_per_req`) instead of the
  backend's static `self.num_draft_tokens`.
- Keep the change narrow to target-verify metadata construction. The GDN path
  already follows the runtime window length.

Validation after applying that fix to triton target-verify metadata:

- Same prompt/output, same 80-row fixed verify window, CUDA graph still disabled.
- DVR output ids stayed `[271, 248068, 271, 248069, 271, 89454, 4384, 22188]`.
- Acceptance improved from `0.2` to `0.5333333333333333`.
- The layer 3 attention error disappeared:
  - before: layer 3 `verify vs full-prefill max = 0.100098`, first bad pos 16
  - after: layer 3 `verify vs full-prefill max = 0.000976562`, no bad pos
- GDN core errors also collapsed:
  - before: GDN layer 4 core max `0.0419922`, first bad pos 16
  - after: GDN layer 4 core max `0.00195312`, no bad pos
- Remaining aligned layer-output max after the fix:

| layer | type | verify vs full-prefill max | first pos >1e-2 |
|---:|---|---:|---|
| 0 | GDN | 0 | - |
| 1 | GDN | 0.000244141 | - |
| 2 | GDN | 0.000488281 | - |
| 3 | ATTN | 0.000976562 | - |
| 4 | GDN | 0.00140381 | - |
| 5 | GDN | 0.0012207 | - |
| 6 | GDN | 0.00134277 | - |
| 7 | ATTN | 0.00128174 | - |
| 8 | GDN | 0.00115967 | - |
| 9 | GDN | 0.00109863 | - |
| 10 | GDN | 0.00195312 | - |
| 11 | ATTN | 0.00109863 | - |
| 12 | GDN | 0.00134277 | - |
| 13 | GDN | 0.00195312 | - |
| 14 | GDN | 0.00170898 | - |
| 15 | ATTN | 0.00244141 | - |
| 16 | GDN | 0.00390625 | - |
| 17 | GDN | 0.00256348 | - |
| 18 | GDN | 0.00585938 | - |
| 19 | ATTN | 0.0078125 | - |
| 20 | GDN | 0.00585938 | - |
| 21 | GDN | 0.00488281 | - |
| 22 | GDN | 0.00390625 | - |
| 23 | ATTN | 0.0117188 | 16 |

Conclusion after this fix:

- The large `>=8 token` failure was a fixed-window/full-attention metadata bug,
  not a GDN recurrent-vs-chunkwise kernel failure.
- There is still a smaller final-layer discrepancy around one bf16 quantum
  scale (`~1e-2`) at the last attention layer for this example. That should be
  investigated separately after the current metadata bug is kept fixed.

## 2026-05-15 Follow-up: fixed window vs final partial scoring

After committing `e01ab8da4 dvr: align chunk verify graph window`, I reran the
GDN no-graph path and traced a short `max_new_tokens=8` case. The scoring oracle
request had `#cached-token: 0`, so the short mismatch is not caused by radix
cache reusing polluted DVR state.

Trace setup:

- Model: `/home/hwj/Qwen3.5-0.8B`
- DVR: `DECODE_VERIFY_ROLLBACK`, topk=1, draft tokens=16
- Verify: chunk-boundary fixed physical window, `FLA_CHUNK_SIZE + draft = 80`
- CUDA graph: disabled
- Trace layers: 0..23

Observed result:

- The generated output length was 8, so the oracle full-prefill scoring request
  ran over `prompt + output = 17` tokens.
- DVR target verify still ran over the full fixed verify input:
  `verified_token + 16 draft_token + padding_token = 80` physical rows.
- For the first GDN layer, the first 17 rows of
  `verify_chunkwise_core` exactly matched the 17-row ordinary scoring prefill.
- For the second GDN layer, q/k/v/g/beta and GDN core still matched for the first
  17 rows.
- The first visible difference appeared in layer 1 **after** the GDN core,
  i.e. in the layer post-processing path (gated norm / output projection / MLP /
  layer communicator), starting at physical row 13.
- This row maps to the first returned-logprob mismatch:
  output index 5 compares the logits for scoring position 14, whose hidden state
  depends on row 13.

Control experiment:

- Restarting with `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=1`
  did not change the mismatch:
  `max_new_tokens=8` still failed with max abs `0.0032685399055480957`.
- `max_new_tokens=16` passed strictly because the oracle scoring length
  (`prompt + 16 = 25`) lines up with the useful verify rows for a full draft
  round much better than the 8-token final partial case.

Current interpretation:

- The remaining short-window mismatch is not a GDN state replay bug. It happens
  before post-verify state commit matters.
- It is a "future draft rows are present during verify" issue. DVR computes a
  longer speculative verify window than the final oracle scoring request when
  generation stops in the middle of a draft block. For pure attention Qwen3 this
  is batch-invariant enough to stay bitwise equal; for Qwen3.5 GDN, at least one
  post-GDN dense/norm path is still sensitive to the physical row count.
- Therefore, strict `full prefill scoring == returned logprobs` for GDN final
  partial blocks requires either:
  1. not forwarding future draft/padding rows through row-count-sensitive dense
     post-processing, or
  2. recomputing/overwriting final partial returned logprobs with an ordinary
     prefill-equivalent scoring path, or
  3. making the remaining Qwen3.5 post-GDN dense/norm path fully batch-invariant
     for `M=17` vs `M=80`.

This is separate from the cross-chunk state/conv/qkvg replay path. The replay
path still needs to use one fixed-window view:

- `pos_before = verified_tail_lens`
- `pos_after = pos_before + accepted_tokens`
- live temporal state uses recurrent replay up to `pos_after`
- chunk boundary state uses chunkwise replay at physical row `FLA_CHUNK_SIZE - 1`
- live conv uses `intermediate_conv_window[pos_after - 1]`
- boundary conv uses `intermediate_conv_window[FLA_CHUNK_SIZE - 1]`
