# DVR Qwen3.5 Divergence Summary

Date: 2026-05-15

Branch: `dvr-v3-controlled-migration`

Base checkpoint: `b5f8ca694 dvr: stabilize qwen3 batch baseline`

This report measures the current Qwen3.5/GDN DVR failure mode. It intentionally
does not fix the token-8 divergence. The oracle is full-prefill scoring over
`prompt_ids + output_ids`.

## Runtime

- model: `/home/hwj/Qwen3.5-0.8B`
- `SGLANG_ENABLE_SPEC_V2=1`
- `--mamba-scheduler-strategy extra_buffer`
- `--max-running-requests 1`
- cuda graph disabled for this measurement
- DeepGEMM disabled
- sampling: `temperature=0.7`, `top_p=0.9`, `top_k=-1`, `ignore_eos=True`

## Main Result

The current GDN DVR path is not close to the Qwen3 attention baseline:

- token IDs still match the generated output against full-prefill scoring.
- returned logprobs diverge early.
- acceptance collapses to roughly `1-3%`.
- for long generation, returned logprobs are incomplete: generation can continue
  to 128/256 tokens, but the API returns fewer comparable logprob entries.

## 128-token Run

- generated tokens: `128`
- returned comparable logprobs: `85`
- first nonzero diff: token index `8`
- max absolute logprob diff: `10.7216`
- mean absolute logprob diff over comparable entries: `3.12484`
- KL-like absolute sum: `38.9946`
- accept rate: `0.019727891156462583`
- accepted token count: `29`
- verify count: `98`

16-token buckets over comparable logprobs:

| tokens | max_abs | mean_abs | p95_abs |
|---:|---:|---:|---:|
| 000-015 | 5.70112 | 1.22756 | 4.45498 |
| 016-031 | 10.7216 | 4.94561 | 8.06951 |
| 032-047 | 7.44827 | 3.85426 | 6.71217 |
| 048-063 | 7.5184 | 2.46206 | 6.98555 |
| 064-079 | 7.77195 | 2.77309 | 7.72336 |
| 080-084 | 5.78547 | 4.28205 | 5.1051 |

Top token diffs:

| rank | token_index | abs_diff |
|---:|---:|---:|
| 1 | 24 | 10.7216 |
| 2 | 16 | 8.06951 |
| 3 | 74 | 7.77195 |
| 4 | 77 | 7.72336 |
| 5 | 55 | 7.5184 |

## 256-token Probe

The standalone 256-token probe generated all 256 tokens but returned only about
183 comparable logprob entries. Treat this as a second symptom rather than a
clean full-length KL curve.

- generated tokens: `256`
- first nonzero diff: token index `5`
- max absolute logprob diff over comparable entries: `17.6402`
- mean absolute logprob diff over comparable entries: `3.83509`
- KL-like absolute sum over comparable entries: `101.89`
- accept rate: `0.01467304625199362`
- accepted token count: `46`
- verify count: `209`
- largest observed diff: token index `73`, abs diff `17.6402`

## Interpretation

The failure is already visible inside the first DVR round:

- short prompt, 128-token run first diverges at output token `8`.
- long 256-token probe first diverges at output token `5`.
- acceptance is close to one token per verify after the first few steps.

This points to target-verify logits not being full-prefill-equivalent before
post-verify state repair can help. The next debugging step should instrument
the fixed-window GDN target verify path and compare, per layer:

- pre-conv hidden input
- conv output
- q/k/v/g/beta cache rows
- chunkwise GDN output
- post-GDN RMSNorm/gating output

The first useful breakpoint is around output token `5-8`, before long-range
state checkpointing becomes the dominant variable.
