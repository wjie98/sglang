# DVR V3 Baseline Results

Baseline date: 2026-05-13

Branch:

- `dvr-clean-v3`
- upstream base when tested: `a929eb728`

Purpose:

- Establish the non-DVR deterministic inference baseline before implementing
  DVR.
- Compare normal generation returned logprobs against full-prefill scoring
  oracle.

## Server Command

No DVR was enabled. Only upstream deterministic inference was enabled.

```bash
SGLANG_RETURN_ORIGINAL_LOGPROB=True \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 \
PYTHONPATH=python \
conda run --no-capture-output -n dvr_dev python -m sglang.launch_server \
  --model-path /home/hwj/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 30124 \
  --page-size 1 \
  --mem-fraction-static 0.45 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --enable-deterministic-inference \
  --skip-server-warmup
```

Runtime notes:

- Server selected pytorch sampling because deterministic inference was enabled.
- DeepGEMM was disabled for local 5090 stability.
- CUDA graph was enabled for normal decode.
- Piecewise CUDA graph was disabled by default in this launch.

## Oracle Method

1. Generate with non-greedy sampling:
   - `temperature=0.7`
   - `top_p=0.9`
   - `top_k=-1`
   - fixed `sampling_seed`
2. Request `return_logprob=True`, `logprob_start_len=0`.
3. Concatenate `prompt_ids + output_ids`.
4. Run `/generate` with `max_new_tokens=0`.
5. Compare generation `output_token_logprobs` with the output tail of oracle
   `input_token_logprobs`.

## Results

Qwen3-0.6B, no DVR, deterministic inference:

```text
bs=1 max_new=1   generated=1   max_abs_diff=0.0                 kl_abs_sum=0.0
bs=1 max_new=8   generated=8   max_abs_diff=0.0                 kl_abs_sum=0.0
bs=1 max_new=16  generated=16  max_abs_diff=0.0                 kl_abs_sum=0.0
bs=1 max_new=20  generated=20  max_abs_diff=0.0                 kl_abs_sum=0.0
bs=1 max_new=63  generated=63  max_abs_diff=0.08078992366790771 kl_abs_sum=0.31039525696242837
bs=1 max_new=64  generated=64  max_abs_diff=0.08078992366790771 kl_abs_sum=0.31345841661820095
bs=1 max_new=80  generated=80  max_abs_diff=0.08078992366790771 kl_abs_sum=0.33227394685215583
bs=1 max_new=128 generated=128 max_abs_diff=0.1059185266494751  kl_abs_sum=0.46962405937296603
bs=2 idx=0 max_new=16 generated=16 max_abs_diff=0.0             kl_abs_sum=0.0
bs=2 idx=1 max_new=20 generated=20 max_abs_diff=0.0             kl_abs_sum=0.0
bs=2 idx=0 max_new=80 generated=80 max_abs_diff=0.08078992366790771 kl_abs_sum=0.33227394685215583
bs=2 idx=1 max_new=64 generated=64 max_abs_diff=0.14081013202667236 kl_abs_sum=0.3408170256885948
```

Additional check for the `bs=1, max_new=80` sequence:

```text
repeat full-prefill scoring maxdiff: 0.0
first decode-vs-prefill difference:
  output token index: 24
  token id: 624
  decode logprob: -2.8694844245910645
  prefill logprob: -2.8349967002868652
  diff: -0.03448772430419922
max decode-vs-prefill diff: 0.08078992366790771
```

## Interpretation

The full-prefill oracle is stable: repeated full-prefill scoring of the same
sequence produced strict equality.

The non-DVR deterministic decode path is strict-equal to full prefill for short
generations in this baseline, but diverges for longer generations. This makes
the DVR objective concrete even for the pure-attention baseline:

- generation returned logprobs should be verify/prefill-derived;
- strict KL=0 should be tested against the full-prefill oracle;
- a DVR implementation should eliminate the long-generation decode-vs-prefill
  drift shown above.

