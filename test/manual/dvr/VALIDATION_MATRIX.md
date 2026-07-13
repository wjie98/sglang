# DVR validation matrix

This file records the fixed validation protocol used while polishing the DVR
integration.  The tests are intentionally split into server launch commands and
small clients, because the 35B and 80B cases are machine dependent and should
reuse SGLang's normal server and benchmark tooling.

## Environment

Use the development environment and model paths below unless a machine-specific
note says otherwise.

```bash
conda activate dvr_dev
export PYTHONPATH=python
export SGLANG_RETURN_ORIGINAL_LOGPROB=True
```

Models:

```text
/mnt/data/hwj/Qwen3.5-0.8B
/mnt/data/hwj/Qwen3.5-35B-A3B
/mnt/data/hwj/Qwen3-Next-80B-A3B-Instruct
```

Datasets:

```text
/mnt/data/hwj/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json
/mnt/data/hwj/LongBench-v2/data.json
```

## Static checks

Run these before and after each code batch.

```bash
test/manual/dvr/scripts/run_static_unit_checks.sh
```

## 0.8B self-DVR KL smoke

Run both spec-v1 compatibility mode and spec-v2 overlap mode.  The expected
result is `ALL_OK True` with `maxdiff=0.0` and `kl_proxy=0.0`.

Fixed script:

```bash
test/manual/dvr/scripts/run_0p8b_self_dvr_kl.sh
```

Server:

```bash
SGLANG_RETURN_ORIGINAL_LOGPROB=True \
PYTHONPATH=python conda run --no-capture-output -n dvr_dev python -m sglang.launch_server \
  --model-path /mnt/data/hwj/Qwen3.5-0.8B \
  --host 127.0.0.1 --port 30124 \
  --speculative-algorithm DECODE_VERIFY_ROLLBACK \
  --speculative-num-draft-tokens 16 \
  --page-size 1 \
  --mem-fraction-static 0.45 \
  --attention-backend triton \
  --linear-attn-backend triton \
  --sampling-backend pytorch \
  --enable-deterministic-inference \
  --cuda-graph-bs 1 2 4 \
  --cuda-graph-max-bs 4 \
  --max-running-requests 8 \
  --skip-server-warmup
```

Add `--disable-overlap-schedule` for the self-DVR v1 compatibility worker.
Omit it for the spec-v2 overlap worker.

Client:

```bash
PYTHONPATH=python conda run --no-capture-output -n dvr_dev python \
  test/manual/dvr/test_dvr_batch_kl.py \
  --base-url http://127.0.0.1:30124 \
  --request-modes concurrent,batch \
  --prompt-token-lengths 2,63,64,65 \
  --max-new 1,8,16,17,63,64,65 \
  --limit-cases 12 \
  --concurrent-workers 4 \
  --ignore-eos
```

The fixed script also runs concurrent and batch `prompt_len=65`, `max_new=512`
cases. Add a separate fixed case for 513 tokens when a change specifically
touches end-of-chunk termination behavior.

One-token synthetic prompts are intentionally not part of the positive DVR
smoke matrix. The first self-draft graph step reaches the internal
`seq_len<=2` GDN state-input boundary for that case, so DVR now rejects it
explicitly instead of running a slow eager path. Normal chat-template prompts
are much longer and do not hit this edge.

## 35B self-DVR and DVR-EAGLE

Use the 35B model for MTP/EAGLE first.  Qwen3.5 35B has MTP weights and is the
smallest local model that exercises the target plus MTP path.

Fixed EAGLE/MTP smoke script:

```bash
test/manual/dvr/scripts/run_35b_mtp_eagle_smoke.sh
```

Fixed no-DVR/self-DVR/DVR-EAGLE throughput comparison:

```bash
test/manual/dvr/scripts/run_35b_dvr_throughput.sh
```

It uses 8 ShareGPT requests, 512 generated tokens, concurrency 4, and runs both
returned-logprob modes.  Self-v1/v2 also run the TP=4 boundary KL client before
the benchmark.  EAGLE correctness and per-prompt acceptance remain the
responsibility of `run_35b_mtp_eagle_smoke.sh`; the throughput script measures
the five matching server modes without duplicating that matrix.

Self-DVR server:

```bash
SGLANG_RETURN_ORIGINAL_LOGPROB=True \
PYTHONPATH=python conda run --no-capture-output -n dvr_dev python -m sglang.launch_server \
  --model-path /mnt/data/hwj/Qwen3.5-35B-A3B \
  --host 127.0.0.1 --port 30135 \
  --tp-size 4 \
  --speculative-algorithm DECODE_VERIFY_ROLLBACK \
  --speculative-num-draft-tokens 16 \
  --page-size 1 \
  --context-length 4096 \
  --max-total-tokens 8192 \
  --mem-fraction-static 0.72 \
  --attention-backend triton \
  --linear-attn-backend triton \
  --sampling-backend pytorch \
  --enable-deterministic-inference \
  --cuda-graph-bs 1 2 4 \
  --cuda-graph-max-bs 4 \
  --max-running-requests 4 \
  --skip-server-warmup
```

Add `--disable-overlap-schedule` for the self-DVR v1 compatibility worker.
Omit it for the spec-v2 overlap worker.

DVR-EAGLE server:

```bash
SGLANG_RETURN_ORIGINAL_LOGPROB=True \
PYTHONPATH=python conda run --no-capture-output -n dvr_dev python -m sglang.launch_server \
  --model-path /mnt/data/hwj/Qwen3.5-35B-A3B \
  --host 127.0.0.1 --port 30135 \
  --tp-size 4 \
  --speculative-algorithm DECODE_VERIFY_ROLLBACK_EAGLE \
  --speculative-draft-model-path /mnt/data/hwj/Qwen3.5-35B-A3B \
  --speculative-num-draft-tokens 2 \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --page-size 1 \
  --context-length 4096 \
  --max-total-tokens 8192 \
  --mem-fraction-static 0.72 \
  --attention-backend triton \
  --linear-attn-backend triton \
  --sampling-backend pytorch \
  --enable-deterministic-inference \
  --cuda-graph-bs 1 2 4 \
  --cuda-graph-max-bs 4 \
  --max-running-requests 4 \
  --skip-server-warmup
```

EAGLE client, returned-logprob path:

```bash
PYTHONPATH=python conda run --no-capture-output -n dvr_dev python \
  test/manual/dvr/test_dvr_eagle_acceptance.py \
  --base-url http://127.0.0.1:30135 \
  --prompt-token-lengths 63,64,65 \
  --max-new 4,16,65 \
  --cache-mode flush-each \
  --check-kl \
  --min-accept-rate 0.96 \
  --ignore-eos \
  --seed 2032
```

EAGLE client, no returned-logprob output path:

```bash
PYTHONPATH=python conda run --no-capture-output -n dvr_dev python \
  test/manual/dvr/test_dvr_eagle_acceptance.py \
  --base-url http://127.0.0.1:30135 \
  --prompt-token-lengths 63,64,65 \
  --max-new 4,16,65 \
  --cache-mode flush-each \
  --no-return-logprob \
  --min-accept-rate 0.96 \
  --ignore-eos \
  --seed 2032
```

The sync and overlap DVR-EAGLE modes are both spec-v2 semantics.  In this tree
`--disable-overlap-schedule` selects synchronous v2 execution for DVR-EAGLE, and
the default overlap scheduler selects overlap v2 execution.  DVR-EAGLE v1 is not
a supported matrix entry.

The `--seed 2032`, `prompt_len=65`, `max_new=65` entry is the fixed regression
for seeded MTP boundary acceptance.  It crosses the GDN chunk boundary and must
keep `accept_rate >= 0.96` in both returned-logprob and no-return-logprob
paths.  A lower value usually means the deterministic verify coin stream or the
MTP suffix-boundary hidden/state seed no longer matches target prefill
semantics.

The fixed ShareGPT cases use `accept_rate >= 0.70`.  Their historical range is
roughly `0.73-1.00`; using the synthetic-boundary threshold for real prompts
would turn normal MTP model quality into a false correctness failure.

## 80B self-DVR throughput

Use Qwen3-Next 80B for long-output self-DVR throughput.  The fixed script is
the source of truth for this matrix:

```bash
test/manual/dvr/scripts/run_80b_self_dvr_throughput.sh
```

The reproduced口径 is 16 requests, 1024 generated tokens, ShareGPT
`max_concurrency=3`, and fixed LongBench custom-cache input with
`max_concurrency=2`.  The server command pins `--max-mamba-cache-size 16`; do
not omit it when comparing against the reference numbers.  The default script
run includes the matching normal no-DVR baseline before DVR v1 and v2.

Compare every DVR run with a no-DVR baseline launched with the same backend, TP
size, context length, request rate, request count, and output length.  Report:

```text
effective_dvr_throughput = output_throughput * accept_rate
dvr_ratio = effective_dvr_throughput / no_dvr_output_throughput
```

For every throughput run, keep the raw benchmark JSONL and server log.  The
server log is the source of DVR accept rate and verifies whether draft decode
uses CUDA graph and non-deterministic decode performance knobs.

## Regression note: returned logprobs

`return_logprob=True` is part of the deterministic-inference validation surface,
not an optional slow path that can be ignored for throughput.  It must not change
the DVR state lifecycle or the draft input state used by the next iteration.

For self-DVR, returned logprobs must stay side-effect-free: keep live GDN state
commit on the same fast self-draft path as `return_logprob=False`, and populate
`next_token_logprobs` from the verifier logits before normal result processing.
Do not use output-layer final repair or accepted-suffix replay as the self-DVR
commit path just because returned logprobs are requested; that changes the next
draft state, lowers the acceptance rate, and cuts long-output throughput.

The fixed 80B ShareGPT 16x1024 reference results are:

```text
spec-v1 return_logprob=True: 152.62 tok/s, accept length 14.58
spec-v2 return_logprob=True: 154.08 tok/s, accept length 14.75
spec-v1 return_logprob=False: 159.91 tok/s, accept length 14.58
spec-v2 return_logprob=False: 159.65 tok/s, accept length 14.68
```

The expected overhead of returned logprobs is only the scoring/output overhead;
the accept length should remain aligned with the no-logprob run.
