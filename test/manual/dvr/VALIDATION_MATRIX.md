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
  --page-size 64 \
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

It uses 8 ShareGPT requests, 512 generated tokens, concurrency 3, and runs both
returned-logprob modes.  Self-v1/v2 also run the TP=4 boundary KL client before
the benchmark.  EAGLE correctness and per-prompt acceptance remain the
responsibility of `run_35b_mtp_eagle_smoke.sh`; the throughput script measures
the six matching server modes without duplicating that matrix.

Self-DVR server:

```bash
SGLANG_RETURN_ORIGINAL_LOGPROB=True \
PYTHONPATH=python conda run --no-capture-output -n dvr_dev python -m sglang.launch_server \
  --model-path /mnt/data/hwj/Qwen3.5-35B-A3B \
  --host 127.0.0.1 --port 30135 \
  --tp-size 4 \
  --speculative-algorithm DECODE_VERIFY_ROLLBACK \
  --speculative-num-draft-tokens 16 \
  --page-size 64 \
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
  --page-size 64 \
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

The reproduced setup is 16 requests, 1024 generated tokens, ShareGPT
`max_concurrency=3`, and fixed LongBench custom-cache input with
`max_concurrency=2`.  The server command pins `--max-mamba-cache-size 16`; do
not omit it when comparing against the reference numbers.  The default script
run includes the matching normal no-DVR baseline before DVR v1 and v2.

Compare every DVR run with a no-DVR baseline launched with the same scheduler
mode, page size, backend, radix setting, TP size, context length, request rate,
request count, output length, and returned-logprob mode. Report:

```text
acceptance_fraction = accept_length / speculative_num_draft_tokens
target_tps = matching_baseline_tps * acceptance_fraction
dvr_ratio = dvr_output_throughput / matching_baseline_tps
target_efficiency = dvr_output_throughput / target_tps
```

The DVR output throughput already includes accepted-token acceleration; do not
multiply it by acceptance again. For every throughput run, keep the raw
benchmark JSONL and server log. The
server log is the source of DVR accept rate and verifies whether draft decode
uses CUDA graph and non-deterministic decode performance knobs.

## NVLink and attention-backend qualification

The H20 qualification matrix runs full-attention backends `triton`, `fa3`, and
`flashinfer`; GDN linear-attention prefill remains Triton because exact chunk
boundary export is a separate requirement. FlashInfer runs both baseline and
DVR with radix disabled. No DVR backend-specific cache workaround is allowed.

Run 35B at batch sizes 1, 4, and 8 to expose launch-bound and throughput-bound
behavior. Run 80B ShareGPT and LongBench with at least 1024 generated tokens.
For each backend keep separate sync and overlap baselines and both
`return_logprob` modes. Record `nvidia-smi topo -m`; a PCIe-only run is not an
NVLink custom-all-reduce result.

The release goal is `target_efficiency >= 0.95` on H20/NVLink without reducing
accepted length or server concurrency. Current A40 results are diagnostic only:
custom all-reduce is expected to remain disabled on four PCIe-only GPUs.

## Self-draft chain CUDA graph gate

Use `scripts/profile_dvr_server.sh` before replacing the per-step self-draft
graph with a graph that captures the entire decode/sampling chain. The script
reports the number of graph launches per draft iteration, GPU kernel busy
fraction, and a conservative perfect-chain speedup ceiling. A chain graph is
eligible only if a prototype preserves all sampling modes and Triton/FA3/
FlashInfer plus Hybrid/GDN metadata semantics, does not reduce token-pool or
request capacity, and demonstrates a stable end-to-end gain of at least 5%.

The A40 0.8B measurement on 2026-07-14 rejected this optimization. The current
15-step path launched 15 graphs per iteration, but draft GPU kernel utilization
was already about 96.4%, limiting an ideal chain to roughly 1.04x before its own
overhead. A capture-only probe over 15 consecutive full target forwards grew
the self-draft graph from about 0.10 GB to 0.26 GB for capture batches
`[1,2,4,8]`. That probe did not yet include capture-safe sampling or correct
multi-step Hybrid/GDN metadata, so a production implementation could only add
cost and backend-specific complexity. Keep the existing one-step graph unless
an H20/NVLink profile crosses the gate; do not carry an unmeasured chain graph
or an eager fallback in production.

The same trace bounds the current one-per-chain draft performance context at
about 78 microseconds to enter and 37 microseconds for exit plus intervening
verify glue. This is below 0.2% of an iteration. Do not replace it with a
separately allocated full-attention/Hybrid backend merely to avoid temporary
field restoration: that would duplicate CUDA graph workspaces and complicate
GDN adapter ownership for a smaller gain. Reconsider only if
`draft_context_gate` is materially larger on the target server.

## Self-draft state rebuild and synchronization gate

The A40 profile also justified one narrower GDN optimization. Rebuilding the
accepted self-draft state directly from request-local state-window slots avoids
materializing per-layer gathered inputs and a separate final-state tensor. On
the fixed 0.8B trace, rollback CPU time fell from about 1.86 ms to 1.57 ms and
rollback GPU time from about 0.70 ms to 0.41 ms. Keep this fused write because
it reduces real work without changing the draft/verify/rollback contract.

Do not remove the GPU-length resolution at the start of target state restore
merely because self draft already has a host length mirror. CUDA graph replay
returns before the 15 draft forwards finish on the device; the measured restore
call spent about 36 ms waiting for that remaining draft work. Reusing the host
mirror removed the wait but reproducibly allowed target restore and pool
reclamation to race unfinished draft graph writes, ending in device-side index
assertions. The wait overlaps useful draft GPU work and is not 36 ms of idle
device time. Revisit it only together with an explicit, backend-independent
graph completion/ownership contract and an end-to-end throughput gain.

## Development and release tests

The full 0.8B/35B/80B matrix is a development qualification suite and may rely
on local model and dataset paths supplied through environment variables. The
release patch should retain only parameterized DVR unit contracts, one compact
gated-linear integration smoke, one reusable server launcher, and one KL/
acceptance client. Do not copy benchmark scheduling or throughput calculations
into Python clients; invoke SGLang's `bench_serving` and keep its raw JSONL.

DVR deliberately rejects one-token synthetic prompts on gated linear-state
models. Upstream's generation-based `/health` probe uses `[0]` as its prompt,
so DVR deployments should set `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false`
and use the resulting non-generating `/health` endpoint. The fixed scripts poll
`/v1/models` for readiness and must not use `/health_generate`.

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

The fixed A40 PCIe 80B reference results from 2026-07-14 are below. These use
16 requests with 1024 generated tokens each; ShareGPT uses concurrency 3 and
LongBench uses concurrency 2. Compare H20/NVLink runs by configuration rather
than treating these PCIe numbers as a release target.

```text
configuration                         ShareGPT tok/s   LongBench tok/s   accept length
sync baseline, return_logprob=False        209.10           172.46          n/a
sync baseline, return_logprob=True         208.36           171.10          n/a
overlap baseline, return_logprob=False     226.13           187.04          n/a
overlap baseline, return_logprob=True      225.19           186.04          n/a
spec-v1, return_logprob=False              161.54           129.67       14.88/14.82
spec-v1, return_logprob=True               161.15           129.42       14.87/14.78
spec-v2, return_logprob=False              160.10           128.74       14.89/14.80
spec-v2, return_logprob=True               159.45           128.49       14.84/14.80
```

The expected overhead of returned logprobs is only the scoring/output overhead;
the accept length should remain aligned with the no-logprob run.
