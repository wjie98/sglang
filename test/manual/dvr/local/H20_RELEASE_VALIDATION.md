# DVR H20/NVLink release qualification

This guide drives the checked-in local scripts. It deliberately keeps machine
paths and release thresholds outside the upstream-facing DVR documentation.

## 1. Environment

Run from the repository root. Override paths for the machine under test:

```bash
export CONDA_ENV=dvr_dev
export MODEL_0P8B=/path/to/Qwen3.5-0.8B
export MODEL_35B=/path/to/Qwen3.5-35B-A3B
export MODEL_80B=/path/to/Qwen3-Next-80B-A3B-Instruct
export SHAREGPT_DATASET=/path/to/ShareGPT.json
export LONGBENCH_DATASET=/path/to/LongBench
export RESULT_ROOT=/path/to/dvr-results
```

Each script sets this worktree's `PYTHONPATH`, writes commit and GPU topology
metadata, and checks that the requested server configuration was actually
resolved. Keep `CUDA_VISIBLE_DEVICES`, TP size, backend, and result directory in
the final report. The scripts use `RANDOM_SEED=2026` for server startup; keep it
fixed together with the benchmark client seed.

## 2. Static gate

```bash
bash test/manual/dvr/local/scripts/run_static_unit_checks.sh
```

This must pass before allocating a multi-GPU model.

## 3. State-transition closure

The GPU matrix validates one target-state transaction. Do not interpret a
passing token-only request or an unchanged prompt checkpoint as sufficient
coverage.

### 3.1 State ownership

- Full-attention KV uses the ordinary request-to-token and Radix pools.
- The target live recurrent slot contains the state used by ordinary target
  execution. It is not a cross-request checkpoint by itself.
- With Radix enabled, the existing Mamba extra-buffer pool provides two
  physical checkpoint slots. DVR records which exact 64-token boundary each
  request-owned logical slot contains. Radix donation may replace a physical
  slot, so physical indices are resolved only after donation is processed.
- With Radix disabled, one request-local checkpoint is sufficient. It is
  discarded with the request; the next request must report no Radix reuse and
  perform an ordinary full prefill.
- Each GDN layer retains at most one unclosed 64-token tail plus the current
  draft rows as `q/k/v/g/beta`. Self-draft additionally saves request-owned
  convolution state. EAGLE/MTP owns a separate draft cache and does not write
  target state-input windows.

### 3.2 Closed transition

1. **Target EXTEND prepare:** clear stale logical ownership for reused request
   slots. Radix-enabled execution uses upstream Mamba tracking; Radix-disabled
   execution installs the request-local tracking slot.
2. **Target EXTEND state initialization:** after deferred Mamba COW resolves a
   warm Radix hit into the request's live slot, copy that exact boundary into
   DVR's request-owned checkpoint on the forward stream. This whole-pool copy
   runs eagerly before a prefill CUDA Graph; it must not be captured as
   data-dependent indexing inside each GDN layer. Target EXTEND then consumes
   the unclosed suffix and retains only the target inputs after the last
   64-token boundary.
3. **Target EXTEND finish:** publish exactly one authoritative boundary from an
   aligned live state, an upstream tracked state, the zero state, or the
   captured Radix prefix. Missing all four sources is a correctness error.
4. **First-decode lifecycle fence:** under overlap, process the unfinished
   prefill result before `get_next_batch_to_run()` constructs another batch.
   Result processing can either finish and release a request or donate its
   cached slot to Radix and install a replacement in the request/device mapping.
   Batch filtering, allocation, draft preparation, and verify must all observe
   that final request state.
5. **Draft:** select the newest logical boundary and resolve its current
   physical slot. Self-draft backs up convolution state before provisional fast
   decode; EAGLE/MTP advances only its separate draft state.
6. **Target verify:** restore the exact target boundary and self-draft
   convolution state, then run deterministic GDN EXTEND over `64 + D`. The
   Triton chunk kernel exports the state before each chunk, so `h[:, 1]` is the
   exact state after the first 64 rows. Returned outputs correspond only to the
   `D` logical verify rows.
7. **Rollback/commit:** commit convolution state at the last accepted input.
   If `tail + accepted >= 64`, write `h[:, 1]` to the alternate checkpoint and
   shift the state-input window by 64. Rebuild target live temporal state only
   for self-draft; EAGLE/MTP keeps target and draft state separate.
8. **Release/re-hit:** publish the newest exact checkpoint not later than the
   visible committed prefix. Radix truncates its stored token KV to the same
   length. A new request matches that aligned state and recomputes the remaining
   suffix through ordinary EXTEND. If output trimming leaves no retained
   checkpoint at or before the visible prefix, skip this insertion and retain
   normal inference semantics.

### 3.3 Required invariants

- Before every non-empty draft, each GDN request has an exact checkpoint.
- A finished request is never captured in a subsequent DVR decode batch; its
  prefill result is consumed exactly once before that batch is constructed.
- At phase boundaries, `boundary_length + tail_length` equals the target's
  committed state length.
- The boundary and `q/k/v/g/beta` window come from the same accepted target
  history; rejected rows never become authoritative.
- A physical checkpoint mapping cannot change between draft preparation and
  rollback.
- Warm-prefix boundary initialization uses the final physical source and
  destination slots after COW. It executes before both eager and graphed target
  EXTEND and covers every layer's temporal and convolution state.
- `D <= 64`, so one verify transaction crosses at most one recurrent chunk.
- `return_logprob=True` and `False` use the same state transition. Logprob
  materialization must not introduce another replay or commit path.
- Cache inability is a cache miss, not an inference fallback. It may reduce the
  next request's cached prefix but must not alter generated tokens or logits.

For a generated prefix of `P` prompt tokens and `G` generated tokens, the probe
request must normally report at least
`floor((P + G - 1) / 64) * 64` cached tokens. The `-1` follows SGLang's committed
KV convention for the sampled bonus token. Do not weaken this check to the old
prompt checkpoint: doing so hides a failure to publish generated boundaries.

The boundary-focused cases in `run_0p8b_self_dvr_kl.sh` cover first-decode
donation (`prompt=126/127`, `max_new=17/65`), request-slot reuse, interleaved
ownership, generated-prefix re-hit, stop/grammar trimming, and 512-token
generation. Run them with Radix enabled first, then repeat with
`DISABLE_RADIX_CACHE=1` to confirm full-prefill semantics.

The explicit prefill-result lifecycle case concurrently submits prompt lengths
`2/63/64/65` with `max_new_tokens=1`. Every request must finish during EXTEND;
none may survive into draft/verify with released KV or Mamba state.

### 3.4 Warm-prefix prefill graph regression

Boundary initialization and donation ownership are independent contracts. A
donation fence can resolve the correct physical slots while a captured
per-layer copy still initializes no state: the dummy prefill batch has zero
prefix lengths, so boolean indexing captures a zero-element operation whose
shape cannot become nonzero at replay time. Replacing the indexing with a
fixed-grid kernel is also insufficient unless every captured input has a
stable address.

The implementation therefore performs one eager whole-pool copy after deferred
COW and before dispatching either eager target EXTEND or its prefill graph. The
copy reads the current request-to-Mamba mapping, translates both endpoints to
physical slots, excludes rows owned by ordinary prefill tracking, and covers
all temporal and convolution state. No DVR boundary copy remains in the GDN
layer body.

Qualification must include cold and warm requests for prompt lengths
`63/64/65` and `126/127/129`, mixed batches where only some rows need the copy,
and generated-prefix re-hit. Compare temporal, convolution, and cached tail
state directly when diagnosing a failure. Disabling prefill graphs is a useful
control but is not an acceptable release workaround. Strict divergence at the
second generated token is characteristic of an uninitialized warm boundary;
`max_new_tokens=1` alone cannot validate this transition.

## 4. DeepGEMM preparation

On H20, compile the exact target-verify GEMM shapes before CUDA graph capture.
Use a cache dedicated to the source revision and driver/toolchain combination:

```bash
export SGLANG_DG_CACHE_DIR=/path/to/cache
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=1
DRAFT_TOKEN_COUNTS="2 16 32" \
bash test/manual/dvr/local/scripts/prepare_h20_deep_gemm.sh
```

For the formal run, retain the cache and set:

```bash
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export REQUIRE_PRECOMPILED_DEEPGEMM=1
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=0
```

Do not use a globally DeepGEMM-disabled run as the release performance number.
If formal serving enters another exhaustive precompile session, the cache is
incomplete and the result is invalid.

## 5. Correctness matrix

Run self-DVR first:

```bash
MODEL_PATH="${MODEL_0P8B}" \
RESULT_ROOT="${RESULT_ROOT}/0p8b-self" \
bash test/manual/dvr/local/scripts/run_0p8b_self_dvr_kl.sh

MODEL_PATH="${MODEL_0P8B}" \
DRAFT_TOKENS=32 \
RESULT_ROOT="${RESULT_ROOT}/0p8b-self-dvr32" \
bash test/manual/dvr/local/scripts/run_0p8b_self_dvr_kl.sh
```

Required coverage:

- synchronous and overlap workers;
- `return_logprob=True` and `False`;
- prompt and output lengths around 63/64/65;
- one-token prompt sentinel;
- concurrent shared-prefix requests and request-slot reuse;
- at least one 512-token generation;
- Radix enabled, plus a separately labelled Radix-disabled run.

For the Radix-enabled run, `dvr_radix_lifecycle.py` must confirm that generated
boundaries advance the reusable checkpoint. A result that only reuses the
original prompt boundary is a failure even when KL remains zero. For the
Radix-disabled run, do not expect any cross-request recurrent-state reuse.

Then run the Qwen3.5 MTP path:

```bash
MODEL_PATH="${MODEL_35B}" \
DRAFT_MODEL_PATH="${MODEL_35B}" \
RESULT_ROOT="${RESULT_ROOT}/35b-eagle" \
bash test/manual/dvr/local/scripts/run_35b_mtp_eagle_smoke.sh
```

Both sync and overlap must satisfy the strict full-prefill KL oracle. Compare
aggregate real-data acceptance instead of requiring identical stochastic token
trajectories across batch shapes or schedules.

## 6. Throughput matrix

Run matched normal, ordinary deterministic, and DVR configurations:

```bash
MODEL_PATH="${MODEL_35B}" \
RESULT_ROOT="${RESULT_ROOT}/35b-throughput" \
bash test/manual/dvr/local/scripts/run_35b_dvr_throughput.sh

MODEL_PATH="${MODEL_80B}" \
RESULT_ROOT="${RESULT_ROOT}/80b-throughput" \
bash test/manual/dvr/local/scripts/run_80b_self_dvr_throughput.sh
```

For each backend and logprob mode, compare only configurations with identical
TP, scheduling, Radix, page size, request count, output length, effective
concurrency, and cache policy. The scripts default to
`FLUSH_CACHE_EACH_RUN=1`: warmup compiles kernels, then the measured requests
start from an empty prefix cache. Cover Triton and FA3 where the model and
machine support them.

Run self-DVR with both chain widths. Keep separate result directories because
the target-verify graph and DeepGEMM shapes differ:

```bash
for draft_tokens in 16 32; do
  MODEL_PATH="${MODEL_80B}" \
  DRAFT_TOKENS="${draft_tokens}" \
  RESULT_ROOT="${RESULT_ROOT}/80b-dvr${draft_tokens}" \
  bash test/manual/dvr/local/scripts/run_80b_self_dvr_throughput.sh
done
```

`dvr16` remains the latency-oriented release configuration. `dvr32` is also a
diagnostic: if it recovers target efficiency while draft decode remains
unchanged, the remaining gap is fixed verify/state cost rather than an overlap
scheduler bubble. Report its acceptance and absolute throughput as well: a
longer draft chain can improve implementation efficiency while reducing both.

For a production-warm-cache measurement, run a second, clearly labelled matrix
with `FLUSH_CACHE_EACH_RUN=0`. Do not compare one policy's DVR result with the
other policy's baseline. Every JSON row contains `cache_report`; reject a cold
comparison if prompt/cache totals differ unexpectedly between modes.

Report:

```text
acceptance_fraction = accept_length / draft_tokens
target_tps = matching_normal_tps * acceptance_fraction
target_efficiency = dvr_tps / target_tps
det_speedup = dvr_tps / matching_ordinary_deterministic_tps
```

The NVLink target is `target_efficiency >= 0.95` after warmup. DVR must also be
faster than the matched ordinary deterministic baseline. Record lower results
rather than changing benchmark knobs in place.

## 7. Log audit

Before accepting a result, confirm from the server log that:

- the expected Triton or FA3 backend was selected;
- GDN linear-attention prefill uses Triton;
- requested CUDA graph batch sizes were captured without fallback;
- target prefill/verify remain deterministic;
- provisional draft capture uses the fast decode settings;
- custom all-reduce is confined to provisional draft graphs;
- self-draft logs `DVR self-draft FlashInfer all-reduce fusion backend:` with
  `auto`, `trtllm`, or the explicitly requested backend on an eligible H20/TP
  configuration. `disabled` is valid only when the user explicitly disabled
  fusion or the upstream normal-decode eligibility policy also rejects it;
- the server did not reduce `max_running_requests`;
- no illegal memory access, device assertion, or graph replay fallback occurred;
- formal serving did not enter an unplanned DeepGEMM precompile session.
- page-major recurrent-state storage, ReplaySSM, streaming sessions, and int8
  recurrent checkpoints were not enabled for GDN DVR.

Attach `run_metadata.txt`, `summary.txt`, server logs, script overrides, and the
exact commit to the qualification report.

## 8. Decode-only profiler

Profile already-running normal-sync, normal-overlap, DVR-sync, and DVR-overlap
servers independently. Use a unique output directory for every trace:

```bash
BASE_URL=http://127.0.0.1:30000 \
SERVER_LOG=/path/to/server.log \
PROFILE_BATCH_SIZE=2 \
RESULT_ROOT="${RESULT_ROOT}/profiles/dvr16-triton-overlap" \
bash test/manual/dvr/local/scripts/profile_dvr_server.sh
```

Repeat for Triton and FA3. The decisive checks are:

- self-draft time per draft step is close to normal non-deterministic overlap
  decode on the same backend;
- `unaccounted_gap_ms` is near zero;
- FA3 D2H copies and host `verify_prepare` are reported for context, but they
  are shared with upstream topk=1 speculative paths and are not a DVR-specific
  optimization target without a matched-path regression;
- reducing target-verify and state-maintenance time, not graph launch count,
  explains any dvr32 gain.

Use `PROFILE_BATCH_SIZE` values that match the throughput matrix. The helper
waits for asynchronous trace export and rejects stale traces from an earlier
run. For finite workloads, compare steady-state iteration periods separately
from pipeline fill/drain time.

Do not implement a whole-chain self-draft graph solely to remove launches. The
existing profiler reports `perfect_chain_speedup_ceiling`; prior A40 traces put
that ceiling near 1.015x.

## 9. Throughput diagnosis on NVLink

The original DVR reference used this break-even condition:

```text
normal_decode_time * step / accept_rate
  + deterministic_prefill_verify_time
  < deterministic_decode_time * step
```

For the stronger release target, measure one complete proposal block. With
`D` proposed tokens and `D - 1` self-draft graph replays:

```text
T_dvr = (D - 1) * T_draft + T_verify + T_maintenance
target_efficiency = D * T_normal_decode / T_dvr
```

Acceptance cancels from `target_efficiency`. At `D=16` and
`T_draft=T_normal_decode`, reaching `target_efficiency >= 0.95` leaves only
`1.84 * T_normal_decode` for deterministic verify, state maintenance, sampling,
rollback, and scheduling combined.

### 9.1 Current evidence

A matched A40, TP1, Triton trace measured:

```text
normal overlap decode       3.574 ms / step
self-draft                 53.212 ms / block = 3.547 ms / step
deterministic target verify 15.779 ms / block
other DVR maintenance       about 0.9 ms / block
```

The draft kernels and per-step latency match ordinary non-deterministic decode;
target verify is about 4.4 decode steps and dominates the remaining gap. The
measured stage times predict target efficiency near 0.82, consistent with the
observed result. Greedy throughput runs use `temperature=0`; their verify
sampling stage was about 0.058 ms per block and does not enter stochastic
rejection sampling.

H20/TP adds two machine-specific checks that A40/TP1 cannot cover:

- Latest upstream can auto-enable FlashInfer all-reduce plus residual/RMSNorm
  fusion for Qwen3Next/Qwen3.5 on SM90. Deterministic target initialization
  disables it globally. DVR preserves the user's pre-deterministic request,
  resolves the same upstream auto policy after model/topology defaults settle,
  pre-initializes the FlashInfer workspaces before graph capture, and captures
  fusion only into the provisional self-draft graph. This communication fusion
  is independent of the Triton/FA3 attention backend. Compare fused kernel and
  separate all-reduce/norm counts between normal decode and the draft graph;
  deterministic target prefill/verify must still use the unfused path.
- A globally disabled batch-invariant DeepGEMM disproportionately slows the
  `bs * D` deterministic verify GEMMs. Formal results require the exact verify
  shapes to be precompiled as described in section 4.

Also verify that custom all-reduce really appears in the captured draft graph.
Unsupported messages may otherwise fall back to the deterministic process's
NCCL tree configuration.

A matched A40 dvr32, batch-size-2 profile also separates overlap scheduling
from model work:

```text
                         sync v1       overlap v2
iteration period         136.961 ms     134.158 ms
self-draft               120.409 ms     119.107 ms
target verify             13.626 ms      13.617 ms
maintenance                1.405 ms       1.399 ms
unaccounted gap             1.521 ms       0.034 ms
```

The v2 result has effectively no scheduler bubble and is slightly faster than
v1. Its draft graph is 99.1% GPU-busy; even a perfect whole-chain graph has only
about a 1.009x ceiling. Treat a large H20 deficit as scheduling only when the
new trace reports a material `unaccounted_gap_ms`. Otherwise attribute it to
`r_draft`, deterministic verify, or measured state maintenance.

### 9.2 Stage ratios

For every matched normal/DVR profile, report:

```text
r_draft  = draft_block_ms / (D - 1) / normal_decode_ms
r_verify = target_verify_ms / normal_decode_ms
r_maint  = maintenance_ms / normal_decode_ms
predicted_efficiency = D / ((D - 1) * r_draft + r_verify + r_maint)
```

Interpret the result as follows:

- `r_draft > 1.03`: a normal decode optimization or collective is missing from
  draft capture.
- `r_draft` near one with high `r_verify`: inspect DeepGEMM, deterministic
  attention, and verify GEMM shapes.
- `r_maint > 0.5`: inspect state copy/commit, sampling, and scheduler spans.
- dvr32 recovering efficiency while `r_draft` stays unchanged indicates fixed
  verify cost rather than a draft graph launch problem.

### 9.3 H20 profiler matrix

Use identical model, TP, backend, batch size, prompt, output length, and cache
policy for these runs:

1. normal overlap;
2. self-DVR v2 with dvr16;
3. self-DVR v2 with dvr32;
4. normal overlap with FlashInfer all-reduce fusion forcibly disabled, as a
   diagnostic only;
5. DVR with custom all-reduce enabled and disabled;
6. DeepGEMM precompiled/enabled versus a separately labelled diagnostic run.

After the stage profiler localizes the gap, capture one steady-state request
with Nsight Systems and summarize it with:

```bash
nsys stats \
  --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum,cuda_api_sum \
  /path/to/report.nsys-rep
```

Inspect FlashInfer fused all-reduce/RMSNorm kernels, custom-all-reduce versus
NCCL kernels, standalone RMSNorm counts, DeepGEMM versus
`matmul_kernel_persistent` in target verify, attention split counts, and GPU
gaps between the 15 draft graph replays. Profile stochastic rejection sampling
separately with `temperature > 0`; it cannot explain a greedy benchmark gap.

For the fusion A/B, first confirm that the normal run and DVR self-draft select
the same FlashInfer communication backend. The diagnostic normal-fusion-off run
should remove any draft/normal difference caused by this feature. If patched
DVR still has `r_draft > 1.03`, inspect residual NCCL fallback and custom-AR
coverage before changing scheduler or state code.
