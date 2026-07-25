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
- The target request's live Mamba slot owns two deliberate timestamps:
  temporal recurrent state is the latest exact 64-token boundary, while
  convolution state is the accepted endpoint.
- Existing Mamba ping-pong slots remain ordinary Radix publication buffers.
  DVR records the logical boundary in each lane but never uses those lanes as
  target-verify inputs or changes their upstream ownership rules.
- With Radix disabled, publication metadata is absent. The active request
  still keeps its live boundary and private DVR state; a later request performs
  an ordinary full prefill.
- Each GDN layer retains at most one unclosed 64-token tail plus the current
  draft rows as `k/v/g/beta` transition inputs; candidate `q` is supplied
  directly to verify. Self-draft additionally owns one request-indexed
  recurrent workspace and private convolution state. EAGLE/MTP owns a separate
  upstream draft cache and does not write target state-input windows.

### 3.2 Closed transition

1. **Target EXTEND prepare:** clear stale logical ownership for reused request
   rows. A GDN warm prefix must be chunk-aligned; otherwise the request
   recomputes from the nearest ordinary compatible prefix.
2. **Target EXTEND:** write the latest exact recurrent boundary to the target
   live temporal slot, leave target convolution state at the accepted endpoint,
   cache only post-boundary `k/v/g/beta`, and seed private self-draft state. If
   EXTEND creates a new exact boundary, copy the matching temporal and
   convolution checkpoint to an ordinary Radix tracking lane.
3. **Draft:** self-draft advances only its private recurrent and convolution
   state. EAGLE/MTP advances only its upstream-owned draft cache.
4. **Target verify:** read the target live boundary without overwriting it and
   run the dedicated deterministic DVR GDN operator over the fixed `64 + D`
   physical window. Return only the `D` logical rows and stage the possible
   exact `h64` in the now-idle self-draft workspace.
5. **Rollback/commit:** commit convolution state at the last accepted input.
   If `tail + accepted >= 64`, publish staged `h64` into the target live slot
   and selected Radix lane, then compact the state-input window by 64. Rebuild
   private self-draft state from the new boundary plus accepted tail;
   EAGLE/MTP skips this target reconstruction.
6. **Overlap publication:** all state writes are enqueued before the worker
   returns its FutureMap. No accepted length or state metadata is read back to
   the host. Only request release waits on the existing rollback event.
7. **Release/re-hit:** select the newest recorded checkpoint not later than the
   visible committed prefix. Radix truncates its stored token KV to the same
   length. A new request matches that aligned state and recomputes the remaining
   suffix through ordinary EXTEND. If output trimming leaves no retained
   checkpoint at or before the visible prefix, skip this insertion and retain
   normal inference semantics.

### 3.3 Required invariants

- Before every non-empty draft, each GDN request has an exact checkpoint.
- Result processing may finish a request after overlap has constructed one
  extra batch. Existing WAR/result-lifetime fencing must protect its request
  slots until that forward no longer reads them.
- At phase boundaries, `boundary_length + tail_length` equals the target's
  committed state length.
- The boundary and `k/v/g/beta` window come from the same accepted target
  history; rejected rows never become authoritative.
- The target live temporal slot is the only target-verify boundary source.
  Radix lane rebinding cannot change it.
- Every published boundary contains matching temporal and convolution state at
  the same exact 64-token position.
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
Radix copy (`prompt=126/127`, `max_new=17/65`), request-slot reuse, interleaved
ownership, generated-prefix re-hit, stop/grammar trimming, and 512-token
generation. Run them with Radix enabled first, then repeat with
`DISABLE_RADIX_CACHE=1` to confirm full-prefill semantics.

The explicit prefill-result lifecycle case concurrently submits prompt lengths
`2/63/64/65` with `max_new_tokens=1`. Every request must finish during EXTEND;
none may survive into draft/verify with released KV or Mamba state.

### 3.4 Warm-prefix prefill graph regression

Warm-prefix state comes through the ordinary target EXTEND path. DVR does not
capture a data-dependent whole-pool COW or alter Radix slot donation. The
request's target live slot is established by normal pool mapping before DVR
records its logical boundary and transition tail.

Qualification must include cold and warm requests for prompt lengths
`63/64/65` and `126/127/129`, mixed batches where only some rows need the copy,
and generated-prefix re-hit. Compare temporal, convolution, and cached tail
state directly when diagnosing a failure. Disabling prefill graphs or Radix is
a useful control but is not an acceptable release workaround.

### 3.5 Isolated self-draft state lifecycle

The implemented state lifecycle is specified in `STATE_LIFECYCLE_REDESIGN.md`.
Do not qualify it by observing only unchanged logits: the intended performance
result depends on removing the old target-state restore transaction, not
reproducing it behind different names.

One self-DVR block has these state boundaries:

1. Target EXTEND seeds one request-indexed recurrent workspace and private
   convolution state. This full-state initialization does not repeat in
   steady-state DVR blocks. DVR allocates one workspace state per layer/request,
   independent of `D`, not one state per proposal token.
2. Run all provisional recurrent decode steps only on that workspace.
3. Run the dedicated DVR GDN verify read-only from the target live 64-token
   boundary over the fixed `64 + D` transition-input window, then reuse the
   same workspace for exported `h64` after draft has finished.
4. Commit accepted convolution state for both self and EAGLE.
5. Rebuild the private self-draft recurrent state from the newest exact boundary plus
   accepted `k/v/g/beta`, with at most 63 rows. EAGLE skips this target-
   temporal operation and finalizes its own cache through the upstream worker.
6. Publish and compact only when accepted history crosses 64 tokens.

Add these NVTX ranges to the existing profile report:

```text
draft_state_copy
target_verify
verify_state_pack
verify_boundary_stage
verify_output_gather
draft_state_rebuild
state_window_compact
boundary_publish
```

The old `dvr_state_restore` range must disappear. `draft_state_rebuild` must
never process more than 63 transition rows, and non-crossing requests must not
move a full `64 + D` state-input window. All ranges remain on the forward
stream without accept-length or tail-length host readback.

The migration-specific correctness matrix must force:

- `tail_len=0/1/48/63`;
- `accept_len=1/D-1/D`;
- crossing and non-crossing requests in the same batch;
- prompt and generated-prefix boundaries at `1/63/64/65`;
- cold prefill, warm Radix re-hit, Radix checkpoint copy, request-slot reuse,
  and Radix off;
- self-DVR v1/v2 and EAGLE sync/overlap; and
- `return_logprob=True/False` with 512- and 1024-token generations.

For self-draft, compare the rebuilt private draft state before the next proposal block
as well as target checkpoint and logits. For EAGLE, compare target checkpoint,
accepted target hidden state, draft-cache finalization, proposal probabilities,
and `accept_index`. A real-data EAGLE acceptance result of exactly 1.0 is not a
pass until proposal and metric accounting are independently validated.

On the 35B H20 BS3 profile, report `draft_state_copy` only for target EXTEND;
it must not appear once per steady-state decode block. Report
`verify_state_pack`, `boundary_state_write`, `verify_output_gather`,
`draft_state_rebuild`, `state_window_compact`, and `boundary_publish`
separately. The old reference is 2.351 ms/block, including a 0.843 ms restore
that must disappear. The pre-optimization verify state-I/O trace attributed
about 1.764 ms/block and 570 launches to per-layer metadata, five window writes,
five full-window reads, h64 staging, and output gather. The current path resolves
metadata once per verify, uses three pack launches per GDN layer, writes the
selected FP32 chunk accumulator from the DVR-specific recurrent producer, and
directly gathers the logical outputs. Establish the new threshold from
measurement rather than assuming those launches or the 64-row-bounded rebuild
are free.

The direct producer side output is now part of the test snapshot and must be
profiled rather than reimplemented. If state packing remains material, evaluate
a segmented DVR verify separately. If rebuild remains material, tune its
existing private kernel against the observed tail-length histogram before
adding multiple graph variants. Neither decision may add tail-length D2H, CPU
branching, or assumptions about Radix being enabled.

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

### 9.4 State-I/O optimization recheck

For the first profile after the verify state-I/O change, use Qwen3.5-35B,
Triton, TP1 or the previously measured topology, BS3, DVR16, and the same prompt
and output corpus as the old trace. Report both per-block time and launch count
for:

```text
verify_state_pack
boundary_state_write
verify_output_gather
draft_state_rebuild
state_window_compact
boundary_publish
```

Also report the parent `target_verify` and full iteration period. Confirm that
there are three state pack launches per GDN layer rather than ten generic
advanced-index operations, and that padded graph lanes do not write dummy slot
zero. Run BS1 and BS3 because reducing launch count may help small batches while
state bandwidth dominates larger ones.

The boundary output belongs to the DVR-specific recurrent producer; verify that
ordinary FLA/GDN calls never enter it. Only proceed to segmented GDN verify if
`verify_state_pack` plus unused-output compute remains material and a prototype
preserves exact outputs and h64 for every tail length `0..63`. Only tune or
bucket the private rebuild kernel if `draft_state_rebuild` remains a material
fraction of the full block. This ordering keeps further optimization optional
and prevents a machine-specific result from narrowing ordinary backend
behavior.
