# DVR H20/NVLink Release Qualification

This document is the final qualification procedure for the DVR development
branch before preparing a clean release branch. Run the checked-in scripts
without changing their fixed benchmark parameters. Put every intentional
parameter variation in a separate result directory.

The qualification covers:

- normal no-DVR inference
- self-DVR synchronous/spec-v1 compatibility and overlap/spec-v2
- DVR-EAGLE with the Qwen3.5 MTP weights
- exact returned-logprob replay (`KL=0` in the DVR test terminology)
- long generation and GDN chunk boundaries
- Triton, FA3, and FlashInfer full-attention backends
- radix enabled and the FlashInfer-required radix-disabled configuration
- custom all-reduce ownership on an NVLink topology
- throughput against the matching scheduler/backend baseline

## H20 performance regression retest

For the current v5-versus-v6 throughput investigation, use this order. Do not
reuse results from a server launched before the DeepGEMM cache was prepared.

1. Record the commit, topology, datasets, and toolchain from section 2.
2. Remove the run-specific DeepGEMM cache and run all three graph-disabled
   prewarm commands in section 2. Wait for every `PREWARM_COMPLETE` marker.
3. Export the four frozen DeepGEMM variables exactly as shown. Keep DeepGEMM
   enabled for both correctness and throughput servers.
4. Run static checks, then the 0.8B and 35B correctness matrices. Stop if any
   graph capture, KL, boundary, or acceptance check fails.
5. Run 80B Triton and FA3 dvr16, followed by dvr32 in separate result roots.
   Each script launches its own matching sync/overlap baseline; never reuse a
   baseline from another backend, draft length, TP size, or returned-logprob
   setting.
6. Repeat the two best complete matrices three times and report the median.
7. Only if a prewarmed DeepGEMM server still fails graph capture, preserve that
   log and run a separate diagnostic with
   `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`. A disabled-DeepGEMM run
   is diagnostic evidence, not a release throughput result.

Interpret the comparison before changing code again:

- normal acceptance with low target efficiency means execution overhead, not a
  draft-quality problem;
- dvr32 recovering while dvr16 remains low points to fixed verify/scheduling
  cost that the longer draft amortizes;
- runtime `DeepGEMM JIT Pre-Compile`, a new cache artifact during graph startup,
  or a graph fallback invalidates the measurement;
- a steady-state D2H sequence-length copy between `dvr_prepare` and verify is a
  regression; initial prefill/checkpoint binding D2H is allowed;
- custom all-reduce must appear only in provisional self-draft, never target
  prefill or target verify.

## Implementation audit snapshot

At the upstream base used by this branch, the production runtime delta is 22
files, 3,050 added lines, 61 deleted lines, and 2,989 net lines. Most code is in
six DVR-owned modules:

- `dvr_worker.py`: shared draft, target verify, and rollback execution
- `dvr_gdn.py`: GDN state-input windows and exact target verify
- `dvr_state_flow.py`: request checkpoint/commit lifecycle
- `dvr_draft_cuda_graph_runner.py`: DVR-only graph and draft decode policy
- `dvr_server_args.py`: validation and gated-linear defaults
- `dvr_state.py`: generic linear-state adapter contract for future KDA-like
  backends

The remaining runtime changes are narrow integration points for algorithm
selection, pool sizing, result-side checkpoint commit, GDN dispatch, and CUDA
graph selection. Ordinary no-DVR execution is guarded from these paths with two
intentional shared semantic changes:

1. FLA chunk boundary tensors preserve the incoming recurrent-state dtype. This
   is needed for fp32 GDN checkpoints and is also the correct generic dtype
   contract.
2. EAGLE verification treats a supplied `draft_probs` tensor as the proposal
   distribution and applies min-p consistently. This is shared with stochastic
   DVR sampling and therefore needs ordinary EAGLE/rejection-sampling CI as
   well as DVR tests.

Current deliberate restrictions are CUDA-only execution, no DP attention, no
PDMux, no disaggregation, chain mode with topk=1, Triton linear-attention
prefill for exact GDN boundary export, at most one 64-token GDN chunk per
verify, and mandatory draft graphs for self-draft models with linear state.
Plain transformers may use eager self-draft when graphs are disabled. FlashInfer
may disable radix cache; DVR correctness must come from request-local checkpoints
rather than radix.

The minimum effective prompt length is one token. SGLang rejects an empty
`input_ids` request before scheduling; an entrypoint that accepts empty text must
first normalize it to a model-valid BOS or template token.

## 1. Release gates

Do not prepare the release branch until all applicable gates pass.

### Correctness

1. `run_static_unit_checks.sh` passes.
2. 0.8B self-DVR v1 and v2 pass TP=1 and TP=4 for every available required
   full-attention backend. All boundary, batch, concurrent, and 512-token cases
   report `ALL_OK True`; this includes one-token prompts whose first iteration
   uses a zero-step, one-root target-verify sentinel before normal drafting.
   These cases use the same strict full-prefill KL oracle.
3. 35B DVR-EAGLE sync and overlap report `kl_failed: 0` and
   `accept_failed: 0` with both returned-logprob modes.
4. The seeded 35B MTP boundary acceptance remains at least 0.96, and fixed
   ShareGPT acceptance remains at least 0.70. Sync and overlap acceptance must
   agree within 0.02 absolute.
5. 35B self-DVR TP=4 boundary KL passes before throughput is accepted.
6. Every 80B run completes all 16 requests and generates 1024 tokens per
   request. Self-DVR accept length remains at least 14.4/16 on ShareGPT and
   LongBench.
7. `return_logprob=True` must not alter accepted length by more than 0.05 or
   introduce a different GDN state lifecycle.
8. No server log contains an illegal memory access, device-side assertion,
   missing checkpoint, CUDA graph fallback, or reduced requested concurrency.

### Performance

Always compare like with like:

- self-v1 against `baseline_sync`
- self-v2 against `baseline_overlap`
- EAGLE-sync against `baseline_sync`
- EAGLE-overlap against `baseline_overlap`
- identical backend, radix setting, TP size, page size, dataset, concurrency,
  output length, and returned-logprob mode

Use:

```text
acceptance_fraction = accept_length / speculative_num_draft_tokens
target_tps = matching_baseline_tps * acceptance_fraction
target_efficiency = dvr_tps / target_tps
```

The H20/NVLink release target is `target_efficiency >= 0.95`. Any result below
0.90 is a failed performance gate, not measurement noise. Values from 0.90 to
0.95 require a profile and an explained follow-up result.

For the final selected backend, run three complete repetitions. Report the
median and coefficient of variation; CV should be at most 3%. Spec-v2 absolute
throughput should be no lower than spec-v1 beyond 2% measurement noise. A
returned-logprob throughput difference greater than 5% also requires profiling.

The current branch does not capture the full 15-step self-draft chain as one
CUDA graph. The A40 gate measured only a 1.04x theoretical launch-overhead
ceiling and at least 0.16 GB extra capture memory. Do not add or enable a chain
graph on H20 unless a new profile demonstrates an end-to-end gain above 5%
without reducing request or token-pool capacity.

## 2. Fixed inputs and environment

Choose four fully NVLink-connected H20 GPUs for TP=4. Do not mix MIG devices or
GPUs used by another process. Replace paths only through environment variables.

```bash
export DVR_REPO=/path/to/sglang-dvr-v6-performance-opt
export CONDA_ENV=dvr_dev
export CUDA_VISIBLE_DEVICES=0,1,2,3

export MODEL_0P8B=/mnt/data/hwj/Qwen3.5-0.8B
export MODEL_35B=/mnt/data/hwj/Qwen3.5-35B-A3B
export MODEL_80B=/mnt/data/hwj/Qwen3-Next-80B-A3B-Instruct
export SHAREGPT_DATASET=/mnt/data/hwj/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json
export LONGBENCH_DATASET=/path/to/longbench_16x1400_custom.jsonl

cd "${DVR_REPO}"
export RUN_ID="$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD)"
export RESULT_BASE=/mnt/data/hwj/dvr-h20-validation/${RUN_ID}
mkdir -p "${RESULT_BASE}"

# Use a cache dedicated to this source/driver qualification run. Do not reuse
# an unknown cache from an earlier SGLang or DeepGEMM build.
export SGLANG_DG_CACHE_DIR=/mnt/data/hwj/deep-gemm-cache/${RUN_ID}
```

The fixed scripts use `/v1/models` for readiness so health generation is not
part of benchmark timing. DVR nevertheless supports the upstream one-token
generating health probe through its one-root target-verify sentinel.

Before running tests, save the machine and source state:

```bash
git remote get-url upstream >/dev/null 2>&1 || \
  git remote add upstream https://github.com/sgl-project/sglang.git
git fetch upstream sglang-miles
git status --short --branch | tee "${RESULT_BASE}/git-status.txt"
git rev-parse HEAD | tee "${RESULT_BASE}/commit.txt"
git merge-base HEAD upstream/sglang-miles | tee "${RESULT_BASE}/upstream-base.txt"
nvidia-smi -L | tee "${RESULT_BASE}/nvidia-smi-L.txt"
nvidia-smi topo -m | tee "${RESULT_BASE}/nvidia-topology.txt"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv | tee "${RESULT_BASE}/gpu.csv"
conda run --no-capture-output -n "${CONDA_ENV}" bash -lc 'gcc --version | head -1; g++ --version | head -1; nvcc --version | tail -1' | tee "${RESULT_BASE}/toolchain.txt"
conda run --no-capture-output -n "${CONDA_ENV}" python -c 'import torch; print(torch.__version__, torch.version.cuda); print([torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])' | tee "${RESULT_BASE}/torch.txt"
test -f "${MODEL_0P8B}/config.json"
test -f "${MODEL_35B}/config.json"
test -f "${MODEL_80B}/config.json"
test -f "${SHAREGPT_DATASET}"
test -f "${LONGBENCH_DATASET}"
sha256sum "${SHAREGPT_DATASET}" "${LONGBENCH_DATASET}" | tee "${RESULT_BASE}/dataset-sha256.txt"
```

The topology file must show NVLink connectivity among all selected TP ranks.
Run the first server for each backend as the real availability check. An import
warning followed by a fallback backend is a failure; do not relabel that run as
FA3 or FlashInfer. Transfer the exact fixed LongBench custom JSONL from the
development machine; regenerating a different 16-request sample invalidates the
comparison.

### DeepGEMM before CUDA graph capture

Do not globally disable DeepGEMM for the release measurement. In deterministic
mode, target prefill/verify uses batch-invariant `bf16_gemm_nn`; disabling it can
penalize DVR verify more than normal decode and invalidates the comparison.

DeepGEMM JIT must also not start for the first time inside CUDA graph startup.
The official wrapper precompiler does not by itself prove that every direct
batch-invariant NN shape has been exercised. Run the checked-in graph-disabled
shape sweep once per model and cache. It uses ordinary deterministic EXTEND with
`M = graph_bs * draft_tokens`, which matches target verify's linear/MoE token
counts without invoking DVR or capturing a graph.

```bash
rm -rf "${SGLANG_DG_CACHE_DIR}"
mkdir -p "${SGLANG_DG_CACHE_DIR}"

CONDA_ENV="${CONDA_ENV}" MODEL_PATH="${MODEL_0P8B}" TP_SIZE=4 \
MAX_GRAPH_BS=4 DRAFT_TOKEN_COUNTS="16" \
RESULT_ROOT="${RESULT_BASE}/deepgemm-prewarm-0p8b" \
  bash test/manual/dvr/scripts/prepare_h20_deep_gemm.sh

CONDA_ENV="${CONDA_ENV}" MODEL_PATH="${MODEL_35B}" TP_SIZE=4 \
MAX_GRAPH_BS=8 DRAFT_TOKEN_COUNTS="2 16" \
RESULT_ROOT="${RESULT_BASE}/deepgemm-prewarm-35b" \
  bash test/manual/dvr/scripts/prepare_h20_deep_gemm.sh

CONDA_ENV="${CONDA_ENV}" MODEL_PATH="${MODEL_80B}" TP_SIZE=4 \
MAX_GRAPH_BS=3 DRAFT_TOKEN_COUNTS="16 32" \
RESULT_ROOT="${RESULT_BASE}/deepgemm-prewarm-80b" \
  bash test/manual/dvr/scripts/prepare_h20_deep_gemm.sh

test -f "${RESULT_BASE}/deepgemm-prewarm-0p8b/PREWARM_COMPLETE"
test -f "${RESULT_BASE}/deepgemm-prewarm-35b/PREWARM_COMPLETE"
test -f "${RESULT_BASE}/deepgemm-prewarm-80b/PREWARM_COMPLETE"
```

Compilation can take tens of minutes on a fresh cache. Wait for each command;
do not kill it because GPU utilization temporarily drops. Each result directory
must contain `PREWARM_COMPLETE`, and its server log must be free of traceback or
CUDA errors.

After all prewarm runs finish, freeze the serving environment:

```bash
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export REQUIRE_PRECOMPILED_DEEPGEMM=1

find "${SGLANG_DG_CACHE_DIR}" -type f -printf '%P %s\n' | sort \
  >"${RESULT_BASE}/deepgemm-cache-before-serving.txt"
```

The fixed scripts now reject an empty cache, a CUDA graph capture/fallback
error, or a serving process that re-enters `DeepGEMM JIT Pre-Compile`. If graph
capture still fails after this exact prewarm, preserve the server log and run a
separate diagnostic with only
`SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`; do not report that diagnostic
throughput as the release result.

## 3. Static validation

```bash
CONDA_ENV="${CONDA_ENV}" \
  bash test/manual/dvr/scripts/run_static_unit_checks.sh \
  2>&1 | tee "${RESULT_BASE}/static-unit.log"
```

Expected result: clean `git diff --check`, successful `py_compile`, and all DVR
unit tests passing.

## 4. 0.8B self-DVR correctness matrix

Run every backend first on one GPU, then on the selected TP=4 NVLink group.
`DISABLE_RADIX_CACHE=auto` keeps radix for Triton/FA3 and disables it for
FlashInfer. Do not override this with a FlashInfer-specific DVR workaround.

```bash
for backend in triton fa3 flashinfer; do
  for tp in 1 4; do
    result="${RESULT_BASE}/0p8b-${backend}-tp${tp}"
    CONDA_ENV="${CONDA_ENV}" \
    MODEL_PATH="${MODEL_0P8B}" \
    TP_SIZE="${tp}" \
    ATTENTION_BACKEND="${backend}" \
    LINEAR_ATTN_BACKEND=triton \
    DISABLE_RADIX_CACHE=auto \
    RESULT_ROOT="${result}" \
      bash test/manual/dvr/scripts/run_0p8b_self_dvr_kl.sh
  done
done
```

Check every `spec_v1_kl.log` and `spec_v2_kl.log` for `ALL_OK True`. The server
logs must show the requested attention backend, page size 64, expected radix
mode, and captured graph batch 4.

## 5. 35B MTP/EAGLE correctness matrix

This is the primary check for MTP hidden-state, draft KV, GDN boundary, and
sync/overlap ownership consistency.

```bash
for backend in triton fa3 flashinfer; do
  result="${RESULT_BASE}/35b-mtp-${backend}"
  CONDA_ENV="${CONDA_ENV}" \
  MODEL_PATH="${MODEL_35B}" \
  DRAFT_MODEL_PATH="${MODEL_35B}" \
  SHAREGPT_DATASET="${SHAREGPT_DATASET}" \
  TP_SIZE=4 \
  MAX_MAMBA_CACHE_SIZE=16 \
  ATTENTION_BACKEND="${backend}" \
  LINEAR_ATTN_BACKEND=triton \
  DISABLE_RADIX_CACHE=auto \
  RESULT_ROOT="${result}" \
    bash test/manual/dvr/scripts/run_35b_mtp_eagle_smoke.sh
done
```

The synthetic boundary and warm-prefix cases must pass both returned-logprob
modes. Compare the per-prompt ShareGPT acceptance values between `sync_v2` and
`overlap_v2`; a consistent difference is an ownership bug even if both remain
above the script's floor.

After the greedy matrix passes, run one stochastic Triton check as a separate
experiment. This validates rejection-sampling probability plumbing; it is not
an acceptance-rate benchmark.

```bash
CONDA_ENV="${CONDA_ENV}" \
MODEL_PATH="${MODEL_35B}" \
DRAFT_MODEL_PATH="${MODEL_35B}" \
SHAREGPT_DATASET="${SHAREGPT_DATASET}" \
TP_SIZE=4 \
ATTENTION_BACKEND=triton \
LINEAR_ATTN_BACKEND=triton \
ACCEPT_TEMPERATURE=0.7 \
ACCEPT_TOP_P=0.9 \
MTP_MIN_ACCEPT_RATE=0.0 \
MTP_REALDATA_MIN_ACCEPT_RATE=0.0 \
RESULT_ROOT="${RESULT_BASE}/35b-mtp-triton-stochastic" \
  bash test/manual/dvr/scripts/run_35b_mtp_eagle_smoke.sh
```

The stochastic run still requires `kl_failed: 0`; acceptance is intentionally
not compared with the greedy floor.

### Shared ordinary-EAGLE regression

`eagle_utils.py` is the only material shared speculative-sampling path changed
by DVR. Run upstream's ordinary NEXTN rejection-sampling test once on the H20
machine. It uses Qwen3.5-9B and may require downloading that additional model.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="${DVR_REPO}/python" \
  conda run --no-capture-output -n "${CONDA_ENV}" \
  python test/registered/spec/eagle/test_eagle_reject_sampling.py \
  2>&1 | tee "${RESULT_BASE}/ordinary-eagle-rejection.log"
```

Its GSM8K score and average accepted length assertions must pass. This is a
non-DVR regression gate and must not be replaced by the DVR-EAGLE smoke above.

## 6. 35B fixed throughput matrix

Run the fixed BS=3 matrix once for all backends. It covers matching no-DVR
sync/overlap baselines, self-v1/v2, EAGLE sync/overlap, and both logprob modes.

```bash
for backend in triton fa3 flashinfer; do
  result="${RESULT_BASE}/35b-throughput-${backend}"
  CONDA_ENV="${CONDA_ENV}" \
  MODEL_PATH="${MODEL_35B}" \
  DRAFT_MODEL_PATH="${MODEL_35B}" \
  SHAREGPT_DATASET="${SHAREGPT_DATASET}" \
  TP_SIZE=4 \
  ATTENTION_BACKEND="${backend}" \
  LINEAR_ATTN_BACKEND=triton \
  DISABLE_RADIX_CACHE=auto \
  DISABLE_CUSTOM_ALL_REDUCE=0 \
  RESULT_ROOT="${result}" \
    bash test/manual/dvr/scripts/run_35b_dvr_throughput.sh
done
```

Do not compare EAGLE's two-token acceptance fraction with self-DVR's 16-token
acceptance fraction. Compare each mode with its own `acceptance_x_baseline`
target printed by the script.

### Large-batch EAGLE follow-up

Use the two fastest compatible backends from the fixed matrix. Run BS=1, 4,
and 8 separately so deterministic target-verify cost and graph coverage are
not hidden by an effective-concurrency reduction.

```bash
for backend in triton fa3; do
  for bs in 1 4 8; do
    result="${RESULT_BASE}/35b-eagle-${backend}-bs${bs}"
    CONDA_ENV="${CONDA_ENV}" \
    MODEL_PATH="${MODEL_35B}" \
    DRAFT_MODEL_PATH="${MODEL_35B}" \
    SHAREGPT_DATASET="${SHAREGPT_DATASET}" \
    TP_SIZE=4 \
    ATTENTION_BACKEND="${backend}" \
    LINEAR_ATTN_BACKEND=triton \
    DISABLE_RADIX_CACHE=auto \
    NUM_PROMPTS=32 \
    OUTPUT_LEN=512 \
    MAX_CONCURRENCY="${bs}" \
    MAX_MAMBA_CACHE_SIZE=32 \
    CONTEXT_LENGTH=4096 \
    MAX_TOTAL_TOKENS=32768 \
    RUN_SELF=0 \
    RESULT_ROOT="${result}" \
      bash test/manual/dvr/scripts/run_35b_dvr_throughput.sh
  done
done
```

If capacity is insufficient, the script must fail. Adjust capacity in a new
result directory; never report a silently reduced BS=8 run.

## 7. 80B long-output matrix

The 80B matrix is the release performance gate. It runs 16 ShareGPT requests
at concurrency 3 and 16 fixed LongBench requests at concurrency 2, with 1024
generated tokens and both returned-logprob modes.

```bash
for backend in triton fa3 flashinfer; do
  result="${RESULT_BASE}/80b-throughput-${backend}"
  CONDA_ENV="${CONDA_ENV}" \
  MODEL_PATH="${MODEL_80B}" \
  SHAREGPT_DATASET="${SHAREGPT_DATASET}" \
  LONGBENCH_CUSTOM_DATASET="${LONGBENCH_DATASET}" \
  TP_SIZE=4 \
  ATTENTION_BACKEND="${backend}" \
  LINEAR_ATTN_BACKEND=triton \
  DISABLE_RADIX_CACHE=auto \
  DISABLE_CUSTOM_ALL_REDUCE=0 \
  RESULT_ROOT="${result}" \
    bash test/manual/dvr/scripts/run_80b_self_dvr_throughput.sh
done
```

Then repeat the fastest two backends with the v5 high-efficiency dvr32
configuration. Keep a separate result directory and do not mix dvr16/dvr32
acceptance denominators:

```bash
for backend in triton fa3; do
  result="${RESULT_BASE}/80b-throughput-${backend}-dvr32"
  CONDA_ENV="${CONDA_ENV}" \
  MODEL_PATH="${MODEL_80B}" \
  SHAREGPT_DATASET="${SHAREGPT_DATASET}" \
  LONGBENCH_CUSTOM_DATASET="${LONGBENCH_DATASET}" \
  TP_SIZE=4 \
  ATTENTION_BACKEND="${backend}" \
  LINEAR_ATTN_BACKEND=triton \
  DISABLE_RADIX_CACHE=auto \
  DISABLE_CUSTOM_ALL_REDUCE=0 \
  DRAFT_TOKENS=32 \
  DRAFT_STEPS=31 \
  RESULT_ROOT="${result}" \
    bash test/manual/dvr/scripts/run_80b_self_dvr_throughput.sh
done
```

For the fastest backend and its runner-up, repeat the full matrix twice more in
new `-r2` and `-r3` result directories. Use the median, not the best run.

Historical H20 numbers from an earlier implementation were approximately
398 tok/s for Triton and 370 tok/s for FA3 on ShareGPT, and 298.45 tok/s for
Triton and 225.32 tok/s for FA3 on LongBench. They are only a gross sanity
check because the code and baseline have changed. The matching baseline and
`target_efficiency` from this run are authoritative.

## 8. Custom all-reduce ownership check

With DVR deterministic inference, target prefill and target verify must retain
the deterministic path with custom all-reduce disabled. Only provisional draft
CUDA graph capture may enable custom all-reduce. First reject any log containing:

```text
DVR draft custom all-reduce setup failed
```

Then run a DVR-only control on the fastest 80B backend with custom all-reduce
disabled. Compare its v1/v2 absolute throughput and accepted length with the
default run; reuse the default run's matching baselines for analysis.

```bash
CONDA_ENV="${CONDA_ENV}" \
MODEL_PATH="${MODEL_80B}" \
SHAREGPT_DATASET="${SHAREGPT_DATASET}" \
LONGBENCH_CUSTOM_DATASET="${LONGBENCH_DATASET}" \
TP_SIZE=4 \
ATTENTION_BACKEND=triton \
LINEAR_ATTN_BACKEND=triton \
DISABLE_CUSTOM_ALL_REDUCE=1 \
RUN_BASELINE=0 \
RUN_DVR=1 \
RESULT_ROOT="${RESULT_BASE}/80b-triton-custom-ar-disabled" \
  bash test/manual/dvr/scripts/run_80b_self_dvr_throughput.sh
```

The enabled run should be faster without changing KL or acceptance. If it is
not, inspect the NVLink topology and custom communicator initialization before
changing DVR code.

If target efficiency remains below 0.95, profile a separately launched server
with `profile_dvr_server.sh`. Do not profile during a throughput run. In the
trace, custom all-reduce kernels may occur in `draft`, but not in prefill or
`verify`. Also record draft, verify, rollback, graph launch count, graph memory,
and GPU kernel busy fraction before proposing another optimization.

For the current synchronization fix, also inspect the steady-state trace for
device-to-host sequence-length copies. A D2H is allowed when a request first
binds a GDN checkpoint or enters target prefill. Repeated D2H copies between
`dvr_prepare`, self-draft, and `dvr_state_restore` are a regression. CUDA event
waits between draft/verify and rollback/next-iteration are expected and must not
appear as host synchronizations.

## 9. Non-DVR upstream A/B

The runtime delta is DVR-guarded except for the recurrent-state dtype
preservation in the FLA chunk kernel and the shared EAGLE proposal-probability
contract. Before release, prove that normal serving has not regressed against
the exact upstream base commit recorded in `upstream-base.txt`.

Create a clean worktree, copy only the fixed manual test harness into it, and
run baseline-only matrices from that worktree. The copied files are untracked
test drivers; its `PYTHONPATH` still points to the clean upstream runtime.

```bash
export UPSTREAM_SHA="$(cat "${RESULT_BASE}/upstream-base.txt")"
export UPSTREAM_WT=/tmp/sglang-dvr-upstream-${UPSTREAM_SHA:0:12}
git worktree add --detach "${UPSTREAM_WT}" "${UPSTREAM_SHA}"
mkdir -p "${UPSTREAM_WT}/test/manual"
cp -a test/manual/dvr "${UPSTREAM_WT}/test/manual/"

cd "${UPSTREAM_WT}"
CONDA_ENV="${CONDA_ENV}" \
MODEL_PATH="${MODEL_80B}" \
SHAREGPT_DATASET="${SHAREGPT_DATASET}" \
LONGBENCH_CUSTOM_DATASET="${LONGBENCH_DATASET}" \
TP_SIZE=4 \
ATTENTION_BACKEND=triton \
LINEAR_ATTN_BACKEND=triton \
RUN_BASELINE=1 \
RUN_DVR=0 \
RESULT_ROOT="${RESULT_BASE}/80b-upstream-baseline-triton" \
  bash test/manual/dvr/scripts/run_80b_self_dvr_throughput.sh

cd "${DVR_REPO}"
git worktree remove --force "${UPSTREAM_WT}"
```

Compare this with `80b-throughput-triton` baseline rows. A repeatable no-DVR
throughput regression above 2% blocks release. Also run one normal generation
smoke on both trees and compare greedy output token IDs.

## 10. Log audit and result report

Run the following audit after all servers have stopped:

```bash
# The prewarm server is expected to compile kernels. Audit only the formal
# serving logs, where any new DeepGEMM precompile session invalidates the run.
rg -n "Traceback|illegal memory access|device-side assert|missing a boundary checkpoint|DVR self-draft decode requires|custom all-reduce setup failed|max_running_requests was reduced|Entering DeepGEMM JIT Pre-Compile session" \
  "${RESULT_BASE}" --glob '**/logs/*server.log' \
  | tee "${RESULT_BASE}/error-audit.txt"

find "${SGLANG_DG_CACHE_DIR}" -type f -printf '%P %s\n' | sort \
  >"${RESULT_BASE}/deepgemm-cache-after-serving.txt"
diff -u "${RESULT_BASE}/deepgemm-cache-before-serving.txt" \
  "${RESULT_BASE}/deepgemm-cache-after-serving.txt" \
  | tee "${RESULT_BASE}/deepgemm-cache-serving-diff.txt"

pgrep -af 'sglang.launch_server|sglang::scheduler|sglang::tokenizer' \
  | tee "${RESULT_BASE}/remaining-processes.txt" || true
```

An empty error audit and no remaining server process are required. For every
server log also verify:

- requested `attention_backend`
- `page_size=64`
- expected `disable_radix_cache`
- expected `disable_overlap_schedule`
- largest requested CUDA graph batch was captured
- no effective concurrency reduction
- self-draft used its dedicated CUDA graph with no eager fallback

The final report should contain one row per backend, dataset, mode, and logprob
setting with:

```text
commit
GPU topology
backend and radix mode
mode and scheduler mode
return_logprob
baseline tok/s
DVR tok/s
accept length and fraction
acceptance_x_baseline
target_efficiency
median/CV for repeated runs
KL failures
server/log result paths
```

Release is ready only when correctness passes on all required backends, the
selected production backend reaches the H20 target, no-DVR A/B is neutral, and
all remaining exceptions are documented as explicit unsupported configurations
rather than silent fallbacks.
