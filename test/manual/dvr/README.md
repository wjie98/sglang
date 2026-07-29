# Decode-Verify-Rollback

Decode-Verify-Rollback (DVR) is a speculative decoding mode for deterministic
target execution. The target model performs deterministic prefill and verify;
provisional draft decoding uses the normal fast decode configuration. DVR can
therefore reproduce generated tokens with a later target prefill, including on
models with gated recurrent linear-attention layers.

DVR has one deterministic target contract. Target prefill and target verify use
SGLang's upstream deterministic MoE, attention, and collective policies. DVR
does not install a second target kernel configuration. Only provisional
self-draft graph capture may select ordinary fast decode optimizations;
rejection sampling corrects its proposal distribution back to the unchanged
target policy.

Two draft backends are available:

- `DECODE_VERIFY_ROLLBACK`: the target model drafts from its committed state.
- `DECODE_VERIFY_ROLLBACK_EAGLE`: an EAGLE or MTP model produces draft tokens.

## Execution phases and numerical policy

DVR keeps forward semantics, numerical policy, and CUDA graph layout separate:

| Phase | Forward mode | Numerical policy | Graph layout |
| --- | --- | --- | --- |
| Target prefill | `EXTEND` | Authoritative deterministic target | Dynamic prefill tokens |
| Self draft | `DECODE` | Provisional normal fast path | One token per request |
| Target verify | `TARGET_VERIFY` | Authoritative deterministic target | Fixed draft tokens per request |
| Rollback/commit | No model forward | Commit target-selected state | GPU state transition |

`TARGET_VERIFY` is classified as EXTEND and attention dispatches it through
`forward_extend`. Its runner reuses decode CUDA graph infrastructure only
because speculative verification has a fixed number of tokens per request and
requires fixed-address speculative metadata. It does not use decode attention
mathematics.

The resolved server configuration is the immutable target contract. DVR does
not switch target prefill or verify into a private numerical mode. The sole
exception is provisional draft graph capture, which temporarily selects normal
fast model and collective kernels. Runtime graph replay must not mutate the
process-wide deterministic dispatcher or server configuration. Deterministic
request sampling remains enabled independently of this model-kernel choice.

## Supported configuration

- CUDA execution with pipeline parallel size one.
- Triton or FlashAttention 3 for all full-attention phases.
- Triton linear-attention prefill for models with GDN layers.
- Chain proposals (`speculative_eagle_topk=1`) and exact rejection sampling.
- Radix cache enabled or disabled. Disabling Radix keeps ordinary full-prefill
  semantics and uses request-local recurrent checkpoints while the request is
  active.
- Overlap scheduling enabled by default. Pass `--disable-overlap-schedule` for
  the synchronous compatibility path.

For GDN models, `page_size` must equal the FLA chunk size (64). This keeps
Radix prefix boundaries identical to the recurrent checkpoints used by verify.
Grammar-constrained decoding and dynamic token penalties
(`frequency_penalty`, `presence_penalty`, `repetition_penalty`, and
`min_new_tokens`) are not supported by DVR. Requests using them are rejected
rather than approximated. Static `logit_bias` is supported.

DVR also rejects adaptive or decoupled speculative execution, multi-layer
EAGLE, and reduced-vocabulary token maps. Self-draft reuses the target model and
therefore rejects a separate draft-model path. Generic speculative argument
handling owns unrelated draft-window and mixed-chunk validation. Self-draft has
no eager fallback: when `max_running_requests` is unset it uses the decode CUDA
graph `max_bs`, and an explicitly larger value is rejected.

## Start a server

The launcher uses the current Python environment. Set `PYTHONPATH` explicitly
when the environment contains an editable SGLang installation from another
worktree:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH=/models/Qwen3.5-35B-A3B \
SERVER_MODE=self \
TP_SIZE=4 \
ATTENTION_BACKEND=fa3 \
MAX_RUNNING_REQUESTS=3 \
MAX_MAMBA_CACHE_SIZE=16 \
CONTEXT_LENGTH=8192 \
MAX_TOTAL_TOKENS=6144 \
CUDA_GRAPH_BS="1 2 3" \
CUDA_GRAPH_MAX_BS_DECODE=3 \
test/manual/dvr/launch_server.sh
```

To use the model's MTP layers as an EAGLE draft model:

```bash
MODEL_PATH=/models/Qwen3.5-35B-A3B \
SERVER_MODE=eagle \
TP_SIZE=4 \
test/manual/dvr/launch_server.sh
```

`DRAFT_MODEL_PATH` defaults to `MODEL_PATH` in EAGLE mode. Set it explicitly
when using a separate EAGLE checkpoint.

`SERVER_MODE` selects a matched server configuration:

| Mode | Behavior |
| --- | --- |
| `normal` | Ordinary non-deterministic serving |
| `deterministic` | Ordinary deterministic serving without DVR |
| `self` | Self-draft DVR |
| `eagle` | DVR with an EAGLE/MTP draft model |

`DVR_MODE` remains an alias for `SERVER_MODE` for older commands. Useful
environment overrides include `SERVER_HOST`, `PORT`, `TP_SIZE`, `DRAFT_TOKENS`,
`DRAFT_STEPS`, `PAGE_SIZE`, `ATTENTION_BACKEND`, `LINEAR_ATTN_BACKEND`,
`SAMPLING_BACKEND`, `MAX_RUNNING_REQUESTS`, `MAX_MAMBA_CACHE_SIZE`,
`MEM_FRACTION_STATIC`, `CONTEXT_LENGTH`, `MAX_TOTAL_TOKENS`, `CUDA_GRAPH_BS`,
`CUDA_GRAPH_MAX_BS_DECODE`, `DISABLE_OVERLAP`, `DISABLE_RADIX_CACHE`,
`DISABLE_CUSTOM_ALL_REDUCE`, and `RANDOM_SEED`. Additional SGLang arguments may
be appended to the command.

The launcher does not activate Conda, select GPUs, configure datasets, or
modify DeepGEMM environment variables. It prints the fully expanded command so
the server configuration can be attached to a test report.

`--disable-overlap-schedule` does not select a separate legacy spec-v1 worker.
Current SGLang uses the same DVR worker and V2 result schema in both modes; the
synchronous mode only disables scheduler overlap.

The launcher passes `--enable-deterministic-inference` explicitly for both DVR
modes. This selects SGLang's ordinary deterministic target kernels; DVR then
restores normal fast decode settings only while capturing and replaying
provisional draft work. Do not combine DVR with
`--enable-prefill-only-deterministic-inference`: target verify is deterministic
as well as prefill. DVR and ordinary deterministic servers must produce exact
prefill scores for the same forced sequence; do not work around a mismatch with
DVR-specific target MoE tiles.

## Request example

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Explain deterministic speculative decoding.",
    "sampling_params": {
      "max_new_tokens": 128,
      "temperature": 1.0,
      "top_p": 0.95,
      "top_k": 20,
      "sampling_seed": 2026
    },
    "return_logprob": true
  }'
```

Formal correctness, acceptance, profiling, and throughput qualification must
use explicit non-greedy parameters. Qwen3.5 uses
`temperature=1.0, top_p=0.95, top_k=20`; Qwen3-Next uses
`temperature=0.7, top_p=0.8, top_k=20`. Keep a fixed per-request
`sampling_seed` across matched runs. Greedy sampling is permitted only for
setup such as DeepGEMM precompilation or for a separately labelled diagnostic;
it underestimates proposal and rejection-sampling cost and may hide acceptance
regressions.

For stochastic RL logprob qualification, set
`SGLANG_RETURN_ORIGINAL_LOGPROB=True` before server startup when the oracle
expects logprobs before temperature scaling.

Within one fixed execution shape, repeat the same prompt and `sampling_seed` at
least three times and require identical DVR output. Run both exact-logprob gates
on every output. This catches accidental use of process-global RNG in draft or
rejection sampling. Output identity across sync/overlap or different batch
shapes is not a DVR requirement unless the matched ordinary deterministic path
also guarantees it.

## Recurrent-state lifecycle

For gated linear-attention models, the ordinary token KV cache and the
recurrent state have different lifetimes. DVR keeps the full-attention KV in
SGLang's existing request and Radix pools. It treats only a recurrent state at
an exact 64-token boundary as a persistent checkpoint and keeps the unclosed
tail as request-owned `k/v/g/beta` transition inputs. Cached `q` is unnecessary:
verify consumes `q` only for the current candidate outputs.

The state transition is:

1. Target EXTEND establishes the latest exact boundary and caches the inputs
   after that boundary in the target request's live Mamba slot. Its convolution
   state remains at the accepted endpoint. With Radix disabled, a later request
   performs an ordinary full prefill.
2. A boundary produced by EXTEND or accepted verify may also be copied into an
   ordinary Radix tracking lane. DVR records its logical length but does not
   change the request-to-Mamba mapping or Radix ownership rules.
3. Target EXTEND seeds one request-owned self-draft recurrent workspace and
   convolution state. Self-draft mutates only this private state; EAGLE/MTP
   advances its upstream-owned draft cache. Neither backend changes the
   authoritative target boundary.
4. Target verify reads the exact boundary without overwriting it and runs
   deterministic EXTEND over one fixed `64 + draft_tokens` window. The first 64
   rows reproduce the original chunk partition; only logical draft rows are
   returned, and the exported boundary state is staged in the now-idle
   self-draft workspace.
5. Rollback commits only accepted rows. Crossing a 64-token boundary publishes
   the staged boundary into the target live slot and, when enabled, a Radix
   tracking lane, then compacts the input window. Self-draft rebuilds its
   private endpoint from the newest exact boundary plus the accepted tail;
   EAGLE/MTP skips this target recurrent-state reconstruction.
6. Request release publishes the newest exact checkpoint no later than the
   visible committed prefix. Radix stores only that aligned prefix; a later
   request rebuilds the non-aligned suffix through ordinary EXTEND. If no such
   checkpoint is available, insertion is skipped rather than changing the
   inference result. Under overlap, release first orders itself after the
   already-enqueued rollback event; this is a GPU stream dependency on the
   finishing request, not a steady-state host synchronization.

At every draft boundary, `checkpoint_length + tail_length` must equal the
committed target history. The checkpoint and cached state inputs must originate
from that same history. Target verify reads only the active request's live
boundary; ordinary Radix publication slots are never used as verify inputs.

## Compatibility contracts

The registered tests intentionally protect behavior rather than private buffer
layouts. The CUDA graph tests are the exception where a narrow integration
contract is required: upstream backend changes must not cause target
deterministic settings to leak into provisional draft capture.

In particular:

- target prefill and verify remain deterministic;
- target MoE execution remains identical to upstream ordinary deterministic
  inference;
- draft graph capture restores environment, backend, batch-invariant, MoE, and
  collective state when it exits;
- weight updates requested with `recapture_cuda_graph=true` rebuild the complete
  DVR graph set only after target and draft weights are updated;
- custom all-reduce may be captured for provisional draft execution but is not
  enabled for target prefill or verify;
- Triton and FA3 draft decode retain their normal split configuration;
- the target live boundary is request-owned; only exact boundaries are copied
  to Radix, while DVR leaves ordinary Radix slot allocation and rebinding
  semantics unchanged.

GDN DVR does not currently support page-major recurrent-state storage,
ReplaySSM, streaming sessions, or int8 recurrent checkpoints. These modes
change the representation or lifetime of the exact state that rollback needs;
the server rejects them instead of silently running with an incomplete state.

Fast regression tests live under `test/registered/unit/`. Release qualification
must additionally cover long-generation KL, acceptance, and throughput on the
target NVLink hardware.

## NVLink qualification

### Qualification order

Use a clean process for every server row and record `git rev-parse HEAD` before
testing. Run the gates in this order so an early correctness failure does not
contaminate later throughput results:

1. Qwen3.5-35B self-DVR with Triton and FA3: both exact-logprob gates, boundary
   prompts, 512/1024-token generation, `return_logprob=false/true`, Radix
   enabled, and one Radix-disabled diagnostic.
2. Qwen3.5-35B MTP/EAGLE: sync and overlap correctness, repeated-seed output,
   acceptance, and throughput. Start with the model-recommended draft length;
   report any additional draft-length sweep separately.
3. Qwen3-Next-80B self-DVR: Triton/FA3, custom-all-reduce on/off, ShareGPT and
   long-input throughput. FlashInfer all-reduce fusion must remain disabled in
   both rows.
4. Only after all correctness gates pass, run the matched normal,
   deterministic, self-DVR, and DVR-EAGLE throughput rows.

For RL weight-update qualification, run one self-DVR row and one same-checkpoint
MTP row after:

```bash
curl http://127.0.0.1:30000/update_weights_from_disk \
  -H 'Content-Type: application/json' \
  -d "{
    \"model_path\": \"${MODEL_PATH}\",
    \"recapture_cuda_graph\": true,
    \"flush_cache\": true
  }"
```

After the endpoint returns success, repeat prompt length 65 with output lengths
65 and 512. Logs must show complete draft and target-verify graph capture, and
the measured requests must not fall back to eager execution. This gate covers
self-draft and MTP stored in the same checkpoint; independently versioned
EAGLE checkpoints require a separately matched update procedure.

### Keep the comparison matched

Start a fresh server for each row. Keep model revision, visible GPUs, TP size,
attention backend, page size, context and token capacity, CUDA graph batch
sizes, Radix policy, custom all-reduce policy, dataset, output length,
concurrency, and random seed identical. Also keep the explicit
model-recommended temperature, top-p, top-k, and request seeds identical.

Pair scheduling modes as follows:

| DVR row | Implementation-efficiency baseline | User-facing baseline |
| --- | --- | --- |
| self/eagle with overlap | normal with overlap | deterministic with overlap |
| self/eagle with `DISABLE_OVERLAP=1` | normal with `DISABLE_OVERLAP=1` | deterministic with `DISABLE_OVERLAP=1` |

The normal baseline measures the cost of one fast non-deterministic decode
step. The ordinary deterministic baseline measures DVR's user-facing speedup.
Its prefill path is also an independent target-policy reference: DVR prefill
must equal ordinary deterministic prefill on the exact sequence generated by
DVR. A failure is a release blocker even when DVR decode still matches its own
clean-prefill replay.

For example, launch these configurations one at a time with the same common
environment:

```bash
conda activate dvr_dev
export MODEL_PATH=/models/Qwen3.5-35B-A3B
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TP_SIZE=4
export PAGE_SIZE=64
export ATTENTION_BACKEND=fa3
export LINEAR_ATTN_BACKEND=triton
export SAMPLING_BACKEND=pytorch
export MAX_RUNNING_REQUESTS=3
export MAX_MAMBA_CACHE_SIZE=16
export MEM_FRACTION_STATIC=0.9
export CONTEXT_LENGTH=8192
export MAX_TOTAL_TOKENS=6144
export CUDA_GRAPH_BS="1 2 3"
export CUDA_GRAPH_MAX_BS_DECODE=3
export RANDOM_SEED=2026

SERVER_MODE=normal test/manual/dvr/launch_server.sh
SERVER_MODE=deterministic test/manual/dvr/launch_server.sh
SERVER_MODE=self DRAFT_TOKENS=16 test/manual/dvr/launch_server.sh
SERVER_MODE=eagle DRAFT_TOKENS=2 test/manual/dvr/launch_server.sh
```

Stop the previous server before starting the next command. Repeat the matrix
with `ATTENTION_BACKEND=triton`. Repeat only the required sync rows with
`DISABLE_OVERLAP=1`; do not label this path as a separate worker generation.

### DeepGEMM on H20

Do not globally disable batch-invariant DeepGEMM for a formal performance run.
Target verify uses `batch_size * draft_tokens` rows and is affected more than
single-token decode when these GEMMs fall back.

Before graph capture, precompile every tested target-verify shape into a cache
dedicated to the commit, driver, CUDA, and compiler combination. Formal serving
should retain that cache and use:

```bash
export SGLANG_DG_CACHE_DIR=/path/to/commit-specific-cache
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=0
```

If server startup enters an exhaustive DeepGEMM precompile session during CUDA
graph capture, the cache is incomplete and that performance result is invalid.
Diagnostic runs with DeepGEMM disabled must be labelled separately.

### Collective and MoE determinism checks

Self-DVR intentionally leaves FlashInfer all-reduce fusion disabled in draft
graphs. Unlike custom all-reduce, this fusion can change the tensor-parallel
reduction path relative to the deterministic target and sharply reduce
acceptance. Custom all-reduce may remain enabled for draft capture. Confirm the
resolved server configuration contains
`flashinfer_allreduce_fusion_backend=None`; do not restore the fusion as a
normal-decode optimization.

DVR does not override upstream target MoE kernel geometry. Active
batch-invariant mode selects the same target policy used by ordinary
deterministic inference. Self-draft capture temporarily disables that active
mode so its MoE kernels may use the ordinary optimized configuration; capture
must restore the target mode and every mode-aware cache before target work. On
H20:

- compare target MoE configuration before draft capture, inside self-draft
  capture, and after capture; the first and last values must be the upstream
  deterministic configuration;
- run Qwen3-Next-80B TP acceptance with custom all-reduce enabled and disabled,
  keeping FlashInfer all-reduce fusion disabled.

An A40 can exercise policy restoration tests but cannot qualify H20
collectives or FA3 performance.

### Correctness matrix

Run both `return_logprob=false` and `true`. At minimum cover:

- prompt lengths `1`, `63`, `64`, and `65`;
- generated lengths `65`, `512`, and `1024` with `ignore_eos=true`;
- batch sizes or concurrency `1` and the performance batch size;
- Radix enabled, generated-prefix reuse, and one separately labelled
  `DISABLE_RADIX_CACHE=1` control;
- self-DVR with 16 draft tokens in synchronous and overlap scheduling;
- Qwen3.5 MTP/EAGLE in synchronous and overlap scheduling;
- Triton and FA3 full-attention backends with Radix enabled;
- one ordinary non-DVR overlap request in both logprob return modes.

Every row has two separate strict checks:

1. **DVR-to-prefill:** decode with DVR, concatenate the accepted generated
   tokens to the prompt, and run a clean target prefill over that exact sequence.
2. **DVR-prefill-to-deterministic-prefill:** stop the DVR server, start the
   matched ordinary deterministic server, and prefill that same concatenated
   sequence with `max_new_tokens=0`. Compare its generated-token positions with
   the DVR replay prefill; ordinary deterministic decode is not executed.

For the fixed HTTP client, both checks require identical token IDs and counts,
selected-token `maxdiff=0`, and `kl_proxy=0`. The proxy is the historical
selected-token logprob check; it is not full-vocabulary KL. Report
full-vocabulary `KL=0` only when separate instrumentation captures and compares
the complete target distributions. If full logits are available, they must
also be bitwise equal.

Start both servers with `SGLANG_RETURN_ORIGINAL_LOGPROB=True`. Compare the same
forced sequence rather than independently sampled trajectories. A completed
artifact must contain both `dvr_gate.exact=true` and `det_gate.exact=true`;
missing or failed gates block throughput qualification. Ordinary deterministic
GDN decode and chunk prefill use different upstream kernels and are not part of
this two-stage gate. The second check does not replace the Miles
prefill-to-trainer parity gate.

Use the fixed client for both phases:

```bash
# Run while the DVR server is active. This writes the forced sequence and
# proves DVR decode -> DVR replay prefill.
python test/manual/dvr/check_logprob_parity.py dvr \
  --artifact /results/qwen35_triton_parity.json \
  --prompt-length 64 \
  --max-new-tokens 512

# Stop DVR, launch the matched deterministic server, then score the saved
# sequence with a prefill-only request. No new trajectory is sampled.
python test/manual/dvr/check_logprob_parity.py det \
  --artifact /results/qwen35_triton_parity.json
```

A passing Radix-disabled control is diagnostic only; the Radix-enabled row must
also pass both checks. Before release, repeat the second phase against a clean
checkout of the pinned `sglang-miles` commit and record that commit with the
artifact; this prevents matching changes on both development server modes from
masking target-policy drift.

For real-data EAGLE testing, record proposal count, accepted draft count,
request-level `spec_accept_length`, and the histogram returned in response
metadata. An exact acceptance length of `1.0` on every request is suspicious and
must not be accepted without checking draft probabilities and accounting.

### Throughput command

Use the same command for every server row, changing only the output filename.
The following is a ShareGPT example:

```bash
python -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --dataset-name sharegpt \
  --dataset-path /datasets/ShareGPT.json \
  --tokenizer "${MODEL_PATH}" \
  --num-prompts 16 \
  --sharegpt-output-len 1024 \
  --warmup-requests 0 \
  --request-rate inf \
  --max-concurrency 3 \
  --disable-stream \
  --disable-tqdm \
  --cache-report \
  --flush-cache \
  --seed 2026 \
  --output-file /results/row.jsonl
```

Add `--return-logprob` for the matching logprob row. `bench_serving` reports
`avg_spec_accept_length` from server-lifetime counters. Do not mix earlier
requests or multiple benchmark variants into that value. The example relies on
server-startup graph warmup and uses a fresh server with no client warmup. If
additional warmup requests are required, use request-level acceptance metadata
for the measured corpus or restart before the measured row.

Report:

```text
acceptance_fraction = accept_length / draft_tokens
target_tps = matching_normal_tps * acceptance_fraction
target_efficiency = dvr_tps / target_tps
det_speedup = dvr_tps / matching_deterministic_tps
```

The NVLink performance target is `target_efficiency >= 0.95` after warmup, and
DVR should be faster than the matched ordinary deterministic baseline. Always
report absolute throughput and acceptance as well; a high target efficiency
does not compensate for an acceptance regression.

Record `git rev-parse HEAD`, `nvidia-smi topo -m`, GPU clocks, driver and CUDA
versions, all exported environment variables above, and the expanded launcher
command with every result.

### FA3 scheduling note

FA3 currently follows upstream behavior and keeps a CPU mirror of sequence
lengths for self-DVR. The mirror gives FA3 an accurate `max_seq_len_k` for its
split heuristic. The small D2H occurs once per DVR proposal block and may
overlap the post-publish rollback tail. Its presence is not a correctness
failure and should not be patched out during qualification.

If FA3 throughput regresses, measure the host wait, the GPU gap between rollback
and the next draft, the selected split configuration, and per-step attention
time. Treat the mirror as a DVR bottleneck only when a matched A/B demonstrates
a material unhidden gap. Triton does not require this CPU mirror.

### Log audit

Before accepting a result, retain the expanded launcher command and verify:

- every full-attention phase resolves to Triton or FA3;
- GDN prefill/verify uses the Triton linear-attention backend;
- all measured decode batch sizes were captured without eager fallback;
- target prefill and verify are deterministic while self-draft uses normal fast
  decode settings;
- an RL weight update with graph recapture rebuilt all DVR-owned graph runners
  before the first post-update request;
- custom all-reduce and communication fusion, when eligible, are confined to
  provisional self-draft capture rather than target prefill or verify;
- `max_running_requests` was not silently reduced;
- the requested Radix policy was active;
- no unexpected DeepGEMM precompile, illegal memory access, device assertion,
  or CUDA graph replay fallback occurred.

### Local pre-push gate

Run the stable unit suite from the repository root:

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/batch_invariant_ops/test_batch_invariant_ops.py \
  test/registered/unit/layers/test_dvr_gdn.py \
  test/registered/unit/model_executor/test_dvr_cuda_graph_runner.py \
  test/registered/unit/server_args/test_dvr_server_args.py \
  test/registered/unit/speculative/test_dvr_state.py \
  test/registered/unit/speculative/test_dvr_worker.py \
  test/registered/unit/speculative/test_dvr_sampling.py \
  test/registered/unit/managers/test_batch_result_processor_spec_grammar.py \
  test/registered/unit/model_executor/test_pool_configurator.py \
  test/registered/unit/layers/moe/test_fused_moe_triton_config.py
```

Also run `git diff --check` and `py_compile` for every changed Python file. The
manual extracted-model matrix is specified in `AGENTS.md`; it is a functional
gate, not a release performance benchmark.
