# DVR GDN Adapter Hook Plan

This plan continues from commit `4ebe18301` on branch
`dvr-v4-performance-opt`. The goal is to reduce the remaining DVR-specific
control flow in `gdn_backend.py` by turning the adapter into a small set of
hooks that decide internally whether DVR is active.

## Goals

- Keep `gdn_backend.py` close to normal GDN flow. It should call adapter hooks
  at natural extension points and fall back to the normal path when the hook
  returns `None`.
- Move DVR activation checks, `batch_size`/`draft_token_num` derivation, and
  intermediate-cache lookup into the adapter.
- Avoid additional registries or private setup methods.
- Keep public adapter names DVR-oriented and backend-agnostic enough for future
  KDA/Mamba-like modules.
- Do not implement the grouped recurrent Triton kernel in this patch. Record it
  as a later performance step.

## Phase 1: Adapter Context Helpers

Add lightweight helpers in `DVRGatedStateAdapter`:

- `is_verify_enabled(state_cache, is_target_verify)`;
- `verify_shape(seq_len, spec_info)`;
- internal access to `intermediate_ssm` and `intermediate_conv_window`.

Expected effect:

- `gdn_backend.py` no longer computes `is_dvr_target_verify`;
- `gdn_backend.py` no longer passes intermediate state/cache tensors into DVR
  adapter calls.

Validation:

- `py_compile`
- `git diff --check`

Status: completed. The adapter now owns DVR verify activation checks,
`batch_size`/`draft_token_num` derivation, and intermediate-cache lookup for
the GDN verify hooks.

## Phase 2: Convert Verify Conv to an Optional Hook

Change `run_verify_conv(...)` into an optional hook:

- return `None` when DVR is not active;
- otherwise run the DVR draft-suffix conv path and export conv windows.

`gdn_backend.py` structure:

```python
dvr_mixed_qkv = self.dvr_state_adapter.prepare_verify_conv(...)
if dvr_mixed_qkv is not None:
    mixed_qkv = dvr_mixed_qkv
else:
    # existing target-verify conv path
```

Validation:

- static checks;
- Qwen3.5 no-graph smoke `max_new=17`.

Status: completed. `prepare_verify_conv(...)` returns `None` for non-DVR paths
and runs the DVR conv-window export only when the state window exists.

## Phase 3: Convert Chunkwise Verify to an Optional Hook

Change `run_verify_chunkwise(...)` into an optional hook:

- return `None` when DVR is not active;
- otherwise run DVR's internal `CHUNK_SIZE + draft` chunkwise scan and return
  only the draft suffix.

`gdn_backend.py` structure:

```python
dvr_out = self.dvr_state_adapter.run_verify_chunkwise(...)
if dvr_out is not None:
    core_attn_out = dvr_out
else:
    # existing target_verify kernel
```

Validation:

- Qwen3.5 no-graph `max_new=17,65,129`;
- Qwen3 attention no-graph `max_new=17,65,129`.

Status: completed. `run_verify_chunkwise(...)` now behaves as an optional hook;
the normal target-verify kernel remains the fallback.

## Phase 4: CUDA Graph Regression

Run:

- Qwen3.5 cuda graph `max_new=17,65,129,257`.

Expected result: strict `maxdiff=0.0`, `kl=0.0`.

Status: completed.

## Phase 5: Report and Commit

Update this file with validation results, commit the change, and refresh the v4
patch-view worktree.

Status: completed.

## Validation Log

Static checks:

```bash
PYTHONPYCACHEPREFIX=/tmp/sglang_pycache \
conda run -n dvr_dev python -m py_compile \
  python/sglang/srt/layers/attention/linear/dvr_state_adapter.py \
  python/sglang/srt/layers/attention/linear/gdn_backend.py \
  python/sglang/srt/speculative/dvr_worker.py \
  python/sglang/srt/speculative/dvr_utils.py

git diff --check
```

Result: passed.

Qwen3.5 no-graph:

```bash
SGLANG_RETURN_ORIGINAL_LOGPROB=True \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 \
SGLANG_ENABLE_SPEC_V2=1 \
PYTHONPATH=python \
conda run --no-capture-output -n dvr_dev python -m sglang.launch_server \
  --model-path /home/hwj/Qwen3.5-0.8B \
  --host 127.0.0.1 \
  --port 30124 \
  --speculative-algorithm DECODE_VERIFY_ROLLBACK \
  --speculative-num-steps 15 \
  --speculative-num-draft-tokens 16 \
  --speculative-eagle-topk 1 \
  --page-size 16 \
  --mem-fraction-static 0.75 \
  --attention-backend triton \
  --linear-attn-backend triton \
  --sampling-backend pytorch \
  --enable-deterministic-inference \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --skip-server-warmup

PYTHONPYCACHEPREFIX=/tmp/sglang_pycache \
PYTHONPATH=python \
conda run -n dvr_dev python test/manual/dvr/test_dvr_batch_kl.py \
  --base-url http://127.0.0.1:30124 \
  --max-new 17,65,129
```

Result:

```text
case=0 max_new=17  maxdiff=0.0 kl=0.0 accept=1.0
case=1 max_new=65  maxdiff=0.0 kl=0.0 accept=1.0
case=2 max_new=129 maxdiff=0.0 kl=0.0 accept=1.0
ALL_OK True
```

Qwen3 attention no-graph:

```bash
SGLANG_RETURN_ORIGINAL_LOGPROB=True \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 \
SGLANG_ENABLE_SPEC_V2=1 \
PYTHONPATH=python \
conda run --no-capture-output -n dvr_dev python -m sglang.launch_server \
  --model-path /home/hwj/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 30124 \
  --speculative-algorithm DECODE_VERIFY_ROLLBACK \
  --speculative-num-steps 15 \
  --speculative-num-draft-tokens 16 \
  --speculative-eagle-topk 1 \
  --page-size 16 \
  --mem-fraction-static 0.45 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --enable-deterministic-inference \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --skip-server-warmup

PYTHONPYCACHEPREFIX=/tmp/sglang_pycache \
PYTHONPATH=python \
conda run -n dvr_dev python test/manual/dvr/test_dvr_batch_kl.py \
  --base-url http://127.0.0.1:30124 \
  --max-new 17,65,129
```

Result:

```text
case=0 max_new=17  maxdiff=0.0 kl=0.0 accept=1.0
case=1 max_new=65  maxdiff=0.0 kl=0.0 accept=1.0
case=2 max_new=129 maxdiff=0.0 kl=0.0 accept=0.88
ALL_OK True
```

Qwen3.5 cuda graph:

```bash
SGLANG_RETURN_ORIGINAL_LOGPROB=True \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 \
SGLANG_ENABLE_SPEC_V2=1 \
PYTHONPATH=python \
conda run --no-capture-output -n dvr_dev python -m sglang.launch_server \
  --model-path /home/hwj/Qwen3.5-0.8B \
  --host 127.0.0.1 \
  --port 30124 \
  --speculative-algorithm DECODE_VERIFY_ROLLBACK \
  --speculative-num-steps 15 \
  --speculative-num-draft-tokens 16 \
  --speculative-eagle-topk 1 \
  --page-size 16 \
  --mem-fraction-static 0.75 \
  --attention-backend triton \
  --linear-attn-backend triton \
  --sampling-backend pytorch \
  --enable-deterministic-inference \
  --disable-piecewise-cuda-graph \
  --skip-server-warmup

PYTHONPYCACHEPREFIX=/tmp/sglang_pycache \
PYTHONPATH=python \
conda run -n dvr_dev python test/manual/dvr/test_dvr_batch_kl.py \
  --base-url http://127.0.0.1:30124 \
  --max-new 17,65,129,257
```

Result:

```text
case=0 max_new=17  maxdiff=0.0 kl=0.0 accept=1.0
case=1 max_new=65  maxdiff=0.0 kl=0.0 accept=1.0
case=2 max_new=129 maxdiff=0.0 kl=0.0 accept=1.0
case=3 max_new=257 gen=179 maxdiff=0.0 kl=0.0 accept=0.8238095238095238
ALL_OK True
```

Notes:

- Long-generation cases may stop before `max_new` after EOS; the strict oracle
  still matched all returned tokens/logprobs exactly.
- The grouped recurrent Triton kernel remains deferred. This patch only moves
  adapter hook ownership and keeps the current callable path.

## Deferred Performance Step

`recurrent_state_from_qkvg_beta` should eventually be replaced by a dedicated
grouped recurrent Triton kernel, similar in spirit to the reference DVR branch.
That optimization should keep the adapter API unchanged and only swap the
callable bound to `DVRStateKernels.recurrent_state`.
