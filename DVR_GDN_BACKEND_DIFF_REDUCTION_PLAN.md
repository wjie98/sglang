# DVR GDN Backend Diff Reduction Plan

This plan continues from commit `f2b57146a` on branch
`dvr-v4-performance-opt`. The purpose is to reduce the visible DVR surface in
`gdn_backend.py` while keeping the current strict KL=0 behavior.

## Goals

- Make `gdn_backend.py` look close to the original GDN backend: it should mainly
  run conv, split tensors, call kernels, and expose one post-verify commit entry.
- Move DVR rolling-window details into
  `python/sglang/srt/layers/attention/linear/dvr_state_adapter.py`.
- Avoid Mamba/FLA names in the adapter public surface. Concrete callables may
  still come from FLA-backed GDN kernels today.
- Avoid over-abstracting. Use one concrete DVR adapter instead of protocols,
  registries, or extra private setup methods.
- Keep the recurrent state kernel optimization as a separate performance phase.
  It should not be mixed with this structural cleanup.

## Phase 1: Inline Kernel Binding

Replace the private `_set_dvr_state_kernels()` method in GDN backend with direct
adapter construction:

```python
self.dvr_state_adapter = DVRGatedStateAdapter(
    kernels=DVRStateKernels(
        chunk_scan=self.kernel_dispatcher.extend,
        recurrent_state=self.kernel_dispatcher.recurrent_state_from_qkvg_beta,
    )
)
```

Expected result:

- no runtime kernel registration helper in `gdn_backend.py`;
- no repeated setup call in decode/extend.

Validation:

- `py_compile`
- `git diff --check`

Status: completed. GDN backend now binds DVR kernels directly when constructing
`DVRGatedStateAdapter`; the repeated private setup method was removed.

## Phase 2: Move Local DVR Helpers Into Adapter

Remove these GDN-backend helper methods:

- `_cache_dvr_extend_state_tail`
- `_run_dvr_verify_conv`
- `_run_dvr_verify_chunk_scan`

Add adapter methods that keep the same behavior:

- `maybe_cache_extend_tail(...)`
- `run_verify_conv(...)`
- `run_verify_chunkwise(...)`

The GDN backend will still compute q/k/v/g/beta and pass layer metadata, but it
will no longer own DVR state-input cache writes, conv-window export, or
chunkwise verify window mechanics.

Validation:

- `py_compile`
- `git diff --check`
- Qwen3.5 no-graph smoke: `max_new=17`

Status: completed. The three backend-local DVR helpers were removed. Extend-tail
cache export, target-verify conv-window export, and chunkwise verify suffix
selection now live behind adapter methods.

## Phase 3: Keep Commit Entry Minimal

Keep `commit_dvr_state_after_verify()` in GDN backend because `dvr_worker.py`
needs a backend entry point. The body should remain a one-call bridge to the
adapter, passing the full speculative state cache and scatter callable.

Validation:

- Qwen3.5 no-graph: `max_new=17,65,129`
- Qwen3 attention no-graph: `max_new=17,65,129`

Status: completed. `commit_dvr_state_after_verify()` remains as the backend
entry used by `dvr_worker.py`, but the body is still a one-call bridge to the
adapter.

## Phase 4: CUDA Graph Regression

Run the same GDN test with CUDA graph enabled:

- Qwen3.5 cuda graph: `max_new=17,65,129,257`

Expected result: strict `maxdiff=0.0`, `kl=0.0`.

Status: completed.

## Phase 5: Recurrent Kernel Follow-up

The current `recurrent_state_from_qkvg_beta` path is functionally correct but
not the final performance shape. The reference DVR branch uses a more direct
grouped recurrent kernel. A future performance patch should:

- add a grouped recurrent Triton kernel for DVR live-state rebuild;
- keep the adapter API unchanged by swapping only the callable bound into
  `DVRStateKernels.recurrent_state`;
- compare latency and strict KL against the current implementation.

This phase is intentionally not implemented in this structural cleanup.

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
case=1 max_new=65  maxdiff=0.0 kl=0.0 accept=0.9733333333333334
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
case=2 max_new=129 maxdiff=0.0 kl=0.0 accept=0.8533333333333334
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
case=2 max_new=129 gen=69  maxdiff=0.0 kl=0.0 accept=0.7047619047619048
case=3 max_new=257 gen=233 maxdiff=0.0 kl=0.0 accept=0.8549019607843137
ALL_OK True
```

Notes:

- Some long-generation cases stopped before `max_new` because EOS was produced.
  The strict full-prefill oracle still matched all returned tokens/logprobs.
- `recurrent_state_from_qkvg_beta` remains the existing callable path. The
  dedicated grouped Triton recurrent kernel is recorded as a follow-up
  performance patch rather than part of this diff-reduction change.
