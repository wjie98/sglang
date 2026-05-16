# DVR GDN Adapter Boundary Plan

This file tracks the next adapter cleanup on top of `dvr-v4-performance-opt`.
The goal is to keep the verified behavior unchanged while making the GDN DVR
integration look like a small model adapter rather than scattered DVR logic in
`gdn_backend.py`.

## Requirements

- Keep the code close to the current working v4 behavior: Qwen3 and Qwen3.5
  must keep strict KL=0 after the refactor.
- Do not rename or reshape the existing memory-pool fields in this round. That
  would increase merge risk and does not improve the adapter boundary.
- Do not introduce Mamba- or FLA-specific names in the adapter public surface.
  The concrete kernels may still come from FLA today, but the adapter should
  describe DVR concepts.
- Avoid abstract base classes. A small concrete adapter is easier to review and
  closer to SGLang's style.
- Keep request-level speculative logic in `dvr_worker.py`; keep layer tensor
  production in `gdn_backend.py`; move DVR rolling-window mechanics into the
  adapter.

## Phase 1: Add a Gated-State Adapter Entry

Add `DVRGatedStateAdapter` in
`python/sglang/srt/layers/attention/linear/dvr_state_adapter.py`.

Responsibilities:

- bind chunkwise and recurrent state kernels through `DVRStateKernels`;
- cache extend q/k/v/g/beta tails into the DVR state window;
- run the fixed `CHUNK_SIZE + draft` chunkwise verify window and return only
  the draft suffix to the normal target-verify path;
- compute commit plans and rebuild live recurrent state after accept/reject;
- keep conv-window export helpers available for the backend.

The first version can still use q/k/v/g/beta internally. The public boundary is
the adapter object, so future KDA/Mamba-like modules can add their own adapter
without learning the DVR worker details.

Validation after Phase 1:

- `py_compile` for `dvr_state_adapter.py`, `gdn_backend.py`, `dvr_worker.py`,
  and `dvr_utils.py`.
- `git diff --check`.

Status: completed. Added `DVRGatedStateAdapter` as the model-facing entry for
GDN-like gated linear-state layers. The adapter now owns the rolling state-input
window, chunkwise verify call, boundary-state write, and live-state rebuild.

## Phase 2: Collapse GDN Backend Calls Onto the Adapter

Change `gdn_backend.py` so it instantiates and calls the adapter instead of
manually composing the low-level DVR helpers.

Expected backend shape:

- `self.dvr_state_adapter.set_kernels(...)` near kernel dispatcher setup;
- extend path calls `self.dvr_state_adapter.cache_extend_tail(...)`;
- target verify calls `self.dvr_state_adapter.run_chunkwise_verify(...)`;
- verify postprocess calls `self.dvr_state_adapter.commit_after_verify(...)`.

Keep GDN-specific tensor generation in the backend:

- convolution execution;
- splitting `mixed_qkv`;
- gated `g/beta` computation.

Validation after Phase 2:

- same static checks as Phase 1;
- Qwen3.5 no-graph smoke with `max_new=17`.

Status: completed. `gdn_backend.py` now calls the adapter for extend-tail
caching, target-verify chunk scan, and verify postprocess commit. The backend
still owns GDN tensor production and conv execution.

## Phase 3: Naming and Local Readability Cleanup

Rename only local helpers where it reduces ambiguity:

- remove `qkvg_beta` from backend method names where the adapter boundary makes
  that detail local;
- prefer `state_window`, `verified_tail_lens`, and `draft_token_num`;
- avoid new global helper functions unless they are used in more than one
  location.

Do not edit unrelated runner, scheduler, memory-pool, or backend files.

Validation after Phase 3:

- static checks;
- Qwen3.5 no-graph `max_new=17,65,129`;
- Qwen3 attention no-graph `max_new=17,65,129`.

Status: completed. Local backend helper names were cleaned up:
`_set_dvr_state_kernels` and `_cache_dvr_extend_state_tail` now describe the
adapter boundary instead of exposing q/k/v/g/beta in the backend method names.

## Phase 4: CUDA Graph Regression

Run the existing DVR CUDA graph path with the refactored adapter.

Validation:

- Qwen3.5 cuda graph `max_new=17,65,129,257`, strict KL=0;
- if successful, refresh the v4 patch-view worktree for IDE review.

Status: completed. Qwen3.5 cuda graph passed strict KL=0 for
`max_new=17,65,129,257`.

## Phase 5: Report and Commit

Update this file with the executed status, commit the refactor, and report:

- files changed;
- exact validation commands and outcomes;
- any remaining adapter limitations.

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
case=3 max_new=257 gen=248 maxdiff=0.0 kl=0.0 accept=0.8851851851851852
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
case=1 max_new=65  maxdiff=0.0 kl=0.0 accept=0.84
case=2 max_new=129 maxdiff=0.0 kl=0.0 accept=0.8814814814814815
ALL_OK True
```

Notes:

- The old public `--speculative-dvr-chunk-boundary-verify` flag has already
  been removed in v4. `DECODE_VERIFY_ROLLBACK` now selects the DVR path.
- Qwen3.5 required `--mem-fraction-static 0.75` locally. `0.45` failed during
  memory-pool sizing before serving started.
