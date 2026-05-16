# DVR GDN Adapter Hook Rename Plan

This plan continues from commit `297a63c0f` on branch
`dvr-v4-performance-opt`. The previous patch moved DVR verify activation,
shape derivation, and intermediate cache lookup into the adapter. This patch
finishes the public hook naming and reduces the remaining fallback-only local
variables in `gdn_backend.py`.

## Goals

- Make adapter calls describe GDN extension points rather than DVR internals.
- Keep `gdn_backend.py` responsible for ordinary GDN flow and fallback paths.
- Keep DVR rolling-window, chunkwise verify, and state-input cache details in
  `dvr_state_adapter.py`.
- Avoid new registration helpers, protocols, registries, or abstract classes.
- Do not change numerical kernels in this patch.

## Phase 1: Rename Adapter Hooks

Rename adapter methods:

- `prepare_verify_conv(...)` -> `maybe_process_target_verify_conv(...)`
- `run_verify_chunkwise(...)` -> `maybe_process_target_verify_state(...)`
- `maybe_cache_extend_tail(...)` -> `maybe_cache_extend_state_inputs(...)`

The methods still return `None` when DVR is not enabled, so normal target
verify remains the fallback path.

Validation:

- `py_compile`
- `git diff --check`

Status: completed. Hook names now describe GDN integration points:
`maybe_process_target_verify_conv`,
`maybe_process_target_verify_state`, and
`maybe_cache_extend_state_inputs`.

## Phase 2: Delay Fallback-Only Variables in GDN Backend

Move fallback-only variables closer to the fallback branch:

- `intermediate_conv_window_cache`
- `intermediate_state_indices`
- `intermediate_state_cache`
- `batch_size`
- `draft_token_num`

The DVR hook already derives or reads these values internally. Keeping these
variables only where fallback needs them makes the backend read less like a DVR
implementation.

Validation:

- Qwen3.5 no-graph `max_new=17,65,129`
- Qwen3 attention no-graph `max_new=17,65,129`

Status: completed. Fallback-only intermediate variables are now created in the
fallback branch rather than before the DVR hook.

## Phase 3: CUDA Graph Regression

Run:

- Qwen3.5 cuda graph `max_new=17,65,129,257`

Expected result: strict `maxdiff=0.0`, `kl=0.0`.

Status: completed.

## Phase 4: Report, Commit, Refresh Patch View

Update this file with validation outcomes, commit the patch, and refresh
`/home/hwj/dvr_qwen3_5/sglang-dvr-v4-patch-view`.

Status: completed.

## Deferred Kernel Work

`recurrent_state_from_qkvg_beta` should still be optimized later with a
dedicated grouped recurrent Triton kernel. That is a performance patch and
should only swap the callable bound in `DVRStateKernels`, leaving these hook
interfaces unchanged.

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

```text
case=0 max_new=17  maxdiff=0.0 kl=0.0 accept=1.0
case=1 max_new=65  maxdiff=0.0 kl=0.0 accept=0.8666666666666667
case=2 max_new=129 maxdiff=0.0 kl=0.0 accept=1.0
ALL_OK True
```

Qwen3 attention no-graph:

```text
case=0 max_new=17  maxdiff=0.0 kl=0.0 accept=1.0
case=1 max_new=65  maxdiff=0.0 kl=0.0 accept=1.0
case=2 max_new=129 maxdiff=0.0 kl=0.0 accept=0.9407407407407408
ALL_OK True
```

Qwen3.5 cuda graph:

```text
case=0 max_new=17  maxdiff=0.0 kl=0.0 accept=1.0
case=1 max_new=65  maxdiff=0.0 kl=0.0 accept=1.0
case=2 max_new=129 gen=107 maxdiff=0.0 kl=0.0 accept=0.8
case=3 max_new=257 gen=252 maxdiff=0.0 kl=0.0 accept=0.9176470588235294
ALL_OK True
```

Notes:

- Long-generation cases can stop before `max_new` after EOS; all returned
  tokens/logprobs still match the full-prefill oracle exactly.
- No numerical kernel was changed in this patch.
