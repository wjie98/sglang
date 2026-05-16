# DVR State Ops Preset Plan

Created: 2026-05-17
Branch: `dvr-v4-performance-opt`
Worktree: `/home/hwj/dvr_qwen3_5/sglang-dvr-clean-v4`

## Goal

Introduce a DVR operator layer for gated/RNN-like state layers. The adapter
should not try to be fully generic across every possible recurrent layer.
Instead, it should consume a small preset object for known layer families such
as GDN, KDA, and Mamba.

This keeps the code clearer:

- model backends only choose a DVR preset and call a few adapter hooks;
- the adapter owns DVR window/state lifecycle;
- the ops layer owns concrete kernel choices and low-level recurrent rebuild
  details;
- future GDN/KDA/Mamba differences are explicit presets, not hidden inside a
  large generic interface.

## Current Problem

The current v4 code is correct in tested modes, but the boundary is still not
ideal:

- `GDNKernelDispatcher.recurrent_state_from_qkvg_beta` is DVR-specific but lives
  in `gdn_backend.py`.
- `DVRStateKernels` is defined in `dvr_state_adapter.py`, mixing the adapter
  lifecycle with low-level kernel selection.
- GDN backend imports FLA recurrent functions only for DVR state rebuild.
- Future KDA/Mamba support would either add more DVR details into each backend
  or make the adapter overly generic.

## Target Structure

### `dvr_state_ops.py`

New file:

`python/sglang/srt/layers/attention/linear/dvr_state_ops.py`

Responsibilities:

- define `DVRStateOps`;
- provide finite presets such as `DVRStateOps.for_gdn(...)`;
- contain GDN recurrent rebuild helper currently in `GDNKernelDispatcher`;
- later host a dedicated Triton recurrent rebuild kernel.

Initial API:

```python
@dataclass
class DVRStateOps:
    chunk_scan: Optional[Callable]
    recurrent_state: Optional[Callable]
    verify_conv: Optional[Callable]
    state_scatter: Optional[Callable]
    chunk_size: int

    @classmethod
    def for_gdn(cls, kernel_dispatcher): ...

    def scan_chunkwise(...): ...
    def rebuild_recurrent_state(...): ...
```

The name intentionally avoids `fla` and `mamba`. GDN currently uses FLA-like
kernels, but DVR should depend on the semantic operation, not on one concrete
library name.

### `dvr_state_adapter.py`

Responsibilities remain:

- rolling `verified + draft + padding` q/k/v/g/beta window;
- chunk-boundary state write;
- conv-window export;
- live state rebuild and commit after verify.

Changes:

- import `DVRStateOps` from `dvr_state_ops.py`;
- remove `DVRStateKernels`;
- call `state_ops.scan_chunkwise` and `state_ops.rebuild_recurrent_state`;
- keep hook names and behavior unchanged.

### `gdn_backend.py`

Target DVR-specific surface:

```python
self.dvr_state_adapter = DVRGatedStateAdapter(
    DVRStateOps.for_gdn(self.kernel_dispatcher)
)
```

GDN backend should no longer define or import DVR recurrent rebuild helpers.

## Phase 1: Structural Migration Only

Status: completed.

Changes:

- add `dvr_state_ops.py`;
- move `DVRStateKernels` to `DVRStateOps`;
- move GDN recurrent rebuild helper from `GDNKernelDispatcher` to
  `rebuild_gdn_state_from_qkvg_beta`;
- update GDN backend to construct the adapter through `DVRStateOps.for_gdn`;
- update imports and tests.

Expected behavior:

- no intended math or scheduling change;
- current KL=0 behavior must remain unchanged.

Validation:

- `py_compile` for:
  - `dvr_state_ops.py`
  - `dvr_state_adapter.py`
  - `gdn_backend.py`
  - `test/manual/dvr/test_dvr_gdn_recurrent_state.py`
- `git diff --check`
- `test/manual/dvr/test_dvr_gdn_recurrent_state.py`, expecting `max_diff=0.0`
- Qwen3.5 no-graph KL: `max_new=17,65,129`
- Qwen3 no-graph KL: `max_new=17,65,129`
- Qwen3.5 cuda graph KL: `max_new=17,65,129,257`

Results:

- Structural migration completed:
  - `DVRStateOps` lives in `dvr_state_ops.py`;
  - `DVRStateKernels` was removed from `dvr_state_adapter.py`;
  - GDN backend constructs `DVRStateOps.for_gdn(...)`;
  - GDN backend no longer defines `recurrent_state_from_qkvg_beta`.
- `py_compile`: passed.
- `git diff --check`: passed.
- Recurrent state validation before enabling the dedicated Triton kernel:
  `max_diff=0.0`.
- Qwen3.5 no-graph before enabling the dedicated Triton kernel:
  `max_new=17,65,129`, all `maxdiff=0.0`, `kl=0.0`.
- Qwen3 attention no-graph:
  `max_new=17,65,129`, all `maxdiff=0.0`, `kl=0.0`.

## Phase 2: Dedicated DVR Recurrent Kernel

Status: completed.

Rationale:

The current varlen wrapper removes Python grouping, but still builds a packed
varlen stream with `nonzero` and temporary q/k/v/g/beta tensors. A dedicated
DVR recurrent rebuild kernel can consume dense `[rows, max_tokens, ...]`
windows plus `token_count` directly.

Proposed API:

```python
def rebuild_gdn_state_from_qkvg_beta_triton(
    q, k, v, g, beta, *, initial_state, token_count
) -> torch.Tensor:
    ...
```

Kernel semantics:

- one launch for mixed token counts;
- `token_count=0` copies the initial state;
- computes final state only, not output activations;
- preserves current FLA recurrent math:
  fp32 state accumulation, q/k l2norm, decay `g`, beta update.

Validation:

- compare Triton kernel against Phase 1 varlen wrapper with exact
  `max_diff=0.0`;
- then repeat Phase 1 endpoint KL tests.

Risk:

- exact KL=0 depends on matching the existing recurrent update order. If the
  dedicated kernel is not bitwise identical, keep the Phase 1 varlen wrapper and
  defer the kernel optimization.

Results:

- Added `rebuild_gdn_state_from_qkvg_beta_triton` in `dvr_state_ops.py`.
- The GDN preset now uses the dedicated Triton kernel by default.
- Manual validation:
  `test/manual/dvr/test_dvr_gdn_recurrent_state.py` reported
  `max_diff=0.0 triton_max_diff=0.0`.
- Qwen3.5 no-graph with Triton recurrent rebuild enabled:
  `max_new=17,65,129`, all `maxdiff=0.0`, `kl=0.0`.
- Qwen3.5 cuda graph with Triton recurrent rebuild enabled:
  `max_new=17,65,129,257`, all `maxdiff=0.0`, `kl=0.0`;
  the longest case generated 213 tokens because of EOS.

## Phase 3: Patch View And Report

Status: pending commit.

Changes:

- commit each passing phase separately;
- refresh `/home/hwj/dvr_qwen3_5/sglang-dvr-v4-patch-view`;
- report exact files, validation results, and remaining risks.
