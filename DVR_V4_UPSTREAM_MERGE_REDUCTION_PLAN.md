# DVR v4 Upstream Merge Reduction Plan

This plan continues from `dvr-v4-performance-opt` after the GDN internal
`FLA_CHUNK_SIZE + draft` verify window was moved inside the FLA scan.

## Goals

1. Keep the verified KL=0 behavior for Qwen3 attention-only and Qwen3.5 GDN.
2. Reduce the remaining GDN memory overhead.
3. Move DVR-specific GDN mechanics behind small Mamba/DVR helpers.
4. Keep the upstream merge surface small and easy to review.

## Step 1: Direct Boundary State Buffer

Current v4 stores GDN target-verify chunk boundary state in
`intermediate_ssm[:, :, FLA_CHUNK_SIZE]`-style storage. DVR only needs the
single state at the first chunk boundary of the fixed verify window, so store it
in a one-slot boundary state buffer when `enable_dvr_qkvg_beta_cache=True`.

Validation:

- Static Python compile.
- `git diff --check`.
- Qwen3.5 no-graph KL=0 for lengths crossing 64.
- Qwen3.5 CUDA graph KL=0 for lengths crossing 64.

## Step 2: Grouped Recurrent State Helper

The current live-state rebuild calls the recurrent FLA path per layer. Add a
small Triton/grouped helper next to chain reject sampling so DVR can rebuild
accepted live states with fewer launches. The first version may keep the same
semantics and fall back to the current implementation if shape assumptions are
not met.

Validation:

- Compare Qwen3.5 strict KL=0 before and after.
- Include unequal accepted lengths in batch if practical.

## Step 3: Move DVR/GDN Mechanics Into Mamba DVR Helpers

Keep generic `gdn_backend.py` readable by moving q/k/v/g/beta cache writes,
internal verify scan preparation, boundary-state writeback, and conv-step
mapping into `mamba_dvr_utils.py`. The backend should expose data produced by
GDN kernels; request-level accept/reject semantics stay in `dvr_worker.py`.

Validation:

- Static checks after each move.
- Qwen3 and Qwen3.5 KL=0 smoke tests.

## Step 4: Upstream Diff Hygiene

Do not include root-level experimental docs in a final upstream patch series.
Keep developer notes in archive/local files. Keep only manual test scripts that
are useful for reproducing KL=0.

Validation:

- `git diff --name-only upstream/sglang-miles..HEAD` review.

## Step 5: Server Args Cleanup

Keep DVR launch defaults in one server-args helper:

- normalize DVR chunk page size;
- force `mamba_scheduler_strategy=extra_buffer` for GDN DVR;
- force `mamba_track_interval=FLA_CHUNK_SIZE`;
- force fp32 Mamba/GDN SSM state;
- disable piecewise CUDA graph for DVR until explicitly supported;
- reject unsupported DP attention paths.

Validation:

- Launch Qwen3 and Qwen3.5 with minimal DVR flags.
- Confirm warnings are centralized and deterministic.

## Deferred: FLA Determinism Patch Split

The FLA determinism edits should eventually be separated into their own commit
or upstream discussion:

- fp32 chunk boundary `h`;
- `chunk_o` cast when `h` is fp32;
- gated RMSNorm `MAX_ROWS_PER_BLOCK=1`.

Do not change this in the current pass.
