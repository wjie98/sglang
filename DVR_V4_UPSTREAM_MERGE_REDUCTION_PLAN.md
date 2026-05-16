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

## Step 2: Compact Boundary State Buffer

Status: completed in `dvr: compact GDN boundary state cache`.

DVR now stores only the first chunk-boundary state exported by the internal
GDN `64 + draft` scan. This keeps the deterministic state commit path unchanged
while avoiding the old 64-token SSM intermediate buffer.

## Step 3: DVR Sampling Kernel

Move chain reject sampling into a DVR sampling helper and add a Triton fast path
with the existing torch implementation as fallback. Keep the public call
contract identical to EAGLE verify so self-DVR remains close to the upstream
speculative decoding flow.

Validation:

- Static Python compile and `git diff --check`.
- Qwen3/Qwen3.5 strict KL=0 smoke tests.
- Batch test where practical.

## Step 4: Move DVR/GDN Mechanics Into Mamba DVR Helpers

Status: in progress. The live-state rebuild has been moved into
`rebuild_mamba_dvr_live_state_grouped`, which keeps recurrent decode semantics
but groups layers x requests by accepted length before calling the recurrent FLA
kernel.

Keep generic `gdn_backend.py` readable by moving q/k/v/g/beta cache writes,
internal verify scan preparation, boundary-state writeback, and conv-step
mapping into `mamba_dvr_utils.py`. The backend should expose data produced by
GDN kernels; request-level accept/reject semantics stay in `dvr_worker.py`.

Validation:

- Static checks after each move.
- Qwen3 and Qwen3.5 KL=0 smoke tests.

## Step 5: Upstream Diff Hygiene

Do not include root-level experimental docs in a final upstream patch series.
Keep developer notes in archive/local files. Keep only manual test scripts that
are useful for reproducing KL=0.

Validation:

- `git diff --name-only upstream/sglang-miles..HEAD` review.

## Step 6: Server Args Cleanup

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
