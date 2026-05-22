# DVR v5 Temporary Machine Debugging Guide

This is a temporary handoff document for moving DVR v5 debugging to a new
machine. It records the environment, launch flags, benchmark setup, and
performance lessons from the current A40 machine. It is not intended as a
polished user-facing SGLang document.

## Worktrees and Environments

- Current v5 worktree: `/home/hwj/dvr_qwen3_5/sglang-dvr-clean-v5`
- Original DVR worktree: `/home/hwj/dvr_qwen3_5/sglang`
- Patch-view worktree: `/home/hwj/dvr_qwen3_5/sglang-dvr-v5-patch-view`
- v5 conda env: `dvr_dev`
- Original DVR conda env: `dvr_ori`

Use the v5 env for all current-branch tests:

```bash
conda run --no-capture-output -n dvr_dev <command>
```

The current machine used a conda-provided compiler for CUDA extension builds:

```bash
conda install -n dvr_dev -c conda-forge gcc_linux-64 gxx_linux-64
conda install -n dvr_ori -c conda-forge gcc_linux-64 gxx_linux-64
```

For local builds, export the conda compiler through NVCC:

```bash
export NVCC_PREPEND_FLAGS="-ccbin /home/hwj/.miniconda3/envs/dvr_dev/bin/x86_64-conda-linux-gnu-g++ -allow-unsupported-compiler"
```

If downloading datasets or models from this machine, unset proxy variables
first when proxy traffic is limited:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

## Local Models and Dataset

- Small smoke model: `/mnt/data/hwj/Qwen3.5-0.8B`
- Qwen3.5 MoE model: `/mnt/data/hwj/Qwen3.5-35B-A3B/`
- Primary DVR performance model: `/mnt/data/hwj/Qwen3-Next-80B-A3B-Instruct`
- ShareGPT dataset:
  `/mnt/data/hwj/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json`

The 80B tests below use Qwen3-Next-80B-A3B-Instruct with tensor parallel 4.

## Recommended Runtime Defaults

Use Triton for both attention and linear attention when validating DVR
correctness. FlashInfer can be useful for some normal serving paths, but in the
current DVR experiments Triton was the reliable backend for KL=0 style checks.

Recommended common server flags for 80B A40 TP=4 tests:

```bash
python -m sglang.launch_server \
  --model-path /mnt/data/hwj/Qwen3-Next-80B-A3B-Instruct \
  --host 127.0.0.1 \
  --port <PORT> \
  --tp-size 4 \
  --context-length 8192 \
  --max-total-tokens 6144 \
  --mem-fraction-static 0.90 \
  --max-running-requests 4 \
  --max-mamba-cache-size 16 \
  --attention-backend triton \
  --linear-attn-backend triton \
  --cuda-graph-bs 1 2 3 4 \
  --cuda-graph-max-bs 4 \
  --skip-server-warmup
```

Recommended DVR additions:

```bash
--enable-deterministic-inference \
--speculative-algorithm DECODE_VERIFY_ROLLBACK \
--speculative-num-steps 15 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 16
```

Recommended environment:

```bash
export PYTHONPATH=python
export SGLANG_RETURN_ORIGINAL_LOGPROB=True
export SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE=64
export NVCC_PREPEND_FLAGS="-ccbin /home/hwj/.miniconda3/envs/dvr_dev/bin/x86_64-conda-linux-gnu-g++ -allow-unsupported-compiler"
```

`SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE=64` is the current conservative choice.
On this A40 machine, tile 32 and tile 64 were effectively tied, while larger
tiles were slower:

| Tile size | 80B DVR output throughput |
| --- | ---: |
| 32 | 158.45 tok/s |
| 64 | 158.35 to 158.58 tok/s |
| 128 | 156.70 tok/s |
| 256 | 154.73 tok/s |
| 512 | 153.44 tok/s |
| 1024 | 149.35 tok/s |
| 2048 | 146.99 tok/s |

Do not disable DeepGEMM or JIT by default on the A40 machine. The current tests
showed those paths can run, and disabling them did not improve the observed DVR
throughput.

## Benchmark Command Shape

Use a longer output sequence for throughput tests, otherwise DVR overhead can
dominate the result. The latest 80B comparison used ShareGPT with 16 prompts,
1024 output tokens, and concurrency 3:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port <PORT> \
  --dataset-name sharegpt \
  --dataset-path /mnt/data/hwj/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 16 \
  --sharegpt-output-len 1024 \
  --sharegpt-context-len 8192 \
  --max-concurrency 3 \
  --seed 2026 \
  --warmup-requests 1
```

Run both `return_logprob=False` and `return_logprob=True` variants. In recent
80B tests the logprob variant was close to the no-logprob variant, so a large
gap usually points to a configuration or code-path issue rather than expected
overhead.

## Current Best 80B Results on A40 TP=4

The latest post-custom-allreduce test directory was:
`/home/hwj/dvr_qwen3_5/dvr_v5_post_custom_ar_fulltest_20260523`.

Common setup:

- Model: `/mnt/data/hwj/Qwen3-Next-80B-A3B-Instruct`
- TP: 4
- Backend: Triton attention and Triton linear attention
- Dataset: ShareGPT
- Prompts: 16
- Output length: 1024 tokens
- Max concurrency: 3
- `SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE=64`

Observed throughput:

| Mode | return_logprob | Output throughput | Total throughput | Mean TTFT | Mean TPOT |
| --- | --- | ---: | ---: | ---: | ---: |
| Normal non-DVR, non-deterministic | False | 214.76 tok/s | 275.34 tok/s | 202.25 ms | 12.64 ms |
| Normal non-DVR, non-deterministic | True | 212.91 tok/s | 272.98 tok/s | 190.73 ms | 12.76 ms |
| DVR | False | 158.58 tok/s | 203.31 tok/s | 370.10 ms | 17.08 ms |
| DVR | True | 157.11 tok/s | 201.43 tok/s | 366.10 ms | 17.23 ms |

The current DVR-to-normal ratio is about 73.8% on this A40 TP=4 box. That is
below the original-branch H200 claim of roughly 80%, but the comparison is not
hardware-equivalent.

Important interpretation:

- The A40 topology is PCIe-only for TP=4. The runtime logs say custom allreduce
  is disabled because it is unsupported on more than two PCIe-only GPUs.
- Therefore, keeping DVR custom allreduce enabled in server args is expected to
  be mostly a no-op on this machine.
- A full NVLink or H200 system is the right place to validate whether custom
  allreduce closes the gap.

## Correctness Smoke Tests

Latest smoke results:

- 0.8B DVR KL smoke:
  `/home/hwj/dvr_qwen3_5/dvr_v5_post_custom_ar_fulltest_20260523/qwen35_0p8b_dvr_kl.log`
  - `ALL_OK True`
  - `maxdiff=0.0`
  - `kl_proxy=0.0`
- 80B DVR KL smoke:
  `/home/hwj/dvr_qwen3_5/dvr_v5_post_custom_ar_fulltest_20260523/qwen3next80b_dvr_tile64_kl.log`
  - `ALL_OK True`
  - `maxdiff=0.0`
  - `kl_proxy=0.0`

The preferred KL=0 style check is:

1. Decode normally.
2. Concatenate the generated tokens back onto the prompt.
3. Re-run prefill.
4. Compare the decode logits with the corresponding prefill logits.

Use Triton backends for this check first. Backend mask differences can make
FlashInfer comparisons noisy even when the high-level behavior is correct.

## DVR-Specific Performance Notes

The current v5 implementation intentionally keeps the self-draft path different
from the deterministic verify path:

- The DVR verify path remains deterministic.
- The self-draft decode path uses the dedicated DVR self-draft CUDA graph
  context and temporarily relaxes deterministic decode behavior.
- The self-draft CUDA graph capture skips unnecessary attention backend graph
  state initialization.
- Mamba scheduler tracking uses the lighter `no_buffer` strategy for self-draft
  capture.
- DVR verify uses `CaptureHiddenMode.NULL` with dummy zero-width hidden states
  instead of `CaptureHiddenMode.FULL`, avoiding full hidden-state materialization
  when the target path only needs token/logit verification.

If throughput regresses, check these first:

1. Self-draft accidentally using deterministic kernels again.
2. Self-draft Mamba tracking running more than necessary.
3. DVR verify falling back to `CaptureHiddenMode.FULL`.
4. Triton decode split tile reset to a large default.
5. CUDA graph batch sizes not covering the benchmark concurrency.
6. Custom allreduce disabled by topology or unsupported GPU interconnect.
7. Hidden server flags left over from experiments, especially backend,
   determinism, radix-cache, and memory-fraction settings.

## Custom Allreduce Policy

The original DVR branch keeps custom allreduce available during DVR
deterministic inference. The v5 branch now follows the same policy at the
server-argument level:

- DVR + deterministic + TP>1 keeps `disable_custom_all_reduce=False` unless the
  user explicitly disables it.
- It sets `NCCL_ALGO=allreduce:tree`.
- On unsupported hardware, SGLang can still disable the custom path at runtime.

Do not add temporary runtime mutation of the TP group's communicator unless a
new experiment proves it is necessary. Keeping the policy in server args is less
invasive and easier to reason about.

## Minimal Validation Checklist After Moving Machines

1. Confirm branch and worktree:

   ```bash
   cd /home/hwj/dvr_qwen3_5/sglang-dvr-clean-v5
   git status --short --branch
   git log --oneline -5
   ```

2. Confirm imports and edited files compile:

   ```bash
   conda run --no-capture-output -n dvr_dev python -m py_compile \
     python/sglang/srt/model_executor/dvr_draft_cuda_graph_runner.py \
     python/sglang/srt/server_args.py \
     python/sglang/srt/speculative/dvr_worker.py
   ```

3. Run the 0.8B DVR KL smoke.
4. Run the 80B DVR KL smoke.
5. Run 80B normal non-DVR throughput with no logprob and with logprob.
6. Run 80B DVR throughput with no logprob and with logprob.
7. Compare output throughput, not just total throughput.
8. Check server logs for custom allreduce support and unexpected backend
   fallbacks.

## Commit Baseline

This document was added after these relevant DVR v5 commits:

- `325915ada dvr: keep custom allreduce available`
- `d0bfaf19c dvr: reduce self-draft tracking overhead`
- `91019ed7b dvr: run self draft with nondeterministic kernels`
- `679bbb5b6 dvr: reject zero-step chain mode`
- `20734ba23 dvr: fix GDN state input cache head shapes`

