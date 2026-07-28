# DVR development notes

## Fixed validation sampling

- Use the fixed DVR validation scripts. Do not replace them with one-off
  commands when comparing correctness, acceptance, or throughput.
- Formal DVR tests must use explicit non-greedy sampling and a per-request
  `sampling_seed`. Greedy sampling misses rejection-sampling cost and can hide
  proposal-distribution regressions.
- For Qwen3.5 use its recommended `temperature=1.0`, `top_p=0.95`, and
  `top_k=20`.
- For Qwen3-Next use its recommended `temperature=0.7`, `top_p=0.8`, and
  `top_k=20`.
- Greedy requests are allowed only for setup tasks such as DeepGEMM
  precompilation or for an explicitly labelled diagnostic. Never mix them into
  release KL, acceptance, or throughput summaries.
- Keep model revision, server seed, request seeds, dataset, prompt/output
  lengths, concurrency, scheduling mode, backend, Radix policy, and
  `return_logprob` handling matched across compared rows.

## Required regression gates

- Before every push, run `git diff --check`, compile every changed Python file,
  and run the DVR unit suite listed in `test/manual/dvr/README.md`.
- Use the extracted Qwen3.5 test model for the local functional gate. Cover
  self-DVR sync/overlap, DVR-EAGLE/MTP sync/overlap, Radix enabled/disabled,
  prompt lengths `1/63/64/65`, generated lengths `2/17/65/512`, concurrent and
  batched requests, and `return_logprob=true/false`.
- The extracted model has deliberately incomplete weights. It is valid for
  graph, state-lifecycle, and strict replay tests, but its EAGLE acceptance and
  throughput are not release metrics.
- Also launch one ordinary non-DVR overlap server and verify both logprob return
  modes. DVR changes must not break the default serving path.
- When weight-update code changes, call `/update_weights_from_disk` with
  `recapture_cuda_graph=true`, then repeat one boundary and one long-generation
  test. Self-draft, EAGLE/MTP draft, and target-verify graphs must all be
  recaptured without eager fallback.

## NVLink release gates

- Follow the ordered H20 matrix in `test/manual/dvr/README.md`. Do not substitute
  A40 results for FP8, FA3, NVLink collective, or release-throughput results.
- For a fixed execution shape, repeat the same non-greedy prompt and request
  seed at least three times and require identical DVR output. Do not require
  output identity across sync/overlap or batch shapes unless the matched
  ordinary deterministic path provides that contract.
- Compare DVR overlap with normal overlap and deterministic overlap. Report
  absolute throughput, accepted length, acceptance-normalized efficiency, and
  DVR/deterministic speedup.
- Keep FlashInfer all-reduce fusion disabled for self-DVR. Test custom
  all-reduce independently; it may be captured only in provisional draft
  graphs.
- Run the H20 FP8 matrix with `USE_TRITON_W8A8_FP8_KERNEL` unset. Target
  deterministic dense W8A8 uses a fixed Triton tile, while provisional
  self-draft intentionally restores the ordinary fast dispatch.
