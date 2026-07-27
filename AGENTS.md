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
