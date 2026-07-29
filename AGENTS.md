# DVR development notes

## Execution policy

- Treat forward semantics, numerical policy, and CUDA graph layout as separate
  concerns. `TARGET_VERIFY` is an EXTEND forward executed in a fixed-token,
  per-request graph; inheriting decode graph infrastructure does not make it a
  decode kernel.
- Resolved `ServerArgs` define the immutable authoritative target policy.
  Target prefill and target verify always inherit it.
- Provisional draft work is the only DVR exception. Process-wide operator,
  environment, and collective state may change only during serialized draft
  graph capture. Runtime replay may select backend-local metadata, but must not
  mutate `ServerArgs`, deterministic environment state, or the process-wide
  batch-invariant dispatcher.
- Sampling determinism is independent of model kernel policy. Draft and
  rejection random numbers must remain request-seed and absolute-position
  based even when provisional model kernels use the normal fast path.

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
- Every correctness run has two independent exact-logprob gates:
  1. DVR target output versus a clean target prefill of the same accepted
     sequence.
  2. DVR target prefill versus ordinary deterministic prefill of that same
     forced sequence and token positions.
  Both require identical token IDs and counts, selected-token `maxdiff=0`, and
  `kl_proxy=0`; passing one does not waive the other. The proxy is computed
  from selected-token logprobs and must not be reported as full-vocabulary KL.
  Claim full-vocabulary `KL=0` only when the test captures and compares the
  complete target distributions.
- Generate the forced sequence only once, with DVR. The deterministic server
  scores that sequence through a prefill-only request (`max_new_tokens=0`); do
  not independently sample a second trajectory or substitute ordinary
  deterministic decode for this gate.
- Run these phases with `test/manual/dvr/check_logprob_parity.py` so every
  comparison uses the same token indexing and exact-logprob convention. Start
  both servers with `SGLANG_RETURN_ORIGINAL_LOGPROB=True`; the completed
  artifact must contain both `dvr_gate.exact=true` and `det_gate.exact=true`.
  Do not start throughput qualification when either gate is missing or false.
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
  A40 results for FA3, NVLink collective, or release-throughput results.
- For a fixed execution shape, repeat the same non-greedy prompt and request
  seed at least three times and require identical DVR output. Run both
  exact-logprob gates on every output. Do not require output identity across
  sync/overlap or batch shapes unless the matched ordinary deterministic path
  provides that contract.
- Compare DVR overlap with normal overlap and deterministic overlap. Report
  absolute throughput, accepted length, acceptance-normalized efficiency, and
  DVR/deterministic speedup.
- Keep FlashInfer all-reduce fusion disabled for self-DVR. Test custom
  all-reduce independently; it may be captured only in provisional draft
  graphs.
- DVR full-attention phases support only Triton and FlashAttention 3.
  Do not introduce a decode-only phase split to bypass this contract.
- DVR target prefill and verify use the unmodified upstream deterministic
  policy. Only provisional self-draft graph capture may select the ordinary
  fast MoE path. A backend or model that fails either exact-logprob gate is
  unsupported; do not add a weaker DVR-specific target policy.
