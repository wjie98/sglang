# DVR KL Testing Guide

This guide is the only intended carry-over from previous clean-port work. Use
it to test the new v3 implementation; do not copy v2 implementation details.

## Server Template

Minimum attention-only DVR launch:

```bash
PYTHONPATH=python \
conda run --no-capture-output -n dvr_dev python -m sglang.launch_server \
  --model-path /home/hwj/Qwen3-0.6B \
  --speculative-algorithm DECODE_VERIFY_ROLLBACK \
  --speculative-num-steps 15 \
  --speculative-num-draft-tokens 16 \
  --speculative-eagle-topk 1 \
  --page-size 1 \
  --enable-deterministic-inference
```

Minimum GDN DVR launch:

```bash
PYTHONPATH=python \
conda run --no-capture-output -n dvr_dev python -m sglang.launch_server \
  --model-path /home/hwj/Qwen3.5-0.8B \
  --speculative-algorithm DECODE_VERIFY_ROLLBACK \
  --speculative-num-steps 15 \
  --speculative-num-draft-tokens 16 \
  --speculative-eagle-topk 1 \
  --speculative-dvr-chunk-boundary-verify \
  --page-size 1 \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-track-interval 64 \
  --mamba-ssm-dtype float32 \
  --linear-attn-backend triton \
  --enable-deterministic-inference
```

`SGLANG_RETURN_ORIGINAL_LOGPROB=True` is needed by the strict-KL oracle because
the test compares returned generation logprobs with a full-prefill scoring pass.
It is not a serving requirement.

On the local 5090 test machine, keep these DeepGEMM workarounds in test
commands until the upstream kernel issue is resolved:

```bash
SGLANG_ENABLE_JIT_DEEPGEMM=0
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
```

The longer templates below include host/port, memory, and warmup settings for
repeatable local testing. They are not the semantic minimum for DVR.

Attention-only model:

```bash
SGLANG_RETURN_ORIGINAL_LOGPROB=True \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 \
PYTHONPATH=python \
conda run --no-capture-output -n dvr_dev python -m sglang.launch_server \
  --model-path /home/hwj/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 30124 \
  --speculative-algorithm DECODE_VERIFY_ROLLBACK \
  --speculative-num-steps 15 \
  --speculative-num-draft-tokens 16 \
  --speculative-eagle-topk 1 \
  --page-size 1 \
  --mem-fraction-static 0.45 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --enable-deterministic-inference \
  --skip-server-warmup
```

Adjust the CLI names if v3 chooses a smaller DVR-specific flag, but tests should
still explicitly pass `--enable-deterministic-inference`. Do not make DVR
silently override the new upstream deterministic-inference handling.

## Generation Request

Use non-greedy sampling when validating DVR because deterministic greedy output
can hide sampling-path bugs:

```python
generation_payload = {
    "text": "Explain deterministic verification in one short sentence:",
    "sampling_params": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,
        "max_new_tokens": 80,
        "sampling_seed": 2026,
    },
    "return_logprob": True,
    "logprob_start_len": 0,
}
```

## Full-Prefill Scoring Oracle

After generation, concatenate `prompt_ids + output_ids` and score the whole
sequence with `max_new_tokens=0`:

```python
scoring_payload = {
    "input_ids": prompt_ids + output_ids,
    "sampling_params": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,
        "max_new_tokens": 0,
        "sampling_seed": 2026,
    },
    "return_logprob": True,
    "logprob_start_len": 0,
}
```

The tail of `scoring["meta_info"]["input_token_logprobs"]` corresponding to
`output_ids` is the strict oracle for returned generation logprobs.

## Minimal Test Script Pattern

```python
import json
import math
import urllib.request


def post_json(url, payload, timeout=180):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def token_ids(items):
    return [int(item[1]) for item in items]


def logprobs(items):
    return [float(item[0]) for item in items]


def score_output(base_url, prompt_ids, output_ids, sampling_params):
    scoring = post_json(
        f"{base_url}/generate",
        {
            "input_ids": prompt_ids + output_ids,
            "sampling_params": {
                "temperature": sampling_params["temperature"],
                "top_p": sampling_params["top_p"],
                "top_k": sampling_params["top_k"],
                "max_new_tokens": 0,
                "sampling_seed": sampling_params["sampling_seed"],
            },
            "return_logprob": True,
            "logprob_start_len": 0,
        },
    )
    input_items = scoring["meta_info"]["input_token_logprobs"]
    scored_ids = token_ids(input_items)
    first_output = len(scored_ids) - len(output_ids)
    oracle_items = input_items[first_output:]
    assert token_ids(oracle_items) == output_ids
    return logprobs(oracle_items)


def compare(output_logprobs, oracle_logprobs):
    max_abs_diff = 0.0
    kl_abs_sum = 0.0
    for lhs, rhs in zip(output_logprobs, oracle_logprobs):
        max_abs_diff = max(max_abs_diff, abs(lhs - rhs))
        kl_abs_sum += abs(math.exp(lhs) * (lhs - rhs))
    return max_abs_diff, kl_abs_sum
```

For strict DVR correctness on a deterministic prefill-compatible path:

```python
assert max_abs_diff == 0.0
```

Do not relax this for bs=1 attention-only tests. For early batched GDN bring-up,
record the difference first, but the target remains strict equality against the
full-prefill oracle.

## Suggested Coverage

Run both single-request and batched cases:

- `bs=1`, `max_new_tokens`: `1, 8, 16, 20, 63, 64, 80, 128`
- `bs=2`, mixed prompt lengths and mixed `max_new_tokens`
- GDN model with random prompt lengths around chunk boundaries
- GDN model with generated length greater than one chunk
- Temporary hacked acceptance-rate mismatch to validate per-request boundary
  state commit and tail shifting

Useful metrics to print from each response:

- generated token count
- max absolute logprob diff
- KL-like absolute sum
- `spec_accept_rate`
- `spec_accept_token_num`
- `spec_verify_ct`
