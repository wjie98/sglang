"""Manual DVR output-logprob vs full-prefill oracle test.

This script borrows the correctness oracle from the original DVR reference
branch:

1. Generate with DVR enabled and return output token logprobs.
2. Concatenate prompt + generated output.
3. Run a separate max_new_tokens=0 scoring request over the concatenated ids.
4. Compare the generated output logprobs with the suffix input logprobs from
   the full-prefill scoring request.

For DVR the expected result is strict equality by default: token ids match,
max selected-token logprob diff is 0, and the selected-token KL proxy is 0.
The "KL" printed here is not a full-vocabulary KL; it is a convenient proxy
over emitted tokens. With strict DVR correctness, every selected logprob is
identical, so this proxy is also exactly 0.

Operational notes learned while debugging DVR:

- Start the server separately. For Blackwell/5090 tests, disable DeepGEMM:
  SGLANG_ENABLE_JIT_DEEPGEMM=0
  SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
- Set SGLANG_RETURN_ORIGINAL_LOGPROB=True when comparing returned logprobs.
- Use non-greedy sampling, e.g. temperature=0.7, to test rejection sampling and
  returned-logprob alignment. Greedy-only tests can miss bugs in the sampler.
- The oracle request flushes radix cache by default. This makes the oracle a
  true full-prefill scoring pass instead of a cached-prefix scoring pass.
- Use lengths around and beyond FLA_CHUNK_SIZE=64 for GDN models. The defaults
  include 63/64/65/129 to cover pre-boundary, boundary, post-boundary, and
  multi-chunk state commits.
- --warm-cache is useful for checking radix-cache compatibility on the DVR
  generation path. The oracle is still flushed unless --no-flush-oracle is set.
- Generation requests use logprob_start_len=-1. Setting logprob_start_len=0
  asks SGLang to return full prompt logprobs and intentionally limits prefix
  matching to length 0, so cached_tokens will stay 0 even when radix cache works.
  The oracle still uses logprob_start_len=0 because it must score the whole
  prompt+output sequence with a true full-prefill pass.

Example:

SGLANG_RETURN_ORIGINAL_LOGPROB=True \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 \
PYTHONPATH=python conda run --no-capture-output -n dvr_dev \
python -m sglang.launch_server \
  --model-path /home/hwj/Qwen3.5-0.8B \
  --host 127.0.0.1 --port 30124 \
  --speculative-algorithm DECODE_VERIFY_ROLLBACK \
  --speculative-num-draft-tokens 16 \
  --page-size 1 \
  --mem-fraction-static 0.45 \
  --attention-backend triton \
  --linear-attn-backend triton \
  --sampling-backend pytorch \
  --enable-deterministic-inference \
  --cuda-graph-bs 1 2 4 \
  --cuda-graph-max-bs 4 \
  --max-running-requests 8 \
  --skip-server-warmup

PYTHONPATH=python conda run -n dvr_dev python \
  test/manual/dvr/test_dvr_batch_kl.py \
  --base-url http://127.0.0.1:30124 \
  --request-modes concurrent,batch \
  --max-new 1,8,16,17,63,64,65,80,129
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import threading
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_BASE = "http://127.0.0.1:30124"

# Keep these local and deterministic. The original reference script can load
# LongBench for large-scale testing, but this manual script should run without
# network or dataset dependencies.
DEFAULT_PROMPTS = [
    "Explain deterministic verification in one short sentence:",
    "List three properties of a reliable cache in concise prose:",
    (
        "In a shared-prefix batch, every request starts with this same setup. "
        "Now explain why prefix reuse should not change returned logprobs."
    ),
    (
        "In a shared-prefix batch, every request starts with this same setup. "
        "Now describe why a chunk boundary must be handled carefully."
    ),
    (
        "Write a compact technical note about numerical reproducibility in "
        "hybrid attention and linear recurrent models. "
        * 12
    ),
]


@dataclasses.dataclass(frozen=True)
class Case:
    case_id: int
    mode: str
    prompt_index: int
    max_new: int
    seed: int
    text: Optional[str] = None
    input_ids: Optional[List[int]] = None


@dataclasses.dataclass
class CaseResult:
    case: Case
    sampling_params: Dict[str, Any]
    response: Dict[str, Any]


def post(base_url: str, payload: Dict[str, Any], timeout: int = 600) -> Any:
    req = urllib.request.Request(
        base_url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"POST /generate failed: {exc.code} {body}") from exc


def flush_cache(base_url: str):
    req = urllib.request.Request(base_url + "/flush_cache", data=b"{}")
    try:
        urllib.request.urlopen(req, timeout=60).read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"POST /flush_cache failed: {exc.code} {body}") from exc


def as_response_list(response: Any) -> List[Dict[str, Any]]:
    return response if isinstance(response, list) else [response]


def token_ids(items: Sequence[Any]) -> List[int]:
    return [int(x[1]) for x in items if x is not None and x[1] is not None]


def logprobs(items: Sequence[Any]) -> List[float]:
    return [float(x[0]) for x in items if x is not None and x[0] is not None]


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_modes(value: str) -> List[str]:
    modes = [x.strip() for x in value.split(",") if x.strip()]
    allowed = {"concurrent", "batch"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unknown request mode(s): {unknown}. Allowed: {sorted(allowed)}")
    return modes


def sampling_params(args: argparse.Namespace, max_new: int, seed: int) -> Dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": max_new,
        "sampling_seed": seed,
        "ignore_eos": args.ignore_eos,
    }


def prompt_token_ids(base_url: str, text: str) -> List[int]:
    # max_new_tokens=0 is also a lightweight tokenizer round trip. Flush later
    # if the test needs an uncached generation path.
    resp = post(
        base_url,
        {
            "text": text,
            "sampling_params": {"max_new_tokens": 0},
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": 0,
        },
    )
    return token_ids(resp["meta_info"]["input_token_logprobs"])


def load_input_ids(path: Optional[str]) -> Optional[List[List[int]]]:
    if path is None:
        return None
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, list) for x in data):
        raise ValueError("--input-ids-json must contain a JSON list of token-id lists")
    return [[int(tok) for tok in row] for row in data]


def build_cases(
    *,
    modes: Sequence[str],
    max_new_values: Sequence[int],
    seed_base: int,
    prompts: Sequence[str],
    input_ids: Optional[Sequence[List[int]]],
    limit_cases: Optional[int],
) -> List[Case]:
    cases: List[Case] = []
    prompt_count = len(input_ids) if input_ids is not None else len(prompts)
    for mode in modes:
        for max_new in max_new_values:
            for prompt_index in range(prompt_count):
                case = Case(
                    case_id=len(cases),
                    mode=mode,
                    prompt_index=prompt_index,
                    max_new=max_new,
                    seed=seed_base + len(cases),
                    text=None if input_ids is not None else prompts[prompt_index],
                    input_ids=None if input_ids is None else list(input_ids[prompt_index]),
                )
                cases.append(case)
                if limit_cases is not None and len(cases) >= limit_cases:
                    return cases
    return cases


def request_for_case(case: Case, params: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "sampling_params": params,
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "logprob_start_len": -1,
    }
    if case.input_ids is not None:
        payload["input_ids"] = case.input_ids
    else:
        payload["text"] = case.text
    return payload


def generate_one(base_url: str, args: argparse.Namespace, case: Case) -> CaseResult:
    params = sampling_params(args, case.max_new, case.seed)
    response = as_response_list(post(base_url, request_for_case(case, params)))[0]
    return CaseResult(case=case, sampling_params=params, response=response)


def generate_concurrent(
    base_url: str, args: argparse.Namespace, cases: Sequence[Case]
) -> List[CaseResult]:
    if not cases:
        return []
    start = threading.Event()

    def run(case: Case) -> CaseResult:
        start.wait()
        return generate_one(base_url, args, case)

    max_workers = min(args.concurrent_workers, len(cases))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run, case) for case in cases]
        start.set()
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return sorted(results, key=lambda x: x.case.case_id)


def generate_batch(
    base_url: str, args: argparse.Namespace, cases: Sequence[Case]
) -> List[CaseResult]:
    # The HTTP batch API applies one sampling_params object to the whole list, so
    # group by max_new. This still exercises scheduler batching and shared-prefix
    # radix-cache behavior while keeping request semantics explicit.
    grouped: Dict[int, List[Case]] = defaultdict(list)
    for case in cases:
        grouped[case.max_new].append(case)

    results: List[CaseResult] = []
    for max_new, group in grouped.items():
        seed = args.seed + 100000 + max_new
        params = sampling_params(args, max_new, seed)
        payload: Dict[str, Any] = {
            "sampling_params": params,
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": -1,
        }
        if group[0].input_ids is not None:
            payload["input_ids"] = [case.input_ids for case in group]
        else:
            payload["text"] = [case.text for case in group]
        responses = as_response_list(post(base_url, payload))
        if len(responses) != len(group):
            raise RuntimeError(
                f"Batch response length mismatch for max_new={max_new}: "
                f"{len(responses)} vs {len(group)}"
            )
        for case, response in zip(group, responses, strict=True):
            results.append(CaseResult(case=case, sampling_params=params, response=response))
    return sorted(results, key=lambda x: x.case.case_id)


def prefill_prompts(base_url: str, prompt_ids: Sequence[List[int]]):
    if not prompt_ids:
        return
    post(
        base_url,
        {
            "input_ids": list(prompt_ids),
            "sampling_params": {"max_new_tokens": 0},
        },
    )


def score_oracle(
    base_url: str,
    prompt_ids: List[int],
    output_ids: List[int],
    params: Dict[str, Any],
    *,
    flush_oracle: bool,
) -> Tuple[List[int], List[float]]:
    if flush_oracle:
        flush_cache(base_url)
    scoring_params = dict(params)
    scoring_params["max_new_tokens"] = 0
    scoring = post(
        base_url,
        {
            "input_ids": prompt_ids + output_ids,
            "sampling_params": scoring_params,
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": 0,
        },
    )
    oracle_items = scoring["meta_info"]["input_token_logprobs"][-len(output_ids) :]
    return token_ids(oracle_items), logprobs(oracle_items)


def first_failure(
    output_ids: List[int],
    oracle_ids: List[int],
    output_lps: List[float],
    oracle_lps: List[float],
    tolerance: float,
) -> Optional[str]:
    if len(output_ids) != len(oracle_ids):
        return f"token length mismatch output={len(output_ids)} oracle={len(oracle_ids)}"
    if len(output_lps) != len(oracle_lps):
        return f"logprob length mismatch output={len(output_lps)} oracle={len(oracle_lps)}"
    for i, (lhs, rhs) in enumerate(zip(output_ids, oracle_ids)):
        if lhs != rhs:
            return f"id mismatch at {i}: output={lhs} oracle={rhs}"
    for i, (lhs, rhs) in enumerate(zip(output_lps, oracle_lps)):
        diff = abs(lhs - rhs)
        if diff > tolerance:
            return (
                f"logprob mismatch at {i}: output={lhs} oracle={rhs} diff={diff}"
            )
    return None


def selected_token_kl_proxy(output_lps: Iterable[float], oracle_lps: Iterable[float]) -> float:
    return sum(abs(math.exp(a) * (a - b)) for a, b in zip(output_lps, oracle_lps))


def evaluate_results(
    base_url: str,
    args: argparse.Namespace,
    results: Sequence[CaseResult],
    prompt_ids_by_index: Sequence[List[int]],
) -> bool:
    all_ok = True
    for item in results:
        case = item.case
        meta = item.response.get("meta_info", {})
        output_items = meta.get("output_token_logprobs", [])
        output_ids = token_ids(output_items)
        output_lps = logprobs(output_items)
        prompt_ids = (
            list(case.input_ids)
            if case.input_ids is not None
            else list(prompt_ids_by_index[case.prompt_index])
        )
        oracle_ids, oracle_lps = score_oracle(
            base_url,
            prompt_ids,
            output_ids,
            item.sampling_params,
            flush_oracle=not args.no_flush_oracle,
        )
        diffs = [abs(a - b) for a, b in zip(output_lps, oracle_lps)]
        maxdiff = max(diffs or [0.0])
        kl_proxy = selected_token_kl_proxy(output_lps, oracle_lps)
        failure = first_failure(
            output_ids, oracle_ids, output_lps, oracle_lps, args.max_diff_tol
        )
        ok = (
            failure is None
            and maxdiff <= args.max_diff_tol
            and kl_proxy <= args.kl_tol
        )
        all_ok = all_ok and ok
        print(
            "case={case_id} mode={mode} prompt={prompt_index} prompt_len={prompt_len} "
            "max_new={max_new} gen={gen} ok={ok} first={first} maxdiff={maxdiff} "
            "kl_proxy={kl_proxy} accept={accept} accepted={accepted} verify_ct={verify_ct} "
            "cached={cached}".format(
                case_id=case.case_id,
                mode=case.mode,
                prompt_index=case.prompt_index,
                prompt_len=len(prompt_ids),
                max_new=case.max_new,
                gen=len(output_ids),
                ok=ok,
                first=failure,
                maxdiff=maxdiff,
                kl_proxy=kl_proxy,
                accept=meta.get("spec_accept_rate"),
                accepted=meta.get("spec_accept_token_num"),
                verify_ct=meta.get("spec_verify_ct"),
                cached=meta.get("cached_tokens"),
            ),
            flush=True,
        )
        if args.dump_failure_json and not ok:
            print(
                json.dumps(
                    {
                        "case": dataclasses.asdict(case),
                        "output_ids": output_ids,
                        "oracle_ids": oracle_ids,
                        "output_logprobs": output_lps,
                        "oracle_logprobs": oracle_lps,
                    },
                    indent=2,
                ),
                flush=True,
            )
    print(f"ALL_OK {all_ok}", flush=True)
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument(
        "--request-modes",
        default="concurrent,batch",
        help="Comma-separated subset of concurrent,batch.",
    )
    parser.add_argument(
        "--max-new",
        default="1,8,16,17,63,64,65,80,129",
        help="Comma-separated generation lengths. Include values around 64.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--concurrent-workers", type=int, default=16)
    parser.add_argument("--max-diff-tol", type=float, default=0.0)
    parser.add_argument("--kl-tol", type=float, default=0.0)
    parser.add_argument(
        "--input-ids-json",
        default=None,
        help="Optional JSON list of input-id lists, similar to the original LongBench DVR test cache.",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Use only the first N prompts/input-id rows.",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="Stop after N generated cases. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Prefill prompts before generation to exercise radix-cache hits.",
    )
    parser.add_argument(
        "--no-flush-start",
        action="store_true",
        help="Do not flush cache before generation.",
    )
    parser.add_argument(
        "--no-flush-oracle",
        action="store_true",
        help="Do not flush before full-prefill oracle scoring.",
    )
    parser.add_argument("--dump-failure-json", action="store_true")
    args = parser.parse_args()

    modes = parse_modes(args.request_modes)
    max_new_values = parse_int_list(args.max_new)
    input_ids = load_input_ids(args.input_ids_json)
    prompts = DEFAULT_PROMPTS
    if input_ids is not None and args.limit_prompts is not None:
        input_ids = input_ids[: args.limit_prompts]
    elif args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]

    if input_ids is None:
        prompt_ids = [prompt_token_ids(args.base_url, prompt) for prompt in prompts]
    else:
        prompt_ids = [list(row) for row in input_ids]

    if not args.no_flush_start:
        flush_cache(args.base_url)
    if args.warm_cache:
        prefill_prompts(args.base_url, prompt_ids)

    cases = build_cases(
        modes=modes,
        max_new_values=max_new_values,
        seed_base=args.seed,
        prompts=prompts,
        input_ids=input_ids,
        limit_cases=args.limit_cases,
    )

    results: List[CaseResult] = []
    for mode in modes:
        mode_cases = [case for case in cases if case.mode == mode]
        if mode == "concurrent":
            results.extend(generate_concurrent(args.base_url, args, mode_cases))
        elif mode == "batch":
            results.extend(generate_batch(args.base_url, args, mode_cases))
        else:
            raise AssertionError(mode)

    results.sort(key=lambda x: x.case.case_id)
    if not evaluate_results(args.base_url, args, results, prompt_ids):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
