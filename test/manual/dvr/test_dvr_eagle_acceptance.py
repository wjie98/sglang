"""Fixed-protocol DVR-EAGLE acceptance and KL smoke test.

This script is intentionally server-side agnostic: start an SGLang server with
DVR-EAGLE first, then run this client against it.  It makes radix-cache state an
explicit test variable so regressions around cached chunk-boundary tails are
reproducible instead of being hidden in request order.

Typical use:

PYTHONPATH=python conda run --no-capture-output -n dvr_dev python \
  test/manual/dvr/test_dvr_eagle_acceptance.py \
  --base-url http://127.0.0.1:31835 \
  --prompt-token-lengths 63,64,65 \
  --max-new 4,16,65,256 \
  --cache-mode flush-each \
  --check-kl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


DEFAULT_BASE = "http://127.0.0.1:30124"


@dataclass(frozen=True)
class Case:
    prompt_len: int
    max_new: int
    seed: int


def post(base_url: str, payload: dict[str, Any], timeout: int = 900) -> Any:
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


def flush_cache(base_url: str) -> None:
    req = urllib.request.Request(base_url + "/flush_cache?timeout=10", data=b"{}")
    try:
        urllib.request.urlopen(req, timeout=60).read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"POST /flush_cache failed: {exc.code} {body}") from exc


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def token_ids(items: Sequence[Any]) -> list[int]:
    return [int(x[1]) for x in items if x is not None and x[1] is not None]


def logprobs(items: Sequence[Any]) -> list[float]:
    return [float(x[0]) for x in items if x is not None and x[0] is not None]


def prompt_token_ids(base_url: str, text: str) -> list[int]:
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


def make_boundary_inputs(base_url: str, lengths: Sequence[int]) -> dict[int, list[int]]:
    max_len = max(lengths)
    text = (
        "DVR EAGLE boundary acceptance validation prompt with stable token "
        "content. It is repeated to build exact input-id prefixes for chunk "
        "and radix-cache tests. "
    )
    ids = prompt_token_ids(base_url, text)
    while len(ids) < max_len:
        text += text
        ids = prompt_token_ids(base_url, text)
    return {length: ids[:length] for length in lengths}


def warm_cache(base_url: str, prompt_ids: Iterable[list[int]]) -> None:
    post(
        base_url,
        {
            "input_ids": list(prompt_ids),
            "sampling_params": {"max_new_tokens": 0},
        },
    )


def sampling_params(args: argparse.Namespace, max_new: int, seed: int) -> dict[str, Any]:
    return {
        "max_new_tokens": max_new,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "sampling_seed": seed,
        "ignore_eos": args.ignore_eos,
    }


def run_generate(
    base_url: str,
    prompt_ids: list[int],
    params: dict[str, Any],
    *,
    return_logprob: bool,
) -> dict[str, Any]:
    return post(
        base_url,
        {
            "input_ids": prompt_ids,
            "sampling_params": params,
            "return_logprob": return_logprob,
            "return_text_in_logprobs": False,
            "logprob_start_len": -1,
        },
    )


def score_oracle(
    base_url: str,
    prompt_ids: list[int],
    output_ids: list[int],
    params: dict[str, Any],
) -> tuple[list[int], list[float]]:
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


def selected_token_kl_proxy(output_lps: list[float], oracle_lps: list[float]) -> float:
    return sum(abs(math.exp(a) * (a - b)) for a, b in zip(output_lps, oracle_lps))


def digest(ids: Sequence[int]) -> str:
    payload = ",".join(str(x) for x in ids).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def maybe_check_kl(
    base_url: str,
    prompt_ids: list[int],
    output_items: Sequence[Any],
    params: dict[str, Any],
    tolerance: float,
) -> tuple[bool | None, float | None, float | None, str | None]:
    if not output_items:
        return None, None, None, None
    output_ids = token_ids(output_items)
    output_lps = logprobs(output_items)
    oracle_ids, oracle_lps = score_oracle(base_url, prompt_ids, output_ids, params)
    if output_ids != oracle_ids:
        return False, None, None, "token_ids"
    if len(output_lps) != len(oracle_lps):
        return False, None, None, "logprob_len"
    maxdiff = max([abs(a - b) for a, b in zip(output_lps, oracle_lps)] or [0.0])
    kl_proxy = selected_token_kl_proxy(output_lps, oracle_lps)
    return maxdiff <= tolerance and kl_proxy <= tolerance, maxdiff, kl_proxy, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--prompt-token-lengths", default="63,64,65")
    parser.add_argument("--max-new", default="4,16,65,256")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--cache-mode",
        choices=("flush-once", "flush-each", "warm-all"),
        default="flush-each",
        help=(
            "flush-each isolates no-cache acceptance; flush-once exposes natural "
            "sequential radix hits; warm-all forces cached-prefix generation."
        ),
    )
    parser.add_argument("--check-kl", action="store_true")
    parser.add_argument("--kl-tol", type=float, default=0.0)
    parser.add_argument("--no-return-logprob", action="store_true")
    args = parser.parse_args()

    prompt_lengths = parse_int_list(args.prompt_token_lengths)
    max_new_values = parse_int_list(args.max_new)
    prompt_ids_by_len = make_boundary_inputs(args.base_url, prompt_lengths)

    if args.cache_mode != "flush-each":
        flush_cache(args.base_url)
    if args.cache_mode == "warm-all":
        warm_cache(args.base_url, prompt_ids_by_len.values())

    rows = []
    case_id = 0
    for max_new in max_new_values:
        for prompt_len in prompt_lengths:
            if args.cache_mode == "flush-each":
                flush_cache(args.base_url)
            case = Case(
                prompt_len=prompt_len,
                max_new=max_new,
                seed=args.seed + case_id,
            )
            params = sampling_params(args, case.max_new, case.seed)
            prompt_ids = prompt_ids_by_len[case.prompt_len]
            resp = run_generate(
                args.base_url,
                prompt_ids,
                params,
                return_logprob=not args.no_return_logprob or args.check_kl,
            )
            meta = resp.get("meta_info", {})
            output_items = meta.get("output_token_logprobs", [])
            output_ids = token_ids(output_items)
            kl_ok = maxdiff = kl_proxy = kl_error = None
            if args.check_kl:
                kl_ok, maxdiff, kl_proxy, kl_error = maybe_check_kl(
                    args.base_url,
                    prompt_ids,
                    output_items,
                    params,
                    args.kl_tol,
                )
            row = {
                "case": case_id,
                "prompt_len": case.prompt_len,
                "max_new": case.max_new,
                "gen": meta.get("completion_tokens", len(output_ids)),
                "cached": meta.get("cached_tokens"),
                "accept_rate": meta.get("spec_accept_rate"),
                "accept_length": meta.get("spec_accept_length"),
                "correct_drafts": meta.get("spec_num_correct_drafts"),
                "proposed_drafts": meta.get("spec_num_proposed_drafts"),
                "verify_ct": meta.get("spec_verify_ct"),
                "histogram": meta.get("spec_correct_drafts_histogram"),
                "hash": digest(output_ids),
                "kl_ok": kl_ok,
                "maxdiff": maxdiff,
                "kl_proxy": kl_proxy,
                "kl_error": kl_error,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            case_id += 1

    failed = [row for row in rows if row["kl_ok"] is False]
    print("SUMMARY " + json.dumps({"cases": len(rows), "kl_failed": len(failed)}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
