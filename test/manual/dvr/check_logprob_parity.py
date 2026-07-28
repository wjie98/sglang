#!/usr/bin/env python3

"""Run the two exact-logprob gates for one DVR-generated sequence."""

import argparse
import json
import math
import urllib.request
from pathlib import Path
from typing import Any


def post_json(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={} if payload is None else {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        body = response.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def flush_cache(base_url: str) -> None:
    result = post_json(base_url, "/flush_cache")
    if isinstance(result, str) and "Cache flushed" not in result:
        raise RuntimeError(f"Cache flush failed: {result}")


def token_logprobs(response: dict[str, Any], field: str) -> list[tuple[float, int]]:
    values = []
    for item in response["meta_info"][field]:
        logprob, token_id = item[:2]
        if logprob is None:
            continue
        values.append((float(logprob), int(token_id)))
    return values


def compare(
    name: str,
    expected: list[tuple[float, int]],
    actual: list[tuple[float, int]],
) -> dict[str, Any]:
    if len(expected) != len(actual):
        raise AssertionError(f"{name}: length {len(actual)} != {len(expected)}")

    diffs = []
    for index, ((expected_lp, expected_id), (actual_lp, actual_id)) in enumerate(
        zip(expected, actual, strict=True)
    ):
        if expected_id != actual_id:
            raise AssertionError(
                f"{name}: token {index} differs: {actual_id} != {expected_id}"
            )
        diffs.append(abs(expected_lp - actual_lp))

    maxdiff = max(diffs, default=0.0)
    kl_proxy = sum(
        abs(math.exp(expected_lp) * (expected_lp - actual_lp))
        for (expected_lp, _), (actual_lp, _) in zip(
            expected, actual, strict=True
        )
    )
    result = {
        "tokens": len(expected),
        "maxdiff": maxdiff,
        "kl_proxy": kl_proxy,
        "exact": maxdiff == 0.0,
    }
    print(f"{name}: {json.dumps(result, sort_keys=True)}")
    if not result["exact"]:
        first = next(index for index, diff in enumerate(diffs) if diff != 0.0)
        raise AssertionError(
            f"{name}: first mismatch at generated token {first}: "
            f"{actual[first][0]} != {expected[first][0]}"
        )
    return result


def load_prompt(args: argparse.Namespace) -> list[int]:
    if args.input_ids_json is None:
        return list(
            range(
                args.prompt_token_start,
                args.prompt_token_start + args.prompt_length,
            )
        )
    payload = json.loads(args.input_ids_json.read_text())
    return list(payload["input_ids"] if isinstance(payload, dict) else payload)


def request_payload(
    input_ids: list[int],
    *,
    max_new_tokens: int,
    args: argparse.Namespace,
    sampling_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if sampling_params is None:
        sampling_params = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "sampling_seed": args.sampling_seed,
        }
    return {
        "input_ids": input_ids,
        "sampling_params": {
            **sampling_params,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "logprob_start_len": 0,
    }


def run_dvr(args: argparse.Namespace) -> None:
    prompt_ids = load_prompt(args)
    flush_cache(args.base_url)
    generated = post_json(
        args.base_url,
        "/generate",
        request_payload(prompt_ids, max_new_tokens=args.max_new_tokens, args=args),
    )
    output_ids = list(generated["output_ids"])
    decode_scores = token_logprobs(generated, "output_token_logprobs")
    forced_ids = prompt_ids + output_ids

    flush_cache(args.base_url)
    replay = post_json(
        args.base_url,
        "/generate",
        request_payload(
            forced_ids,
            max_new_tokens=0,
            args=args,
        ),
    )
    replay_scores = token_logprobs(replay, "input_token_logprobs")
    replay_generated_scores = replay_scores[len(prompt_ids) - 1 :]
    if len(replay_generated_scores) != len(output_ids):
        raise AssertionError(
            "DVR replay did not return one input logprob per generated token: "
            f"{len(replay_generated_scores)} != {len(output_ids)}"
        )
    dvr_gate = compare(
        "DVR decode -> DVR prefill", decode_scores, replay_generated_scores
    )

    artifact = {
        "prompt_ids": prompt_ids,
        "output_ids": output_ids,
        "forced_ids": forced_ids,
        "sampling_params": request_payload(
            prompt_ids, max_new_tokens=args.max_new_tokens, args=args
        )["sampling_params"],
        "dvr_decode_scores": decode_scores,
        "dvr_prefill_scores": replay_generated_scores,
        "dvr_gate": dvr_gate,
    }
    args.artifact.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Wrote {args.artifact}")


def run_det(args: argparse.Namespace) -> None:
    artifact = json.loads(args.artifact.read_text())
    forced_ids = artifact["forced_ids"]
    prompt_length = len(artifact["prompt_ids"])

    flush_cache(args.base_url)
    replay = post_json(
        args.base_url,
        "/generate",
        request_payload(
            forced_ids,
            max_new_tokens=0,
            args=args,
            sampling_params=artifact["sampling_params"],
        ),
    )
    replay_scores = token_logprobs(replay, "input_token_logprobs")
    det_generated_scores = replay_scores[prompt_length - 1 :]
    expected = [tuple(item) for item in artifact["dvr_prefill_scores"]]
    det_gate = compare(
        "DVR prefill -> deterministic prefill",
        expected,
        det_generated_scores,
    )
    artifact["det_prefill_scores"] = det_generated_scores
    artifact["det_gate"] = det_gate
    args.artifact.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Updated {args.artifact}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("dvr", "det"))
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input-ids-json", type=Path)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--prompt-token-start", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=65)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sampling-seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.phase == "dvr":
        run_dvr(arguments)
    else:
        run_det(arguments)
