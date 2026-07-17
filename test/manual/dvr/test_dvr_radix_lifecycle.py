"""Focused DVR radix ownership and nearest-checkpoint replay validation."""

from __future__ import annotations

import argparse
import uuid
from typing import Any

from test_dvr_batch_kl import (
    first_failure,
    flush_cache,
    logprobs,
    make_prompt_token_boundary_inputs,
    post,
    prompt_token_ids,
    score_oracle,
    token_ids,
)


def sampling_params(max_new: int, seed: int, *, greedy: bool = False) -> dict[str, Any]:
    return {
        "temperature": 0.0 if greedy else 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": max_new,
        "sampling_seed": seed,
        "ignore_eos": True,
    }


def generate(
    base_url: str,
    input_ids: list[int],
    *,
    max_new: int,
    seed: int,
    rid: str | None = None,
    greedy: bool = False,
    stop_token_id: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    regex: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = sampling_params(max_new, seed, greedy=greedy)
    if temperature is not None:
        params["temperature"] = temperature
    if top_k is not None:
        params["top_k"] = top_k
    if regex is not None:
        params["regex"] = regex
    if stop_token_id is not None:
        params["ignore_eos"] = False
        params["stop_token_ids"] = [stop_token_id]
    payload = {
        "input_ids": input_ids,
        "sampling_params": params,
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "logprob_start_len": -1,
    }
    if rid is not None:
        payload["rid"] = rid
    return post(base_url, payload), params


def output(response: dict[str, Any]) -> tuple[list[int], list[float]]:
    items = response["meta_info"]["output_token_logprobs"]
    return token_ids(items), logprobs(items)


def assert_exact_oracle(
    base_url: str,
    prompt_ids: list[int],
    response: dict[str, Any],
    params: dict[str, Any],
) -> None:
    output_ids, output_lps = output(response)
    oracle_ids, oracle_lps = score_oracle(
        base_url,
        prompt_ids,
        output_ids,
        params,
        flush_oracle=True,
    )
    failure = first_failure(output_ids, oracle_ids, output_lps, oracle_lps, 0.0)
    if failure is not None:
        raise AssertionError(failure)


def alternate_prompt(base_url: str, length: int) -> list[int]:
    text = (
        "A distinct DVR request must never inherit another request's recurrent state. "
    )
    ids = prompt_token_ids(base_url, text)
    while len(ids) < length:
        text += text
        ids = prompt_token_ids(base_url, text)
    return ids[:length]


def test_same_rid_slot_reuse(base_url: str, slot_cycles: int) -> None:
    first_prompt = make_prompt_token_boundary_inputs(base_url, [65])[0]
    second_prompt = alternate_prompt(base_url, 65)
    reused_rid = "dvr-radix-aba"
    flush_cache(base_url)

    generate(base_url, first_prompt, max_new=1, seed=10, rid=reused_rid)
    for index in range(slot_cycles):
        generate(
            base_url,
            second_prompt,
            max_new=1,
            seed=20 + index,
            rid=f"dvr-slot-cycle-{index}",
        )

    response, params = generate(
        base_url,
        second_prompt,
        max_new=65,
        seed=100,
        rid=reused_rid,
    )
    assert_exact_oracle(base_url, second_prompt, response, params)
    print("same_rid_slot_reuse ok=True")


def test_generated_prefix_boundaries(
    base_url: str, prompt_lengths: list[int], generated_lengths: list[int]
) -> None:
    prompts = make_prompt_token_boundary_inputs(base_url, prompt_lengths)
    case_id = 0
    for prompt in prompts:
        for generated_len in generated_lengths:
            flush_cache(base_url)
            seed_response, _ = generate(
                base_url,
                prompt,
                max_new=generated_len,
                seed=1000 + case_id,
                rid=f"dvr-boundary-seed-{case_id}",
            )
            seed_ids, _ = output(seed_response)
            if len(seed_ids) != generated_len:
                raise AssertionError(
                    f"seed length mismatch: expected={generated_len}, actual={len(seed_ids)}"
                )
            prefix = prompt + seed_ids
            response, params = generate(
                base_url,
                prefix,
                max_new=17,
                seed=2000 + case_id,
                rid=f"dvr-boundary-probe-{case_id}",
            )
            cached = int(response["meta_info"].get("cached_tokens") or 0)
            # DVR retains the prompt checkpoint but does not publish an overlap
            # decode slot that may be ahead of the result finishing the request.
            # Standard EXTEND rebuilds the generated suffix on the probe.
            expected = (len(prompt) - 1) // 64 * 64
            if cached < expected:
                raise AssertionError(
                    f"nearest checkpoint miss: prompt={len(prompt)}, "
                    f"generated={generated_len}, expected={expected}, actual={cached}"
                )
            assert_exact_oracle(base_url, prefix, response, params)
            print(
                f"boundary case={case_id} prompt={len(prompt)} "
                f"generated={generated_len} cached={cached} ok=True"
            )
            case_id += 1


def test_stop_boundary(base_url: str) -> None:
    prompt = make_prompt_token_boundary_inputs(base_url, [126])[0]
    selected = None
    for seed in range(3000, 3010):
        flush_cache(base_url)
        seed_response, _ = generate(
            base_url,
            prompt,
            max_new=32,
            seed=seed,
            temperature=1.3,
            top_k=100,
        )
        seed_ids, _ = output(seed_response)
        for position in range(1, min(5, len(seed_ids))):
            token = seed_ids[position]
            if token not in seed_ids[:position]:
                selected = seed, position, token
                break
        if selected is not None:
            break
    if selected is None:
        raise AssertionError("No reproducible early stop token was generated.")
    seed, stop_position, stop_token = selected

    flush_cache(base_url)
    response, params = generate(
        base_url,
        prompt,
        max_new=96,
        seed=seed,
        stop_token_id=stop_token,
        temperature=1.3,
        top_k=100,
        rid=f"dvr-stop-{uuid.uuid4().hex}",
    )
    stopped_ids, _ = output(response)
    if len(stopped_ids) != stop_position + 1 or stopped_ids[-1] != stop_token:
        raise AssertionError(
            f"stop boundary mismatch: expected={stop_position + 1}, "
            f"actual={len(stopped_ids)}, stop_token={stop_token}"
        )
    assert_exact_oracle(base_url, prompt, response, params)
    print(f"stop_boundary generated={len(stopped_ids)} ok=True")


def test_grammar_boundary(base_url: str) -> None:
    prompt = make_prompt_token_boundary_inputs(base_url, [126])[0]
    flush_cache(base_url)
    response, _ = generate(
        base_url,
        prompt,
        max_new=16,
        seed=4000,
        regex="( Yes){2}",
        rid=f"dvr-grammar-{uuid.uuid4().hex}",
    )
    generated_ids, _ = output(response)
    if not generated_ids or len(generated_ids) >= 16:
        raise AssertionError(
            f"grammar did not terminate inside verify: generated={len(generated_ids)}"
        )

    # Grammar sampling reports probabilities after applying its vocabulary mask,
    # whereas the prefill oracle reports raw model probabilities. Instead, reuse
    # the completed request as a radix prefix and score an unconstrained probe.
    # This catches a speculative checkpoint published beyond the grammar finish.
    prefix = prompt + generated_ids
    probe, params = generate(
        base_url,
        prefix,
        max_new=17,
        seed=4001,
        rid=f"dvr-grammar-probe-{uuid.uuid4().hex}",
    )
    cached = int(probe["meta_info"].get("cached_tokens") or 0)
    expected = (len(prompt) - 1) // 64 * 64
    if cached < expected:
        raise AssertionError(
            f"grammar checkpoint miss: expected={expected}, actual={cached}"
        )
    assert_exact_oracle(base_url, prefix, probe, params)
    print(
        f"grammar_boundary generated={len(generated_ids)} "
        f"cached={cached} ok=True"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30124")
    parser.add_argument("--slot-cycles", type=int, default=8)
    parser.add_argument("--prompt-lengths", default="63,64,65")
    parser.add_argument(
        "--generated-lengths",
        default="1,2,62,63,64,65,66,126,127,128,129",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Scan every generated-prefix length from 1 through 129.",
    )
    args = parser.parse_args()

    prompt_lengths = [int(value) for value in args.prompt_lengths.split(",")]
    generated_lengths = (
        list(range(1, 130))
        if args.exhaustive
        else [int(value) for value in args.generated_lengths.split(",")]
    )
    test_same_rid_slot_reuse(args.base_url, args.slot_cycles)
    test_generated_prefix_boundaries(args.base_url, prompt_lengths, generated_lengths)
    test_stop_boundary(args.base_url)
    test_grammar_boundary(args.base_url)
    print("ALL_OK True")


if __name__ == "__main__":
    main()
