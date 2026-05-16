import concurrent.futures
import argparse
import json
import math
import threading
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:30124"
DEFAULT_PROMPTS = [
    "Explain deterministic verification in one short sentence:",
    "List three properties of a reliable cache in concise prose:",
    "Write a compact paragraph about batch inference and scheduling:",
    "Give a short technical note about numerical reproducibility in transformers:",
]


def post(base_url, payload, timeout=600):
    req = urllib.request.Request(
        base_url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def token_ids(items):
    return [int(x[1]) for x in items if x is not None and x[1] is not None]


def logprobs(items):
    return [float(x[0]) for x in items if x is not None and x[0] is not None]


def prompt_token_ids(base_url, text):
    resp = post(
        base_url,
        {
            "text": text,
            "sampling_params": {"max_new_tokens": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        }
    )
    return token_ids(resp["meta_info"]["input_token_logprobs"])


def generate_one(base_url, prompts, max_new, barrier, i):
    barrier.wait()
    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,
        "max_new_tokens": max_new[i],
        "sampling_seed": 2026 + i,
    }
    output = post(
        base_url,
        {
            "text": prompts[i],
            "sampling_params": sampling_params,
            "return_logprob": True,
            "logprob_start_len": 0,
        }
    )
    return i, sampling_params, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--max-new", default="17,65,129,257")
    args = parser.parse_args()

    max_new = [int(x) for x in args.max_new.split(",") if x]
    prompts = [DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)] for i in range(len(max_new))]
    prompt_ids = [prompt_token_ids(args.base_url, prompt) for prompt in prompts]
    barrier = threading.Barrier(len(prompts))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [
            executor.submit(generate_one, args.base_url, prompts, max_new, barrier, i)
            for i in range(len(prompts))
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort()

    all_ok = True
    for i, sampling_params, output in results:
        output_items = output["meta_info"]["output_token_logprobs"]
        output_ids = token_ids(output_items)
        output_logprobs = logprobs(output_items)
        scoring = post(
            args.base_url,
            {
                "input_ids": prompt_ids[i] + output_ids,
                "sampling_params": {
                    "temperature": sampling_params["temperature"],
                    "top_p": sampling_params["top_p"],
                    "top_k": sampling_params["top_k"],
                    "max_new_tokens": 0,
                    "sampling_seed": sampling_params["sampling_seed"],
                },
                "return_logprob": True,
                "logprob_start_len": 0,
            }
        )
        oracle_items = scoring["meta_info"]["input_token_logprobs"][-len(output_ids) :]
        oracle_ids = token_ids(oracle_items)
        oracle_logprobs = logprobs(oracle_items)
        diffs = [abs(a - b) for a, b in zip(output_logprobs, oracle_logprobs)]
        maxdiff = max(diffs or [0.0])
        first = next((j for j, diff in enumerate(diffs) if diff != 0.0), None)
        kl_abs_sum = sum(
            abs(math.exp(a) * (a - b))
            for a, b in zip(output_logprobs, oracle_logprobs)
        )
        ok = oracle_ids == output_ids and maxdiff == 0.0
        all_ok = all_ok and ok
        meta = output.get("meta_info", {})
        print(
            "case={} prompt_len={} max_new={} gen={} ids={} first={} "
            "maxdiff={} kl={} accept={} accepted={} verify_ct={}".format(
                i,
                len(prompt_ids[i]),
                max_new[i],
                len(output_ids),
                oracle_ids == output_ids,
                first,
                maxdiff,
                kl_abs_sum,
                meta.get("spec_accept_rate"),
                meta.get("spec_accept_token_num"),
                meta.get("spec_verify_ct"),
            ),
            flush=True,
        )
    print(f"ALL_OK {all_ok}", flush=True)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
