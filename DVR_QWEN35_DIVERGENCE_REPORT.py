import argparse
import json
import math
import statistics
import urllib.request
from pathlib import Path


PROMPTS = [
    "Explain deterministic verification in one short sentence:",
    "Write a long numbered technical checklist about chunkwise recurrent inference, deterministic verification, state rollback, cache ownership, and batch scheduling. Keep writing detailed items without concluding early:",
]


def post(base_url, payload, timeout=900):
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
        },
    )
    return token_ids(resp["meta_info"]["input_token_logprobs"])


def bar(value, max_value, width=28):
    if max_value <= 0:
        return ""
    n = int(round(width * value / max_value))
    return "#" * n + "." * (width - n)


def bucket_stats(diffs, start, end):
    vals = diffs[start:end]
    if not vals:
        return None
    return {
        "range": f"{start:03d}-{end - 1:03d}",
        "max": max(vals),
        "mean": statistics.fmean(vals),
        "p95": sorted(vals)[int((len(vals) - 1) * 0.95)],
    }


def run_case(base_url, prompt, max_new, seed):
    prompt_ids = prompt_token_ids(base_url, prompt)
    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,
        "max_new_tokens": max_new,
        "sampling_seed": seed,
        "ignore_eos": True,
    }
    output = post(
        base_url,
        {
            "text": prompt,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "logprob_start_len": 0,
        },
    )
    output_items = output["meta_info"]["output_token_logprobs"]
    output_ids = token_ids(output_items)
    output_logprobs = logprobs(output_items)
    scoring = post(
        base_url,
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
    oracle_items = scoring["meta_info"]["input_token_logprobs"][-len(output_ids) :]
    oracle_ids = token_ids(oracle_items)
    oracle_logprobs = logprobs(oracle_items)
    diffs = [abs(a - b) for a, b in zip(output_logprobs, oracle_logprobs)]
    weighted = [abs(math.exp(a) * (a - b)) for a, b in zip(output_logprobs, oracle_logprobs)]
    first = next((i for i, diff in enumerate(diffs) if diff != 0.0), None)
    top = sorted(enumerate(diffs), key=lambda x: x[1], reverse=True)[:10]
    return {
        "prompt_len": len(prompt_ids),
        "max_new": max_new,
        "generated": len(output_ids),
        "returned_logprobs": len(output_logprobs),
        "oracle_logprobs": len(oracle_logprobs),
        "compared": len(diffs),
        "ids_match": oracle_ids == output_ids,
        "first": first,
        "maxdiff": max(diffs or [0.0]),
        "mean": statistics.fmean(diffs) if diffs else 0.0,
        "kl_abs_sum": sum(weighted),
        "accept": output.get("meta_info", {}).get("spec_accept_rate"),
        "accepted": output.get("meta_info", {}).get("spec_accept_token_num"),
        "verify_ct": output.get("meta_info", {}).get("spec_verify_ct"),
        "diffs": diffs,
        "weighted": weighted,
        "top": top,
    }


def render_report(cases):
    lines = ["# DVR Qwen3.5 Divergence Report", ""]
    lines.append("Strict oracle: returned generation logprobs vs full-prefill scoring logprobs.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| case | prompt_len | max_new | gen | ret_lp | cmp | ids | first_diff | max_abs | mean_abs | kl_abs_sum | accept | accepted | verify_ct |")
    lines.append("|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, case in enumerate(cases):
        lines.append(
            f"| {i} | {case['prompt_len']} | {case['max_new']} | {case['generated']} | "
            f"{case['returned_logprobs']} | {case['compared']} | {case['ids_match']} | "
            f"{case['first']} | {case['maxdiff']:.6g} | "
            f"{case['mean']:.6g} | {case['kl_abs_sum']:.6g} | {case['accept']} | "
            f"{case['accepted']} | {case['verify_ct']} |"
        )
    lines.append("")
    for i, case in enumerate(cases):
        lines.append(f"## Case {i}: max_new={case['max_new']}")
        lines.append("")
        max_bucket = max(
            [bucket_stats(case["diffs"], s, min(s + 16, len(case["diffs"])))["max"] for s in range(0, len(case["diffs"]), 16)]
            or [0.0]
        )
        lines.append("### 16-token buckets")
        lines.append("")
        lines.append("| tokens | max_abs | mean_abs | p95_abs | bar |")
        lines.append("|---:|---:|---:|---:|:---|")
        for start in range(0, len(case["diffs"]), 16):
            stat = bucket_stats(case["diffs"], start, min(start + 16, len(case["diffs"])))
            if stat is None:
                continue
            lines.append(
                f"| {stat['range']} | {stat['max']:.6g} | {stat['mean']:.6g} | "
                f"{stat['p95']:.6g} | `{bar(stat['max'], max_bucket)}` |"
            )
        lines.append("")
        lines.append("### Top token diffs")
        lines.append("")
        lines.append("| rank | token_index | abs_diff |")
        lines.append("|---:|---:|---:|")
        for rank, (idx, diff) in enumerate(case["top"], start=1):
            lines.append(f"| {rank} | {idx} | {diff:.6g} |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30125")
    parser.add_argument("--max-new", default="128,256")
    parser.add_argument("--output", default="DVR_QWEN35_DIVERGENCE_REPORT.md")
    args = parser.parse_args()
    max_news = [int(x) for x in args.max_new.split(",") if x]
    cases = [
        run_case(args.base_url, PROMPTS[i % len(PROMPTS)], max_new, 2026 + i)
        for i, max_new in enumerate(max_news)
    ]
    report = render_report(cases)
    Path(args.output).write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
