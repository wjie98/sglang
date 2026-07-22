#!/usr/bin/env python3
"""Export a two-layer Qwen3.5 checkpoint for DVR integration smoke tests."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


MODEL_INDEX = "model.safetensors.index.json"
MODEL_FILE = "model.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gdn-layer", type=int)
    parser.add_argument("--attention-layer", type=int)
    parser.add_argument("--max-position-embeddings", type=int, default=8192)
    return parser.parse_args()


def layer_types(text_config: dict) -> list[str]:
    explicit = text_config.get("layer_types")
    if explicit is not None:
        return explicit
    interval = int(text_config["full_attention_interval"])
    return [
        "full_attention" if (layer + 1) % interval == 0 else "linear_attention"
        for layer in range(int(text_config["num_hidden_layers"]))
    ]


def first_layer(types: list[str], kind: str) -> int:
    try:
        return types.index(kind)
    except ValueError as exc:
        raise ValueError(f"Source model has no {kind!r} layer: {types}") from exc


def remap_weight(
    name: str, *, gdn_layer: int, attention_layer: int
) -> str | None:
    language_prefix = "model.language_model."
    if name.startswith("mtp."):
        return name
    if name.startswith("model.visual."):
        if name.startswith("model.visual.blocks."):
            return name if name.startswith("model.visual.blocks.0.") else None
        return name
    if not name.startswith(language_prefix):
        return None

    layers_prefix = f"{language_prefix}layers."
    if not name.startswith(layers_prefix):
        return name

    source_to_target = ((gdn_layer, 0), (attention_layer, 1))
    for source_layer, target_layer in source_to_target:
        source_prefix = f"{layers_prefix}{source_layer}."
        if name.startswith(source_prefix):
            return f"{layers_prefix}{target_layer}." + name[len(source_prefix) :]
    return None


def load_selected_weights(
    source: Path,
    weight_map: dict[str, str],
    *,
    gdn_layer: int,
    attention_layer: int,
) -> dict[str, torch.Tensor]:
    selected: dict[str, tuple[str, str]] = {}
    for source_name, shard in weight_map.items():
        target_name = remap_weight(
            source_name,
            gdn_layer=gdn_layer,
            attention_layer=attention_layer,
        )
        if target_name is None:
            continue
        if target_name in selected:
            raise ValueError(f"Duplicate destination weight: {target_name}")
        selected[target_name] = (shard, source_name)

    tensors: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for target_name, (shard, source_name) in selected.items():
        by_shard.setdefault(shard, []).append((target_name, source_name))

    for shard, names in by_shard.items():
        with safe_open(source / shard, framework="pt", device="cpu") as weights:
            for target_name, source_name in names:
                tensors[target_name] = weights.get_tensor(source_name).contiguous()
    return tensors


def copy_support_files(source: Path, output: Path) -> None:
    excluded = {"config.json", MODEL_INDEX, "README.md"}
    for path in source.iterdir():
        if not path.is_file() or path.name in excluded:
            continue
        if path.name.startswith("model") and ".safetensors" in path.name:
            continue
        shutil.copy2(path, output / path.name)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads((source / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError(f"Unexpected source architecture: {config.get('architectures')}")
    text_config = config["text_config"]
    source_types = layer_types(text_config)
    gdn_layer = (
        args.gdn_layer
        if args.gdn_layer is not None
        else first_layer(source_types, "linear_attention")
    )
    attention_layer = (
        args.attention_layer
        if args.attention_layer is not None
        else first_layer(source_types, "full_attention")
    )
    if source_types[gdn_layer] != "linear_attention":
        raise ValueError(f"Layer {gdn_layer} is not GDN: {source_types[gdn_layer]}")
    if source_types[attention_layer] != "full_attention":
        raise ValueError(
            f"Layer {attention_layer} is not full attention: "
            f"{source_types[attention_layer]}"
        )

    text_config["num_hidden_layers"] = 2
    text_config["full_attention_interval"] = 2
    text_config["layer_types"] = ["linear_attention", "full_attention"]
    text_config["max_position_embeddings"] = args.max_position_embeddings
    text_config["mtp_num_hidden_layers"] = 1
    # SGLang registers Qwen3.5 through its multimodal wrapper, whose constructor
    # inspects the first vision block even for text-only serving. Retain one block
    # to preserve that contract; text smoke requests never execute the vision path.
    config["vision_config"]["depth"] = 1
    config["vision_config"]["deepstack_visual_indexes"] = []
    config["dvr_smoke_model"] = {
        "source": str(source),
        "target_layers": {"0": gdn_layer, "1": attention_layer},
        "purpose": "DVR GDN/full-attention/MTP integration testing only",
    }

    index = json.loads((source / MODEL_INDEX).read_text())
    tensors = load_selected_weights(
        source,
        index["weight_map"],
        gdn_layer=gdn_layer,
        attention_layer=attention_layer,
    )
    required_prefixes = (
        "model.language_model.layers.0.linear_attn.",
        "model.language_model.layers.1.self_attn.",
        "model.visual.blocks.0.",
        "model.visual.patch_embed.",
        "mtp.layers.0.self_attn.",
    )
    for prefix in required_prefixes:
        if not any(name.startswith(prefix) for name in tensors):
            raise ValueError(f"Exported checkpoint is missing {prefix}")

    save_file(tensors, output / MODEL_FILE, metadata={"format": "pt"})
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": {name: MODEL_FILE for name in sorted(tensors)},
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / MODEL_INDEX).write_text(json.dumps(output_index, indent=2) + "\n")
    manifest = {
        "source": str(source),
        "gdn_source_layer": gdn_layer,
        "attention_source_layer": attention_layer,
        "target_layer_types": ["linear_attention", "full_attention"],
        "mtp_layers": 1,
        "vision_layers": 1,
        "weight_count": len(tensors),
        "weight_bytes": total_size,
    }
    (output / "smoke_model_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    copy_support_files(source, output)
    (output / "README.md").write_text(
        f"""# Qwen3.5 DVR two-layer smoke model

This checkpoint is intentionally inaccurate. It retains one target GDN layer,
one target full-attention layer, and the source MTP layer solely for fast
SGLang/DVR integration tests. The original vocabulary, tokenizer, hidden size,
and tensor-parallel layout are retained. The vision tower is reduced to the
single block required by the Qwen3.5 wrapper; use text requests and do not pass
`--enable-multimodal`.

Source layer mapping: GDN {gdn_layer} -> 0, full attention
{attention_layer} -> 1.

This model can validate checkpoint loading, TP sharding, GDN and attention
execution, self-DVR, DVR-EAGLE/MTP, CUDA graph capture, and the decode-vs-prefill
KL oracle. Do not use its text quality, acceptance rate, or throughput as a
model-quality result.

From the SGLang repository root, start a minimal TP4 self-DVR server with:

```bash
MODEL={output}
CUDA_VISIBLE_DEVICES=0,1,2,3 \\
SGLANG_RETURN_ORIGINAL_LOGPROB=True \\
PYTHONPATH="$PWD/python" \\
conda run --no-capture-output -n dvr_dev python -m sglang.launch_server \\
  --model-path "$MODEL" --host 127.0.0.1 --port 30124 --tp-size 4 \\
  --context-length 2048 --max-total-tokens 4096 --max-running-requests 4 \\
  --max-mamba-cache-size 32 --mem-fraction-static 0.3 \\
  --attention-backend triton --linear-attn-backend triton \\
  --sampling-backend pytorch --enable-deterministic-inference \\
  --speculative-algorithm DECODE_VERIFY_ROLLBACK \\
  --speculative-num-draft-tokens 16 --page-size 64 \\
  --cuda-graph-bs 1 2 4 --cuda-graph-max-bs-decode 4 \\
  --skip-server-warmup
```

Run a short boundary/KL check in another shell:

```bash
PYTHONPATH="$PWD/python" conda run --no-capture-output -n dvr_dev \\
python test/manual/dvr/local/clients/dvr_batch_kl.py \\
  --base-url http://127.0.0.1:30124 --request-modes batch \\
  --prompt-token-lengths 63,64,65 --max-new 16 --ignore-eos
```

For DVR-EAGLE/MTP, replace the three self-DVR speculative options with:

```bash
--speculative-algorithm DECODE_VERIFY_ROLLBACK_EAGLE \\
--speculative-draft-model-path "$MODEL" \\
--speculative-num-draft-tokens 2 --speculative-num-steps 1 \\
--speculative-eagle-topk 1
```
"""
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
