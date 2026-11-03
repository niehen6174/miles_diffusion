"""Encode a training JSONL into the bounded-memory cache consumed by Miles PDD."""

import argparse
import json
from pathlib import Path

import torch

from miles.distillation.prompt_cache import atomic_json, file_sha256, split_encoded_batch


def read_prompts(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or any(
        row.get("split") != "train"
        or not isinstance(row.get("id"), str)
        or not row["id"].strip()
        or not isinstance(row.get("input"), str)
        or not row["input"].strip()
        for row in rows
    ):
        raise ValueError("Expected nonempty training rows with string id and input")
    if len({row["id"] for row in rows}) != len(rows) or len({row["input"] for row in rows}) != len(rows):
        raise ValueError("Training IDs and prompts must be unique")
    return rows


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=64)
    args = parser.parse_args()
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")
    rows = read_prompts(args.prompts)
    source_hash = file_sha256(args.prompts)
    args.output.mkdir(parents=True, exist_ok=False)

    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline.from_pretrained(
        args.model, transformer=None, vae=None, torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipe.text_encoder.to("cuda").eval().requires_grad_(False)
    shards = []
    for start in range(0, len(rows), args.shard_size):
        batch = rows[start : start + args.shard_size]
        conditions = [
            split_encoded_batch(*pipe.encode_prompt(prompt=row["input"], device="cuda"), 1024)[0] for row in batch
        ]
        path = args.output / f"shard-{start:06d}.pt"
        ids = [row["id"] for row in batch]
        torch.save({"ids": ids, "conditions": conditions}, path)
        shards.append(
            {"start": start, "count": len(batch), "file": path.name, "ids": ids, "sha256": file_sha256(path)}
        )
        print(f"Encoded {start + len(batch)}/{len(rows)}", flush=True)
    if file_sha256(args.prompts) != source_hash:
        raise ValueError("Prompt file changed during encoding")
    # Only publish a readable cache after every shard has been written.
    atomic_json(
        args.output / "manifest.json",
        {
            "format_version": 1,
            "completed": True,
            "signature": {"model_revision": args.model.name, "resolution": 1024},
            "splits": {"train": {"count": len(rows), "source_sha256": source_hash, "shards": shards}},
        },
    )


if __name__ == "__main__":
    main()
