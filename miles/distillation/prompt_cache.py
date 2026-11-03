"""Verified, bounded-memory text conditioning for large PDD prompt sets."""

import hashlib
import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def validate_condition(condition, resolution):
    embeddings = condition["encoder_hidden_states"]
    if embeddings.ndim != 3 or embeddings.shape[0] != 1 or embeddings.shape[1] < 1:
        raise ValueError("invalid text embedding shape")
    if embeddings.device.type != "cpu" or embeddings.dtype != torch.bfloat16 or not torch.isfinite(embeddings).all():
        raise ValueError("expected finite CPU BF16 embeddings")
    length = embeddings.shape[1]
    if condition["encoder_hidden_states_mask"] is not None:
        raise ValueError("unpadded single-prompt conditions must have no attention mask")
    mask = condition["img_mask"]
    if mask.dtype != torch.bool or mask.shape != (1, length + (resolution // 32) ** 2):
        raise ValueError("invalid image mask")
    if mask[:, :length].any() or not mask[:, length:].all():
        raise ValueError("invalid text/image mask positions")
    if condition["img_shapes"] != [[(1, resolution // 16, resolution // 16)]]:
        raise ValueError("image geometry mismatch")


def split_encoded_batch(embeddings, attention_mask, image_mask, resolution):
    """Remove batch padding so cached rows match the native single-prompt API."""
    result = []
    for i in range(embeddings.shape[0]):
        keep = (
            torch.ones(embeddings.shape[1], dtype=torch.bool, device=embeddings.device)
            if attention_mask is None
            else attention_mask[i].bool()
        )
        text = embeddings[i, keep].unsqueeze(0).detach().cpu().contiguous()
        text_mask = image_mask[i, keep].unsqueeze(0).cpu().bool()
        condition = {
            "encoder_hidden_states": text,
            "encoder_hidden_states_mask": None,
            "img_mask": torch.cat([text_mask, torch.ones(1, (resolution // 32) ** 2, dtype=torch.bool)], dim=1),
            "img_shapes": [[(1, resolution // 16, resolution // 16)]],
        }
        validate_condition(condition, resolution)
        result.append(condition)
    return result


class ShardedConditions:
    """Read-only integer-indexed conditions, with at most max_shards resident."""

    def __init__(self, root, split, *, model_revision=None, resolution=None, max_shards=2):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if not self.manifest.get("completed") or self.manifest.get("format_version") != 1:
            raise ValueError("cache is incomplete or has an unsupported format")
        signature = self.manifest["signature"]
        if model_revision is not None and signature["model_revision"] != model_revision:
            raise ValueError("cache model revision mismatch")
        if resolution is not None and signature["resolution"] != resolution:
            raise ValueError("cache resolution mismatch")
        if max_shards < 1:
            raise ValueError("max_shards must be positive")
        self.resolution = signature["resolution"]
        self.max_shards = max_shards
        self.shards = self.manifest["splits"][split]["shards"]
        self.count = self.manifest["splits"][split]["count"]
        expected = 0
        for shard in self.shards:
            if shard["start"] != expected or shard["count"] < 1:
                raise ValueError("noncontiguous shard index")
            expected += shard["count"]
            relative = Path(shard["file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe shard path")
        if expected != self.count:
            raise ValueError("shard row count mismatch")
        self.starts = [s["start"] for s in self.shards]
        self._resident = OrderedDict()
        self._verified = set()

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < self.count:
            raise IndexError(index)
        number = bisect_right(self.starts, index) - 1
        info = self.shards[number]
        if number not in self._resident:
            path = self.root / info["file"]
            if number not in self._verified:
                if file_sha256(path) != info["sha256"]:
                    raise ValueError(f"corrupt cache shard: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if payload["ids"] != info["ids"] or len(payload["conditions"]) != info["count"]:
                raise ValueError("shard IDs or count do not match manifest")
            if number not in self._verified:
                for condition in payload["conditions"]:
                    validate_condition(condition, self.resolution)
                self._verified.add(number)
            self._resident[number] = payload["conditions"]
            while len(self._resident) > self.max_shards:
                self._resident.popitem(last=False)
        self._resident.move_to_end(number)
        return self._resident[number][index - info["start"]]
