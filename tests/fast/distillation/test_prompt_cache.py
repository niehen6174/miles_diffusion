import json

import pytest
import torch

from miles.distillation.prompt_cache import (
    ShardedConditions,
    atomic_json,
    file_sha256,
    split_encoded_batch,
)


def test_batch_unpadding_keeps_each_prompt_and_geometry():
    embeds = torch.arange(2 * 4 * 3).reshape(2, 4, 3).to(torch.bfloat16)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    result = split_encoded_batch(embeds, mask, torch.zeros(2, 4, dtype=torch.bool), 1024)
    for index, length in enumerate([2, 4]):
        assert torch.equal(result[index]["encoder_hidden_states"], embeds[index : index + 1, :length])
        assert result[index]["encoder_hidden_states_mask"] is None
        assert result[index]["img_mask"].shape == (1, length + 1024)
        assert result[index]["img_shapes"] == [[(1, 64, 64)]]
    with pytest.raises(ValueError, match="positions"):
        split_encoded_batch(embeds, mask, torch.ones(2, 4, dtype=torch.bool), 1024)
    embeds[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        split_encoded_batch(embeds, mask, torch.zeros(2, 4, dtype=torch.bool), 1024)


def build_cache(root):
    shards = []
    for i in range(3):
        conditions = split_encoded_batch(
            torch.full((2, 3, 4), i, dtype=torch.bfloat16), None, torch.zeros(2, 3, dtype=torch.bool), 1024
        )
        path = root / f"shard-{i}.pt"
        ids = [f"id-{i}-0", f"id-{i}-1"]
        torch.save({"ids": ids, "conditions": conditions}, path)
        shards.append({"start": 2 * i, "count": 2, "file": path.name, "sha256": file_sha256(path), "ids": ids})
    atomic_json(
        root / "manifest.json",
        {
            "completed": True,
            "format_version": 1,
            "signature": {"model_revision": "revision", "resolution": 1024},
            "splits": {"train": {"count": 6, "shards": shards}},
        },
    )


def test_shards_random_access_lru_and_integrity(tmp_path):
    build_cache(tmp_path)
    dataset = ShardedConditions(tmp_path, "train", max_shards=1, model_revision="revision", resolution=1024)
    for index in [0, 5, 2, 1, 4]:
        assert (dataset[index]["encoder_hidden_states"] == index // 2).all()
        assert len(dataset._resident) == 1
    for index in [-1, 6]:
        with pytest.raises(IndexError):
            dataset[index]
    with pytest.raises(ValueError, match="revision"):
        ShardedConditions(tmp_path, "train", model_revision="other")
    with (tmp_path / "shard-1.pt").open("ab") as stream:
        stream.write(b"corruption")
    fresh = ShardedConditions(tmp_path, "train")
    with pytest.raises(ValueError, match="corrupt"):
        fresh[2]


def test_incomplete_and_index_gap_rejected(tmp_path):
    build_cache(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["completed"] = False
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="incomplete"):
        ShardedConditions(tmp_path, "train")
    manifest["completed"] = True
    manifest["splits"]["train"]["shards"][1]["start"] = 3
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="noncontiguous"):
        ShardedConditions(tmp_path, "train")
