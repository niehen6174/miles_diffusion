"""CPU-only prompt producer for PDD; frozen cached embeddings, no rollout engine."""

import json
import logging
from functools import lru_cache
from pathlib import Path

from miles.distillation.prompt_cache import ShardedConditions, file_sha256
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.utils import tracking_utils
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cache(cache_path, prompt_path, model_path, resolution):
    cache = ShardedConditions(cache_path, "train", model_revision=Path(model_path).name, resolution=resolution)
    if file_sha256(prompt_path) != cache.manifest["splits"]["train"]["source_sha256"]:
        raise ValueError("PDD prompt file must match the encoded training split exactly")
    rows = [json.loads(line) for line in Path(prompt_path).read_text().splitlines()]
    ids = [key for shard in cache.shards for key in shard["ids"]]
    if [r["id"] for r in rows] != ids or any(r["split"] != "train" for r in rows):
        raise ValueError("PDD cache IDs or training split mismatch")
    lookup = {r["input"]: i for i, r in enumerate(rows)}
    if len(lookup) != len(rows):
        raise ValueError("PDD training prompts must be unique")
    return cache, lookup


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        raise ValueError("PDD training-only producer does not run image evaluation")
    cache, lookup = _cache(args.pdd_train_cache, args.prompt_data, args.hf_checkpoint, args.diffusion_height)
    groups = data_source.get_samples(args.rollout_batch_size)
    for group in groups:
        for sample in group:
            index = lookup[sample.prompt]
            sample.train_metadata = {
                "pdd_pair": {"sample_index": sample.index, "cache_index": index, "cond_kwargs": cache[index]}
            }
            sample.status = Sample.Status.COMPLETED
    return RolloutFnTrainOutput(samples=groups, metrics={"pdd_trajectories": sum(map(len, groups))})


def convert_samples_to_train_data(args, samples):
    return {"train_data": [s.train_metadata["pdd_pair"] for s in samples]}


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    values = {f"rollout/{k}": v for k, v in (rollout_extra_metrics or {}).items()}
    values["rollout/step"] = compute_rollout_step(args, rollout_id)
    tracking_utils.log(args, values, step_key="rollout/step")
    logger.info("PDD trajectory batch %d: %s (%.1fs)", rollout_id, values, rollout_time)
    return True
