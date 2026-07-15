from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import numpy as np
import ray
import torch
from PIL import Image

from miles.utils.misc import SingletonMeta
from miles.utils.processing_utils import cfhw_to_fhwc, image_or_video_to_uint8
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def sample_frame_indices(num_total_frames: int, num_frames: int | None) -> list[int]:
    if num_total_frames <= 0:
        raise ValueError(f"video has no frames: {num_total_frames}")
    if num_frames is None or num_total_frames <= num_frames:
        return list(range(num_total_frames))
    if num_frames == 1:
        return [num_total_frames // 2]
    step = (num_total_frames - 1) / (num_frames - 1)
    return [int(round(i * step)) for i in range(num_frames)]


def _feature_tensor(features):
    # transformers <5.0 returns a plain tensor; >=5.0 returns BaseModelOutputWithPooling.
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output") and isinstance(features.pooler_output, torch.Tensor):
        return features.pooler_output
    raise TypeError(f"Cannot extract embedding tensor from {type(features)!r}")


def _sample_to_rgb_hwc_uint8_frames(sample: Sample, num_frames: int | None) -> list[np.ndarray]:
    cfhw = sample.generated_output
    if cfhw is None:
        raise ValueError("generated_output is None")

    fhwc = image_or_video_to_uint8(cfhw_to_fhwc(cfhw.detach().cpu()))
    indices = sample_frame_indices(fhwc.shape[0], num_frames)
    return [np.ascontiguousarray(fhwc[i].numpy()) for i in indices]


class PickScoreScorer(torch.nn.Module):
    """CLIP PickScore for (prompt, image) pairs; raw logits scaled to ~0-1."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        processor_path: str,
        model_path: str,
    ) -> None:
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> list[float]:
        image_inputs = self.processor(images=list(images), return_tensors="pt", padding=True)
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        if "pixel_values" in image_inputs:
            image_inputs["pixel_values"] = image_inputs["pixel_values"].float()

        text_inputs = self.processor(
            text=list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        image_embs = _feature_tensor(self.model.get_image_features(**image_inputs))
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        text_embs = _feature_tensor(self.model.get_text_features(**text_inputs))
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        scores = self.model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)
        # Flow-Factory convention: scale raw PickScore logits (~0-26) to ~0-1.
        scores = scores.float() / 26.0
        return [float(score) for score in scores.detach().cpu()]


@ray.remote
class PickScoreRewardActor:
    def __init__(
        self,
        *,
        processor_path: str,
        model_path: str,
    ) -> None:
        gpu_ids = ray.get_gpu_ids()
        use_cuda = bool(gpu_ids) and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        device = "cuda" if use_cuda else "cpu"
        self.scorer = PickScoreScorer(
            device=device,
            processor_path=processor_path,
            model_path=model_path,
        )

    def score_batch(self, images: list, prompts: list[str]) -> list[float]:
        pil_images = [Image.fromarray(image) if isinstance(image, np.ndarray) else image for image in images]
        return self.scorer(prompts, pil_images)


class AsyncPickScorePool(metaclass=SingletonMeta):
    """Ray actor pool for GPU PickScore reward inference."""

    def __init__(self, args) -> None:
        num_workers = args.pickscore_num_workers
        num_gpus_per_worker = args.pickscore_num_gpus_per_worker
        self._batch_size = args.pickscore_batch_size
        self._actors = [
            PickScoreRewardActor.options(
                num_cpus=1,
                num_gpus=num_gpus_per_worker,
                scheduling_strategy="DEFAULT",
            ).remote(
                processor_path=args.pickscore_processor_path,
                model_path=args.pickscore_model_path,
            )
            for _ in range(num_workers)
        ]
        self._round_robin_index = 0
        logger.info(
            "Initialized PickScore actor pool with %d workers, %.3f GPUs/worker, batch_size=%d.",
            num_workers,
            num_gpus_per_worker,
            self._batch_size,
        )

    def _next_actor(self):
        i = self._round_robin_index % len(self._actors)
        self._round_robin_index += 1
        return self._actors[i]

    async def score(self, images: list, prompts: list[str]) -> list[float]:
        refs = []
        for start in range(0, len(images), self._batch_size):
            end = start + self._batch_size
            refs.append(self._next_actor().score_batch.remote(images[start:end], prompts[start:end]))

        loop = asyncio.get_running_loop()
        chunked_scores = await loop.run_in_executor(None, ray.get, refs)
        return [float(score) for chunk in chunked_scores for score in chunk]


async def pickscore_rm(args, samples: Sequence[Sample]) -> list[float]:
    pool = AsyncPickScorePool(args)
    images: list[np.ndarray] = []
    prompts: list[str] = []
    frame_counts: list[int] = []
    for sample in samples:
        frames = _sample_to_rgb_hwc_uint8_frames(sample, args.pickscore_num_frames)
        images.extend(frames)
        prompts.extend([sample.prompt] * len(frames))
        frame_counts.append(len(frames))

    flat_scores = await pool.score(images, prompts)
    scores: list[float] = []
    offset = 0
    for count in frame_counts:
        scores.append(float(sum(flat_scores[offset : offset + count]) / count))
        offset += count
    return scores
