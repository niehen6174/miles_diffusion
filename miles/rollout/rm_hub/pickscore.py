from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence

import numpy as np
import ray
import torch
from PIL import Image

from miles.utils.misc import SingletonMeta
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _sample_to_rgb_hwc_uint8(sample: Sample) -> np.ndarray:
    generated_output = sample.generated_output
    if generated_output is None:
        raise ValueError("generated_output is None")
    if generated_output.ndim != 4:
        raise ValueError(
            f"generated_output must be 4D [C, F, H, W], got {tuple(generated_output.shape)}"
        )

    frame_chw = generated_output.detach().cpu()[:, 0, :, :]
    if frame_chw.shape[0] == 1:
        frame_chw = frame_chw.repeat(3, 1, 1)
    elif frame_chw.shape[0] > 3:
        frame_chw = frame_chw[:3]
    elif frame_chw.shape[0] != 3:
        raise ValueError(f"generated_output channel dimension must be 1, 3, or >3, got {frame_chw.shape[0]}")

    hwc = frame_chw.float().numpy().transpose(1, 2, 0)
    if float(hwc.max()) <= 1.0 + 1e-3:
        hwc = hwc * 255.0
    return np.ascontiguousarray(hwc.clip(0, 255).astype(np.uint8))


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported PickScore dtype: {dtype_name}")


def _required_arg(args, name: str) -> str:
    value = getattr(args, name, None)
    if value is None or value == "":
        raise ValueError(f"--{name.replace('_', '-')} must be set when --rm-type pickscore.")
    return value


class PickScoreScorer(torch.nn.Module):
    """Small local copy of Flow-GRPO's PickScore scorer.

    The scorer consumes final PIL images and prompt strings, then returns one
    scalar reward per prompt/image pair.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        processor_path: str,
        model_path: str,
    ) -> None:
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.dtype = dtype
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)
        self.model = self.model.to(dtype=dtype)

    def _to_device(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        moved = {}
        for key, value in inputs.items():
            if torch.is_floating_point(value):
                moved[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                moved[key] = value.to(device=self.device)
        return moved

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> list[float]:
        if len(images) != len(prompts):
            raise ValueError(f"Images({len(images)}) and prompts({len(prompts)}) must have the same length")

        image_inputs = self.processor(
            images=list(images),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = self.processor(
            text=list(prompts),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = self._to_device(image_inputs)
        text_inputs = self._to_device(text_inputs)

        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)
        scores = scores.diag() / 26.0
        return [float(score) for score in scores.detach().cpu()]


@ray.remote
class PickScoreRewardActor:
    def __init__(
        self,
        *,
        dtype_name: str = "fp32",
        processor_path: str,
        model_path: str,
    ) -> None:
        gpu_ids = ray.get_gpu_ids()
        use_cuda = bool(gpu_ids) and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        device = "cuda" if use_cuda else "cpu"
        logger.info(
            "Initializing PickScore actor on device=%s ray_gpu_ids=%s CUDA_VISIBLE_DEVICES=%s",
            device,
            gpu_ids,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        self.scorer = PickScoreScorer(
            device=device,
            dtype=_dtype_from_name(dtype_name),
            processor_path=processor_path,
            model_path=model_path,
        )

    def score_batch(self, images: list[np.ndarray], prompts: list[str]) -> list[float]:
        pil_images = [Image.fromarray(image) for image in images]
        return self.scorer(prompts, pil_images)


class AsyncPickScorePool(metaclass=SingletonMeta):
    """Ray actor pool for GPU PickScore reward inference."""

    def __init__(self, args) -> None:
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. PickScore RM requires Ray for PickScoreRewardActor.")

        num_workers = int(getattr(args, "pickscore_num_workers", 1) or 1)
        if num_workers <= 0:
            raise ValueError(f"pickscore_num_workers must be > 0, got {num_workers}")

        num_gpus_per_worker = float(getattr(args, "pickscore_num_gpus_per_worker", 1.0))
        if num_gpus_per_worker < 0:
            raise ValueError(f"pickscore_num_gpus_per_worker must be >= 0, got {num_gpus_per_worker}")

        self._batch_size = int(getattr(args, "pickscore_batch_size", 8) or 8)
        if self._batch_size <= 0:
            raise ValueError(f"pickscore_batch_size must be > 0, got {self._batch_size}")

        dtype_name = getattr(args, "pickscore_dtype", "fp32")
        processor_path = _required_arg(args, "pickscore_processor_path")
        model_path = _required_arg(args, "pickscore_model_path")
        self._actors = [
            PickScoreRewardActor.options(
                num_cpus=1,
                num_gpus=num_gpus_per_worker,
                scheduling_strategy="DEFAULT",
            ).remote(
                dtype_name=dtype_name,
                processor_path=processor_path,
                model_path=model_path,
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

    async def score(self, images: list[np.ndarray], prompts: list[str]) -> list[float]:
        refs = []
        for start in range(0, len(images), self._batch_size):
            end = start + self._batch_size
            refs.append(self._next_actor().score_batch.remote(images[start:end], prompts[start:end]))

        loop = asyncio.get_running_loop()
        chunked_scores = await loop.run_in_executor(None, ray.get, refs)
        return [float(score) for chunk in chunked_scores for score in chunk]


async def pickscore_rm(args, samples: Sequence[Sample]) -> list[float]:
    pool = AsyncPickScorePool(args)
    images = [_sample_to_rgb_hwc_uint8(sample) for sample in samples]
    prompts = [sample.prompt for sample in samples]
    return await pool.score(images, prompts)
