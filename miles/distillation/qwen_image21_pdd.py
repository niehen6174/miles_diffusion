"""Experimental PDD adapter for Qwen-Image 2.1 T2I (not Qwen-Image 1.0).

The single-stream transformer emits text + target image tokens. Slice the target
tail before exposing [interval, batch, image_tokens, channels] to the PDD loss.
The caller supplies the same img_mask/img_shapes/text conditioning as Miles' 2.1 config.
"""

from __future__ import annotations

import json

import torch
from torch import nn

from .pdd import ParallelLinear


def checkpoint_model_config(transformer: nn.Module) -> dict:
    """Store constructor fields only, excluding HF metadata such as a pathlib checkpoint path."""
    return json.loads(json.dumps({key: value for key, value in transformer.config.items() if not key.startswith("_")}))


class TokenParallelProjection(nn.Module):
    def __init__(self, source: nn.Linear, num_intervals: int):
        super().__init__()
        self.parallel = ParallelLinear(source, num_intervals)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # [N, B, S, C] -> [B, S, N*C]; transformer keeps joint sequence semantics.
        values = self.parallel(hidden)
        return values.permute(1, 2, 0, 3).flatten(2)


class QwenImage21PDDStudent(nn.Module):
    """Convert an owned unsharded transformer; its input timestep is sigma, not sigma*1000."""

    def __init__(self, transformer: nn.Module, num_intervals: int):
        super().__init__()
        if not isinstance(transformer.proj_out, nn.Linear):
            raise TypeError("expected an unmodified Qwen-Image 2.1 nn.Linear proj_out")
        if transformer.config.patch_size != 1:
            raise ValueError("Qwen-Image 2.1 PDD currently requires unpatched latent tokens")
        self.channels = transformer.proj_out.out_features
        if self.channels != transformer.config.in_channels:
            raise ValueError("PDD requires matching latent input/output channels")
        self.num_intervals = num_intervals
        transformer.proj_out = TokenParallelProjection(transformer.proj_out, num_intervals)
        self.transformer = transformer

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, **conditioning) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError("expected packed latent tokens [batch, image_tokens, channels]")
        if conditioning.get("kv_cache") is not None or conditioning.get("kv_cache_mode") is not None:
            raise ValueError("PDD training does not support the inference prefix KV cache")
        dtype = self.transformer.img_in.weight.dtype
        conditioning = {
            key: value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
            for key, value in conditioning.items()
        }
        output = self.transformer(
            hidden_states=x.to(dtype),
            timestep=sigma.expand(x.shape[0]),
            return_dict=False,
            **conditioning,
        )[0]
        target = output[:, -x.shape[1] :]
        return target.reshape(x.shape[0], x.shape[1], self.num_intervals, self.channels).permute(2, 0, 1, 3)
