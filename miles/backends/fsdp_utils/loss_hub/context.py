"""Shared handles passed into diffusion loss functions."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DiffusionLossContext:
    """Train-side handles for a diffusion ``LossFunction``.

    Kept free of Ray / FSDP actor internals so losses are unit-testable and
    swappable via ``--custom-loss-function-path``.
    """

    models: dict[str, torch.nn.Module]
    model: torch.nn.Module
    train_pipeline_config: Any
    sde_backend: Any
    scheduler: Any
    args: Namespace
    forward_dtype: torch.dtype
    device: torch.device
