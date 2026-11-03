"""Parallel Decoding Distillation (arXiv:2607.26004, equations 9--15).

The grid may increase (paper time) or decrease (diffusers sigma). The teacher
must return derivatives with respect to that same coordinate. Predictions use
shape [interval, batch, ...]; interval indices are absolute, not block-relative.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


def validate_grid(grid: torch.Tensor) -> None:
    if grid.ndim != 1 or grid.numel() < 2 or not torch.isfinite(grid).all():
        raise ValueError("grid must be a finite 1D tensor with at least two points")
    delta = grid.diff()
    if not ((delta > 0).all() or (delta < 0).all()):
        raise ValueError("grid must be strictly monotonic")


def shifted_sigma_grid(num_intervals: int, shift: float = 1.0, *, device=None) -> torch.Tensor:
    if num_intervals < 1 or not 0 < shift < float("inf"):
        raise ValueError("num_intervals and finite shift must be positive")
    sigma = torch.linspace(1, 0, num_intervals + 1, device=device, dtype=torch.float64)
    return (shift * sigma / (1 + (shift - 1) * sigma)).float()


class ParallelLinear(nn.Module):
    """N independent output projections initialized from one pretrained layer."""

    def __init__(self, source: nn.Linear, num_intervals: int):
        super().__init__()
        if num_intervals < 1:
            raise ValueError("num_intervals must be positive")
        self.weight = nn.Parameter(source.weight.detach().unsqueeze(0).repeat(num_intervals, 1, 1))
        self.bias = (
            nn.Parameter(source.bias.detach().unsqueeze(0).repeat(num_intervals, 1))
            if source.bias is not None
            else None
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        result = torch.einsum("...i,koi->k...o", hidden, self.weight)
        if self.bias is not None:
            result = result + self.bias.reshape(self.bias.shape[0], *([1] * (hidden.ndim - 1)), -1)
        return result


@torch.no_grad()
def teacher_mean_velocity(
    teacher: Callable, x: torch.Tensor, time: torch.Tensor, delta: torch.Tensor, solver: str = "midpoint"
) -> torch.Tensor:
    """Teacher closure includes conditioning/CFG; never build a teacher gradient graph."""
    if solver not in {"euler", "midpoint"}:
        raise ValueError("solver must be euler or midpoint")
    velocity = teacher(x.detach(), time)
    if velocity.shape != x.shape:
        raise ValueError("teacher output must match the latent shape exactly")
    if solver == "midpoint":
        velocity = teacher(x.detach() + 0.5 * delta * velocity, time + 0.5 * delta)
        if velocity.shape != x.shape:
            raise ValueError("teacher output must match the latent shape exactly")
    return velocity.detach()


def block_update(x: torch.Tensor, predictions: torch.Tensor, grid: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Accumulate signed interval lengths in fp32 (or fp64 for fp64 inputs)."""
    if predictions.shape != (len(grid) - 1, *x.shape):
        raise ValueError("predictions must have shape [num_intervals, *latent.shape]")
    if not 0 <= start <= end < len(grid):
        raise ValueError("invalid block bounds")
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    h = grid.diff()[start:end].to(device=x.device, dtype=dtype)
    return x.to(dtype) + torch.einsum("k,k...->...", h, predictions[start:end].to(dtype))


@dataclass
class PDDLossResult:
    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    teacher_state: torch.Tensor
    target: torch.Tensor


def pdd_loss(
    predictions: torch.Tensor,
    x_start: torch.Tensor,
    grid: torch.Tensor,
    start: int,
    selected: int,
    teacher: Callable,
    *,
    solver: str = "midpoint",
) -> PDDLossResult:
    """One uniformly sampled intra-block interval; shared indices across this microbatch.

    The caller supplies predictions from ONE student forward at (x_start, grid[start]).
    Earlier heads construct the teacher input under stop-gradient. Only the selected
    head and its shared backbone receive gradients from this Monte Carlo loss.
    """
    validate_grid(grid)
    if not 0 <= start <= selected < len(grid) - 1:
        raise ValueError("require 0 <= start <= selected < num_intervals")
    with torch.no_grad():
        x_selected = block_update(x_start.detach(), predictions.detach(), grid, start, selected)
        target = teacher_mean_velocity(teacher, x_selected, grid[selected], grid.diff()[selected], solver)
    error = (predictions[selected].float() - target.float()).square()
    per_sample = error.flatten(1).mean(1)
    return PDDLossResult(per_sample.mean(), per_sample, x_selected, target)


@torch.no_grad()
def sample_pdd(student: Callable, noise: torch.Tensor, grid: torch.Tensor, block_size: int) -> torch.Tensor:
    """Reference sampler; ceil(N/block_size) student calls, with a clipped final block."""
    validate_grid(grid)
    if not 1 <= block_size < len(grid):
        raise ValueError("block_size must be in [1, num_intervals]")
    x = noise.clone()
    for start in range(0, len(grid) - 1, block_size):
        predictions = student(x, grid[start])
        x = block_update(x, predictions, grid, start, min(start + block_size, len(grid) - 1))
    return x
