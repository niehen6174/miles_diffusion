"""Diffusion algorithm plugin interfaces.

Model-family concerns stay in ``TrainPipelineConfig``; algorithm plugins own
collection contracts, train-example schema, and loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class CollectionSpec:
    """What the collector / rollout engine must provide for this algorithm."""

    mode: str  # "offline" | "online"
    needs_reward: bool = True
    needs_trajectory: bool = True
    needs_logprob: bool = True
    sampler: str = "sde"  # "sde" | "ode" | "any"
    return_denoising_env: bool = True
    sync_weights_to_rollout: bool = True


@dataclass
class TrainLossContext:
    """Train-side handles passed into ``compute_loss``."""

    models: dict[str, torch.nn.Module]
    model: torch.nn.Module
    train_pipeline_config: Any
    sde_backend: Any | None
    scheduler: Any | None
    args: Any
    forward_dtype: torch.dtype
    device: torch.device


@dataclass
class LossOutput:
    """Reserved return type for ``compute_loss`` (loss + metrics).

    **Unused in the Flow-GRPO PR:** ``compute_loss`` still returns a bare
    ``Tensor`` and appends into the caller's ``log_stats`` (actor-compatible).
    Kept for a later cleanup that unifies the return shape.
    """

    loss_sum: torch.Tensor
    log_stats: dict[str, list[torch.Tensor]] = field(default_factory=dict)


@dataclass
class TrainSignals:
    """Reward-derived training signals attached to samples before ``build_train_data``.

    Not classification labels: e.g. GRPO advantages or NFT soft +/− weights.
    """

    raw_rewards: list[float]
    advantages: list[float] | None = None
    # Reserved for DiffusionNFT soft positive/negative weights; Flow-GRPO ignores it.
    nft_signals: list[float] | None = None


@runtime_checkable
class DiffusionAlgorithm(Protocol):
    name: str

    def validate_args(self, args) -> None: ...

    def collection_spec(self) -> CollectionSpec:
        """Return acquisition contract; see ``CollectionSpec`` — not fully consumed yet."""
        ...

    def postprocess_rewards(self, args, samples: list) -> TrainSignals: ...

    def build_train_data(self, args, samples: list, signals: TrainSignals) -> dict[str, Any]: ...

    def validate_train_batch(self, batch: list[dict]) -> list[str]: ...

    def compute_loss(
        self,
        ctx: TrainLossContext,
        batch: list[dict],
        *,
        log_stats: dict[str, list[torch.Tensor]],
        pad_to_len: int | None = None,
    ) -> torch.Tensor: ...

    def prepare_rollout_data(self, rollout_data: dict, ctx: TrainLossContext) -> None:
        """Optional hook before the micro-batch loop (e.g. sync scheduler meta)."""
        ...
