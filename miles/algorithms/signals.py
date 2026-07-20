"""Shared reward → train-signal helpers for GRPO-style algorithms."""

from __future__ import annotations

import torch

from miles.algorithms.base import TrainSignals
from miles.utils.types import Sample


def grpo_group_advantages(args, samples: list[Sample]) -> TrainSignals:
    """Group-relative advantage normalization (Flow-GRPO default)."""
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    rewards_flat = torch.tensor(raw_rewards, dtype=torch.float)
    rewards = rewards_flat.view(-1, args.n_samples_per_prompt)

    if args.globalize_reward_mean:
        mean = rewards_flat.mean()
    else:
        mean = rewards.mean(dim=-1, keepdim=True)
    centered = rewards - mean

    if args.grpo_std_normalization:
        if args.globalize_reward_std:
            std = rewards_flat.std()
        else:
            std = rewards.std(dim=-1, keepdim=True)
        centered = centered / (std + 1e-4)

    return TrainSignals(raw_rewards=raw_rewards, advantages=centered.flatten().tolist())
