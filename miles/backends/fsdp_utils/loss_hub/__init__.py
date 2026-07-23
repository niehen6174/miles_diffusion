"""Diffusion training building blocks (miles customization style).

Loss / advantage math lives here; the FSDP actor only schedules micro-batches
and calls ``get_diffusion_loss_function``. Swap pieces via ``--loss-type`` /
``--custom-loss-function-path`` (and the rollout ``*-path`` hooks).
"""

from miles.backends.fsdp_utils.loss_hub.advantages import grpo_normalize_rewards
from miles.backends.fsdp_utils.loss_hub.context import DiffusionLossContext
from miles.backends.fsdp_utils.loss_hub.losses import (
    DiffusionLossFunction,
    flow_grpo_ppo_loss,
    get_diffusion_loss_function,
)

__all__ = [
    "DiffusionLossContext",
    "DiffusionLossFunction",
    "flow_grpo_ppo_loss",
    "get_diffusion_loss_function",
    "grpo_normalize_rewards",
]
