"""Smoke tests for diffusion loss_hub customization dispatch."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.backends.fsdp_utils.loss_hub import (
    flow_grpo_ppo_loss,
    get_diffusion_loss_function,
    grpo_normalize_rewards,
)
from miles.utils.types import Sample


def _args(**overrides):
    base = dict(
        loss_type="flow_grpo_ppo",
        custom_loss_function_path=None,
        n_samples_per_prompt=2,
        globalize_reward_mean=False,
        globalize_reward_std=False,
        grpo_std_normalization=True,
        reward_key=None,
    )
    base.update(overrides)
    return Namespace(**base)


class TestGetDiffusionLossFunction:
    def test_default_flow_grpo(self):
        assert get_diffusion_loss_function(_args()) is flow_grpo_ppo_loss

    def test_policy_loss_alias(self):
        assert get_diffusion_loss_function(_args(loss_type="policy_loss")) is flow_grpo_ppo_loss

    def test_custom_path(self):
        fn = get_diffusion_loss_function(
            _args(
                loss_type="custom_loss",
                custom_loss_function_path="miles.backends.fsdp_utils.loss_hub.losses.flow_grpo_ppo_loss",
            )
        )
        assert fn is flow_grpo_ppo_loss

    def test_custom_requires_path(self):
        with pytest.raises(ValueError, match="custom-loss-function-path"):
            get_diffusion_loss_function(_args(loss_type="custom_loss", custom_loss_function_path=None))


class TestGrpoNormalizeRewards:
    def test_group_mean_centering(self):
        samples = [Sample(reward=r) for r in (1.0, 3.0, 10.0, 10.0)]
        raw, norm = grpo_normalize_rewards(_args(grpo_std_normalization=False), samples)
        assert raw == [1.0, 3.0, 10.0, 10.0]
        # group0 mean=2 → [-1, +1]; group1 mean=10 → [0, 0]
        assert norm == [-1.0, 1.0, 0.0, 0.0]
