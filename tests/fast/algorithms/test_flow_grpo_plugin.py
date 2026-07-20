from argparse import Namespace
from types import SimpleNamespace

from miles.algorithms.flow_grpo import FlowGRPOAlgorithm
from miles.algorithms.labels import grpo_group_advantages
from miles.algorithms.registry import builtin_algorithm_names, load_algorithm, resolve_algorithm_class_path
from miles.utils.types import Sample


def test_builtin_is_flow_grpo_only():
    assert builtin_algorithm_names() == ["flow_grpo"]


def test_load_flow_grpo_algorithm():
    args = Namespace(
        diffusion_algorithm="flow_grpo",
        diffusion_algorithm_path=None,
        use_lora=False,
        diffusion_kl_beta=0.0,
    )
    assert resolve_algorithm_class_path(args).endswith("FlowGRPOAlgorithm")
    algo = load_algorithm(args)
    assert isinstance(algo, FlowGRPOAlgorithm)
    spec = algo.collection_spec()
    assert spec.mode == "online"
    assert spec.needs_logprob is True
    assert spec.needs_trajectory is True


def test_grpo_group_advantages():
    args = SimpleNamespace(
        n_samples_per_prompt=2,
        globalize_reward_mean=False,
        globalize_reward_std=False,
        grpo_std_normalization=True,
        reward_key=None,
    )
    samples = [
        Sample(prompt="a", reward=1.0),
        Sample(prompt="a", reward=3.0),
    ]
    labels = grpo_group_advantages(args, samples)
    assert len(labels.advantages) == 2
    assert abs(sum(labels.advantages)) < 1e-5
