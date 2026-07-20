"""Diffusion algorithm plugins.

Currently ships Flow-GRPO only; SFT / AWM / DiffusionNFT land in follow-up PRs.
"""

from miles.algorithms.base import CollectionSpec, DiffusionAlgorithm, TrainLossContext, TrainSignals
from miles.algorithms.registry import builtin_algorithm_names, load_algorithm, resolve_algorithm_class_path

__all__ = [
    "CollectionSpec",
    "DiffusionAlgorithm",
    "TrainLossContext",
    "TrainSignals",
    "builtin_algorithm_names",
    "load_algorithm",
    "resolve_algorithm_class_path",
]
