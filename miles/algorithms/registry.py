"""Resolve ``--diffusion-algorithm`` / ``--diffusion-algorithm-path`` to a class.

PR1 ships Flow-GRPO only; later PRs register additional builtins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miles.algorithms.base import DiffusionAlgorithm

_BUILTIN: dict[str, str] = {
    "flow_grpo": "miles.algorithms.flow_grpo.FlowGRPOAlgorithm",
}


def resolve_algorithm_class_path(args) -> str:
    """Return the dotted class path for the selected diffusion algorithm."""
    if getattr(args, "diffusion_algorithm_path", None):
        return args.diffusion_algorithm_path
    name = getattr(args, "diffusion_algorithm", None) or "flow_grpo"
    key = str(name).strip().lower()
    if key not in _BUILTIN:
        raise ValueError(
            f"Unknown --diffusion-algorithm {name!r}; choose one of {sorted(_BUILTIN)} "
            "or pass --diffusion-algorithm-path. "
            "SFT/AWM/NFT land in follow-up PRs."
        )
    return _BUILTIN[key]


def load_algorithm(args) -> DiffusionAlgorithm:
    from miles.utils.misc import load_function

    path = resolve_algorithm_class_path(args)
    cls = load_function(path)
    algo = cls() if isinstance(cls, type) else cls
    algo.validate_args(args)
    return algo


def builtin_algorithm_names() -> list[str]:
    return sorted(_BUILTIN)
