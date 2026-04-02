from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class LazyTensor(ABC):
    """Deferred load: **base64 ``str``** matching :func:`~miles.utils.diffusion_rollout_response.tensor_to_base64`.

    Public API: :meth:`resolve` → **CPU** :class:`torch.Tensor`.
    """

    @abstractmethod
    def resolve(self) -> torch.Tensor:
        """Materialize to a CPU tensor."""
        raise NotImplementedError


@dataclass
class SafetensorsBase64LazyTensor(LazyTensor):
    """Tensor wire: base64 of safetensors (default key ``"t"``) or ``torch.save`` bytes."""

    b64: str
    tensor_key: str | None = None

    def resolve(self) -> torch.Tensor:
        from miles.utils.diffusion_rollout_response import decode_tensor_base64

        return decode_tensor_base64(self.b64, tensor_key=self.tensor_key)


def safetensors_b64_lazy_tensor(b64: str, *, tensor_key: str | None = None) -> SafetensorsBase64LazyTensor:
    """Construct the only supported :class:`LazyTensor` implementation."""
    return SafetensorsBase64LazyTensor(b64=b64, tensor_key=tensor_key)


# Tensor field: either deferred safetensors+b64 or already materialized (e.g. after ``resolve()``).
RolloutTensorRef = LazyTensor | torch.Tensor


def resolve_maybe_lazy(value: RolloutTensorRef | None) -> torch.Tensor | None:
    """If ``value`` is :class:`LazyTensor`, call :meth:`LazyTensor.resolve`; else CPU-copy tensor."""
    if value is None:
        return None
    if isinstance(value, LazyTensor):
        return value.resolve()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    raise TypeError(type(value))


@dataclass
class RolloutDebugTensors:
    rollout_variance_noises: RolloutTensorRef | None = None
    rollout_prev_sample_means: RolloutTensorRef | None = None
    rollout_noise_std_devs: RolloutTensorRef | None = None
    rollout_model_outputs: RolloutTensorRef | None = None


@dataclass
class CondKwargs:
    txt_seq_lens: list[int] | None = None
    freqs_cis: list[RolloutTensorRef] | None = None
    img_shapes: list[list[tuple[int, int, int]]] | None = None
    encoder_hidden_states: list[RolloutTensorRef] | None = None


@dataclass
class DenoisingStatic:
    image_kwargs: Any | None = None
    pos_cond_kwargs: CondKwargs | None = None
    neg_cond_kwargs: CondKwargs | None = None
    guidance: Any | None = None


@dataclass
class DenoisingTrajectory:
    latent_model_inputs: RolloutTensorRef | None = None
    timesteps: RolloutTensorRef | None = None


@dataclass
class DenoisingEnv:
    """Matches ``denoising_env`` in ``POST /rollout/images`` (``static`` + ``trajectory``)."""

    static: DenoisingStatic | None = None
    trajectory: DenoisingTrajectory | None = None


@dataclass
class Sample:
    """The sample generated.

    Diffusion image rollout: fill from sglang-diffusion ``POST /rollout/images`` via
    :meth:`from_rollout_image_response` or :meth:`apply_rollout_image_response` (see
    :mod:`miles.utils.diffusion_rollout_response`).
    """

    group_index: int | None = None
    index: int | None = None
    # correlation id from rollout engine (e.g. UUID string)
    request_id: str | None = None
    # prompt
    prompt: str = ""
    # reproducibility
    seed: int | None = None
    # Lazy: :class:`SafetensorsBase64LazyTensor` (safetensors+b64 ``str``); eager: :class:`torch.Tensor`
    generated_output: RolloutTensorRef | None = None
    rollout_log_probs: RolloutTensorRef | None = None
    rollout_debug_tensors: RolloutDebugTensors | None = None
    denoising_env: DenoisingEnv | None = None

    inference_time_s: float | None = None
    peak_memory_mb: float | None = None

    reward: dict[str, Any] | None = None
    weight_versions: list[str] = field(default_factory=list)

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        # Indicates a recoverable or non-critical failure during generation (e.g., tool call failure,
        # external API error, parsing error). Unlike ABORTED, FAILED samples may still contain partial
        # valid output and can be retried or handled gracefully.
        FAILED = "failed"

    status: Status = Status.PENDING

    metadata: dict = field(default_factory=dict)
    # metadata used during training, e.g., what loss to use for this sample.
    train_metadata: dict | None = None

    non_generation_time: float = 0.0  # time spent in non-generation steps

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        return value

    @staticmethod
    def from_dict(data: dict):
        data = dict(data)
        data["status"] = Sample.Status(data["status"])
        field_names = set(Sample.__dataclass_fields__.keys())
        init_data = {k: v for k, v in data.items() if k in field_names}
        sample = Sample(**init_data)

        for key, value in data.items():
            if key not in field_names:
                setattr(sample, key, value)

        return sample

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]
