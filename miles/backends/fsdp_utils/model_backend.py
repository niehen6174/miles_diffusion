"""Model backend: owns model-side behavior for the FSDP trainer.

Selected via ``--model-backend-path`` (miles custom-function style); the
family config declares the default. Three concerns, all properties of the
concrete modeling rather than of the training loop:

  - ``load_models_and_scheduler``: checkpoint -> ``({component: model}, scheduler)``
  - ``enable_gradient_checkpointing``: how this model turns on grad ckpt
  - ``fsdp_no_split_modules``: which block classes FSDP wraps

Defaults implement the diffusers protocol (see ``models/__init__.py``); a
native model overrides methods here instead of retrofitting its instances.
"""

from __future__ import annotations

import abc
from typing import Any

import torch
from diffusers import DiffusionPipeline


class ModelBackend(abc.ABC):
    def __init__(self, train_pipeline_config):
        self.config = train_pipeline_config

    @abc.abstractmethod
    def load_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        """Return ``({component: model}, scheduler)`` on CPU."""

    def enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        """Turn on grad checkpointing; default = the diffusers protocol method."""
        model.enable_gradient_checkpointing()

    def fsdp_no_split_modules(self, model: torch.nn.Module) -> list[str]:
        """Block class names FSDP wraps; default = the model's own declaration."""
        return model._no_split_modules

    def set_attention_backend(self, model: torch.nn.Module, backend: str) -> None:
        """Select the DiT attention backend; default = the diffusers protocol method."""
        model.set_attention_backend(backend)


class DiffusersModelBackend(ModelBackend):
    """Load trainable components from a diffusers pipeline checkpoint."""

    def load_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        pipeline = DiffusionPipeline.from_pretrained(
            args.hf_checkpoint,
            torch_dtype=master_dtype,
            trust_remote_code=True,
            text_encoder=None,
            vae=None,
            tokenizer=None,
        )
        raw_models: dict[str, torch.nn.Module] = {}
        for component in args.update_weight_target_modules:
            sub_model = getattr(pipeline, component, None)
            if sub_model is None:
                raise ValueError(
                    f"--update-weight-target-module: pipeline {args.hf_checkpoint} " f"has no component '{component}'"
                )
            raw_models[component] = sub_model
        scheduler = pipeline.scheduler
        del pipeline
        return raw_models, scheduler
