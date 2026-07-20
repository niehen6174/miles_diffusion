"""Shared train-side DiT forward helpers used by algorithm plugins."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from miles.algorithms.base import TrainLossContext


def cast_cond_to_dtype(cond: dict, dtype: torch.dtype) -> dict:
    """Cast floating-point tensors to the model's compute dtype; leave masks alone."""
    out: dict = {}
    for k, v in cond.items():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            out[k] = v.to(dtype)
        else:
            out[k] = v
    return out


def append_rollout_train_abs_diff_stats(
    log_stats: dict[str, list],
    prefix: str,
    train: torch.Tensor,
    rollout: torch.Tensor,
) -> torch.Tensor:
    bsz = train.shape[0]
    diff = (train.reshape(bsz, -1).float() - rollout.reshape(bsz, -1).float()).abs()
    ref_max = rollout.reshape(bsz, -1).float().abs().max() + 1e-30
    mean_abs_diff = diff.mean().detach()
    log_stats[f"{prefix}_max_abs_diff"].append(diff.max().detach())
    log_stats[f"{prefix}_mean_abs_diff"].append(mean_abs_diff)
    log_stats[f"{prefix}_rel_max"].append((diff.max() / ref_max).detach())
    return mean_abs_diff


def select_model_for_timesteps(
    ctx: TrainLossContext,
    timesteps: torch.Tensor,
    *,
    guidance_scale: float,
    num_train_timesteps: int,
) -> tuple[str, torch.nn.Module, float]:
    """Pick the DiT component (Wan dual-expert aware) and maybe retarget guidance."""
    train_pipeline_config = ctx.train_pipeline_config
    if len(ctx.models) == 1:
        component, model = next(iter(ctx.models.items()))
        return component, model, guidance_scale

    components = {train_pipeline_config.component_for_timestep(t, num_train_timesteps) for t in timesteps.tolist()}
    if len(components) > 1:
        raise ValueError(
            f"Micro-batch mixes denoising phases {sorted(components)}; set "
            "--micro-batch-size 1 so each forward is phase-pure (one DiT, one CFG scale)."
        )
    component = components.pop()
    model = ctx.models[component]
    guidance_scale = train_pipeline_config.select_guidance_scale(
        float(timesteps[0]),
        num_train_timesteps,
        guidance_scale,
        ctx.args.diffusion_guidance_scale_2,
    )
    return component, model, guidance_scale


def prepare_cfg_conds(
    ctx: TrainLossContext,
    batch: list[dict],
    *,
    use_cfg: bool,
    pad_to_len: int | None,
) -> tuple[dict | None, dict | None, dict | None, bool]:
    """Build pos / neg / joint cond dicts for a micro-batch."""
    train_pipeline_config = ctx.train_pipeline_config
    device = ctx.device
    forward_dtype = ctx.forward_dtype
    bsz = len(batch)

    pos_list = [
        train_pipeline_config.prepare_cond_kwargs(batch[i]["denoising_env"].pos_cond_kwargs, device)
        for i in range(bsz)
    ]
    neg_list = (
        [
            train_pipeline_config.prepare_cond_kwargs(batch[i]["denoising_env"].neg_cond_kwargs, device)
            for i in range(bsz)
        ]
        if use_cfg
        else None
    )

    cfg_batching = use_cfg and bool(ctx.args.fsdp_cfg_batching)
    joint_cond = None
    pos_cond = None
    neg_cond = None
    if cfg_batching:
        joint_cond = cast_cond_to_dtype(
            train_pipeline_config.collate_cond_for_sample_batch(pos_list + neg_list, device, pad_to_len=pad_to_len),
            forward_dtype,
        )
    else:
        pos_cond = cast_cond_to_dtype(
            train_pipeline_config.collate_cond_for_sample_batch(pos_list, device, pad_to_len=pad_to_len),
            forward_dtype,
        )
        if use_cfg and neg_list is not None:
            neg_cond = cast_cond_to_dtype(
                train_pipeline_config.collate_cond_for_sample_batch(neg_list, device, pad_to_len=pad_to_len),
                forward_dtype,
            )
    return pos_cond, neg_cond, joint_cond, cfg_batching


def compute_noise_pred(
    ctx: TrainLossContext,
    *,
    model: torch.nn.Module,
    latents_input: torch.Tensor,
    timesteps_input: torch.Tensor,
    pos_cond: dict | None,
    neg_cond: dict | None,
    joint_cond: dict | None,
    use_cfg: bool,
    cfg_batching: bool,
    guidance_scale: float,
    true_cfg_scale: float | None,
    disable_adapter: bool = False,
) -> torch.Tensor:
    adapter_ctx = model.disable_adapter() if disable_adapter else nullcontext()
    with adapter_ctx:
        return ctx.train_pipeline_config.compute_noise_pred(
            model=model,
            latents_input=latents_input,
            timesteps_input=timesteps_input,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            joint_cond=joint_cond,
            use_cfg=use_cfg,
            cfg_batching=cfg_batching,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
        )


def resolve_cfg_flags(args: Any) -> tuple[bool, float, float | None]:
    guidance_scale = args.diffusion_guidance_scale
    true_cfg_scale = args.diffusion_true_cfg_scale
    cfg_scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
    use_cfg = cfg_scale > 0
    return use_cfg, guidance_scale, true_cfg_scale


def model_has_disable_adapter(models: dict[str, torch.nn.Module]) -> bool:
    return all(hasattr(m, "disable_adapter") for m in models.values())
