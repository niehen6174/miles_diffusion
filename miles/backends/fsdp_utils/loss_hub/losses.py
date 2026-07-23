"""Diffusion loss building blocks (miles ``loss_hub`` style).

Builtin: ``flow_grpo_ppo`` (reverse-SDE PPO-clip). Custom objectives use
``--loss-type custom_loss --custom-loss-function-path module.fn``.
"""

from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
from typing import Protocol

import torch

from miles.backends.fsdp_utils.loss_hub.context import DiffusionLossContext
from miles.utils.misc import load_function
from miles.utils.train_data_utils import stack_train_pair_rollout_debug


class DiffusionLossFunction(Protocol):
    """Common signature for diffusion loss functions."""

    def __call__(
        self,
        ctx: DiffusionLossContext,
        batch: list[dict],
        *,
        log_stats: dict[str, list[torch.Tensor]],
        pad_to_len: int | None = None,
        write_old_log_prob: bool = False,
        old_log_prob_from_new: bool = False,
    ) -> torch.Tensor | None:
        """Return sum of per-example losses (actor divides by local pair count)."""
        ...


def get_diffusion_loss_function(args: Namespace) -> DiffusionLossFunction:
    """Dispatch the train objective from CLI (miles ``get_loss_function`` style)."""
    loss_type = args.loss_type
    if loss_type == "custom_loss" or args.custom_loss_function_path is not None:
        path = args.custom_loss_function_path
        if not path:
            raise ValueError("--loss-type custom_loss requires --custom-loss-function-path")
        fn = load_function(path)
        if fn is None:
            raise ValueError(f"Failed to load custom loss from {path!r}")
        return fn
    if loss_type in ("flow_grpo_ppo", "policy_loss"):
        # policy_loss kept as alias so legacy scripts that leave the LLM default
        # unchanged still hit Flow-GRPO on the diffusion actor path.
        return flow_grpo_ppo_loss
    raise ValueError(
        f"Unsupported diffusion --loss-type {loss_type!r}. "
        "Use flow_grpo_ppo (default), policy_loss (alias), or custom_loss."
    )


def flow_grpo_ppo_loss(
    ctx: DiffusionLossContext,
    batch: list[dict],
    *,
    log_stats: dict[str, list[torch.Tensor]],
    pad_to_len: int | None = None,
    write_old_log_prob: bool = False,
    old_log_prob_from_new: bool = False,
) -> torch.Tensor | None:
    """One DiT forward + PPO-clip over ``len(batch)`` Flow-GRPO train pairs."""
    args = ctx.args
    device = ctx.device
    forward_dtype = ctx.forward_dtype
    train_pipeline_config = ctx.train_pipeline_config
    models = ctx.models
    sde_backend = ctx.sde_backend
    bsz = len(batch)

    guidance_scale = args.diffusion_guidance_scale
    true_cfg_scale = args.diffusion_true_cfg_scale
    cfg_scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
    use_cfg = cfg_scale > 0
    clip_range = args.diffusion_clip_range
    noise_level = args.diffusion_noise_level
    num_train_timesteps = ctx.scheduler.config.num_train_timesteps
    kl_beta = float(args.diffusion_kl_beta)

    def _stack(key):
        return torch.stack([pair[key] for pair in batch]).to(device=device, dtype=torch.float32)

    latents_microbatch = _stack("latent")
    next_latents_microbatch = _stack("next_latent")
    timesteps_microbatch = _stack("timestep")
    next_timesteps_microbatch = _stack("next_timestep")
    log_prob_old_microbatch = _stack("log_prob_old")

    advantage = torch.tensor(
        [float(pair["advantage"]) for pair in batch],
        device=device,
        dtype=torch.float32,
    )
    advantage = torch.clamp(advantage, -args.diffusion_adv_clip_max, args.diffusion_adv_clip_max)

    if len(models) == 1:
        component, model = next(iter(models.items()))
    else:
        components = {
            train_pipeline_config.component_for_timestep(t, num_train_timesteps)
            for t in timesteps_microbatch.tolist()
        }
        if len(components) > 1:
            raise ValueError(
                f"Micro-batch mixes denoising phases {sorted(components)}; set "
                "--micro-batch-size 1 so each forward is phase-pure (one DiT, one CFG scale)."
            )
        component = components.pop()
        model = models[component]
        guidance_scale = train_pipeline_config.select_guidance_scale(
            float(timesteps_microbatch[0]),
            num_train_timesteps,
            guidance_scale,
            args.diffusion_guidance_scale_2,
        )

    if train_pipeline_config.needs_timestep_scaling:
        timesteps_for_model = timesteps_microbatch / float(num_train_timesteps)
    else:
        timesteps_for_model = timesteps_microbatch

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

    cfg_batching = use_cfg and bool(args.fsdp_cfg_batching)
    joint_cond = None
    pos_cond_microbatch = None
    neg_cond_microbatch = None
    if cfg_batching:
        joint_cond = _cast_cond_to_dtype(
            train_pipeline_config.collate_cond_for_sample_batch(pos_list + neg_list, device, pad_to_len=pad_to_len),
            forward_dtype,
        )
    else:
        pos_cond_microbatch = _cast_cond_to_dtype(
            train_pipeline_config.collate_cond_for_sample_batch(pos_list, device, pad_to_len=pad_to_len),
            forward_dtype,
        )
        if use_cfg and neg_list is not None:
            neg_cond_microbatch = _cast_cond_to_dtype(
                train_pipeline_config.collate_cond_for_sample_batch(neg_list, device, pad_to_len=pad_to_len),
                forward_dtype,
            )

    latents_input = latents_microbatch.to(forward_dtype)
    timesteps_input = timesteps_for_model.to(forward_dtype)

    def _compute_noise_pred(disable_adapter: bool = False) -> torch.Tensor:
        adapter_ctx = model.disable_adapter() if disable_adapter else nullcontext()
        with adapter_ctx:
            return train_pipeline_config.compute_noise_pred(
                model=model,
                latents_input=latents_input,
                timesteps_input=timesteps_input,
                pos_cond=pos_cond_microbatch,
                neg_cond=neg_cond_microbatch,
                joint_cond=joint_cond,
                use_cfg=use_cfg,
                cfg_batching=cfg_batching,
                guidance_scale=guidance_scale,
                true_cfg_scale=true_cfg_scale,
            )

    noise_pred_microbatch = _compute_noise_pred()

    _, log_prob_new_microbatch, prev_sample_mean_new, std_dev_t_new = sde_backend.sde_step_logprob(
        noise_pred_microbatch.float(),
        timesteps_microbatch,
        next_timesteps_microbatch,
        latents_microbatch.float(),
        prev_sample=next_latents_microbatch.float(),
        noise_level=noise_level,
    )

    if write_old_log_prob:
        for pair, log_prob in zip(batch, log_prob_new_microbatch, strict=True):
            pair["log_prob_old"] = log_prob.cpu()
        return None

    log_prob_new = log_prob_new_microbatch
    log_prob_old = log_prob_new.detach() if old_log_prob_from_new else log_prob_old_microbatch
    ratio = torch.exp(log_prob_new - log_prob_old)
    unclipped = -advantage * ratio
    clipped = -advantage * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    per_pair_loss = torch.maximum(unclipped, clipped)
    loss_sum = per_pair_loss.sum()

    kl_loss = loss_sum.new_zeros(())
    if kl_beta > 0:
        with torch.no_grad():
            ref_noise_pred_microbatch = _compute_noise_pred(disable_adapter=True)
            _, _, prev_sample_mean_ref, _ = sde_backend.sde_step_logprob(
                ref_noise_pred_microbatch.float(),
                timesteps_microbatch,
                next_timesteps_microbatch,
                latents_microbatch.float(),
                prev_sample=next_latents_microbatch.float(),
                noise_level=noise_level,
            )
        kl_per_pair = ((prev_sample_mean_new - prev_sample_mean_ref) ** 2).mean(
            dim=tuple(range(1, prev_sample_mean_new.ndim)),
            keepdim=True,
        ) / (2 * std_dev_t_new**2)
        loss_sum = loss_sum + kl_beta * kl_per_pair.sum()
        kl_loss = kl_per_pair.mean()

    with torch.no_grad():
        log_stats["loss"].append((per_pair_loss.mean() + kl_beta * kl_loss).detach())
        log_stats["policy_loss"].append(per_pair_loss.mean().detach())
        log_stats["kl_loss"].append(kl_loss.detach())
        log_stats["loss_abs_mean"].append(per_pair_loss.abs().mean().detach())
        log_stats["adv_abs_mean"].append(advantage.abs().mean().detach())
        log_stats["ratio_abs_minus_1"].append((ratio - 1.0).abs().mean().detach())
        log_stats["approx_kl"].append(0.5 * torch.mean((log_prob_new - log_prob_old) ** 2).detach())
        log_stats["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).detach())
        log_stats["log_prob_new_idx_0"].append(log_prob_new[0].detach())
        log_stats["log_prob_old_idx_0"].append(log_prob_old[0].detach())
        log_prob_mean_abs_diff = torch.mean(torch.abs(log_prob_new - log_prob_old)).detach()
        log_stats["log_prob_mean_abs_diff"].append(log_prob_mean_abs_diff)
        if len(models) > 1:
            log_stats[f"log_prob_mean_abs_diff_{component}"].append(log_prob_mean_abs_diff)

        rollout_model_output = stack_train_pair_rollout_debug(batch, "rollout_step_model_output")
        if rollout_model_output is not None:
            mean_abs_diff = _append_rollout_train_abs_diff_stats(
                log_stats,
                "model_output",
                noise_pred_microbatch.float(),
                rollout_model_output.to(device=device, dtype=torch.float32),
            )
            if len(models) > 1:
                log_stats[f"model_output_mean_abs_diff_{component}"].append(mean_abs_diff)

    return loss_sum


def _append_rollout_train_abs_diff_stats(
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


def _cast_cond_to_dtype(cond: dict, dtype: torch.dtype) -> dict:
    """Cast floating-point tensors to the model's compute dtype; leave bool
    masks / int / list / scalar values untouched.
    """
    out = {}
    for k, v in cond.items():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            out[k] = v.to(dtype=dtype)
        else:
            out[k] = v
    return out
