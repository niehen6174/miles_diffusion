"""Flow-GRPO: reverse-SDE log-prob + PPO-clip."""

from __future__ import annotations

from typing import Any

import torch

from miles.algorithms.base import CollectionSpec, TrainLabels, TrainLossContext
from miles.algorithms.labels import grpo_group_advantages
from miles.algorithms.train_forward_utils import (
    append_rollout_train_abs_diff_stats,
    compute_noise_pred,
    prepare_cfg_conds,
    resolve_cfg_flags,
    select_model_for_timesteps,
)
from miles.utils.train_data_utils import RolloutTrainDataConverter, scheduler_meta_from_rollout, stack_train_pair_rollout_debug
from miles.utils.types import Sample


class FlowGRPOAlgorithm:
    name = "flow_grpo"

    def validate_args(self, args) -> None:
        kl_beta = float(getattr(args, "diffusion_kl_beta", 0.0) or 0.0)
        if kl_beta > 0 and not args.use_lora:
            raise ValueError(
                "--diffusion-kl-beta currently requires --use-lora so the base model can be used as reference."
            )

    def collection_spec(self) -> CollectionSpec:
        return CollectionSpec(
            mode="online",
            needs_reward=True,
            needs_trajectory=True,
            needs_logprob=True,
            sampler="sde",
            return_denoising_env=True,
            sync_weights_to_rollout=True,
        )

    def postprocess_rewards(self, args, samples: list[Sample]) -> TrainLabels:
        return grpo_group_advantages(args, samples)

    def build_train_data(self, args, samples: list[Sample], labels: TrainLabels) -> dict[str, Any]:
        rewards = labels.advantages if labels.advantages is not None else labels.raw_rewards
        return RolloutTrainDataConverter().convert_samples(samples, rewards, labels.raw_rewards)

    def validate_train_batch(self, batch: list[dict]) -> list[str]:
        errors: list[str] = []
        required = ("latent", "next_latent", "timestep", "next_timestep", "log_prob_old", "advantage", "denoising_env")
        for i, pair in enumerate(batch):
            for key in required:
                if key not in pair:
                    errors.append(f"batch[{i}] missing {key}")
        return errors

    def prepare_rollout_data(self, rollout_data: dict, ctx: TrainLossContext) -> None:
        if ctx.scheduler is None:
            return
        num_train_timesteps = ctx.scheduler.config.num_train_timesteps
        scheduler_timesteps, scheduler_sigmas = scheduler_meta_from_rollout(
            rollout_data,
            device=ctx.device,
            num_train_timesteps=num_train_timesteps,
        )
        ctx.scheduler.timesteps = scheduler_timesteps
        ctx.scheduler.sigmas = scheduler_sigmas
        ctx.scheduler._step_index = None
        ctx.scheduler._begin_index = None

    def compute_loss(
        self,
        ctx: TrainLossContext,
        batch: list[dict],
        *,
        log_stats: dict[str, list[torch.Tensor]],
        pad_to_len: int | None = None,
    ) -> torch.Tensor:
        """One DiT forward + PPO loss over ``len(batch)`` train pairs. Returns sum of per-pair losses."""
        if ctx.sde_backend is None:
            raise RuntimeError("Flow-GRPO requires an SDE step backend")

        args = ctx.args
        forward_dtype = ctx.forward_dtype
        train_pipeline_config = ctx.train_pipeline_config
        device = ctx.device
        bsz = len(batch)

        use_cfg, guidance_scale, true_cfg_scale = resolve_cfg_flags(args)
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

        component, model, guidance_scale = select_model_for_timesteps(
            ctx,
            timesteps_microbatch,
            guidance_scale=guidance_scale,
            num_train_timesteps=num_train_timesteps,
        )

        if train_pipeline_config.needs_timestep_scaling:
            timesteps_for_model = timesteps_microbatch / float(num_train_timesteps)
        else:
            timesteps_for_model = timesteps_microbatch

        pos_cond, neg_cond, joint_cond, cfg_batching = prepare_cfg_conds(
            ctx, batch, use_cfg=use_cfg, pad_to_len=pad_to_len
        )

        latents_input = latents_microbatch.to(forward_dtype)
        timesteps_input = timesteps_for_model.to(forward_dtype)

        def _pred(disable_adapter: bool = False) -> torch.Tensor:
            return compute_noise_pred(
                ctx,
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
                disable_adapter=disable_adapter,
            )

        noise_pred_microbatch = _pred()

        _, log_prob_new_microbatch, prev_sample_mean_new, std_dev_t_new = ctx.sde_backend.sde_step_logprob(
            noise_pred_microbatch.float(),
            timesteps_microbatch,
            next_timesteps_microbatch,
            latents_microbatch.float(),
            prev_sample=next_latents_microbatch.float(),
            noise_level=noise_level,
        )

        log_prob_new = log_prob_new_microbatch
        log_prob_old = log_prob_old_microbatch
        ratio = torch.exp(log_prob_new - log_prob_old)
        unclipped = -advantage * ratio
        clipped = -advantage * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        per_pair_loss = torch.maximum(unclipped, clipped)
        loss_sum = per_pair_loss.sum()

        kl_loss = loss_sum.new_zeros(())
        if kl_beta > 0:
            with torch.no_grad():
                ref_noise_pred_microbatch = _pred(disable_adapter=True)
                _, _, prev_sample_mean_ref, _ = ctx.sde_backend.sde_step_logprob(
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
            if len(ctx.models) > 1:
                log_stats[f"log_prob_mean_abs_diff_{component}"].append(log_prob_mean_abs_diff)

            rollout_model_output = stack_train_pair_rollout_debug(batch, "rollout_step_model_output")
            if rollout_model_output is not None:
                mean_abs_diff = append_rollout_train_abs_diff_stats(
                    log_stats,
                    "model_output",
                    noise_pred_microbatch.float(),
                    rollout_model_output.to(device=device, dtype=torch.float32),
                )
                if len(ctx.models) > 1:
                    log_stats[f"model_output_mean_abs_diff_{component}"].append(mean_abs_diff)

        return loss_sum
