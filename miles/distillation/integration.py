"""PDD protocol shared by the Miles actor, loader and cached-prompt producer.

Version 1 checkpoints only at full-trajectory (rollout) boundaries. Eight optimizer
updates carry detached pre-update states; no trajectory is discarded on resume.
"""

from pathlib import Path

import torch

from .pdd import validate_grid
from .prompt_cache import file_sha256
from .qwen_image21_pdd import TokenParallelProjection


def make_grid(model_path, intervals, tokens, device="cpu"):
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler", local_files_only=True
    )
    cfg = scheduler.config
    mu = calculate_shift(tokens, cfg.base_image_seq_len, cfg.max_image_seq_len, cfg.base_shift, cfg.max_shift)
    scheduler.set_timesteps(intervals, device=device, sigmas=np.linspace(1.0, 1.0 / intervals, intervals), mu=mu)
    grid = scheduler.sigmas.float()
    validate_grid(grid)
    return grid


def install_student(model, args, *, materialize_weights):
    if not isinstance(model.proj_out, torch.nn.Linear) or model.config.patch_size != 1:
        raise ValueError("PDD needs an unmodified Qwen-Image 2.1 transformer")
    if model.proj_out.out_features != model.config.in_channels:
        raise ValueError("PDD requires equal input and output latent channels")
    model.proj_out = TokenParallelProjection(model.proj_out, args.pdd_num_intervals)
    if materialize_weights and args.pdd_init_checkpoint:
        saved = torch.load(args.pdd_init_checkpoint, map_location="cpu", mmap=True, weights_only=True)
        if saved["num_intervals"] != args.pdd_num_intervals or saved["resolution"] != args.diffusion_height:
            raise ValueError("PDD initial checkpoint protocol mismatch")
        grid = make_grid(args.hf_checkpoint, args.pdd_num_intervals, (args.diffusion_height // 16) ** 2)
        torch.testing.assert_close(saved["grid"].cpu(), grid, rtol=0, atol=0)
        state = saved["student"]
        if not all(k.startswith("transformer.") for k in state):
            raise ValueError("Expected experimental QwenImage21PDDStudent state dict")
        model.load_state_dict({k.removeprefix("transformer."): v for k, v in state.items()}, strict=True)


def expand_trajectories(pairs, *, intervals, block_size):
    """Called AFTER DP splitting; each local window keeps the same ordered prompts."""
    if len({p["sample_index"] for p in pairs}) != len(pairs):
        raise ValueError("PDD needs a distinct lane ID per sampled prompt occurrence")
    return [dict(pair, pdd_start=start) for start in range(0, intervals, block_size) for pair in pairs]


def resume_signature(args):
    fields = (
        "seed",
        "hf_checkpoint",
        "teacher_load",
        "pdd_num_intervals",
        "pdd_block_size",
        "diffusion_height",
        "diffusion_width",
        "diffusion_forward_dtype",
        "fsdp_master_dtype",
        "fsdp_reduce_dtype",
        "micro_batch_size",
        "rollout_batch_size",
        "train_dp_split_mode",
        "sequence_parallel_size",
        "actor_num_nodes",
        "actor_num_gpus_per_node",
        "lr",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "clip_grad",
    )
    return {
        "version": 1,
        "boundary": "complete_trajectory",
        "settings": {k: getattr(args, k) for k in fields},
        "cache_manifest_sha256": file_sha256(Path(args.pdd_train_cache) / "manifest.json"),
        "prompt_sha256": file_sha256(args.prompt_data),
        "grid": make_grid(args.hf_checkpoint, args.pdd_num_intervals, (args.diffusion_height // 16) ** 2).tolist(),
    }


def configure_args(args):
    """Narrow, validated first integration; unsupported modes fail before loading GPUs."""
    if args.diffusion_model_family != "qwen_image21" or args.diffusion_task != "t2i":
        raise ValueError("PDD v1 supports Qwen-Image 2.1 T2I only")
    if not args.train_only or args.train_backend != "fsdp":
        raise ValueError("PDD v1 requires --train-only --train-backend fsdp")
    if args.update_weight_target_modules != ["transformer"]:
        raise ValueError("PDD v1 trains only the transformer component")
    if (args.pdd_num_intervals, args.pdd_block_size) != (32, 4):
        raise ValueError("PDD v1 uses 32 intervals and block size 4 (8 student forwards)")
    if args.diffusion_height != 1024 or args.diffusion_width != 1024:
        raise ValueError("PDD v1 requires 1024 x 1024")
    if args.micro_batch_size != 1 or args.micro_batch_size_sample is not None:
        raise ValueError("PDD v1 requires --micro-batch-size 1 and no legacy sample/timestep tiling")
    if args.sequence_parallel_size != 1:
        raise ValueError("PDD sequence parallelism is not yet validated; use FSDP with SP=1")
    if args.use_lora or args.use_ema or args.ref_load or args.ref_mode != "none":
        raise ValueError("PDD v1 uses full student weights and a frozen teacher; no LoRA, EMA or reference model")
    if args.teacher_lora_adapter_paths or args.lora_adapter_paths:
        raise ValueError("PDD v1 does not accept LoRA adapters")
    if args.diffusion_recompute_old_log_prob or args.diffusion_kl_beta or args.eval_interval is not None:
        raise ValueError("PDD v1 does not use RL log probabilities, KL or rollout evaluation")
    if args.diffusion_guidance_scale != 1 or args.diffusion_true_cfg_scale not in (None, 1):
        raise ValueError("PDD v1 requires CFG=1")
    if not args.pdd_train_cache or not args.prompt_data or not args.rollout_global_dataset:
        raise ValueError("PDD requires --pdd-train-cache, --prompt-data and the global dataset")
    if args.n_samples_per_prompt != 1:
        raise ValueError("PDD requires --n-samples-per-prompt 1")
    if args.no_save_optim or args.no_load_optim or args.no_load_rng:
        raise ValueError("PDD resumption requires optimizer and RNG checkpoints")
    if args.fsdp_master_dtype != "fp32" or args.diffusion_forward_dtype != "bf16":
        raise ValueError("PDD v1 matches the experiment: FP32 master weights and BF16 forward")
    if args.teacher_load is None:
        args.teacher_load = args.hf_checkpoint
    if args.teacher_load != args.hf_checkpoint:
        raise ValueError("PDD v1 teacher must be the same base model as --hf-checkpoint")
    paths = {
        "rollout_function_path": "miles.rollout.pdd_rollout.generate_rollout",
        "custom_convert_samples_to_train_data_path": "miles.rollout.pdd_rollout.convert_samples_to_train_data",
        "custom_rollout_log_function_path": "miles.rollout.pdd_rollout.log_rollout_data",
        "custom_prepare_train_batch_path": "miles.backends.fsdp_utils.loss_hub.pdd.prepare_pdd_batch",
        "custom_loss_function_path": "miles.backends.fsdp_utils.loss_hub.pdd.pdd_loss_formula",
    }
    for key, value in paths.items():
        old = getattr(args, key)
        if old not in (None, value, "miles.rollout.sglang_rollout.generate_rollout"):
            raise ValueError(f"PDD requires {key}={value}")
        setattr(args, key, value)
    if args.num_steps_per_rollout not in (None, 8):
        raise ValueError("PDD performs 8 optimizer updates per rollout trajectory")
    args.num_steps_per_rollout = 8
    if args.global_batch_size not in (None, args.rollout_batch_size):
        raise ValueError("PDD global batch size equals the number of trajectory lanes (rollout batch size)")
    args.global_batch_size = args.rollout_batch_size
