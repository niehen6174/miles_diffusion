"""Real native stage -> PDD loss -> FSDP weight iterator -> native stage smoke.

Uses random tiny weights and synthetic embeddings, no pretrained download. The
worker subprocess isolates SGLang globals/patches from other tests in this suite.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-5-gpu-h200", labels=["fsdp"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/SGLang")
def test_online_stage_train_sync(tmp_path):
    subprocess.run(
        [sys.executable, __file__, str(tmp_path)], check=True, timeout=180, env={**os.environ, "PYTHONPATH": "."}
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["updates"] == 16 and report["teacher_frozen"]
    assert report["weights_exact"] and report["next_rollout_changed"]
    # Same weights, every block; mismatched BF16 timestep rounding gave 1.5-3% on blocks 2 and 4.
    assert max(max(r["pre_update_relative_error"]) for r in report["rollouts"]) < 5e-3


def worker(output):
    import dataclasses
    import socket

    from diffusers import FlowMatchEulerDiscreteScheduler as HFScheduler
    from diffusers import QwenImage21Transformer2DModel as HFModel
    from sglang.multimodal_gen.configs.models.dits.qwenimage21 import QwenImage21ArchConfig, QwenImage21DitConfig
    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import QwenImage21PipelineConfig
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import ComponentResidencyManager
    from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import QwenImage21Transformer2DModel as Native
    from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import build_layout
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
        QwenImage21DenoisingStage,
        prepare_qwen21_mu,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import TimestepPreparationStage
    from sglang.multimodal_gen.runtime.post_training.weights_updater import WeightsUpdater
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from miles.backends.fsdp_utils.configs.qwen_image21 import QwenImage21TrainPipelineConfig
    from miles.backends.fsdp_utils.hf_weight_iterator import FSDPHfWeightIterator
    from miles.backends.fsdp_utils.loss_hub.pdd import pdd_loss_formula, prepare_pdd_batch
    from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext
    from miles.backends.fsdp_utils.metrics import SCHEMA
    from miles.backends.sglang_diffusion_utils.monkey_patches import patch_pdd_qwen_image21
    from miles.distillation.integration import expand_trajectories, install_student, make_grid
    from miles.distillation.pdd import block_update
    from miles.rollout.pdd_online import convert_samples_to_train_data
    from miles.utils.diffusion_rollout_response import _parse_denoising_env, _parse_dit_trajectory
    from miles.utils.types import Sample

    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
    cfg = dict(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        # The real head width: native Q/K norm + RoPE fast paths only fire at 128.
        num_attention_heads=1,
        attention_head_dim=128,
        context_in_dim=16,
        mlp_ratio=2,
        axes_dims_rope=(16, 56, 56),
    )
    model_path = output / "model"
    HFScheduler(use_dynamic_shifting=True).save_pretrained(model_path / "scheduler")
    config = QwenImage21DitConfig(arch_config=QwenImage21ArchConfig(**cfg))
    server = ServerArgs(
        model_path=str(model_path),
        num_gpus=1,
        attention_backend="torch_sdpa",
        pipeline_config=QwenImage21PipelineConfig(dit_config=config, dit_precision="fp32"),
    )
    set_global_server_args(server)
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    patch_pdd_qwen_image21.apply()
    patch_pdd_qwen_image21.apply()  # idempotent in scheduler workers
    torch.manual_seed(72)
    train_model = HFModel(**cfg).cuda()
    teacher = HFModel(**cfg).cuda().bfloat16().eval().requires_grad_(False)
    args = SimpleNamespace(
        pdd_num_intervals=32,
        pdd_block_size=4,
        pdd_init_checkpoint=None,
        pdd_online=True,
        hf_checkpoint=str(model_path),
        diffusion_height=1024,
        diffusion_width=1024,
        seed=42,
        update_weight_buffer_size=4096,
    )
    install_student(train_model, args, materialize_weights=True)
    train_config = QwenImage21TrainPipelineConfig()
    train_config.configure(args)
    train_config.postprocess_model_after_materialize(train_model)
    train_config.postprocess_model_after_materialize(teacher)
    mesh = init_device_mesh("cuda", (1,))
    fully_shard(
        train_model,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False
        ),
    )
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-3, weight_decay=0, foreach=False)
    native = Native(config, {}).cuda().eval()
    native.post_load_weights()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path / "scheduler")
    time_stage = TimestepPreparationStage(scheduler, [prepare_qwen21_mu])
    stage = QwenImage21DenoisingStage(transformer=native, scheduler=scheduler)
    pipeline = SimpleNamespace(
        modules={"transformer": native}, component_residency_strategies={}, _stage_name_mapping={"denoising": stage}
    )
    manager = ComponentResidencyManager(pipeline, server)
    manager.refresh_pipeline(pipeline)
    stage.set_component_residency_manager(manager)
    updater = WeightsUpdater(pipeline)

    def sync():
        iterator = FSDPHfWeightIterator(args, train_model, diffusion_model_family="qwen_image21")
        seen = set()
        for bucket in iterator.iter_hf_weights():
            ok, message = updater.update_weights_from_tensor({"transformer": bucket}, target_modules=["transformer"])
            assert ok, message
            for name, value in bucket:
                torch.testing.assert_close(dict(native.named_parameters())[name], value, rtol=0, atol=0)
                seen.add(name)
        assert seen == set(dict(native.named_parameters()))

    embeddings = torch.randn(1, 3, 16, device="cuda")
    noise = torch.randn(1, 4096, 4, device="cuda").bfloat16().float()
    layout = build_layout([False] * 3, [(1, 64, 64)], config.axes_dims_rope, "cuda")

    @torch.no_grad()
    def rollout():
        batch = Req(
            prompt="tiny PDD",
            height=1024,
            width=1024,
            num_inference_steps=8,
            guidance_scale=1.0,
            true_cfg_scale=1.0,
            do_classifier_free_guidance=False,
            prompt_embeds=[embeddings],
            prompt_embeds_mask=[torch.ones(1, 3, device="cuda", dtype=torch.bool)],
            latents=noise.clone(),
            raw_latent_shape=list(noise.shape),
            rollout=True,
            rollout_sde_type="ode",
            rollout_noise_level=0.0,
            rollout_log_prob_no_const=True,
            rollout_return_dit_trajectory=True,
            rollout_return_denoising_env=True,
            extra={"qwen21_positive": dict(layouts=[layout], prefix_caches=None), "qwen21_mu": 1.15},
        )
        batch = time_stage.forward(batch, server)
        manager.begin_request([stage], batch, server)
        result = stage.forward(batch, server)
        data = result.rollout_trajectory_data
        trace = dataclasses.asdict(data.dit_trajectory)
        assert trace["latents"].shape == (1, 9, 4096, 4)
        trace["latents"] = trace["latents"][0]
        sample = Sample(index=0, prompt="tiny PDD", seed=31)
        sample.dit_trajectory = _parse_dit_trajectory(trace, deserialize_func=lambda x: x)
        sample.denoising_env = _parse_denoising_env(
            dataclasses.asdict(data.denoising_env), deserialize_func=lambda x: x
        )
        return sample

    class Metrics:
        def __init__(self):
            self.values = {}

        def emit_mean(self, name, *, total, count):
            assert name in SCHEMA, f"{name} is not declared in the FSDP metric schema"
            self.values[name] = float(total / count)

    def relative_drift(new, old):
        # Per block: state change after training, relative to that block's own update size.
        new, old = new.float(), old.float()
        return [
            float((new[i] - old[i]).abs().mean() / (old[i + 1] - old[i]).abs().mean().clamp_min(1e-6))
            for i in range(8)
        ]

    sync()
    first = rollout()
    sample = first
    rollouts = []
    for rollout_id in range(2):
        pair = convert_samples_to_train_data(args, [sample])["train_data"][0]
        ctx = DiffusionLossContext(
            models={"transformer": train_model},
            teacher_models={"transformer": teacher},
            train_pipeline_config=train_config,
            sde_backend=None,
            args=args,
            forward_dtype=torch.bfloat16,
            device=torch.device("cuda"),
            rollout_id=rollout_id,
            algorithm_state={"grid": make_grid(str(model_path), 32, 4096, "cuda")},
        )
        # Pure engine/trainer parity: every block at the rollout weights, before any update.
        alignment = []
        with torch.no_grad():
            for block in expand_trajectories([pair], intervals=32, block_size=4):
                prepared = prepare_pdd_batch(ctx, [block])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = train_config.compute_noise_pred(
                        model=train_model,
                        latents_input=prepared.latents,
                        timesteps_input=prepared.timesteps_for_model,
                        pos_cond=prepared.pos_cond,
                        neg_cond=None,
                        joint_cond=None,
                        use_cfg=False,
                        cfg_batching=False,
                        guidance_scale=1.0,
                        true_cfg_scale=1.0,
                    )["visual"]
                x = prepared.latents["visual"]
                heads = raw.reshape(*x.shape[:-1], 32, x.shape[-1]).permute(2, 0, 1, 3)
                start = block["pdd_start"]
                following = block_update(x, heads, prepared.extras["grid"], start, start + 4)
                expected = pair["pdd_trajectory"][start // 4 + 1].cuda().unsqueeze(0)
                alignment.append(
                    float((following - expected).abs().mean() / (expected - x).abs().mean().clamp_min(1e-6))
                )
        blocks = []
        for block in expand_trajectories([pair], intervals=32, block_size=4):
            prepared = prepare_pdd_batch(ctx, [block])
            metrics = Metrics()
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = train_config.compute_noise_pred(
                    model=train_model,
                    latents_input=prepared.latents,
                    timesteps_input=prepared.timesteps_for_model,
                    pos_cond=prepared.pos_cond,
                    neg_cond=None,
                    joint_cond=None,
                    use_cfg=False,
                    cfg_batching=False,
                    guidance_scale=1.0,
                    true_cfg_scale=1.0,
                )
                loss = pdd_loss_formula(
                    ctx, [block], prepared, new_pred=pred, old_pred=None, ref_pred=None, metrics=metrics
                )
            loss.backward()
            assert all(p.grad is None for p in teacher.parameters())
            optimizer.step()
            blocks.append(dict(start=block["pdd_start"], **metrics.values))
        assert not ctx.algorithm_state["lanes"]
        sync()
        previous, sample = sample, rollout()
        rollouts.append(
            dict(
                pre_update_relative_error=alignment,
                blocks=blocks,
                state_drift_after_8_updates=relative_drift(
                    sample.dit_trajectory.latents, previous.dit_trajectory.latents
                ),
            )
        )
    assert not torch.equal(first.dit_trajectory.latents, sample.dit_trajectory.latents)
    report = dict(
        updates=sum(len(r["blocks"]) for r in rollouts),
        teacher_frozen=True,
        weights_exact=True,
        next_rollout_changed=True,
        trajectory_shape=list(sample.dit_trajectory.latents.shape),
        rollouts=rollouts,
        scope="random tiny DiT; real SGLang stages and FSDP; no HTTP/Ray driver or pretrained quality",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    worker(Path(sys.argv[1]))
