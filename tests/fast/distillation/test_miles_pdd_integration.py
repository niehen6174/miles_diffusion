"""Parity with the independent optimizer loop, including carried pre-update states."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from miles.backends.fsdp_utils.loss_hub.pdd import pdd_loss_formula, prepare_pdd_batch
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext
from miles.distillation.integration import expand_trajectories, install_student
from miles.distillation.pdd import ParallelLinear, block_update, pdd_loss


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=2, patch_size=1)
        self.base = nn.Linear(2, 2)
        self.head = ParallelLinear(nn.Linear(2, 2), 32)

    def forward(self, x, sigma):
        return self.head(self.base(x) + sigma.reshape(-1, 1, 1))


class Config:
    def compute_noise_pred(self, *, model, latents_input, timesteps_input, **kw):
        return {"visual": model(latents_input["visual"].float(), timesteps_input["visual"])}


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.calls = []

    def forward(self, x, sigma):
        self.calls.append((x.detach().clone(), sigma.detach().clone()))
        return self.scale * x + sigma.reshape(-1, 1, 1)


class Metrics:
    def emit_mean(self, *args, **kwargs):
        pass


def context(student, teacher, rollout=0):
    return DiffusionLossContext(
        models={"transformer": student},
        teacher_models={"transformer": teacher},
        train_pipeline_config=Config(),
        sde_backend=None,
        device=torch.device("cpu"),
        forward_dtype=torch.float32,
        args=SimpleNamespace(pdd_num_intervals=32, pdd_block_size=4, diffusion_height=32, seed=42),
        rollout_id=rollout,
        algorithm_state={"grid": torch.linspace(1, 0, 33)},
    )


def raw_forward(model, prepared):
    pred = model(prepared.latents["visual"], prepared.timesteps_for_model["visual"])
    return {"visual": pred.permute(1, 2, 0, 3).flatten(2)}


def test_eight_optimizer_updates_match_reference_and_carry_preupdate_states():
    torch.manual_seed(12)
    student = Model()
    reference = copy.deepcopy(student)
    teacher = Teacher()
    ctx = context(student, teacher)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0)
    refopt = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0)
    pairs = [{"sample_index": i, "cond_kwargs": {}} for i in (2, 9)]
    expanded = expand_trajectories(pairs, intervals=32, block_size=4)
    states = {}
    for step in range(8):
        opt.zero_grad()
        refopt.zero_grad()
        for pair in expanded[step * 2 : (step + 1) * 2]:
            prepared = prepare_pdd_batch(ctx, [pair])
            lane, start, k, grid = (prepared.extras[v] for v in ("lane", "start", "selected", "grid"))
            x = prepared.latents["visual"]
            if step:
                torch.testing.assert_close(x, states[lane], rtol=0, atol=0)
            assert not x.requires_grad
            predictions = reference(x, grid[start].reshape(1))
            expected = pdd_loss(
                predictions, x, grid, start, k, lambda state, sigma: teacher(state.bfloat16().float(), sigma.expand(1))
            )
            states[lane] = block_update(x, predictions, grid, start, start + 4).detach()
            loss = pdd_loss_formula(
                ctx,
                [pair],
                prepared,
                new_pred=raw_forward(student, prepared),
                old_pred=None,
                ref_pred=None,
                metrics=Metrics(),
            )
            torch.testing.assert_close(loss, expected.sum(), rtol=0, atol=0)
            (loss / 2).backward()
            (expected.sum() / 2).backward()
        for a, b in zip(student.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        opt.step()
        refopt.step()
    assert ctx.algorithm_state["lanes"] == {}
    assert teacher.scale.grad is None
    for a, b in zip(student.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_resume_boundary_reconstructs_same_noise_and_interval():
    student, teacher = Model(), Teacher()
    pair = {"sample_index": 99, "pdd_start": 0, "cond_kwargs": {}}
    a = prepare_pdd_batch(context(student, teacher, 7), [pair])
    torch.randn(500)  # unrelated process RNG must not affect the resumed trajectory
    b = prepare_pdd_batch(context(student, teacher, 7), [pair])
    torch.testing.assert_close(a.latents["visual"], b.latents["visual"], rtol=0, atol=0)
    assert a.extras["selected"] == b.extras["selected"]
    c = prepare_pdd_batch(context(student, teacher, 8), [pair])
    assert not torch.equal(a.latents["visual"], c.latents["visual"])


def test_heads_installed_on_meta_before_sharding():
    with torch.device("meta"):
        model = nn.Module()
        model.proj_out = nn.Linear(4, 2)
    model.config = SimpleNamespace(patch_size=1, in_channels=2)
    install_student(model, SimpleNamespace(pdd_num_intervals=32))
    assert all(p.is_meta for p in model.parameters())
    assert model.proj_out.parallel.weight.shape == (32, 2, 4)


def test_dp_split_precedes_trajectory_expansion():
    from miles.utils.train_data_utils import TrainDataDPSplitter

    pairs = [{"sample_index": i} for i in range(8)]
    shards = TrainDataDPSplitter().split_by_dp({"train_data": pairs}, 2)
    for shard in shards:
        expanded = expand_trajectories(shard["train_data"], intervals=32, block_size=4)
        for start in range(8):
            window = expanded[start * 4 : (start + 1) * 4]
            assert [p["sample_index"] for p in window] == [p["sample_index"] for p in shard["train_data"]]
            assert all(p["pdd_start"] == start * 4 for p in window)
    with pytest.raises(ValueError, match="distinct lane"):
        expand_trajectories([pairs[0], pairs[0]], intervals=32, block_size=4)


def valid_args():
    return SimpleNamespace(
        diffusion_model_family="qwen_image21",
        diffusion_task="t2i",
        train_only=True,
        train_backend="fsdp",
        update_weight_target_modules=["transformer"],
        pdd_num_intervals=32,
        pdd_block_size=4,
        diffusion_height=1024,
        diffusion_width=1024,
        micro_batch_size=1,
        micro_batch_size_sample=None,
        sequence_parallel_size=1,
        use_lora=False,
        use_ema=False,
        ref_load=None,
        ref_mode="none",
        teacher_lora_adapter_paths=None,
        lora_adapter_paths=None,
        diffusion_recompute_old_log_prob=False,
        diffusion_kl_beta=0,
        eval_interval=None,
        diffusion_guidance_scale=1.0,
        diffusion_true_cfg_scale=1.0,
        pdd_train_cache="cache",
        prompt_data="train.jsonl",
        rollout_global_dataset=True,
        n_samples_per_prompt=1,
        no_save_optim=False,
        no_load_optim=False,
        no_load_rng=False,
        fsdp_master_dtype="fp32",
        diffusion_forward_dtype="bf16",
        teacher_load=None,
        hf_checkpoint="model",
        rollout_function_path="miles.rollout.sglang_rollout.generate_rollout",
        custom_convert_samples_to_train_data_path=None,
        custom_rollout_log_function_path=None,
        custom_prepare_train_batch_path=None,
        custom_loss_function_path=None,
        num_steps_per_rollout=None,
        global_batch_size=None,
        rollout_batch_size=8,
    )


def test_pdd_defaults_keep_eight_lanes_for_all_eight_updates():
    from miles.distillation.integration import configure_args

    args = valid_args()
    configure_args(args)
    assert args.global_batch_size == 8 and args.num_steps_per_rollout == 8
    assert args.teacher_load == args.hf_checkpoint
    assert args.custom_loss_function_path.endswith("pdd.pdd_loss_formula")


@pytest.mark.parametrize(
    "name,value",
    [
        ("sequence_parallel_size", 2),
        ("use_lora", True),
        ("train_only", False),
        ("pdd_block_size", 8),
        ("diffusion_guidance_scale", 4),
        ("no_load_optim", True),
        ("global_batch_size", 1),
        ("num_steps_per_rollout", 1),
        ("diffusion_height", 512),
    ],
)
def test_unvalidated_modes_fail_before_gpu_load(name, value):
    from miles.distillation.integration import configure_args

    args = valid_args()
    setattr(args, name, value)
    with pytest.raises(ValueError):
        configure_args(args)
