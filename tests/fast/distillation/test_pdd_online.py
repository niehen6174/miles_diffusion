"""Online protocol, signed ODE reduction and unmodified cached-mode regressions."""

import copy
from types import SimpleNamespace

import pytest
import torch
from tests.fast.distillation.test_miles_pdd_integration import (
    Metrics,
    Model,
    Teacher,
    context,
    pdd_loss_formula,
    prepare_pdd_batch,
    raw_forward,
    valid_args,
)

from miles.backends.sglang_diffusion_utils.monkey_patches.patch_pdd_qwen_image21 import (
    PDDProjection,
    embedded_sigmas,
    train_matched_timesteps,
)
from miles.distillation.integration import configure_args, expand_trajectories
from miles.distillation.pdd import block_update
from miles.rollout import pdd_online


def online_args():
    args = valid_args()
    vars(args).update(
        pdd_online=True,
        train_only=False,
        pdd_train_cache=None,
        diffusion_num_steps=8,
        diffusion_eval_num_steps=8,
        diffusion_sde_type="ode",
        diffusion_noise_level=0,
        rollout_num_gpus_per_engine=1,
        rollout_microgroup_size=1,
        sglang_dit_precision="fp32",
        diffusion_step_strategy_path=None,
        custom_generate_function_path=None,
        group_rm=False,
        rollout_patch_groups=None,
        custom_rm_path=None,
    )
    return args


def test_online_configuration_trains_all_blocks():
    args = online_args()
    configure_args(args)
    assert args.num_steps_per_rollout == 8
    assert args.rollout_function_path == "miles.rollout.sglang_diffusion_rollout.generate_rollout"
    assert args.rollout_patch_groups == ["pdd_qwen_image21"]
    assert args.rollout_return_full_trajectory


@pytest.mark.parametrize(
    "key,value",
    [
        ("diffusion_num_steps", 40),
        ("diffusion_noise_level", 0.1),
        ("rollout_num_gpus_per_engine", 2),
        ("rollout_microgroup_size", 2),
        ("sglang_dit_precision", "bf16"),
        ("train_only", True),
        ("num_steps_per_rollout", 1),
        ("rollout_patch_groups", ["qwen_image21"]),
    ],
)
def test_online_rejects_unsupported_modes(key, value):
    args = online_args()
    setattr(args, key, value)
    with pytest.raises(ValueError):
        configure_args(args)


def test_native_ode_projection_matches_pdd_and_sees_live_weight_updates():
    torch.manual_seed(8)
    projection = PDDProjection(torch.nn.Linear(3, 2))
    projection.grid = torch.linspace(1, 0, 33).square()
    hidden, state = torch.randn(1, 5, 3), torch.randn(1, 5, 2)
    for start in range(0, 32, 4):
        projection.start = start
        expected = block_update(state, projection.parallel(hidden), projection.grid, start, start + 4)
        actual = state + projection(hidden) * (projection.grid[start + 4] - projection.grid[start])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    before = projection(hidden).clone()
    with torch.no_grad():
        projection.parallel.weight.add_(0.5)
    assert not torch.equal(before, projection(hidden))
    assert set(projection.state_dict()) == {"parallel.weight", "parallel.bias"}


def test_converter_validates_grid_and_complete_trajectory(monkeypatch):
    grid = torch.linspace(1, 0, 33)
    monkeypatch.setattr(pdd_online, "make_grid", lambda *args: grid)
    sample = SimpleNamespace(
        seed=31,
        index=17,
        denoising_env=SimpleNamespace(pos_cond_kwargs={}),
        dit_trajectory=SimpleNamespace(
            sigmas=grid[::4],
            latents=torch.randn(9, 4, 2),
            latent_step_indices=None,
        ),
    )
    args = SimpleNamespace(seed=42, hf_checkpoint="unused", diffusion_height=32)
    pair = pdd_online.convert_samples_to_train_data(args, [sample])["train_data"][0]
    torch.testing.assert_close(pair["pdd_trajectory"], sample.dit_trajectory.latents)
    blocks = expand_trajectories([pair], intervals=32, block_size=4)
    assert [block["pdd_start"] for block in blocks] == list(range(0, 32, 4))
    for field, value in [
        ("sigmas", grid[::4] + 0.01),
        ("latents", torch.randn(8, 4, 2)),
        ("latents", torch.full((9, 4, 2), float("nan"))),
        ("latent_step_indices", torch.arange(1, 10)),
    ]:
        bad = copy.deepcopy(sample)
        setattr(bad.dit_trajectory, field, value)
        with pytest.raises(ValueError):
            pdd_online.convert_samples_to_train_data(args, [bad])


def test_engine_timesteps_reproduce_trainer_bf16_sigma():
    # Native DiT embeds bf16(bf16(t) / 1000); the Diffusers trainer embeds bf16(sigma).
    def mismatch(timesteps, sigmas):
        return ((timesteps.bfloat16() / 1000).float() - sigmas.bfloat16().float()).abs()

    # Block starts that 1000 * sigma rounds away from the trainer's value.
    sigmas = torch.tensor([1.0, 0.956724, 0.904531, 0.840349, 0.759511, 0.654567, 0.512844, 0.310901])
    assert mismatch(sigmas * 1000, sigmas).max() > 0
    assert mismatch(train_matched_timesteps(sigmas), sigmas).max() == 0
    sweep = torch.linspace(1, 0.01, 2000)
    fixed, naive = mismatch(train_matched_timesteps(sweep), sweep), mismatch(sweep * 1000, sweep)
    assert (fixed <= naive).all() and (fixed == 0).float().mean() > 0.95
    # Where no engine timestep reaches bf16(sigma), the trainer adopts the engine's value.
    unreachable = sweep[fixed > 0]
    assert unreachable.numel() > 0
    embedded = embedded_sigmas(train_matched_timesteps(unreachable))
    torch.testing.assert_close(embedded.bfloat16().float(), embedded, rtol=0, atol=0)
    assert (embedded - unreachable).abs().max() < 8e-3


def test_online_loss_consumes_rollout_state_without_carrying_trajectory():
    student, teacher = Model(), Teacher()
    ctx = context(student, teacher)
    ctx.args.pdd_online = True
    ctx.train_pipeline_config.prepare_cond_kwargs = lambda cond, device: cond
    pair = dict(
        sample_index=7,
        pdd_start=12,
        pdd_trajectory=torch.randn(9, 4, 2),
        denoising_env=SimpleNamespace(pos_cond_kwargs={}),
    )
    prepared = prepare_pdd_batch(ctx, [pair])
    torch.testing.assert_close(prepared.latents["visual"][0], pair["pdd_trajectory"][3])
    loss = pdd_loss_formula(
        ctx, [pair], prepared, new_pred=raw_forward(student, prepared), old_pred=None, ref_pred=None, metrics=Metrics()
    )
    loss.backward()
    assert not ctx.algorithm_state["lanes"]
    assert teacher.scale.grad is None
    assert student.base.weight.grad is not None


def test_resume_signature_separates_online_from_cached_protocol(monkeypatch):
    from miles.distillation import integration

    args = online_args()
    vars(args).update(
        seed=42,
        fsdp_reduce_dtype="fp32",
        train_dp_split_mode="uniform",
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        lr=1e-5,
        weight_decay=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        clip_grad=1,
        rollout_seed=8,
        diffusion_generator_device="cuda",
        diffusion_quality="default",
    )
    calls = []

    def digest(path):
        calls.append(str(path))
        return "digest"

    monkeypatch.setattr(integration, "file_sha256", digest)
    monkeypatch.setattr(integration, "make_grid", lambda *args: torch.linspace(1, 0, 33))
    online = integration.resume_signature(args)
    assert online["version"] == 2 and online["cache_manifest_sha256"] is None
    assert calls == [args.prompt_data]
    args.pdd_online = False
    args.pdd_train_cache = "cache"
    cached = integration.resume_signature(args)
    assert cached["version"] == 1 and cached != online
    assert "cache/manifest.json" in calls
