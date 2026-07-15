from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import math

import pytest
import torch

from miles.backends.fsdp_utils.model_backend import LTXModelBackend
from miles.backends.fsdp_utils.models.ltx_geometry import build_ltx_t2v_geometry
from miles.backends.fsdp_utils.sde_step_backend import CpsSdeStepBackend


class TestLTXAttentionBackend:
    # LTX maps --fsdp-attention-backend to ltx_core's AttentionFunction instead of the
    # diffusers set_attention_backend(str) method its native transformers don't have.
    def test_unknown_backend_raises(self):
        pytest.importorskip("ltx_core")
        with pytest.raises(ValueError, match="not an ltx_core backend"):
            LTXModelBackend(None).set_attention_backend(torch.nn.Linear(2, 2), "bogus")

    def test_alias_noops_without_attention_submodule(self):
        pytest.importorskip("ltx_core")
        # "sdpa"/"native" alias -> PYTORCH; a model with no Attention submodule is a no-op
        LTXModelBackend(None).set_attention_backend(torch.nn.Linear(2, 2), "sdpa")  # no raise

    def test_swaps_callable_on_attention_submodule(self):
        pytest.importorskip("ltx_core")
        from ltx_core.model.transformer.attention import Attention

        model = torch.nn.Sequential(Attention(query_dim=16, heads=1, dim_head=8))
        LTXModelBackend(None).set_attention_backend(model, "sdpa_math")
        # SDPA_MATH resolves to a PytorchAttention callable on both paths
        assert type(model[0].attention_function).__name__ == "PytorchAttention"
        assert type(model[0].masked_attention_function).__name__ == "PytorchAttention"


class TestCpsSdeStepBackend:
    # LTX uses CPS dynamics: σ = timestep/divisor straight from the rollout values
    # (no scheduler), and the CPS mean/std kernel must match sgl-d's rollout_sde_type="cps".
    def test_cps_kernel_matches_reference(self):
        torch.manual_seed(0)
        sb = CpsSdeStepBackend(None, sde_timestep_divisor=1000.0)  # no scheduler / no config
        t = torch.tensor([700.0, 300.0])
        nt = torch.tensor([600.0, 0.0])  # terminal σ_next = 0
        x, v, nxt = (torch.randn(2, 128, 8) for _ in range(3))
        _, log_prob, mean, std = sb.sde_step_logprob(v, t, nt, x, prev_sample=nxt, noise_level=0.8)

        sigma, sigma_next = (t / 1000).view(-1, 1, 1), (nt / 1000).view(-1, 1, 1)
        std_t = sigma_next * math.sin(0.8 * math.pi / 2)
        expected_mean = (x - sigma * v) * (1 - sigma_next) + (x + v * (1 - sigma)) * torch.sqrt(
            torch.clamp(sigma_next**2 - std_t**2, min=1e-12)
        )
        torch.testing.assert_close(mean, expected_mean, rtol=0.0, atol=0.0)
        # no-const log_prob = -(prev - mean)^2 mean over non-batch dims
        torch.testing.assert_close(log_prob, (-((nxt - expected_mean) ** 2)).mean(dim=(1, 2)), rtol=0.0, atol=0.0)
        assert log_prob.shape == (2,)


class TestLtxT2VGeometry:
    # T2V geometry is a pure function of latent shape + request constants; it rebuilds the
    # RoPE positions / masks the rollout doesn't send, and hard-checks the token count.
    def test_geometry_shapes_and_token_guard(self):
        # 512x512, 25 frames -> latent 16x16 spatial, (25-1)//8+1 = 4 frames => 4*16*16 = 1024 tokens
        num_tokens = 4 * 16 * 16
        geom = build_ltx_t2v_geometry(
            batch_size=2,
            num_tokens=num_tokens,
            latent_dim=8,
            height=512,
            width=512,
            num_frames=25,
            fps=24.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert geom["denoise_mask"].shape == (2, num_tokens)  # all tokens denoise (T2V)
        assert torch.all(geom["denoise_mask"] == 1.0)
        assert geom["clean_latent"].shape == (2, num_tokens, 8)
        assert torch.all(geom["clean_latent"] == 0.0)  # no conditioning frame in T2V

    def test_token_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="token count mismatch"):
            build_ltx_t2v_geometry(
                batch_size=1,
                num_tokens=999,
                latent_dim=8,
                height=512,
                width=512,
                num_frames=25,
                fps=24.0,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
