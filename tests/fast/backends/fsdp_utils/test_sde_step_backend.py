from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import torch

from miles.backends.fsdp_utils.sde_step_backend import DiffusersSdeStepBackend
from miles.utils.sde_log_prob import sde_step_with_logprob


class _FakeScheduler:
    def __init__(self, num_steps=8):
        self.sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

    def index_for_timestep(self, t):
        return int(torch.argmin((self.sigmas[:-1] - t).abs()))


class TestDiffusersSdeStepBackend:
    # The layered backend (σ resolution + mean/std kernel + Gaussian log_prob) must
    # reproduce the monolithic sde_step_with_logprob bit-for-bit — the refactor contract.
    def test_matches_monolithic_reference(self):
        torch.manual_seed(0)
        sched = _FakeScheduler()
        t = sched.sigmas[[2, 5]]  # per-pair timestep
        nt = sched.sigmas[[3, 6]]  # per-pair next timestep (ignored by the diffusers +1 path)
        v, x, nxt = (torch.randn(2, 4, 6) for _ in range(3))

        backend = DiffusersSdeStepBackend(sched)
        got = backend.sde_step_logprob(v, t, nt, x, prev_sample=nxt, noise_level=0.7)
        want = sde_step_with_logprob(sched, v, t, x, prev_sample=nxt, noise_level=0.7)
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w, rtol=0.0, atol=0.0)
        assert got[1].shape == (2,)
