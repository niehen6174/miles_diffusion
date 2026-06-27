"""bf16 path: ShardedGradScaler(enabled=False) must be a bitwise no-op vs plain
backward + optimizer.step. CPU-only (enabled=False short-circuits before any CUDA op),
so this runs in CI without a GPU. The scaler only matters for fp16 (SD3.5)."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import torch
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


def _run(use_scaler: bool, steps: int = 4):
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 16)  # identical init each call
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
    scaler = ShardedGradScaler(enabled=False)  # bf16 forward -> enabled=False
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randn(8, 16, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss = (model(x) ** 2).mean()
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return [p.detach().clone() for p in model.parameters()]


def test_disabled_scaler_is_bitwise_passthrough():
    plain = _run(use_scaler=False)
    scaled = _run(use_scaler=True)
    assert len(plain) == len(scaled)
    for a, b in zip(plain, scaled, strict=True):
        assert torch.equal(a, b), "ShardedGradScaler(enabled=False) changed the update -- not a no-op!"


def test_disabled_scale_returns_loss_unchanged():
    s = ShardedGradScaler(enabled=False)
    loss = torch.tensor([1.234, -5.0, 0.0])
    assert torch.equal(s.scale(loss), loss)  # no scaling applied


if __name__ == "__main__":
    test_disabled_scaler_is_bitwise_passthrough()
    test_disabled_scale_returns_loss_unchanged()
    print("PASS: ShardedGradScaler(enabled=False) is a bitwise no-op (CPU, no GPU needed)")
