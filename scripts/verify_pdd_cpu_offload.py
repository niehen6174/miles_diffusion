"""Real single-GPU FSDP2 CPU-offload accumulation versus eager AdamW."""

import copy
import json
import socket
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(128, 256)
        self.b = torch.nn.Linear(256, 64)

    def forward(self, x):
        return self.b(torch.nn.functional.silu(checkpoint(self.a, x, use_reentrant=False)))


def main():
    torch.cuda.set_device(0)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.manual_seed(42)
    torch.set_num_threads(4)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=0,
        world_size=1,
        device_id=torch.device("cuda:0"),
    )
    try:
        mesh = init_device_mesh("cuda", (1,))
        reference = Model().cuda()
        offload = copy.deepcopy(reference).cpu()
        policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False)
        for module in (offload.a, offload.b, offload):
            fully_shard(module, mesh=mesh, offload_policy=CPUOffloadPolicy(), mp_policy=policy)
        opt = torch.optim.AdamW(offload.parameters(), lr=1e-4, weight_decay=0, foreach=False)
        refopt = torch.optim.AdamW(reference.parameters(), lr=1e-4, weight_decay=0, foreach=False)
        errors = []
        for _step in range(2):
            opt.zero_grad(set_to_none=True)
            refopt.zero_grad(set_to_none=True)
            for i in range(8):
                x = torch.randn(4, 128, device="cuda")
                y = torch.randn(4, 64, device="cuda")
                offload.set_requires_gradient_sync(i == 7)
                ((offload(x) - y).square().mean() / 8).backward()
                ((reference(x) - y).square().mean() / 8).backward()
            max_grad = 0
            for a, b in zip(offload.parameters(), reference.parameters(), strict=True):
                assert a.device.type == "cpu" and a.grad.device.type == "cpu"
                actual = a.grad.to_local()
                expected = b.grad.cpu()
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                max_grad = max(max_grad, (actual - expected).abs().max().item())
            opt.step()
            refopt.step()
            for a, b in zip(offload.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(a.to_local(), b.cpu(), atol=1e-6, rtol=1e-5)
            errors.append(max_grad)
        print(
            json.dumps(
                dict(
                    passed=True,
                    windows=2,
                    microbatches_per_window=8,
                    max_gradient_errors=errors,
                    optimizer="CPU AdamW",
                    gradient_checkpointing=True,
                )
            )
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
