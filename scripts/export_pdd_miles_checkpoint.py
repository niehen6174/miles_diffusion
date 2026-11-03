"""Export a Miles PDD DCP checkpoint to the existing image-evaluation format.

CPU only. Does not modify training checkpoints. Requires enough host RAM and disk
for a full student state; optimizer shards are not read.
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Miles iter_XXXXXXX directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads((args.checkpoint / "meta.json").read_text())
    protocol = meta["pdd_signature"]
    if protocol["version"] != 1 or protocol["boundary"] != "complete_trajectory":
        raise ValueError("Unsupported PDD checkpoint protocol")
    settings = protocol["settings"]
    config = json.loads((Path(settings["hf_checkpoint"]) / "transformer/config.json").read_text())
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdd-export-", dir=args.output.parent) as temporary:
        merged = Path(temporary) / "model.pt"
        dcp_to_torch_save(args.checkpoint / "model", merged)
        state = torch.load(merged, map_location="cpu", mmap=True, weights_only=True)["model_state"]["model"]
        expected_head = "proj_out.parallel.weight"
        if expected_head not in state or state[expected_head].shape[0] != settings["pdd_num_intervals"]:
            raise ValueError("Checkpoint has no valid PDD parallel projection")
        payload = {
            "student": {"transformer." + k: v for k, v in state.items()},
            "config": config,
            "grid": torch.tensor(protocol["grid"], dtype=torch.float32),
            "num_intervals": settings["pdd_num_intervals"],
            "resolution": settings["diffusion_height"],
            "step": meta["global_step"],
            "source": str(args.checkpoint),
            "step_note": "Miles optimizer updates; does not include experimental initialization updates",
        }
        exported = Path(temporary) / "export.pt"
        torch.save(payload, exported)
        # Publish without overwriting a concurrently created output.
        import os

        os.link(exported, args.output)
    print(json.dumps({"output": str(args.output), "step": meta["global_step"], "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
