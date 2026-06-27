"""Golden fixture for build_tiled_microbatch_schedule -> legacy_tile_2d_grouping.json.

Distinct from gen_legacy_tile_fixture: this cross-checks the 2D tiling function alone,
over a single optimizer window, across many (sample_mb, tstep_mb, iter_order, M, T)
configs -- crucially the NON-degenerate ones a 1D micro_batch_size cannot express
(SD3 tstep_mb=5 < T=10, timestep_major, ragged, single-cell, whole-window). It does NOT
exercise any DP split -- that axis is the 1D fixture's job. Replayed cell-for-cell by
test_tiled_microbatch_schedule.py.

Each config executes origin/main _run_optim_window verbatim (_forward_tile -> recorder,
debug_skip_optimizer_step) over M samples x T sde-steps.
Regenerate: python tests/fixtures/gen_legacy_tile_2d_fixture.py
"""

from __future__ import annotations

import ast
import json
import subprocess
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

LEGACY_REF = "origin/main"  # radixark/miles_diffusion @ a48476c
LEGACY_ACTOR_PATH = "miles/backends/fsdp_utils/actor.py"

# (name, M samples, T sde-steps, sample_microbatch, tstep_microbatch, iter_order)
CONFIGS = [
    {"name": "sd3_like_tstepmb_lt_T", "M": 16, "T": 10, "sample_mb": 8, "tstep_mb": 5, "iter_order": "sample_major"},
    {"name": "ocr_like_tstepmb_eq_T", "M": 12, "T": 2, "sample_mb": 4, "tstep_mb": 2, "iter_order": "sample_major"},
    {"name": "tstepmb_1", "M": 8, "T": 4, "sample_mb": 4, "tstep_mb": 1, "iter_order": "sample_major"},
    {"name": "timestep_major", "M": 8, "T": 6, "sample_mb": 2, "tstep_mb": 3, "iter_order": "timestep_major"},
    {"name": "ragged_sample_chunk", "M": 10, "T": 4, "sample_mb": 4, "tstep_mb": 2, "iter_order": "sample_major"},
    {"name": "ragged_tstep_chunk", "M": 6, "T": 7, "sample_mb": 3, "tstep_mb": 4, "iter_order": "sample_major"},
    {"name": "single_cell", "M": 4, "T": 4, "sample_mb": 1, "tstep_mb": 1, "iter_order": "sample_major"},
    {"name": "whole_window_one_tile", "M": 8, "T": 3, "sample_mb": 8, "tstep_mb": 3, "iter_order": "sample_major"},
    {"name": "timestep_major_ragged", "M": 9, "T": 5, "sample_mb": 4, "tstep_mb": 2, "iter_order": "timestep_major"},
]


# _chunked_indices: VERBATIM from origin/main actor.py:774 (needed by the live-exec'd legacy fn).
def _chunked_indices(total: int, chunk_size: int, device: torch.device) -> list[torch.Tensor]:
    if total <= 0:
        return []
    chunk_size = max(1, chunk_size)
    return [
        torch.arange(start, min(start + chunk_size, total), device=device, dtype=torch.long)
        for start in range(0, total, chunk_size)
    ]


def _load_legacy_run_optim_window():
    src = subprocess.check_output(["git", "show", f"{LEGACY_REF}:{LEGACY_ACTOR_PATH}"], text=True)
    tree = ast.parse(src)
    func_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_optim_window":
            func_src = ast.get_source_segment(src, node)
            break
    if func_src is None:
        raise RuntimeError(f"_run_optim_window not found in {LEGACY_REF}:{LEGACY_ACTOR_PATH}")
    ns = {
        "torch": torch,
        "defaultdict": defaultdict,
        "nullcontext": nullcontext,
        "_chunked_indices": _chunked_indices,
    }
    exec(compile(func_src, "<legacy _run_optim_window>", "exec"), ns)
    return ns["_run_optim_window"]


class _TileRecorder:
    def __init__(self):
        self.args = SimpleNamespace(debug_skip_optimizer_step=True)
        self.tiles: list[tuple[list[int], list[int]]] = []

    def _forward_tile(self, *, sample_indices, tstep_indices, **_kwargs):
        self.tiles.append((sample_indices.tolist(), tstep_indices.tolist()))
        return torch.zeros(())


def main() -> None:
    run_optim_window = _load_legacy_run_optim_window()
    cases = []
    for cfg in CONFIGS:
        m, t = cfg["M"], cfg["T"]
        sample_mb = min(cfg["sample_mb"], m)
        tstep_mb = min(cfg["tstep_mb"], t)
        rec = _TileRecorder()
        grids = {"latents": torch.zeros(m, t, 1), "num_samples_in_window": m, "sde_window_size": t}
        run_optim_window(
            rec,
            grids=grids,
            sample_microbatch=sample_mb,
            tstep_microbatch=tstep_mb,
            iter_order=cfg["iter_order"],
            use_cfg=False,
            guidance_scale=0.0,
            true_cfg_scale=None,
            clip_range=0.0,
            kl_beta=0.0,
            noise_level=0.0,
            num_train_timesteps=1000,
        )
        # _forward_tile reshape is sample-major: sp outer, tp inner.
        tiles = [[[sp, tp] for sp in sample_pos for tp in tstep_pos] for (sample_pos, tstep_pos) in rec.tiles]
        cases.append({**cfg, "tiles": tiles})

    out = {
        "meta": {
            "description": (
                "Real legacy TrainRayActor._run_optim_window 2D tile membership over a single window, "
                "across many (sample_microbatch x tstep_microbatch x iter_order x M x T) configs. Each "
                "cell is [sample_pos, tstep_pos]. build_tiled_microbatch_schedule must reproduce every "
                "tile (pair index = sample_pos * T + tstep_pos)."
            ),
            "legacy_ref": "radixark/miles_diffusion@a48476c (origin/main), _run_optim_window executed live",
        },
        "cases": cases,
    }
    out_path = Path(__file__).parent / "legacy_tile_2d_grouping.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}  ({len(cases)} cases)")


if __name__ == "__main__":
    main()
