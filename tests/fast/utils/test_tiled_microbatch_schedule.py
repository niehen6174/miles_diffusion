"""Cross-check build_tiled_microbatch_schedule against REAL legacy 2D tiles.

Distinct from test_legacy_tile_grouping_golden.py (which adds the DP split): this replays
many (sample_mb x tstep_mb x iter_order x M x T) configs from legacy_tile_2d_grouping.json
-- including non-degenerate tilings a 1D micro_batch_size cannot express -- through the
refactored build_tiled_microbatch_schedule and asserts cell-for-cell equality. No DP
split, no model; pure CPU. See gen_legacy_tile_2d_fixture.py for provenance.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import json
from pathlib import Path

from miles.utils.train_data_utils import build_tiled_microbatch_schedule, reorder_train_pairs_for_tiling

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_tile_2d_grouping.json"


def _cases():
    return json.loads(_FIXTURE.read_text())["cases"]


def test_tiled_schedule_matches_real_legacy_tiles():
    cases = _cases()
    assert cases, "fixture has no cases"
    for case in cases:
        t = case["T"]
        sched = build_tiled_microbatch_schedule(
            num_samples_per_optim_step=case["M"],
            sde_window_size=t,
            num_optim_steps_per_rollout=1,
            sample_microbatch=case["sample_mb"],
            tstep_microbatch=case["tstep_mb"],
            iter_order=case["iter_order"],
        )
        assert len(sched) == 1
        # pair index -> (sample_pos, tstep_pos) within the window
        got = [[[idx // t, idx % t] for idx in mb] for mb in sched[0]]
        assert got == case["tiles"], f"{case['name']}: tile mismatch\n got={got}\n exp={case['tiles']}"


def test_multi_optim_step_offsets():
    sched = build_tiled_microbatch_schedule(
        num_samples_per_optim_step=4,
        sde_window_size=2,
        num_optim_steps_per_rollout=2,
        sample_microbatch=2,
        tstep_microbatch=2,
        iter_order="sample_major",
    )
    assert len(sched) == 2
    assert sched[0][0] == [0, 1, 2, 3]  # step 0, samples 0,1 x tsteps 0,1
    assert sched[1][0] == [8, 9, 10, 11]  # step 1 offset by M*T = 8


def test_tiling_partitions_every_pair_exactly_once():
    sched = build_tiled_microbatch_schedule(
        num_samples_per_optim_step=10,
        sde_window_size=4,
        num_optim_steps_per_rollout=1,
        sample_microbatch=4,
        tstep_microbatch=2,
        iter_order="sample_major",
    )
    flat = [i for mb in sched[0] for i in mb]
    assert sorted(flat) == list(range(10 * 4))  # no gaps, no overlaps


def test_unknown_iter_order_raises():
    try:
        build_tiled_microbatch_schedule(
            num_samples_per_optim_step=4,
            sde_window_size=2,
            num_optim_steps_per_rollout=1,
            sample_microbatch=2,
            tstep_microbatch=2,
            iter_order="bogus",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown iter_order")


def _sample_major_pairs(num_samples, sde_window):
    return [{"sample_index": s, "tag": (s, t)} for s in range(num_samples) for t in range(sde_window)]


def test_reorder_makes_legacy_tiles_contiguous():
    # After reorder, each contiguous tile_size block == the corresponding legacy tile
    # (so the actor's plain contiguous schedule reproduces the 2D tiling).
    data = _sample_major_pairs(8, 4)  # 8 samples x 4 tsteps
    reordered = reorder_train_pairs_for_tiling(
        data, num_optim_steps_per_rollout=1, sample_microbatch=4, tstep_microbatch=2, iter_order="sample_major"
    )
    tiled = build_tiled_microbatch_schedule(
        num_samples_per_optim_step=8,
        sde_window_size=4,
        num_optim_steps_per_rollout=1,
        sample_microbatch=4,
        tstep_microbatch=2,
        iter_order="sample_major",
    )
    tile_size = 4 * 2
    for i, tile in enumerate(tiled[0]):
        chunk = reordered[i * tile_size : (i + 1) * tile_size]
        assert [p["tag"] for p in chunk] == [(idx // 4, idx % 4) for idx in tile]


def test_reorder_multi_optim_step_preserves_step_boundaries():
    data = _sample_major_pairs(8, 2)  # 16 pairs; 2 optim steps -> 4 samples/step
    reordered = reorder_train_pairs_for_tiling(
        data, num_optim_steps_per_rollout=2, sample_microbatch=2, tstep_microbatch=2, iter_order="sample_major"
    )
    assert {p["sample_index"] for p in reordered[:8]} == {0, 1, 2, 3}
    assert {p["sample_index"] for p in reordered[8:]} == {4, 5, 6, 7}


def test_reorder_rejects_ragged_tiles():
    data = _sample_major_pairs(10, 4)  # 10 samples not divisible by sample_mb=4 -> ragged
    try:
        reorder_train_pairs_for_tiling(
            data, num_optim_steps_per_rollout=1, sample_microbatch=4, tstep_microbatch=2, iter_order="sample_major"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for ragged (non-uniform) tiles")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
