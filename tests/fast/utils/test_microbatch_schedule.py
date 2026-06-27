"""CPU unit tests for build_microbatch_schedule (train-data grouping util).

Pure index arithmetic: split one rank's flat sample-major train-pairs into
optimizer steps, then into contiguous micro-batches. No torch / model needed.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from miles.utils.train_data_utils import build_microbatch_schedule


def test_even_division():
    sched = build_microbatch_schedule(num_pairs_per_optim_step=16, num_optim_steps_per_rollout=2, micro_batch_size=8)
    assert sched == [[(0, 8), (8, 16)], [(16, 24), (24, 32)]]


def test_absolute_offsets_across_steps():
    sched = build_microbatch_schedule(num_pairs_per_optim_step=8, num_optim_steps_per_rollout=3, micro_batch_size=4)
    assert sched == [
        [(0, 4), (4, 8)],
        [(8, 12), (12, 16)],
        [(16, 20), (20, 24)],
    ]


def test_count_and_contiguous_coverage():
    sched = build_microbatch_schedule(num_pairs_per_optim_step=256, num_optim_steps_per_rollout=2, micro_batch_size=8)
    assert len(sched) == 2
    assert all(len(step) == 32 for step in sched)  # 256 / 8 = 32 micro-batches per step
    for k, step in enumerate(sched):
        assert step[0][0] == k * 256  # step starts at its absolute offset
        assert step[-1][1] == (k + 1) * 256  # and covers the whole step
        for (_lo, hi), (nlo, _) in zip(step, step[1:], strict=False):
            assert hi == nlo  # contiguous, no gaps/overlaps


def test_raises_when_step_not_divisible_by_microbatch():
    """num_pairs_per_optim_step must be a whole multiple of micro_batch_size, so
    every micro-batch is full and all DP ranks run the same count. A ragged
    remainder (or micro_batch_size > step) must raise, not silently truncate."""
    bad_configs = [
        dict(num_pairs_per_optim_step=10, num_optim_steps_per_rollout=1, micro_batch_size=4),  # remainder 2
        dict(num_pairs_per_optim_step=5, num_optim_steps_per_rollout=2, micro_batch_size=100),  # mbs > step
    ]
    for cfg in bad_configs:
        try:
            build_microbatch_schedule(**cfg)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {cfg}")


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
