"""Interface contract for ``collate_cond_for_sample_batch(..., pad_to_len=...)``.

The refactored flat-pair trainer collates cond **per micro-batch** instead of
once per optimizer window, so to keep bitwise grouping parity it needs to ask
every model config to pad text to one shared width (the legacy window-wide
seq_len) via ``pad_to_len``.  For that to be uniform, ``pad_to_len`` must be part
of the base contract — not just Qwen-Image's override — otherwise the trainer
passing it would ``TypeError`` on the concat-based configs.

Contract:
  * variable-length-padding configs (Qwen-Image) **honor** ``pad_to_len``;
  * fixed-length concat configs (SD3 — and, on their own upstream PR branches,
    Wan2.2 / LTX, which use the identical ``torch.cat`` pattern) **accept and
    ignore** it (it is a no-op for them).

CPU-only, no model forward.

Run:  python -m pytest tests/test_cond_collate_pad_to_len_interface.py -q
"""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="stage-a-cpu", labels=[])

import inspect

import torch

from miles.backends.fsdp_utils.configs.qwen_image import QwenImageTrainPipelineConfig
from miles.backends.fsdp_utils.configs.sd3 import SD3TrainPipelineConfig
from miles.backends.fsdp_utils.configs.train_pipeline_config import TrainPipelineConfig


def test_pad_to_len_is_in_the_base_contract():
    """base + every on-branch config expose pad_to_len (default None)."""
    for cfg in (TrainPipelineConfig, QwenImageTrainPipelineConfig, SD3TrainPipelineConfig):
        params = inspect.signature(cfg.collate_cond_for_sample_batch).parameters
        assert "pad_to_len" in params, f"{cfg.__name__}.collate_cond_for_sample_batch lacks pad_to_len"
        assert params["pad_to_len"].default is None, cfg.__name__


def test_sd3_concat_config_accepts_and_ignores_pad_to_len():
    """SD3 concats fixed-length embeds — pad_to_len must be a no-op, not an error.

    (Wan2.2 / LTX use the same torch.cat collate on their upstream PR branches,
    so this is the stand-in for every fixed-length config.)"""
    conds = [
        {
            "encoder_hidden_states": torch.arange(8 * 4, dtype=torch.float32).reshape(1, 8, 4),
            "pooled_projections": torch.arange(16, dtype=torch.float32).reshape(1, 16),
        },
        {
            "encoder_hidden_states": torch.arange(8 * 4, 2 * 8 * 4, dtype=torch.float32).reshape(1, 8, 4),
            "pooled_projections": torch.arange(16, 32, dtype=torch.float32).reshape(1, 16),
        },
    ]
    base = SD3TrainPipelineConfig.collate_cond_for_sample_batch(None, conds, "cpu")
    padded = SD3TrainPipelineConfig.collate_cond_for_sample_batch(None, conds, "cpu", pad_to_len=999)

    assert set(base) == set(padded)
    for k in base:
        assert torch.equal(base[k], padded[k]), f"pad_to_len changed concat output for {k}"


def _qwen_cond(seq_len: int, dim: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "encoder_hidden_states": torch.randn(1, seq_len, dim, generator=g),
        "txt_seq_lens": [seq_len],
        "img_shapes": [(1, 2, 2)],
    }


def test_qwen_variable_config_honors_pad_to_len():
    """Qwen-Image does variable-length padding — pad_to_len must widen to it."""
    conds = [_qwen_cond(5, seed=0), _qwen_cond(7, seed=1)]
    local = QwenImageTrainPipelineConfig.collate_cond_for_sample_batch(None, conds, "cpu")
    widened = QwenImageTrainPipelineConfig.collate_cond_for_sample_batch(None, conds, "cpu", pad_to_len=20)

    assert local["encoder_hidden_states"].shape[1] == 7  # batch-local max
    assert widened["encoder_hidden_states"].shape[1] == 20  # forced shared width
    assert widened["encoder_hidden_states_mask"].shape[1] == 20


def test_qwen_single_sample_collate_is_clean_broadcast_with_all_true_mask():
    """Single-sample (timestep-stacked) micro-batches now always go through
    collate (the expand_cond_for_timestep_batch shortcut was removed). Collating
    bsz copies of one sample must give each row == that sample's embed at its own
    (un-padded) length, plus an all-True mask -- the all-True mask being a forward
    no-op verified on GPU in tests/manual/check_mask_equivalence.py."""
    cond = _qwen_cond(6, seed=3)
    bsz = 4
    out = QwenImageTrainPipelineConfig.collate_cond_for_sample_batch(None, [cond] * bsz, "cpu")
    enc = out["encoder_hidden_states"]
    assert enc.shape[0] == bsz
    assert enc.shape[1] == 6  # single sample's own length, no extra padding
    for i in range(bsz):
        assert torch.equal(enc[i], cond["encoder_hidden_states"][0])
    assert out["encoder_hidden_states_mask"].all()


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
