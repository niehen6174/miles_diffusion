import pytest
import torch

from miles.utils.diffusion_rollout_response import _normalize_generated_output


@pytest.mark.parametrize(
    ("shape", "permutation"),
    [
        ((3, 8, 12), None),
        ((8, 12, 3), (2, 0, 1)),
        ((3, 5, 8, 12), None),
        ((5, 8, 12, 3), (3, 0, 1, 2)),
    ],
)
def test_normalize_generated_output_to_cfhw(shape, permutation):
    output = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    expected = output if permutation is None else output.permute(permutation)
    if expected.ndim == 3:
        expected = expected.unsqueeze(1)

    actual = _normalize_generated_output(output)

    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected)


def test_normalize_generated_output_rejects_ambiguous_layout():
    with pytest.raises(ValueError, match="layout is ambiguous"):
        _normalize_generated_output(torch.zeros(3, 8, 4))


def test_normalize_generated_output_rejects_missing_channel_dimension():
    with pytest.raises(ValueError, match="no recognizable channel dimension"):
        _normalize_generated_output(torch.zeros(5, 8, 12))
