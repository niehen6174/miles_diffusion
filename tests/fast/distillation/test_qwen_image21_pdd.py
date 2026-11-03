import copy
import io
from pathlib import Path

import pytest
import torch

from miles.distillation.pdd import pdd_loss, shifted_sigma_grid
from miles.distillation.qwen_image21_pdd import QwenImage21PDDStudent, checkpoint_model_config


def make_model():
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "QwenImage21Transformer2DModel"):
        pytest.skip("requires diffusers with Qwen-Image 2.1 support")
    return diffusers.QwenImage21Transformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=16,
        axes_dims_rope=(4, 6, 6),
        mlp_ratio=2,
    )


def conditioning(device="cpu"):
    return {
        "encoder_hidden_states": torch.randn(1, 3, 16, device=device),
        "encoder_hidden_states_mask": torch.tensor([[True, True, False]], device=device),
        "img_mask": torch.tensor([[False, False, False, True, True, True, True]], device=device),
        "img_shapes": [[(1, 4, 4)]],
    }


def test_qwen21_head_init_target_tail_and_checkpointed_backward():
    torch.manual_seed(12)
    teacher = make_model().eval().requires_grad_(False)
    student = QwenImage21PDDStudent(copy.deepcopy(teacher).requires_grad_(True), 4).train()
    student.transformer.enable_gradient_checkpointing()
    cond = conditioning()
    x, grid = torch.randn(1, 16, 4), shifted_sigma_grid(4, 3)

    def teacher_forward(state, sigma):
        return teacher(state, timestep=sigma.expand(1), return_dict=False, **cond)[0][:, -16:]

    predictions = student(x, grid[0], **cond)
    assert predictions.shape == (4, 1, 16, 4)
    for prediction in predictions:
        torch.testing.assert_close(prediction, teacher_forward(x, grid[0]))
    result = pdd_loss(predictions, x, grid, 0, 2, teacher_forward)
    result.loss.backward()
    assert student.transformer.img_in.weight.grad.abs().sum() > 0
    assert student.transformer.proj_out.parallel.weight.grad[2].abs().sum() > 0
    assert student.transformer.proj_out.parallel.weight.grad[[0, 1, 3]].count_nonzero() == 0
    assert all(p.grad is None for p in teacher.parameters())


def test_rejects_training_with_inference_cache():
    student = QwenImage21PDDStudent(make_model(), 4)
    with pytest.raises(ValueError, match="KV cache"):
        student(torch.randn(1, 16, 4), torch.tensor(1.0), kv_cache=object())


def test_checkpoint_config_supports_weights_only_load_with_local_model_path():
    teacher = make_model()
    teacher.register_to_config(_name_or_path=Path("/example/local/checkpoint"))
    student = QwenImage21PDDStudent(teacher, 4)
    checkpoint = io.BytesIO()
    torch.save({"config": checkpoint_model_config(teacher), "student": student.state_dict()}, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=True)
    new_student = QwenImage21PDDStudent(type(teacher).from_config(restored["config"]), 4)
    new_student.load_state_dict(restored["student"])
    for key, value in student.state_dict().items():
        torch.testing.assert_close(new_student.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 CUDA checkpoint audit")
def test_bf16_gradient_checkpointing_preserves_outputs_and_parameter_gradients():
    """Exercise the actual FP32-parameter/BF16-forward training precision policy."""
    torch.manual_seed(91)
    eager = QwenImage21PDDStudent(make_model().cuda(), 4).train()
    checkpointed = copy.deepcopy(eager)
    checkpointed.transformer.enable_gradient_checkpointing()
    cond = conditioning("cuda")
    x = torch.randn(1, 16, 4, device="cuda")
    sigma = torch.tensor(0.683, device="cuda")
    target = torch.randn_like(x)
    outputs = []
    for model in (eager, checkpointed):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(x, sigma, **cond)
            loss = (output[2].float() - target).square().mean()
        loss.backward()
        outputs.append(output.detach())
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    for (name, a), (_, b) in zip(eager.named_parameters(), checkpointed.named_parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0, msg=name)
