"""Tests for optimize modules (quantization, pruning, ONNX helpers)."""

import torch
import torch.nn as nn

from t2i.optimize.quantize import convert_to_fp16, model_size_mb, quantize_dynamic_int8
from t2i.optimize.prune import verify_unet_compatibility


class DummyLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class DummyUNet(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states=None, return_dict=False):
        return (self.conv(sample),)


def test_model_size_mb():
    model = DummyLinearModel()
    size = model_size_mb(model)
    assert size > 0.0
    assert size < 1.0  # Dummy model is small (a few KB)


def test_convert_to_fp16():
    model = DummyLinearModel()
    convert_to_fp16(model)
    for p in model.parameters():
        assert p.dtype == torch.float16


def test_quantize_dynamic_int8():
    model = DummyLinearModel()
    q_model = quantize_dynamic_int8(model)
    assert q_model is not None
    # Verify inference works
    x = torch.randn(2, 32)
    out = q_model(x)
    assert out.shape == (2, 32)


def test_verify_unet_compatibility():
    unet1 = DummyUNet(channels=4)
    unet2 = DummyUNet(channels=4)
    report = verify_unet_compatibility(
        unet1, unet2,
        batch_size=1, latent_channels=4, latent_h=16, latent_w=16,
        seq_len=10, hidden_dim=32, device="cpu",
    )
    assert report["shapes_match"] is True
    assert "max_abs_diff" in report
    assert "mean_abs_diff" in report
