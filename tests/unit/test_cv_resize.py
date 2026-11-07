"""Tests for t2i.cv.resize module."""
import numpy as np
import pytest
from t2i.config import ResizeConfig
from t2i.cv.resize import apply_resize, unsharp_mask, select_generation_resolution


@pytest.fixture
def sample_image() -> np.ndarray:
    """512×512 random RGB image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)


def test_resize_none_passthrough(sample_image):
    cfg = ResizeConfig(mode="none")
    out = apply_resize(sample_image, cfg, target_size=(256, 256))
    assert out.shape == sample_image.shape, "mode=none should not change shape"


def test_resize_lanczos_down(sample_image):
    cfg = ResizeConfig(mode="lanczos_down")
    out = apply_resize(sample_image, cfg, target_size=(256, 256))
    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8


def test_resize_bicubic_up(sample_image):
    small = sample_image[:256, :256, :]
    cfg = ResizeConfig(mode="bicubic_up")
    out = apply_resize(small, cfg, target_size=(512, 512))
    assert out.shape == (512, 512, 3)
    assert out.dtype == np.uint8


def test_resize_adaptive(sample_image):
    cfg = ResizeConfig(mode="adaptive", unsharp_radius=1.0, unsharp_amount=0.5)
    out = apply_resize(sample_image, cfg, target_size=(256, 256))
    assert out.shape == (256, 256, 3)


def test_unsharp_mask_does_not_change_shape(sample_image):
    out = unsharp_mask(sample_image, radius=1.0, amount=0.3)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8


def test_select_generation_resolution_tight_budget():
    res = select_generation_resolution((512, 512), memory_budget_mb=1000.0, estimated_full_res_mb=2000.0)
    assert res == (384, 384), "Should reduce to 384 when budget is tight"


def test_select_generation_resolution_ample_budget():
    res = select_generation_resolution((512, 512), memory_budget_mb=4000.0, estimated_full_res_mb=2000.0)
    assert res == (512, 512), "Should keep 512 when budget is ample"
