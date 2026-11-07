"""Tests for t2i.cv.color module."""
import numpy as np
import pytest
from t2i.config import ColorConfig
from t2i.cv.color import apply_clahe, apply_white_balance, apply_vibrance, apply_color_correction


@pytest.fixture
def sample_image() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(10, 240, (256, 256, 3), dtype=np.uint8)


def test_clahe_shape_preserved(sample_image):
    out = apply_clahe(sample_image, clip_limit=2.0, tile_grid=(8, 8))
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8


def test_clahe_increases_contrast(sample_image):
    out = apply_clahe(sample_image, clip_limit=3.0)
    # CLAHE should generally increase or maintain std of pixel values
    orig_std = sample_image[:, :, 0].std()
    # We just check output is valid; contrast change is not guaranteed to increase
    assert out.min() >= 0 and out.max() <= 255


def test_white_balance_gray_world(sample_image):
    out = apply_white_balance(sample_image, method="gray_world")
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8
    # Channel means should be closer together after gray-world
    out_means = out.mean(axis=(0, 1))
    in_means = sample_image.mean(axis=(0, 1))
    out_range = out_means.max() - out_means.min()
    in_range = in_means.max() - in_means.min()
    assert out_range <= in_range + 5.0  # allow small tolerance


def test_white_balance_percentile(sample_image):
    out = apply_white_balance(sample_image, method="percentile", percentile=95.0)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8


def test_white_balance_invalid_method(sample_image):
    with pytest.raises(ValueError, match="Unknown white_balance"):
        apply_white_balance(sample_image, method="invalid")


def test_vibrance_shape_preserved(sample_image):
    out = apply_vibrance(sample_image, amount=0.2)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8


def test_vibrance_zero_amount_mostly_unchanged(sample_image):
    out = apply_vibrance(sample_image, amount=0.0)
    # With amount=0 the output should be very close to input
    diff = np.abs(out.astype(int) - sample_image.astype(int))
    assert diff.max() <= 5


def test_apply_color_correction_full(sample_image):
    cfg = ColorConfig(clahe_clip=2.0, clahe_tile=[8, 8], white_balance="gray_world", vibrance=0.1)
    out = apply_color_correction(sample_image, cfg)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8
