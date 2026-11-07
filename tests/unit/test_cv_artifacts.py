"""Tests for t2i.cv.artifacts module."""
import numpy as np
import pytest
from t2i.config import ArtifactsConfig
from t2i.cv.artifacts import (
    detect_checkerboard,
    detect_over_smooth_mask,
    detect_over_saturated_mask,
    correct_inside_mask,
    detect_and_correct_artifacts,
)


@pytest.fixture
def checkerboard_image() -> np.ndarray:
    """8×8 checkerboard pattern tiled to 256×256 — high FFT peaks."""
    tile = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    # Tile to 256×256
    tiled = np.tile(tile, (128, 128))
    return np.stack([tiled, tiled, tiled], axis=2)


@pytest.fixture
def smooth_image() -> np.ndarray:
    """Uniform grey image — zero local variance."""
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def saturated_image() -> np.ndarray:
    """Image with fully saturated (255) pixels in a region."""
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    img[0:10, 0:10, :] = 255
    return img


def test_detect_checkerboard_finds_peaks(checkerboard_image):
    count, strength = detect_checkerboard(checkerboard_image, peak_threshold_factor=2.0)
    assert count > 0, "Should detect peaks in a checkerboard pattern"


def test_detect_checkerboard_smooth_no_peaks(smooth_image):
    count, strength = detect_checkerboard(smooth_image, peak_threshold_factor=6.0)
    # Uniform image → very few or no spectral peaks
    assert count == 0 or strength < 10.0


def test_detect_over_smooth_mask(smooth_image):
    mask = detect_over_smooth_mask(smooth_image, local_var_threshold=10.0)
    assert mask.shape == (64, 64)
    # All pixels should be flagged as over-smooth in a uniform image
    assert mask.all()


def test_detect_over_saturated_mask(saturated_image):
    mask = detect_over_saturated_mask(saturated_image)
    assert mask.shape == (64, 64)
    # The 10×10 block at top-left should be flagged
    assert mask[0:10, 0:10].all()
    # Interior non-saturated pixels should not be flagged
    assert not mask[20, 20]


def test_correct_inside_mask_shape(saturated_image):
    mask = detect_over_saturated_mask(saturated_image)
    corrected = correct_inside_mask(saturated_image, mask, method="median")
    assert corrected.shape == saturated_image.shape
    assert corrected.dtype == np.uint8


def test_detect_and_correct_no_correction(smooth_image):
    cfg = ArtifactsConfig(fft_peak_threshold=6.0, local_var_threshold=10.0, correct=False)
    out, report = detect_and_correct_artifacts(smooth_image, cfg)
    assert np.array_equal(out, smooth_image), "No correction → output must equal input"
    assert report["correction_applied"] is False


def test_detect_and_correct_with_correction():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    cfg = ArtifactsConfig(fft_peak_threshold=6.0, local_var_threshold=10.0, correct=True)
    out, report = detect_and_correct_artifacts(img, cfg)
    assert out.shape == img.shape
    assert report["correction_applied"] is True
