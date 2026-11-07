"""Tests for t2i.cv.quality module."""
import numpy as np
import pytest
from t2i.cv.quality import (
    laplacian_sharpness,
    colourfulness,
    rms_contrast,
    histogram_entropy,
    compute_quality_report,
    QualityReport,
)


@pytest.fixture
def uniform_image() -> np.ndarray:
    """Completely flat image — zero sharpness, zero contrast, low entropy."""
    return np.full((128, 128, 3), 128, dtype=np.uint8)


@pytest.fixture
def noisy_image() -> np.ndarray:
    """High-frequency noise image — high sharpness and contrast."""
    rng = np.random.default_rng(99)
    return rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)


def test_laplacian_sharpness_uniform_is_zero(uniform_image):
    score = laplacian_sharpness(uniform_image)
    assert score == pytest.approx(0.0, abs=1e-6)


def test_laplacian_sharpness_noisy_is_high(noisy_image):
    score = laplacian_sharpness(noisy_image)
    assert score > 100.0  # noisy image has high sharpness


def test_colourfulness_returns_nonnegative(noisy_image):
    score = colourfulness(noisy_image)
    assert score >= 0.0


def test_rms_contrast_uniform_is_zero(uniform_image):
    score = rms_contrast(uniform_image)
    assert score == pytest.approx(0.0, abs=1e-4)


def test_rms_contrast_noisy_is_positive(noisy_image):
    score = rms_contrast(noisy_image)
    assert score > 0.0


def test_histogram_entropy_uniform(uniform_image):
    # A uniform image has only one grey level → entropy is 0
    score = histogram_entropy(uniform_image)
    assert score == pytest.approx(0.0, abs=0.01)


def test_histogram_entropy_noisy_is_high(noisy_image):
    score = histogram_entropy(noisy_image)
    assert score > 7.0  # close to 8 bits for a truly uniform histogram


def test_compute_quality_report_returns_dataclass(noisy_image):
    report = compute_quality_report(noisy_image)
    assert isinstance(report, QualityReport)
    assert report.sharpness_laplacian > 0
    assert report.colourfulness >= 0
    assert report.rms_contrast >= 0
    assert report.entropy >= 0
