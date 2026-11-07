"""Tests for t2i.cv.edge_filters module."""
import numpy as np
import pytest
from t2i.config import EdgeConfig
from t2i.cv.edge_filters import (
    guided_filter,
    enhance_detail,
    edge_preservation_score,
    apply_edge_filter,
)


@pytest.fixture
def sample_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(10, 245, (128, 128, 3), dtype=np.uint8)


def test_guided_filter_output_shape(sample_image):
    base = guided_filter(sample_image, radius=4, eps=0.01)
    assert base.shape == sample_image.shape
    assert base.dtype == np.uint8


def test_enhance_detail_output_shape(sample_image):
    base = guided_filter(sample_image, radius=4, eps=0.01)
    enhanced = enhance_detail(sample_image, base, detail_scale=0.3)
    assert enhanced.shape == sample_image.shape
    assert enhanced.dtype == np.uint8


def test_edge_preservation_score_identical_images(sample_image):
    score = edge_preservation_score(sample_image, sample_image, method="sobel")
    assert score == pytest.approx(1.0, abs=1e-3)


def test_edge_preservation_score_canny(sample_image):
    score = edge_preservation_score(sample_image, sample_image, method="canny")
    assert score == pytest.approx(1.0, abs=1e-3)


def test_apply_edge_filter_pipeline(sample_image):
    cfg = EdgeConfig(guided_radius=4, guided_eps=0.01, detail_scale=0.2)
    out = apply_edge_filter(sample_image, cfg)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8
