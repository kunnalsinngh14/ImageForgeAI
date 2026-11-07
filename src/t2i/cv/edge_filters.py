"""
t2i.cv.edge_filters — Edge-aware post-processing with guided filtering.

Implements:
- Guided filter for edge-preserving denoising (falls back to bilateral filter
  when OpenCV-contrib ximgproc is unavailable).
- Detail enhancement via base-layer subtraction and re-addition.
- Edge-preservation score (Sobel/Canny overlap in [0, 1]).

All functions are pure (no in-place modification of inputs).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from t2i.config import EdgeConfig

logger = logging.getLogger(__name__)

# Check whether ximgproc (opencv-contrib) is available
try:
    _guided_filter_fn = cv2.ximgproc.guidedFilter  # type: ignore[attr-defined]
    _HAS_XIMGPROC = True
except AttributeError:
    _HAS_XIMGPROC = False
    logger.warning(
        "cv2.ximgproc.guidedFilter not available. "
        "Falling back to cv2.bilateralFilter. "
        "Install 'opencv-contrib-python' for the guided filter."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_edge_filter(image: np.ndarray, cfg: EdgeConfig) -> np.ndarray:
    """
    Apply guided-filter edge-preserving denoising and detail enhancement.

    Parameters
    ----------
    image:
        [H, W, 3] uint8 RGB image.
    cfg:
        EdgeConfig with guided_radius, guided_eps, and detail_scale.

    Returns
    -------
    np.ndarray
        Filtered [H, W, 3] uint8 RGB image.
    """
    base = guided_filter(image, cfg.guided_radius, cfg.guided_eps)
    enhanced = enhance_detail(image, base, cfg.detail_scale)
    return enhanced


def guided_filter(
    image: np.ndarray,
    radius: int = 8,
    eps: float = 0.01,
) -> np.ndarray:
    """
    Edge-preserving guided filter.

    Uses the image itself as both the guide and the input (self-guided filter),
    which smooths texture while preserving strong edges.  Falls back to a
    bilateral filter when ximgproc is not installed.

    Parameters
    ----------
    image:
        [H, W, 3] uint8 RGB image.
    radius:
        Filter radius in pixels.
    eps:
        Regularisation parameter (controls edge vs. flat region trade-off).
        Small eps → strong edge preservation.

    Returns
    -------
    np.ndarray
        Filtered [H, W, 3] uint8 RGB image (base layer).
    """
    if _HAS_XIMGPROC:
        # guidedFilter signature: (guide, src, radius, eps, dDepth=-1)
        # Using image as its own guide → self-guided smoothing.
        base = cv2.ximgproc.guidedFilter(  # type: ignore[attr-defined]
            guide=image,
            src=image,
            radius=radius,
            eps=eps * 255 * 255,  # scale eps to match uint8 range²
            dDepth=-1,
        )
    else:
        # Bilateral filter as fallback: d=radius*2+1, sigmaColor/Space tuned.
        base = cv2.bilateralFilter(
            image,
            d=radius * 2 + 1,
            sigmaColor=75,
            sigmaSpace=radius,
        )
    return base.astype(np.uint8)


def enhance_detail(
    image: np.ndarray,
    base: np.ndarray,
    detail_scale: float = 0.3,
) -> np.ndarray:
    """
    Enhance fine details by boosting the detail layer (image − base).

    ``result = base + detail_scale * (image − base)``
    When detail_scale > 1 this amplifies texture; < 1 softens it;
    == 1 returns the original image.

    Parameters
    ----------
    image:
        Original [H, W, 3] uint8 RGB image.
    base:
        Base (smooth) layer, same shape.
    detail_scale:
        Strength of detail enhancement.  0 returns the base layer.

    Returns
    -------
    np.ndarray
        [H, W, 3] uint8 RGB image.
    """
    image_f = image.astype(np.float32)
    base_f = base.astype(np.float32)
    detail = image_f - base_f  # detail layer, can be negative
    enhanced = base_f + detail_scale * detail
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def edge_preservation_score(
    original: np.ndarray,
    processed: np.ndarray,
    method: str = "sobel",
) -> float:
    """
    Measure how well edges from the original image are preserved after processing.

    Score = |edge_original ∩ edge_processed| / |edge_original|

    Returns a value in [0, 1]; 1.0 means all original edges are preserved.

    Parameters
    ----------
    original:
        [H, W, 3] uint8 RGB image.
    processed:
        [H, W, 3] uint8 RGB image, same shape.
    method:
        ``"sobel"`` or ``"canny"``.

    Returns
    -------
    float
        Edge preservation score in [0, 1].
    """
    orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)

    if method == "sobel":

        def _sobel_edges(gray: np.ndarray) -> np.ndarray:
            sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(sx**2 + sy**2)
            threshold = magnitude.mean() + magnitude.std()
            return (magnitude > threshold).astype(np.uint8)

        orig_edges = _sobel_edges(orig_gray)
        proc_edges = _sobel_edges(proc_gray)

    elif method == "canny":
        orig_edges = cv2.Canny(orig_gray, 50, 150) // 255
        proc_edges = cv2.Canny(proc_gray, 50, 150) // 255

    else:
        raise ValueError(f"Unknown edge method: {method!r}")

    orig_count = orig_edges.sum()
    if orig_count == 0:
        return 1.0  # no edges in original → trivially preserved

    intersection = (orig_edges & proc_edges).sum()
    return float(intersection) / float(orig_count)
