"""
t2i.cv.resize — Resolution management and adaptive generation path.

All functions are pure (no side effects on the input array) and toggled by
ResizeConfig so their effect can be isolated in experiments.

Strategies:
- ``none``:        pass-through.
- ``lanczos_down``: downscale with Lanczos (better than bilinear for shrinking).
- ``bicubic_up``:  upscale with bicubic interpolation.
- ``adaptive``:    generate at a lower resolution, then upscale to target size
                   with an optional unsharp-mask pass for perceived sharpness.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from t2i.config import ResizeConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_resize(
    image: np.ndarray,
    cfg: ResizeConfig,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Resize *image* according to the ResizeConfig.

    Parameters
    ----------
    image:
        Input image, shape [H, W, 3], dtype uint8.
    cfg:
        ResizeConfig from the loaded YAML.
    target_size:
        (width, height) for the output image.  Required for modes other than
        ``none``.  Ignored when ``mode == "none"``.

    Returns
    -------
    np.ndarray
        Resized (or unchanged) image, [H, W, 3] uint8.
    """
    if cfg.mode == "none" or target_size is None:
        return image

    out_w, out_h = target_size

    if cfg.mode == "lanczos_down":
        resized = _lanczos_resize(image, out_w, out_h)
    elif cfg.mode == "bicubic_up":
        resized = _bicubic_resize(image, out_w, out_h)
    elif cfg.mode == "adaptive":
        resized = _bicubic_resize(image, out_w, out_h)
        if cfg.unsharp_amount > 0:
            resized = unsharp_mask(resized, cfg.unsharp_radius, cfg.unsharp_amount)
    else:
        raise ValueError(f"Unknown resize mode: {cfg.mode!r}")

    logger.debug("Resized %s → (%d, %d) mode=%s", image.shape[:2], out_h, out_w, cfg.mode)
    return resized


def select_generation_resolution(
    target_resolution: tuple[int, int],
    memory_budget_mb: float,
    estimated_full_res_mb: float = 2000.0,
) -> tuple[int, int]:
    """
    Choose a reduced generation resolution when memory is tight.

    When ``estimated_full_res_mb > memory_budget_mb``, returns a 384×384
    resolution; otherwise returns the target resolution unchanged.

    Parameters
    ----------
    target_resolution:
        Desired (width, height) for the output image.
    memory_budget_mb:
        Available memory in MiB.
    estimated_full_res_mb:
        Estimated peak memory for full-resolution generation.

    Returns
    -------
    tuple[int, int]
        (width, height) to use for generation.
    """
    if estimated_full_res_mb > memory_budget_mb:
        logger.info(
            "Memory budget %.0f MB < estimated %.0f MB — dropping to 384×384.",
            memory_budget_mb,
            estimated_full_res_mb,
        )
        return (384, 384)
    return target_resolution


def unsharp_mask(
    image: np.ndarray,
    radius: float = 1.0,
    amount: float = 0.5,
) -> np.ndarray:
    """
    Apply an unsharp mask to enhance perceived sharpness after upscaling.

    Result = image + amount * (image - blurred)

    Parameters
    ----------
    image:
        [H, W, 3] uint8 image.
    radius:
        Gaussian blur sigma in pixels.
    amount:
        Strength of the sharpening effect in [0, 1].

    Returns
    -------
    np.ndarray
        Sharpened image, same shape and dtype as input.
    """
    kernel_size = int(2 * round(3 * radius) + 1)  # 3-sigma rule; must be odd
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), radius)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _lanczos_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Downscale with Lanczos (INTER_LANCZOS4 in OpenCV)."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)


def _bicubic_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Upscale with bicubic interpolation."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
