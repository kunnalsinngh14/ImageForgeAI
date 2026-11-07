"""
t2i.cv.color — Colour correction and enhancement.

Steps applied in order:
1. CLAHE on the L channel of LAB space.
2. White balance (gray-world or percentile).
3. Vibrance / saturation adjustment with oversaturation clamp.

All functions are pure: they do not modify the input array.
Toggles come from ColorConfig loaded from YAML.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from t2i.config import ColorConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_color_correction(image: np.ndarray, cfg: ColorConfig) -> np.ndarray:
    """
    Apply the full colour-correction pipeline in one call.

    Parameters
    ----------
    image:
        Input [H, W, 3] uint8 RGB image.
    cfg:
        ColorConfig with CLAHE, white-balance, and vibrance parameters.

    Returns
    -------
    np.ndarray
        Colour-corrected [H, W, 3] uint8 RGB image.
    """
    out = apply_clahe(image, cfg.clahe_clip, tuple(cfg.clahe_tile))
    out = apply_white_balance(out, cfg.white_balance)
    if cfg.vibrance != 0.0:
        out = apply_vibrance(out, cfg.vibrance)
    return out


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to the L
    channel of the LAB colour space and convert back to RGB.

    Parameters
    ----------
    image:
        [H, W, 3] uint8 RGB image.
    clip_limit:
        CLAHE clip limit.  Higher values → more contrast, more noise.
    tile_grid:
        (tiles_x, tiles_y) grid size for the adaptive equalisation.

    Returns
    -------
    np.ndarray
        [H, W, 3] uint8 RGB image.
    """
    # Convert RGB → LAB (OpenCV uses BGR internally)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    # Split into L, A, B channels
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Apply CLAHE only to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l_ch)

    # Merge back and convert to RGB
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2RGB)


def apply_white_balance(
    image: np.ndarray,
    method: str = "gray_world",
    percentile: float = 95.0,
) -> np.ndarray:
    """
    Apply white balance correction.

    Parameters
    ----------
    image:
        [H, W, 3] uint8 RGB image.
    method:
        ``"gray_world"``: scales each channel so its mean equals the global mean.
        ``"percentile"``: scales so the ``percentile``-th percentile of each
        channel is the same (white-patch assumption).
    percentile:
        Only used when ``method == "percentile"``.

    Returns
    -------
    np.ndarray
        [H, W, 3] uint8 RGB image.
    """
    if method == "none":
        return image

    img_float = image.astype(np.float32)

    if method == "gray_world":
        # Gray-world: assume the average colour of the scene is neutral gray.
        channel_means = img_float.mean(axis=(0, 1))  # [3]
        global_mean = channel_means.mean()
        # Avoid division by zero
        scales = np.where(channel_means > 1e-6, global_mean / channel_means, 1.0)
        balanced = img_float * scales[np.newaxis, np.newaxis, :]

    elif method == "percentile":
        # White-patch: brightest patch should be white.
        channel_pct = np.percentile(img_float, percentile, axis=(0, 1))  # [3]
        # Scale so that the brightest pixel per channel maps to 255.
        scales = np.where(channel_pct > 1e-6, 255.0 / channel_pct, 1.0)
        balanced = img_float * scales[np.newaxis, np.newaxis, :]

    else:
        raise ValueError(f"Unknown white_balance method: {method!r}")

    return np.clip(balanced, 0, 255).astype(np.uint8)


def apply_vibrance(
    image: np.ndarray,
    amount: float = 0.15,
    max_saturation_increase: float = 0.4,
    clip_fraction_threshold: float = 0.02,
) -> np.ndarray:
    """
    Increase vibrance (selective saturation boost for under-saturated pixels).

    Unlike a flat saturation increase, vibrance boosts dull colours more than
    already-vivid ones, reducing the risk of oversaturation.

    After boosting, the clipped-pixel fraction (channels == 255 or 0) is
    checked.  If it exceeds ``clip_fraction_threshold``, the amount is halved
    iteratively until the fraction is acceptable (or ``amount < 0.001``).

    Parameters
    ----------
    image:
        [H, W, 3] uint8 RGB image.
    amount:
        Vibrance strength. Positive = more vivid, negative = less vivid.
    max_saturation_increase:
        Hard cap on the per-pixel saturation multiplier.
    clip_fraction_threshold:
        Maximum fraction of clipped pixels before the amount is reduced.

    Returns
    -------
    np.ndarray
        [H, W, 3] uint8 RGB image.
    """
    # Work in HSV so we can operate directly on the S channel.
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h_ch = hsv[:, :, 0]  # noqa: F841
    s_ch = hsv[:, :, 1]  # [0, 255]
    v_ch = hsv[:, :, 2]  # noqa: F841

    # Normalise S to [0, 1]
    s_norm = s_ch / 255.0

    # Vibrance boost: boost less-saturated pixels more.
    # boost factor = amount * (1 - s_norm) for under-saturated pixels.
    boost = amount * (1.0 - s_norm)
    boost = np.clip(boost, -max_saturation_increase, max_saturation_increase)

    s_boosted = np.clip(s_norm + boost, 0.0, 1.0) * 255.0
    hsv[:, :, 1] = s_boosted

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Clipping check
    clipped_fraction = ((result == 255) | (result == 0)).sum() / result.size
    if clipped_fraction > clip_fraction_threshold and abs(amount) > 0.001:
        logger.debug(
            "Vibrance: clipped fraction %.3f > threshold %.3f; reducing amount.",
            clipped_fraction,
            clip_fraction_threshold,
        )
        return apply_vibrance(image, amount * 0.5, max_saturation_increase, clip_fraction_threshold)

    return result
