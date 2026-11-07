"""
t2i.cv.artifacts — Artifact detection and targeted correction.

Detects three classes of artifacts:
1. Checkerboard patterns — 2D FFT peak analysis on the luminance channel.
2. Over-smoothed regions — local variance below a threshold over large areas.
3. Over-saturated regions — RGB channels clipped to 0 or 255.

Correction is applied only inside flagged masks to minimise quality loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from t2i.config import ArtifactsConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class ArtifactReport:
    """Counts and correction status for one image."""

    # Checkerboard
    fft_peak_count_before: int
    fft_peak_count_after: int
    fft_peak_strength_before: float
    fft_peak_strength_after: float

    # Over-smoothed
    smooth_region_fraction_before: float
    smooth_region_fraction_after: float

    # Over-saturated
    saturated_pixel_fraction_before: float
    saturated_pixel_fraction_after: float

    # Whether correction was applied
    correction_applied: bool


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------


def detect_checkerboard(
    image: np.ndarray,
    peak_threshold_factor: float = 6.0,
) -> tuple[int, float]:
    """
    Detect checkerboard-like periodic artifacts via 2D FFT.

    Computes the FFT of the grayscale (luminance) channel, finds peaks in the
    power spectrum above ``peak_threshold_factor × median``, and reports their
    count and total relative strength.

    Parameters
    ----------
    image:
        [H, W, 3] uint8 RGB image.
    peak_threshold_factor:
        Peaks must exceed this multiple of the spectral median to be counted.

    Returns
    -------
    (peak_count, peak_strength)
        peak_count: number of anomalous spectral peaks.
        peak_strength: mean relative strength of those peaks (0 if none).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    # Exclude the DC component (centre of the shifted spectrum)
    h, w = power.shape
    cy, cx = h // 2, w // 2
    mask = np.ones((h, w), dtype=bool)
    r_dc = max(h, w) // 20  # exclude central 5% of the spectrum
    for y in range(h):
        for x in range(w):
            if (y - cy) ** 2 + (x - cx) ** 2 < r_dc**2:
                mask[y, x] = False

    ac_power = power[mask]
    threshold = np.median(ac_power) * peak_threshold_factor
    peaks = ac_power[ac_power > threshold]

    peak_count = int(len(peaks))
    peak_strength = float(peaks.mean() / (np.median(ac_power) + 1e-8)) if peak_count > 0 else 0.0
    return peak_count, peak_strength


def detect_over_smooth_mask(
    image: np.ndarray,
    local_var_threshold: float = 10.0,
    block_size: int = 16,
) -> np.ndarray:
    """
    Return a boolean mask where local variance is below the threshold.

    Processes the image in non-overlapping blocks; a block is flagged if its
    per-channel variance is below ``local_var_threshold`` in all three channels.

    Parameters
    ----------
    image:
        [H, W, 3] uint8 image.
    local_var_threshold:
        Variance threshold below which a region is considered over-smoothed.
    block_size:
        Side length of the analysis blocks in pixels.

    Returns
    -------
    np.ndarray
        Boolean mask [H, W].  True = over-smoothed.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    img_f = image.astype(np.float32)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img_f[y : y + block_size, x : x + block_size]
            var_per_ch = block.var(axis=(0, 1))  # [3]
            if (var_per_ch < local_var_threshold).all():
                mask[y : y + block_size, x : x + block_size] = True

    return mask


def detect_over_saturated_mask(image: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of pixels where any RGB channel is clipped (0 or 255).

    Parameters
    ----------
    image:
        [H, W, 3] uint8 image.

    Returns
    -------
    np.ndarray
        Boolean mask [H, W].  True = at least one channel is clipped.
    """
    clipped = (image == 0) | (image == 255)  # [H, W, 3]
    return clipped.any(axis=2)  # [H, W]


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------


def correct_inside_mask(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "median",
) -> np.ndarray:
    """
    Apply a local correction filter only within the flagged mask pixels.

    Parameters
    ----------
    image:
        [H, W, 3] uint8 image.
    mask:
        [H, W] boolean mask; True = apply correction.
    method:
        ``"median"``: 3×3 median filter.
        ``"nlm"``: Non-local means (slower, better quality).

    Returns
    -------
    np.ndarray
        [H, W, 3] uint8 image with correction applied only inside the mask.
    """
    result = image.copy()
    if not mask.any():
        return result

    if method == "median":
        filtered = cv2.medianBlur(image, ksize=3)
    elif method == "nlm":
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        filtered_bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
        filtered = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unknown correction method: {method!r}")

    # Apply only inside the mask (broadcast mask over channels)
    mask3 = np.stack([mask, mask, mask], axis=2)
    result[mask3] = filtered[mask3]
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_and_correct_artifacts(
    image: np.ndarray,
    cfg: ArtifactsConfig,
) -> tuple[np.ndarray, dict]:
    """
    Run all artifact detectors; optionally correct flagged regions.

    Parameters
    ----------
    image:
        [H, W, 3] uint8 RGB image.
    cfg:
        ArtifactsConfig with detection thresholds and correction toggle.

    Returns
    -------
    (corrected_image, artifact_report_dict)
        ``corrected_image`` is the input image if ``cfg.correct`` is False.
        ``artifact_report_dict`` is a JSON-serialisable dict.
    """
    # --- Before ---
    fft_before, fft_strength_before = detect_checkerboard(image, cfg.fft_peak_threshold)
    smooth_mask_before = detect_over_smooth_mask(image, cfg.local_var_threshold)
    sat_mask_before = detect_over_saturated_mask(image)

    smooth_frac_before = float(smooth_mask_before.mean())
    sat_frac_before = float(sat_mask_before.mean())

    if not cfg.correct:
        report = ArtifactReport(
            fft_peak_count_before=fft_before,
            fft_peak_count_after=fft_before,
            fft_peak_strength_before=fft_strength_before,
            fft_peak_strength_after=fft_strength_before,
            smooth_region_fraction_before=smooth_frac_before,
            smooth_region_fraction_after=smooth_frac_before,
            saturated_pixel_fraction_before=sat_frac_before,
            saturated_pixel_fraction_after=sat_frac_before,
            correction_applied=False,
        )
        import dataclasses

        return image, dataclasses.asdict(report)

    # --- Correction ---
    corrected = image.copy()

    # Correct over-smoothed regions with NLM inside the mask
    if smooth_mask_before.any():
        corrected = correct_inside_mask(corrected, smooth_mask_before, method="nlm")

    # Correct checkerboard / saturation with median filter
    combined_mask = sat_mask_before
    if combined_mask.any():
        corrected = correct_inside_mask(corrected, combined_mask, method="median")

    # --- After ---
    fft_after, fft_strength_after = detect_checkerboard(corrected, cfg.fft_peak_threshold)
    smooth_mask_after = detect_over_smooth_mask(corrected, cfg.local_var_threshold)
    sat_mask_after = detect_over_saturated_mask(corrected)

    report = ArtifactReport(
        fft_peak_count_before=fft_before,
        fft_peak_count_after=fft_after,
        fft_peak_strength_before=fft_strength_before,
        fft_peak_strength_after=fft_strength_after,
        smooth_region_fraction_before=smooth_frac_before,
        smooth_region_fraction_after=float(smooth_mask_after.mean()),
        saturated_pixel_fraction_before=sat_frac_before,
        saturated_pixel_fraction_after=float(sat_mask_after.mean()),
        correction_applied=True,
    )
    import dataclasses

    return corrected, dataclasses.asdict(report)
