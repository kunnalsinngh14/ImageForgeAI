"""
t2i.cv.quality — No-reference image quality metrics.

Computes at least three metrics per image and packages them into a
QualityReport dataclass that is attached to every GenerationResult.

Metrics implemented:
1. Laplacian-variance sharpness.
2. Colourfulness (Hasler & Süsstrunk, 2003).
3. RMS contrast.
4. Histogram-entropy.
5. BRISQUE / NIQE via pyiqa (optional — skipped gracefully if unavailable).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try importing pyiqa for BRISQUE/NIQE
try:
    import pyiqa  # type: ignore

    _HAS_PYIQA = True
except ImportError:
    _HAS_PYIQA = False
    logger.warning("pyiqa not installed — BRISQUE/NIQE will be None in QualityReport.")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class QualityReport:
    """No-reference quality metrics for one image."""

    # 1. Laplacian variance — higher → sharper.
    sharpness_laplacian: float

    # 2. Colourfulness (Hasler & Süsstrunk) — higher → more colourful.
    colourfulness: float

    # 3. RMS contrast (global) — higher → more contrast.
    rms_contrast: float

    # 4. Histogram entropy (Y channel, 256 bins) — higher → more complex texture.
    entropy: float

    # 5. No-reference metric (BRISQUE or NIQE). Lower BRISQUE → better quality.
    niqe_or_brisque: Optional[float] = None
    niqe_or_brisque_metric: Optional[str] = None  # "brisque" | "niqe" | None


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def laplacian_sharpness(image: np.ndarray) -> float:
    """
    Compute sharpness as the variance of the Laplacian of the grayscale image.

    A higher value indicates a sharper image with more high-frequency content.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def colourfulness(image: np.ndarray) -> float:
    """
    Colourfulness metric (Hasler & Süsstrunk, 2003).

    Computed on the rg and yb opponent colour channels.
    C = sqrt(σ_rgyb² + μ_rgyb²) where σ and μ are std and mean
    of the combined opponent-channel magnitude.
    """
    img_f = image.astype(np.float32)
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    rg = r - g
    yb = 0.5 * (r + g) - b

    sigma_rgyb = math.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mu_rgyb = math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(sigma_rgyb + 0.3 * mu_rgyb)


def rms_contrast(image: np.ndarray) -> float:
    """
    Root-mean-square contrast of the grayscale image, normalised to [0, 1].

    RMS = std(I) / 127.5 so that a uniform image scores 0 and a black/white
    image scores approximately 1.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(gray.std() / 127.5)


def histogram_entropy(image: np.ndarray, bins: int = 256) -> float:
    """
    Shannon entropy of the grayscale histogram.

    Higher entropy → more uniform histogram → richer texture.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist, _ = np.histogram(gray.flatten(), bins=bins, range=(0, 256))
    # Normalise to a probability distribution
    prob = hist / hist.sum()
    prob = prob[prob > 0]  # avoid log(0)
    return float(-np.sum(prob * np.log2(prob)))


def compute_niqe_brisque(image: np.ndarray) -> tuple[Optional[float], Optional[str]]:
    """
    Compute BRISQUE (preferred) or NIQE via pyiqa.

    Returns (score, metric_name).  Returns (None, None) when pyiqa is unavailable.
    """
    if not _HAS_PYIQA:
        return None, None

    import torch
    from PIL import Image as PILImage

    try:
        # pyiqa expects a PIL image or a [0,1] float tensor [1, 3, H, W]
        pil_img = PILImage.fromarray(image)
        metric_name = "brisque"
        metric_fn = pyiqa.create_metric(metric_name, device=torch.device("cpu"))
        score = metric_fn(pil_img).item()
        return float(score), metric_name
    except Exception as exc:
        logger.warning("BRISQUE computation failed: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def compute_quality_report(image: np.ndarray) -> QualityReport:
    """
    Compute all quality metrics for one image.

    Parameters
    ----------
    image:
        [H, W, 3] uint8 RGB image.

    Returns
    -------
    QualityReport
    """
    score_niqe, metric_name = compute_niqe_brisque(image)
    return QualityReport(
        sharpness_laplacian=laplacian_sharpness(image),
        colourfulness=colourfulness(image),
        rms_contrast=rms_contrast(image),
        entropy=histogram_entropy(image),
        niqe_or_brisque=score_niqe,
        niqe_or_brisque_metric=metric_name,
    )
