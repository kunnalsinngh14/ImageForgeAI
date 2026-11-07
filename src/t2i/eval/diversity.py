"""
t2i.eval.diversity — LPIPS diversity score across seeds for a fixed prompt.

Measures how diverse the outputs are for a fixed prompt run with different seeds.
High diversity indicates the model explores the prompt space; low diversity
indicates mode collapse (common with low-step distilled models).

Metric: mean pairwise LPIPS across seed pairs.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_lpips_diversity(
    images: list[np.ndarray],
    device: str | torch.device = "cpu",
) -> dict:
    """
    Compute mean pairwise LPIPS diversity score.

    Parameters
    ----------
    images:
        List of [H, W, 3] uint8 RGB images generated for the same prompt
        with different seeds.
    device:
        Target device for LPIPS model.

    Returns
    -------
    dict
        ``{mean_lpips, std_lpips, n_pairs, all_scores}``
    """
    try:
        import lpips  # type: ignore
    except ImportError as exc:
        raise ImportError("lpips is required: pip install lpips") from exc

    loss_fn = lpips.LPIPS(net="alex").to(torch.device(device))
    loss_fn.eval()

    scores: list[float] = []
    pairs = list(combinations(range(len(images)), 2))

    if len(pairs) == 0:
        raise ValueError("Need at least 2 images to compute LPIPS diversity.")

    for i, j in pairs:
        t1 = _to_lpips_tensor(images[i], device)
        t2 = _to_lpips_tensor(images[j], device)
        with torch.inference_mode():
            score = loss_fn(t1, t2).item()
        scores.append(float(score))

    arr = np.array(scores)
    return {
        "mean_lpips": float(arr.mean()),
        "std_lpips": float(arr.std()),
        "n_pairs": len(scores),
        "all_scores": [round(s, 4) for s in scores],
    }


def _to_lpips_tensor(image: np.ndarray, device: str | torch.device) -> torch.Tensor:
    """
    Convert a uint8 RGB NumPy image [H, W, 3] to an LPIPS-compatible tensor.

    LPIPS expects [1, 3, H, W] float32 in [-1, 1].
    """
    img_float = image.astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor.to(torch.device(device))
