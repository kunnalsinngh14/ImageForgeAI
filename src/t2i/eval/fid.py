"""
t2i.eval.fid — Fréchet Inception Distance computation.

Uses Inception-v3 features (via pytorch-fid or torchmetrics) to compare
distributions of generated and reference images.

Requirements:
  - At least 1,000 generated images.
  - Bootstrap confidence intervals reported.
  - Sanity check: FID between two disjoint real-image sets must be near zero.

Usage::

    python -m t2i.eval.fid \\
        --generated results/fid_run_xxx/images/ \\
        --reference data/fid_subset/images/ \\
        --batch-size 64 \\
        --output results/fid_run_xxx/fid.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_fid_score(
    generated_dir: Path,
    reference_dir: Path,
    batch_size: int = 64,
    device: str = "cpu",
    num_workers: int = 0,
) -> float:
    """
    Compute FID between generated and reference image directories.

    Uses ``pytorch_fid.fid_score.calculate_fid_given_paths`` which runs
    Inception-v3 feature extraction and computes the Fréchet distance.

    Parameters
    ----------
    generated_dir:
        Directory of generated PNG/JPG images.
    reference_dir:
        Directory of reference (real) images.
    batch_size:
        Inception-v3 batch size.
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    float
        FID score. Lower is better.
    """
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths  # type: ignore
    except ImportError as exc:
        raise ImportError("pytorch-fid is required: pip install pytorch-fid") from exc

    paths = [str(generated_dir), str(reference_dir)]
    fid = calculate_fid_given_paths(
        paths,
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=num_workers,
    )
    return float(fid)


def bootstrap_fid_ci(
    generated_dir: Path,
    reference_dir: Path,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.8,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    Estimate FID bootstrap confidence intervals.

    Re-samples ``sample_fraction`` of images in each bootstrap iteration.
    Returns 95% CI (2.5th and 97.5th percentiles).

    Parameters
    ----------
    generated_dir, reference_dir:
        Image directories.
    n_bootstrap:
        Number of bootstrap iterations.
    sample_fraction:
        Fraction of images to sample per iteration.
    Returns
    -------
    dict
        ``{mean, std, ci_low, ci_high, n_bootstrap, sample_fraction}``
    """
    import shutil
    import tempfile

    rng = np.random.default_rng(seed)
    gen_images = sorted(generated_dir.glob("*.png")) + sorted(generated_dir.glob("*.jpg"))
    ref_images = sorted(reference_dir.glob("*.png")) + sorted(reference_dir.glob("*.jpg"))

    if len(gen_images) < 2 or len(ref_images) < 2:
        raise ValueError("Need at least 2 images in each directory for bootstrap FID.")

    fid_values: list[float] = []

    for i in range(n_bootstrap):
        n_gen = max(2, int(len(gen_images) * sample_fraction))
        n_ref = max(2, int(len(ref_images) * sample_fraction))
        sampled_gen = rng.choice(gen_images, size=n_gen, replace=False)
        sampled_ref = rng.choice(ref_images, size=n_ref, replace=False)

        with tempfile.TemporaryDirectory() as tmp_gen, tempfile.TemporaryDirectory() as tmp_ref:
            for img in sampled_gen:
                shutil.copy(img, Path(tmp_gen) / img.name)
            for img in sampled_ref:
                shutil.copy(img, Path(tmp_ref) / img.name)

            try:
                fid = compute_fid_score(Path(tmp_gen), Path(tmp_ref), batch_size, device)
                fid_values.append(fid)
            except Exception as exc:
                logger.warning("Bootstrap iteration %d failed: %s", i, exc)

    arr = np.array(fid_values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci_low_95": float(np.percentile(arr, 2.5)),
        "ci_high_95": float(np.percentile(arr, 97.5)),
        "n_bootstrap": len(fid_values),
        "sample_fraction": sample_fraction,
        "all_fid_values": [round(v, 4) for v in fid_values],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compute FID score.")
    p.add_argument("--generated", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=100)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    fid = compute_fid_score(args.generated, args.reference, args.batch_size, args.device)
    result = {"fid": round(fid, 4), "n_generated": len(list(args.generated.glob("*.png")))}

    if args.bootstrap:
        ci = bootstrap_fid_ci(
            args.generated,
            args.reference,
            args.n_bootstrap,
            batch_size=args.batch_size,
            device=args.device,
        )
        result["bootstrap"] = ci

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"FID: {fid:.2f}  → {args.output}")


if __name__ == "__main__":
    main()
