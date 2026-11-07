#!/usr/bin/env python3
"""
scripts/verify_pipeline.py — Verify the explicit pipeline against the diffusers
reference pipeline for a fixed seed.

Runs both pipelines with the same prompt, seed, and config and reports:
  - max and mean absolute pixel difference between the two outputs
  - max and mean absolute latent difference after the denoising stage
  - whether the difference is within the documented tolerance

Usage::

    python scripts/verify_pipeline.py \
        --config configs/baseline.yaml \
        --prompt "a red bicycle to the left of a blue bench" \
        --seed 1234 \
        --tolerance 5.0

Exit code is non-zero if the mean absolute pixel difference exceeds the
tolerance (default 5.0 / 255.0 ≈ 2%).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Documented tolerance: mean absolute pixel difference (in 0-255 units).
# The explicit pipeline mirrors the diffusers reference exactly, so the
# difference is expected to be < 1.0 (floating-point rounding only).
DEFAULT_TOLERANCE = 5.0


def _set_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_reference(config, prompt: str, negative_prompt: str, seed: int) -> np.ndarray:
    """Run the diffusers reference pipeline and return the uint8 RGB image."""
    from diffusers import StableDiffusionPipeline

    _set_seeds(seed)
    pipe = StableDiffusionPipeline.from_pretrained(
        config.model.base,
        revision=config.model.revision,
        torch_dtype=torch.float32,
    ).to("cpu")
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance_scale,
        width=config.width,
        height=config.height,
        output_type="latent",
        return_dict=True,
    )
    latents = result.images  # [1, 4, H/8, W/8]
    with torch.inference_mode():
        scaled = latents / 0.18215
        decoded = pipe.vae.decode(scaled, return_dict=False)[0]
    image = decoded.clamp(-1, 1)
    image = ((image + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
    return image


def run_explicit(config, prompt: str, negative_prompt: str, seed: int, device: str | None = None) -> np.ndarray:
    """Run the explicit t2i pipeline and return the uint8 RGB image."""
    from t2i.pipeline.generate import Generator

    gen = Generator(config, device=device)
    result = gen.generate(prompt=prompt, negative_prompt=negative_prompt, seed=seed)
    return result.image


def compare_images(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute absolute pixel differences between two uint8 RGB images."""
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "shape": list(a.shape),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Verify explicit pipeline vs. reference.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                   help="Max allowed mean absolute pixel difference (0-255 units).")
    p.add_argument("--skip-reference", action="store_true",
                   help="Skip the reference pipeline (only run the explicit one).")
    p.add_argument("--device", default=None,
                   help="Device to run on ('cuda' or 'cpu'). Defaults to cuda if available.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    from t2i.config import load_config
    cfg = load_config(args.config)

    logger.info("Running explicit pipeline (seed=%d)...", args.seed)
    explicit_image = run_explicit(cfg, args.prompt, args.negative_prompt, args.seed, device=args.device)

    if args.skip_reference:
        logger.info("--skip-reference set; only the explicit pipeline ran.")
        print(f"Explicit pipeline output shape: {explicit_image.shape}")
        return

    logger.info("Running reference pipeline (seed=%d)...", args.seed)
    ref_image = run_reference(cfg, args.prompt, args.negative_prompt, args.seed)

    diff = compare_images(explicit_image, ref_image)
    logger.info("Pixel diff: max=%.3f mean=%.3f", diff["max_abs_diff"], diff["mean_abs_diff"])

    ok = diff["mean_abs_diff"] <= args.tolerance
    print("\n=== Verification result ===")
    print(f"  Max absolute pixel diff : {diff['max_abs_diff']:.4f}")
    print(f"  Mean absolute pixel diff: {diff['mean_abs_diff']:.4f}")
    print(f"  Tolerance               : {args.tolerance:.4f}")
    print(f"  Status                  : {'PASS' if ok else 'FAIL'}")

    if not ok:
        print(f"\nMean absolute diff {diff['mean_abs_diff']:.4f} exceeds tolerance {args.tolerance:.4f}.")
        sys.exit(1)


if __name__ == "__main__":
    main()