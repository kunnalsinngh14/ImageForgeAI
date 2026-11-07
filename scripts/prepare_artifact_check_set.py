"""
scripts/prepare_artifact_check_set.py — Create the 30-image artifact check set.

Generates 30 synthetic test images representing real-world failure modes:
  1-10: Clean scenes (geometric, gradients, contrast variations)
  11-15: Checkerboard / periodic high-frequency FFT artifacts
  16-20: Over-smoothed / low-variance regions
  21-25: Over-saturated / clipped RGB channels (0 or 255)
  26-30: Mixed low-quality (blur + low contrast + noise)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def generate_artifact_check_set(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    rng = np.random.default_rng(1234)

    for i in range(1, 31):
        filename = f"artifact_test_{i:03d}.png"
        filepath = output_dir / filename

        # Base image: gradient + shapes
        x = np.linspace(0, 1, 512)
        y = np.linspace(0, 1, 512)
        xx, yy = np.meshgrid(x, y)

        img = np.zeros((512, 512, 3), dtype=np.float32)
        img[:, :, 0] = (xx * 180 + yy * 50)
        img[:, :, 1] = (yy * 200 + (1 - xx) * 40)
        img[:, :, 2] = ((1 - yy) * 160 + xx * 80)

        # Draw geometric features
        cv2.circle(img, (256, 256), 80, (220, 150, 80), -1)
        cv2.rectangle(img, (60, 60), (180, 180), (80, 200, 120), -1)

        if 11 <= i <= 15:
            # Inject checkerboard pattern in luminance
            freq = 4 * (i - 10)
            checker = np.sin(freq * xx * np.pi * 8) * np.sin(freq * yy * np.pi * 8)
            img[:, :, :] += checker[:, :, None] * 80.0

        elif 16 <= i <= 20:
            # Over-smoothed flat patch
            img[150:350, 150:350, :] = 128.0

        elif 21 <= i <= 25:
            # Over-saturated / clipped region
            img[100:300, 100:300, 0] = 300.0  # clipped to 255
            img[100:300, 100:300, 1] = 0.0
            img[300:450, 300:450, 2] = 320.0

        elif 26 <= i <= 30:
            # Blurry + noisy
            img = cv2.GaussianBlur(img, (15, 15), 5.0)
            noise = rng.normal(0, 15, img.shape)
            img += noise

        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(filepath)
        paths.append(filepath)

    return paths


def main():
    p = argparse.ArgumentParser(description="Create the 30-image artifact test set.")
    p.add_argument("--output", default=Path("data/artifact_check_set"), type=Path)
    args = p.parse_args()

    paths = generate_artifact_check_set(args.output)
    print(f"Generated {len(paths)} artifact test images in {args.output}")


if __name__ == "__main__":
    main()
