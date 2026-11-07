#!/usr/bin/env python3
"""
scripts/capture_env.py — Capture hardware and software environment info
and write results/env.json.

Run this once at the start of M0 to establish the environment record.
Every subsequent results file should include the same hardware info.

Usage::

    python scripts/capture_env.py --output results/env.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path

import psutil
import torch

logger = logging.getLogger(__name__)


def capture_env() -> dict:
    """Collect full hardware and software environment information."""
    # GPU info
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "compute_capability": f"{props.major}.{props.minor}",
            })

    # Python packages
    packages = {}
    for pkg in [
        "torch", "diffusers", "transformers", "accelerate", "safetensors",
        "optimum", "onnxruntime", "onnx", "opencv-python-headless",
        "numpy", "scikit-image", "Pillow", "open-clip-torch", "torchmetrics",
        "pytorch-fid", "pyiqa", "lpips", "psutil", "fastapi", "uvicorn",
        "gradio", "pandas", "matplotlib", "pydantic", "pyyaml", "huggingface-hub",
    ]:
        try:
            import importlib.metadata
            packages[pkg] = importlib.metadata.version(pkg)
        except Exception:
            packages[pkg] = "not installed"

    # Git commit
    git_hash = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except Exception:
        pass

    return {
        "hardware": {
            "cpu_model": platform.processor() or platform.machine(),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "gpu": gpu_info if gpu_info else "none",
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "software": {
            "os": platform.platform(),
            "os_version": platform.version(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "packages": packages,
        },
        "git": {
            "commit_hash": git_hash,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Capture environment info to JSON.")
    p.add_argument("--output", type=Path, default=Path("results/env.json"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    env = capture_env()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2)

    logger.info("Environment info written to %s", args.output)
    print(f"\nCPU: {env['hardware']['cpu_model']}")
    print(f"Cores: {env['hardware']['cpu_cores_physical']} physical / {env['hardware']['cpu_cores_logical']} logical")
    print(f"RAM: {env['hardware']['ram_gb']} GB")
    print(f"GPU: {env['hardware']['gpu']}")
    print(f"CUDA: {env['hardware']['cuda_version'] or 'N/A'}")
    print(f"PyTorch: {env['software']['packages']['torch']}")
    print(f"Diffusers: {env['software']['packages']['diffusers']}")


if __name__ == "__main__":
    main()
