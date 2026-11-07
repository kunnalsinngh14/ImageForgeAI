"""
t2i.eval.size — On-disk model size reporting.

Sums the sizes of all deployed weight files per component (text encoder,
U-Net, VAE) and reports total storage in MiB.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Mapping of subdirectory names to component labels.
_COMPONENT_DIRS = {
    "text_encoder": "text_encoder",
    "unet": "unet",
    "vae": "vae",
    "tokenizer": "tokenizer",
    "scheduler": "scheduler",
}

_WEIGHT_EXTENSIONS = {".safetensors", ".bin", ".pt", ".onnx", ".tflite"}


def measure_component_sizes(models_dir: Path) -> dict:
    """
    Walk the models directory and report on-disk sizes per component.

    Parameters
    ----------
    models_dir:
        Root directory containing model subdirectories (e.g. ``models/``).

    Returns
    -------
    dict
        ``{"components": {name: size_mb}, "total_mb": float}``
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        logger.warning("Models directory does not exist: %s", models_dir)
        return {"components": {}, "total_mb": 0.0}

    component_sizes: dict[str, float] = {}
    total_bytes = 0

    for item in sorted(models_dir.iterdir()):
        if item.is_dir():
            dir_bytes = sum(
                f.stat().st_size
                for f in item.rglob("*")
                if f.is_file() and f.suffix in _WEIGHT_EXTENSIONS
            )
            if dir_bytes > 0:
                component_sizes[item.name] = round(dir_bytes / (1024**2), 2)
                total_bytes += dir_bytes
        elif item.is_file() and item.suffix in _WEIGHT_EXTENSIONS:
            # Top-level weight files (e.g. ONNX exports)
            component_sizes[item.name] = round(item.stat().st_size / (1024**2), 2)
            total_bytes += item.stat().st_size

    return {
        "components": component_sizes,
        "total_mb": round(total_bytes / (1024**2), 2),
    }


def onnx_export_size_mb(export_dir: Path) -> dict:
    """
    Report the size of ONNX export artifacts.

    Parameters
    ----------
    export_dir:
        Directory containing .onnx files.

    Returns
    -------
    dict
        ``{"files": {filename: size_mb}, "total_mb": float}``
    """
    export_dir = Path(export_dir)
    files = {}
    total = 0
    for f in sorted(export_dir.glob("*.onnx")):
        size = f.stat().st_size
        files[f.name] = round(size / (1024**2), 2)
        total += size
    return {"files": files, "total_mb": round(total / (1024**2), 2)}
