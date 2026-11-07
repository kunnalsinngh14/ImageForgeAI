#!/usr/bin/env python3
"""
scripts/download_models.py — Download all required model weights from
Hugging Face Hub, verify SHA-256 checksums, and write models/MANIFEST.json.

Usage::

    python scripts/download_models.py [--models-dir models] [--check-only]

This script must be run once before any pipeline or benchmark can run.
All model revisions are pinned in MANIFEST_SPEC below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

# Redirect Hugging Face cache to D: to preserve space on C:
if Path("D:/").exists():
    os.environ.setdefault("HF_HOME", "D:/.cache/huggingface")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model specification — pin revisions here
# ---------------------------------------------------------------------------

MANIFEST_SPEC: list[dict] = [
    {
        "component": "stable-diffusion-v1-5",
        "hub_id": "runwayml/stable-diffusion-v1-5",
        "revision": "main",  # Update to pinned commit hash after first download
        "subfolders": ["text_encoder", "tokenizer", "unet", "vae", "scheduler"],
        "license": "CreativeML OpenRAIL-M",
        "license_url": "https://huggingface.co/spaces/CompVis/stable-diffusion-license",
        "notes": (
            "OpenRAIL-M: restrictions include no generation of harmful or deceptive content. "
            "See docs/licenses/openrail-m.md."
        ),
    },
    {
        "component": "lcm-lora-sdv1-5",
        "hub_id": "latent-consistency/lcm-lora-sdv1-5",
        "revision": "main",  # Update to pinned commit hash after first download
        "subfolders": [],    # Stored at root of the repo
        "license": "OpenRAIL++",
        "license_url": "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5",
        "notes": "LCM-LoRA for SD v1.5 — 2–8 step sampling.",
    },
    {
        "component": "bk-sdm-small",
        "hub_id": "nota-ai/bk-sdm-small",
        "revision": "main",
        "subfolders": [],
        "license": "CreativeML OpenRAIL-M",
        "license_url": "https://huggingface.co/nota-ai/bk-sdm-small",
        "notes": "Block-pruned distilled U-Net variant.",
    },
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def download_component(
    spec: dict,
    models_dir: Path,
    force: bool = False,
) -> dict:
    """
    Download one model component from Hugging Face Hub.

    Returns a manifest entry dict.
    """
    from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore

    hub_id = spec["hub_id"]
    revision = spec["revision"]
    local_dir = models_dir / spec["component"]
    local_dir.mkdir(parents=True, exist_ok=True)

    if spec["component"] == "stable-diffusion-v1-5":
        # Targeted download of exact required files
        files_to_download = [
            ("model_index.json", None),
            ("scheduler/scheduler_config.json", "scheduler"),
            ("tokenizer/tokenizer_config.json", "tokenizer"),
            ("tokenizer/vocab.json", "tokenizer"),
            ("tokenizer/merges.txt", "tokenizer"),
            ("tokenizer/special_tokens_map.json", "tokenizer"),
            ("text_encoder/config.json", "text_encoder"),
            ("text_encoder/model.safetensors", "text_encoder"),
            ("vae/config.json", "vae"),
            ("vae/diffusion_pytorch_model.safetensors", "vae"),
            ("unet/config.json", "unet"),
            ("unet/diffusion_pytorch_model.safetensors", "unet"),
        ]
        for rel_name, subfolder in files_to_download:
            target_file = local_dir / rel_name
            if target_file.exists() and target_file.stat().st_size > 0 and not force:
                logger.info("  Already present: %s", rel_name)
            else:
                logger.info("  Downloading %s from %s...", rel_name, hub_id)
                fname = rel_name.split("/")[-1]
                hf_hub_download(
                    repo_id=hub_id,
                    filename=fname,
                    subfolder=subfolder,
                    local_dir=str(local_dir),
                )
                logger.info("  Downloaded %s", rel_name)

    elif spec["component"] == "lcm-lora-sdv1-5":
        files_to_download = [
            "pytorch_lora_weights.safetensors",
        ]
        for fname in files_to_download:
            target_file = local_dir / fname
            if target_file.exists() and target_file.stat().st_size > 0 and not force:
                logger.info("  Already present: %s", fname)
            else:
                logger.info("  Downloading %s from %s...", fname, hub_id)
                hf_hub_download(
                    repo_id=hub_id,
                    filename=fname,
                    local_dir=str(local_dir),
                )
                logger.info("  Downloaded %s", fname)

    elif spec["component"] == "bk-sdm-small":
        files_to_download = [
            ("config.json", "unet"),
            ("diffusion_pytorch_model.safetensors", "unet"),
        ]
        for fname, subfolder in files_to_download:
            target_file = local_dir / subfolder / fname
            if target_file.exists() and target_file.stat().st_size > 0 and not force:
                logger.info("  Already present: %s/%s", subfolder, fname)
            else:
                logger.info("  Downloading %s from %s/%s...", fname, hub_id, subfolder)
                try:
                    hf_hub_download(
                        repo_id=hub_id,
                        filename=fname,
                        subfolder=subfolder,
                        local_dir=str(local_dir),
                    )
                    logger.info("  Downloaded %s/%s", subfolder, fname)
                except Exception as e:
                    logger.warning("  Could not download %s for %s: %s", fname, hub_id, e)

    # Collect weight files and compute checksums (exclude .incomplete)
    weight_files = [f for f in local_dir.rglob("*.safetensors") if not f.name.endswith(".incomplete")]
    if not weight_files:
        weight_files = [f for f in local_dir.rglob("*.bin") if not f.name.endswith(".incomplete")]

    file_checksums = {}
    for f in sorted(weight_files):
        rel = str(f.relative_to(models_dir))
        file_checksums[rel] = _sha256_file(f)
        logger.debug("  Checksum %s: %s", rel, file_checksums[rel])

    return {
        "component": spec["component"],
        "hub_id": hub_id,
        "revision": revision,
        "local_dir": str(local_dir.relative_to(models_dir)),
        "license": spec["license"],
        "license_url": spec["license_url"],
        "notes": spec["notes"],
        "files": file_checksums,
    }


def write_manifest(manifest: list[dict], models_dir: Path) -> None:
    """Write models/MANIFEST.json."""
    manifest_path = models_dir / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({"models": manifest}, fh, indent=2)
    logger.info("MANIFEST.json written to %s", manifest_path)


def verify_manifest(models_dir: Path) -> bool:
    """Re-compute checksums and compare against MANIFEST.json."""
    manifest_path = models_dir / "MANIFEST.json"
    if not manifest_path.exists():
        logger.error("MANIFEST.json not found at %s", manifest_path)
        return False

    with open(manifest_path) as fh:
        manifest = json.load(fh)["models"]

    all_ok = True
    for entry in manifest:
        for rel_path, expected_sha in entry.get("files", {}).items():
            full_path = models_dir / rel_path
            if not full_path.exists():
                logger.error("MISSING: %s", full_path)
                all_ok = False
                continue
            actual = _sha256_file(full_path)
            if actual != expected_sha:
                logger.error("CHECKSUM MISMATCH: %s (expected %s, got %s)", rel_path, expected_sha, actual)
                all_ok = False
            else:
                logger.debug("OK: %s", rel_path)

    if all_ok:
        logger.info("All checksums verified successfully.")
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Download and verify model weights.")
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--check-only", action="store_true",
                   help="Only verify existing checksums, do not download.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if already present.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)

    if args.check_only:
        ok = verify_manifest(args.models_dir)
        sys.exit(0 if ok else 1)

    manifest = []
    for spec in MANIFEST_SPEC:
        logger.info("Processing: %s", spec["component"])
        try:
            entry = download_component(spec, args.models_dir, force=args.force)
            manifest.append(entry)
        except Exception as exc:
            logger.error("Failed to download %s: %s", spec["component"], exc)
            sys.exit(1)

    write_manifest(manifest, args.models_dir)
    logger.info("All models downloaded successfully.")


if __name__ == "__main__":
    main()
