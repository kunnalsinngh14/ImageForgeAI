"""
t2i.optimize.prune — Block-pruned U-Net (BK-SDM variants) loader.

Loads a block-pruned distilled U-Net from Hugging Face Hub and verifies that it
accepts the same inputs and produces the same output shape as the full U-Net.

The block-pruned variants (BK-SDM-Small, BK-SDM-Tiny, etc.) were produced by
removing residual blocks from the SD v1.5 U-Net and distilling the pruned model,
yielding smaller and faster models with some quality degradation.

Reference: https://huggingface.co/nota-ai/bk-sdm-small
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from diffusers.models import UNet2DConditionModel

logger = logging.getLogger(__name__)

# Default BK-SDM hub IDs
_BKSDM_SMALL = "nota-ai/bk-sdm-small"
_BKSDM_TINY = "nota-ai/bk-sdm-tiny"
_BKSDM_BASE = "nota-ai/bk-sdm-base"

VARIANT_MAP = {
    "bk-sdm-small": _BKSDM_SMALL,
    "bk-sdm-tiny": _BKSDM_TINY,
    "bk-sdm-base": _BKSDM_BASE,
}


def load_pruned_unet(
    variant: str = "bk-sdm-small",
    revision: Optional[str] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    custom_model_id: Optional[str] = None,
) -> UNet2DConditionModel:
    """
    Load a block-pruned distilled U-Net from Hugging Face Hub.

    Parameters
    ----------
    variant:
        One of ``"bk-sdm-small"``, ``"bk-sdm-tiny"``, ``"bk-sdm-base"``
        (or any key in ``VARIANT_MAP``).
    revision:
        Optional pinned git commit hash for the Hub model.
    device:
        Target device.
    dtype:
        torch dtype for the model weights.
    custom_model_id:
        Override the Hub model ID directly (ignores ``variant``).

    Returns
    -------
    UNet2DConditionModel
        Loaded and evaluated pruned U-Net.
    """
    model_id = custom_model_id or VARIANT_MAP.get(variant)
    if model_id is None:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {list(VARIANT_MAP)} "
            "or set custom_model_id."
        )

    # Check local models directory first
    from pathlib import Path

    local_path = Path("models") / variant
    if (local_path / "unet").is_dir() and (local_path / "unet" / "config.json").exists():
        load_target = str(local_path / "unet")
        subfolder = None
    elif local_path.is_dir() and (local_path / "config.json").exists():
        load_target = str(local_path)
        subfolder = None
    else:
        load_target = model_id
        subfolder = "unet" if "bk-sdm" in model_id else None

    logger.info(
        "Loading pruned U-Net from %s (subfolder=%s, revision=%s)…",
        load_target,
        subfolder,
        revision,
    )
    cfg_path = Path(load_target) / "config.json"
    weights_fp32 = Path(load_target) / "diffusion_pytorch_model.safetensors"
    weights_fp16 = Path(load_target) / "diffusion_pytorch_model.fp16.safetensors"
    # Choose weights matching device and dtype
    if str(device).startswith("cpu") and dtype == torch.float32 and weights_fp32.exists():
        weights_path = weights_fp32
    elif weights_fp16.exists():
        weights_path = weights_fp16
    else:
        weights_path = weights_fp32

    if cfg_path.exists() and weights_path.exists():
        import json
        import safetensors.torch

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        for k in ["_name_or_path", "_class_name", "_diffusers_version"]:
            cfg_dict.pop(k, None)
        with torch.device("meta"):
            unet = UNet2DConditionModel(**cfg_dict)
        sd = safetensors.torch.load_file(str(weights_path), device=str(device))
        unet.load_state_dict(sd, strict=False, assign=True)
        del sd
        if str(device) != "cpu" and unet.device != torch.device(device):
            unet = unet.to(device)
    elif subfolder:
        unet = UNet2DConditionModel.from_pretrained(
            load_target,
            subfolder=subfolder,
            revision=revision,
            torch_dtype=dtype,
        ).to(device)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            load_target,
            revision=revision,
            torch_dtype=dtype,
        ).to(device)
    unet.eval()
    logger.info(
        "Pruned U-Net loaded. Parameter count: %s M",
        f"{sum(p.numel() for p in unet.parameters()) / 1e6:.1f}",
    )
    return unet


def verify_unet_compatibility(
    full_unet: UNet2DConditionModel,
    pruned_unet: UNet2DConditionModel,
    batch_size: int = 1,
    latent_channels: int = 4,
    latent_h: int = 64,
    latent_w: int = 64,
    seq_len: int = 77,
    hidden_dim: int = 768,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Verify that the pruned U-Net accepts the same inputs and produces the same
    output *shape* as the full U-Net.

    Runs both models with the same random inputs and reports:
    - Whether output shapes match.
    - Max and mean absolute difference (since the models differ in weights,
      this is not expected to be small — shape compatibility is what matters).

    Parameters
    ----------
    full_unet, pruned_unet:
        The two U-Net instances to compare.
    batch_size, latent_channels, latent_h, latent_w:
        Latent tensor dimensions.
    seq_len, hidden_dim:
        Text embedding dimensions (77, 768 for CLIP ViT-L/14).
    device, dtype:
        Run the check on this device with this dtype.

    Returns
    -------
    dict
        Compatibility report with keys:
        - ``shapes_match`` (bool)
        - ``full_shape``, ``pruned_shape`` (tuples)
        - ``max_abs_diff``, ``mean_abs_diff`` (float) — output value difference.
    """
    # Build dummy inputs
    # CFG batch: 2 × latent for each U-Net call
    latents = torch.randn(
        batch_size, latent_channels, latent_h, latent_w, dtype=dtype, device=device
    )
    # Duplicate for CFG
    latent_input = torch.cat([latents, latents], dim=0)
    t = torch.tensor([500], dtype=torch.long, device=device)
    embeddings = torch.randn(2, seq_len, hidden_dim, dtype=dtype, device=device)

    with torch.inference_mode():
        out_full = full_unet(latent_input, t, encoder_hidden_states=embeddings, return_dict=False)[
            0
        ]
        out_pruned = pruned_unet(
            latent_input, t, encoder_hidden_states=embeddings, return_dict=False
        )[0]

    shapes_match = out_full.shape == out_pruned.shape
    if shapes_match:
        diff = (out_full - out_pruned).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
    else:
        max_diff = float("nan")
        mean_diff = float("nan")

    report = {
        "shapes_match": shapes_match,
        "full_shape": tuple(out_full.shape),
        "pruned_shape": tuple(out_pruned.shape),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
    }
    logger.info("U-Net compatibility check: %s", report)
    return report
