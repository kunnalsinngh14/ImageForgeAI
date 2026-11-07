"""
t2i.pipeline.decoder — VAE latent-to-pixel decoding for SD v1.5.

Scales the denoised latents by the VAE scaling factor, runs the VAE decoder,
and converts the floating-point output to uint8 RGB.

Tensor shapes:
  latents (input):  [1, 4, H/8, W/8]   (e.g. [1, 4, 64, 64] for 512×512)
  vae output:       [1, 3, H, W]        in range [-1, 1]
  image (output):   [H, W, 3]           uint8 NumPy array
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from diffusers.models import AutoencoderKL

logger = logging.getLogger(__name__)

# SD v1.5 VAE scaling factor.  Latents are divided by this before decoding
# because the VAE was trained with this normalisation applied to its latent space.
VAE_SCALING_FACTOR: float = 0.18215


class Decoder:
    """
    Wraps the VAE decoder for SD v1.5.

    Parameters
    ----------
    vae:
        AutoencoderKL loaded from the SD v1.5 checkpoint.
    device:
        Target device.
    enable_tiling:
        If True, calls ``vae.enable_tiling()`` to process the image in tiles,
        reducing peak memory at the cost of slightly more time. Toggled from
        ``configs/*.yaml`` via the ``vae_tiling`` field.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        device: str | torch.device,
        enable_tiling: bool = False,
    ) -> None:
        self.vae = vae
        self.device = torch.device(device)
        if enable_tiling:
            self.vae.enable_tiling()
            if hasattr(self.vae, "tile_latent_min_size"):
                self.vae.tile_latent_min_size = 32
            if hasattr(self.vae, "tile_sample_min_size"):
                self.vae.tile_sample_min_size = 256
            logger.debug("VAE tiling enabled.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(self, latents: torch.Tensor) -> np.ndarray:
        """
        Decode latents to a uint8 RGB image.

        Steps:
        1. Scale latents: ``latents / VAE_SCALING_FACTOR``.
           The VAE encoder divided by this factor; the decoder expects the
           un-scaled values.
        2. Run ``vae.decode`` to get a [1, 3, H, W] float tensor in [-1, 1].
        3. Clamp, shift, and scale to [0, 255].
        4. Convert to uint8 NumPy [H, W, 3] in RGB order.

        Parameters
        ----------
        latents:
            Denoised latents from the denoiser. Shape: [1, 4, H/8, W/8].

        Returns
        -------
        np.ndarray
            RGB image, shape [H, W, 3], dtype uint8.
        """
        with torch.inference_mode():
            # --- Step 1: Undo the VAE latent normalisation ---
            # Shape stays [1, 4, H/8, W/8]
            dtype = getattr(self.vae, "dtype", torch.float32)
            scaled_latents = (latents / VAE_SCALING_FACTOR).to(dtype=dtype)

            # --- Step 2: Decode ---
            # vae.decode returns a DecoderOutput; .sample is the image tensor.
            # Shape: [1, 3, H, W],  range: approximately [-1, 1]
            try:
                decoded = self.vae.decode(scaled_latents, return_dict=False)[0]
            except RuntimeError as exc:
                if (
                    "DefaultCPUAllocator" in str(exc)
                    or "out of memory" in str(exc)
                    or "allocate" in str(exc)
                ):
                    logger.warning(
                        "Standard VAE decode ran out of memory (%s). Retrying with VAE tiling enabled...",
                        exc,
                    )
                    self.vae.enable_tiling()
                    if hasattr(self.vae, "tile_latent_min_size"):
                        self.vae.tile_latent_min_size = 32
                    if hasattr(self.vae, "tile_sample_min_size"):
                        self.vae.tile_sample_min_size = 256
                    decoded = self.vae.decode(scaled_latents, return_dict=False)[0]
                else:
                    raise

        # --- Step 3: Convert from [-1, 1] to [0, 255] ---
        # Clamp first to avoid artefacts from values slightly outside [-1, 1].
        image_float = decoded.clamp(-1.0, 1.0)  # [1, 3, H, W] in [-1, 1]
        image_float = (image_float + 1.0) / 2.0  # [0, 1]
        image_float = image_float * 255.0  # [0, 255]

        # --- Step 4: To NumPy uint8 [H, W, 3] ---
        # Move to CPU, remove batch dim, permute C×H×W → H×W×C.
        image_np: np.ndarray = (
            image_float[0]  # [3, H, W]
            .permute(1, 2, 0)  # [H, W, 3]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        return image_np


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def load_decoder(
    model_id: str,
    revision: str,
    device: str | torch.device,
    dtype: torch.dtype,
    enable_tiling: bool = False,
) -> Decoder:
    """
    Load the AutoencoderKL (VAE) and return a Decoder.

    Parameters
    ----------
    model_id:
        SD v1.5 Hub id.
    revision:
        Pinned git commit hash.
    device:
        Target device.
    dtype:
        torch dtype for the VAE weights.
    enable_tiling:
        Toggle VAE tiling for lower peak memory.
    """
    from pathlib import Path
    import json
    import safetensors.torch

    p = Path(model_id) / "vae"
    cfg_file = p / "config.json"
    w_fp16 = p / "diffusion_pytorch_model.fp16.safetensors"
    w_fp32 = p / "diffusion_pytorch_model.safetensors"

    if cfg_file.exists() and (w_fp16.exists() or w_fp32.exists()):
        w_file = w_fp16 if w_fp16.exists() else w_fp32
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        for k in ["_name_or_path", "_class_name", "_diffusers_version"]:
            cfg_dict.pop(k, None)
        vae = AutoencoderKL(**cfg_dict)
        sd = safetensors.torch.load_file(str(w_file), device="cpu")
        vae.load_state_dict(sd, strict=False, assign=True)
        del sd
        vae = vae.to(device)
        if str(device) == "cpu" and vae.dtype == torch.float16:
            vae = vae.to(torch.float32)
    else:
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            revision=revision,
            torch_dtype=dtype,
        ).to(device)
    vae.eval()
    return Decoder(vae, device, enable_tiling=enable_tiling)
