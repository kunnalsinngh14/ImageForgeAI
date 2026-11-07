"""
t2i.pipeline.denoiser — Explicit U-Net denoising loop with scheduler support.

Implements the reverse diffusion process for SD v1.5 step-by-step so every
operation can be inspected and explained.

Supported schedulers: DDIM, DPM-Solver++, LCM.
Classifier-free guidance (CFG) is applied explicitly after splitting the
U-Net's batch output.

Tensor shapes (512×512 image, batch=1):
  latents:    [1, 4, 64, 64]    (4-channel latent, 64 = 512 / 8)
  latents_in: [2, 4, 64, 64]    (CFG: duplicated; 1 for no-CFG)
  noise_pred: [2, 4, 64, 64]    (U-Net output for CFG batch)
  noise_guided:[1, 4, 64, 64]   (after guidance application)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, LCMScheduler
from diffusers.models import UNet2DConditionModel

logger = logging.getLogger(__name__)

# Type alias for the optional step callback.
# Signature: (step_index: int, latents: Tensor) -> bool
# Return True to stop early; False to continue.
StepCallback = Callable[[int, torch.Tensor], bool]

# Map config scheduler name → diffusers class
_SCHEDULER_MAP = {
    "ddim": DDIMScheduler,
    "dpm_solver_pp": DPMSolverMultistepScheduler,
    "lcm": LCMScheduler,
}


@dataclass
class DenoiseResult:
    """Return value of :func:`Denoiser.denoise`."""

    # Final denoised latents. Shape: [1, 4, H/8, W/8]
    latents: torch.Tensor

    # Number of denoising steps actually executed (may be < config steps if stopped early).
    steps_executed: int

    # True if the step callback triggered early stopping.
    early_stopped: bool

    # Per-step latency in seconds (len == steps_executed).
    step_times_s: list[float] = field(default_factory=list)


class Denoiser:
    """
    Wraps the U-Net and scheduler to perform the reverse diffusion loop.

    Parameters
    ----------
    unet:
        UNet2DConditionModel loaded from the SD v1.5 checkpoint.
    scheduler_name:
        One of ``"ddim"``, ``"dpm_solver_pp"``, or ``"lcm"``.
    num_inference_steps:
        Number of denoising steps.
    device:
        Target device.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler_name: str,
        num_inference_steps: int,
        device: str | torch.device,
        scheduler_config: Optional[dict] = None,
    ) -> None:
        self.unet = unet
        self.device = torch.device(device)
        self.num_inference_steps = num_inference_steps

        # Build the scheduler from the config name.
        scheduler_cls = _SCHEDULER_MAP.get(scheduler_name)
        if scheduler_cls is None:
            raise ValueError(
                f"Unknown scheduler '{scheduler_name}'. " f"Choose from: {list(_SCHEDULER_MAP)}"
            )
        if scheduler_config:
            self.scheduler = scheduler_cls.from_config(scheduler_config)
        else:
            self.scheduler = scheduler_cls()
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def denoise(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        guidance_scale: float,
        seed: int,
        step_callback: Optional[StepCallback] = None,
    ) -> DenoiseResult:
        """
        Run the full denoising loop.

        Steps per timestep:
        1. Duplicate latents along batch dim for CFG (skip for no-CFG).
        2. Scale the model input by the scheduler's noise sigma.
        3. Forward pass through the U-Net with timestep + text embeddings.
        4. Split output into uncond and cond predictions.
        5. Apply guidance: guided = uncond + scale * (cond - uncond).
        6. Step the scheduler to get the denoised latents for the next step.

        Parameters
        ----------
        latents:
            Initial noisy latents. Shape: [1, 4, H/8, W/8].
        text_embeddings:
            CFG embedding batch from the prompt encoder.
            Shape: [2, 77, 768] for CFG, [1, 77, 768] for no-CFG.
        guidance_scale:
            CFG scale. When == 1.0, the unconditional branch is skipped.
        seed:
            Recorded in the result for traceability (not re-applied here).
        step_callback:
            Optional callable(step_index, latents) → bool.
            Return True to stop the loop early.

        Returns
        -------
        DenoiseResult
        """
        import time

        cfg_enabled: bool = guidance_scale != 1.0
        # Determine batch size fed to the U-Net.
        # CFG: 2 (uncond + cond). No-CFG: 1 (cond only).
        _unet_batch_size = 2 if cfg_enabled else 1

        step_times: list[float] = []
        early_stopped = False

        with torch.inference_mode():
            for i, t in enumerate(self.scheduler.timesteps):
                t_start = time.perf_counter()

                # --- Step 1 & 2: Scale model input by the scheduler ---
                unet_dtype = getattr(self.unet, "dtype", torch.float32)
                latent_single = self.scheduler.scale_model_input(latents, t).to(dtype=unet_dtype)
                text_embeddings = text_embeddings.to(dtype=unet_dtype)

                # --- Step 3, 4 & 5: U-Net forward pass & Guidance ---
                if cfg_enabled:
                    # Sequential evaluation (batch=1) halves peak activation memory to stay well within 4GB VRAM
                    emb_uncond, emb_cond = text_embeddings.chunk(2, dim=0)
                    noise_pred_uncond = self.unet(
                        latent_single,
                        t,
                        encoder_hidden_states=emb_uncond,
                        return_dict=False,
                    )[0]
                    noise_pred_cond = self.unet(
                        latent_single,
                        t,
                        encoder_hidden_states=emb_cond,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = self.unet(
                        latent_single,
                        t,
                        encoder_hidden_states=text_embeddings,
                        return_dict=False,
                    )[0]

                # --- Step 6: Scheduler step ---
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                step_times.append(time.perf_counter() - t_start)

                # --- Optional early-stop callback ---
                if step_callback is not None and step_callback(i, latents):
                    logger.info(
                        "Denoiser early-stopped at step %d / %d",
                        i + 1,
                        len(self.scheduler.timesteps),
                    )
                    early_stopped = True
                    break

        return DenoiseResult(
            latents=latents,
            steps_executed=len(step_times),
            early_stopped=early_stopped,
            step_times_s=step_times,
        )

    @staticmethod
    def initialize_latents(
        batch_size: int,
        latent_channels: int,
        latent_height: int,
        latent_width: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
        scheduler,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Sample initial Gaussian noise and scale by the scheduler's init sigma.

        The seed is applied to the global RNG via ``torch.manual_seed`` by the
        caller (``generate.py``) before this function is invoked, so that the
        explicit pipeline matches the diffusers reference pipeline, which draws
        its initial noise from the global RNG when no ``generator`` is passed.

        Parameters
        ----------
        batch_size, latent_channels, latent_height, latent_width:
            Shape components of the latent tensor.
        dtype:
            Floating-point dtype.
        device:
            Target device.
        seed:
            Torch manual seed for deterministic initialisation (recorded for
            traceability; the RNG must already be seeded by the caller).
        scheduler:
            The diffusers scheduler (used to get ``init_noise_sigma``).
        generator:
            Optional explicit generator.  When ``None`` (default), the global
            RNG is used, matching the diffusers reference pipeline.

        Returns
        -------
        torch.Tensor
            Shape [batch_size, latent_channels, latent_height, latent_width],
            scaled by ``scheduler.init_noise_sigma``.
        """
        latents = torch.randn(
            (batch_size, latent_channels, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        # Scale by init_noise_sigma so the magnitude matches the scheduler's
        # expected starting noise level.  For DDIM this is 1.0; for
        # DPM-Solver++ and LCM it may differ.
        latents = latents * scheduler.init_noise_sigma
        return latents


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def load_denoiser(
    model_id: str,
    revision: str,
    device: str | torch.device,
    dtype: torch.dtype,
    scheduler_name: str,
    num_inference_steps: int,
    variant: str = "full_unet",
    unet_model_id: Optional[str] = None,
) -> Denoiser:
    """
    Load U-Net and return a Denoiser.

    Parameters
    ----------
    model_id:
        SD v1.5 Hub id (for the full U-Net).
    unet_model_id:
        Alternative Hub id for the block-pruned variant (BK-SDM).
    variant:
        ``"full_unet"`` or ``"block_pruned"``.
    """
    if variant == "dreamshaper_v8":
        effective_model_id = "models/dreamshaper-v8"
    elif variant == "block_pruned" and unet_model_id:
        effective_model_id = unet_model_id
    else:
        effective_model_id = model_id
    if variant == "block_pruned":
        from t2i.optimize.prune import load_pruned_unet

        unet = load_pruned_unet(
            variant="bk-sdm-small", custom_model_id=effective_model_id, device=device, dtype=dtype
        )
        return Denoiser(unet, scheduler_name, num_inference_steps, device)

    from pathlib import Path
    import json

    p = Path(effective_model_id)
    if (p / "config.json").exists():
        subfolder = None
        target_dir = p
    elif (p / "unet" / "config.json").exists():
        subfolder = "unet"
        target_dir = p / "unet"
    else:
        subfolder = "unet"
        target_dir = p

    # Memory-safe loading if local config and weights exist
    weights_fp16 = target_dir / "diffusion_pytorch_model.fp16.safetensors"
    weights_fp32 = target_dir / "diffusion_pytorch_model.safetensors"
    cfg_file = target_dir / "config.json"

    if cfg_file.exists() and (weights_fp16.exists() or weights_fp32.exists()):
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        for k in ["_name_or_path", "_class_name", "_diffusers_version"]:
            cfg_dict.pop(k, None)

        # Select weight file: always prefer fp16 (1.71 GB) to eliminate paging file errors and avoid memory duplication
        if weights_fp16.exists():
            w_file = weights_fp16
        elif weights_fp32.exists():
            w_file = weights_fp32
        else:
            w_file = weights_fp16

        logger.info("Loading U-Net from %s using streaming meta-device allocation...", w_file)
        with torch.device("meta"):
            unet = UNet2DConditionModel(**cfg_dict)
        from safetensors import safe_open
        from accelerate.utils import set_module_tensor_to_device

        target_dev = str(device)
        with safe_open(str(w_file), framework="numpy") as f:
            for name in f.keys():
                t = torch.from_numpy(f.get_tensor(name))
                set_module_tensor_to_device(
                    unet,
                    tensor_name=name,
                    device=target_dev,
                    value=t,
                    dtype=dtype,
                )
    else:
        use_variant = (
            "fp16" if (target_dir / "diffusion_pytorch_model.fp16.safetensors").exists() else None
        )
        unet = UNet2DConditionModel.from_pretrained(
            effective_model_id,
            subfolder=subfolder,
            variant=use_variant,
            revision=revision if effective_model_id == model_id else None,
            torch_dtype=dtype,
        ).to(device)
    unet.eval()
    return Denoiser(unet, scheduler_name, num_inference_steps, device)
