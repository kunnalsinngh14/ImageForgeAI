"""
t2i.optimize.lcm_lora — LCM-LoRA integration for few-step sampling.

Loads the LCM-LoRA adapter, merges or attaches it to the SD v1.5 U-Net,
and switches the scheduler to LCMScheduler.

LCM-LoRA enables high-quality image generation in 2–8 steps by distilling
the LCM guidance into a LoRA adapter, eliminating the need for CFG at
inference time (guidance_scale = 1.0).

Reference: https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
"""

from __future__ import annotations

import logging
from typing import Optional

from diffusers import LCMScheduler
from diffusers.models import UNet2DConditionModel
from peft import PeftModel  # type: ignore

logger = logging.getLogger(__name__)

# Default LCM-LoRA hub ID for SD v1.5
_DEFAULT_LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"


def attach_lcm_lora(
    unet: UNet2DConditionModel,
    scheduler,
    lora_id: str = _DEFAULT_LCM_LORA_ID,
    lora_revision: Optional[str] = None,
    merge_weights: bool = True,
) -> None:
    """
    Load and attach the LCM-LoRA weights to the U-Net.

    Sets the scheduler to LCMScheduler (required for LCM step logic).

    Parameters
    ----------
    unet:
        The SD v1.5 U-Net instance.
    scheduler:
        The pipeline's current scheduler.  Its class is replaced in-place
        with LCMScheduler using the same config.
    lora_id:
        Hugging Face Hub id of the LCM-LoRA adapter.
    lora_revision:
        Pinned revision / commit hash.
    merge_weights:
        If True, merge LoRA into the base weights for inference efficiency.
        If False, keep them separate (uses PEFT's set_adapter mechanism).
    """
    logger.info("Loading LCM-LoRA from %s (revision=%s)…", lora_id, lora_revision)

    # Load LoRA via PEFT
    unet_peft = PeftModel.from_pretrained(
        unet,
        lora_id,
        revision=lora_revision,
    )

    if merge_weights:
        # Merge LoRA deltas into base weights for a single forward pass.
        unet_peft.merge_adapter()
        logger.info("LCM-LoRA weights merged into U-Net.")
    else:
        logger.info("LCM-LoRA attached (not merged).")

    # Switch scheduler to LCM
    _switch_to_lcm_scheduler(scheduler)


def _switch_to_lcm_scheduler(scheduler) -> None:
    """
    Replace the scheduler's internals with LCMScheduler config.

    diffusers schedulers expose ``scheduler.config`` as a FrozenDict-like
    object; we build a fresh LCMScheduler from that config.
    """
    try:
        lcm_config = dict(scheduler.config)
        # LCMScheduler may not support all keys from DDIM — keep only valid ones.
        valid_keys = {
            "num_train_timesteps",
            "beta_start",
            "beta_end",
            "beta_schedule",
            "trained_betas",
            "clip_sample",
            "set_alpha_to_one",
            "steps_offset",
        }
        lcm_config = {k: v for k, v in lcm_config.items() if k in valid_keys}
        new_scheduler = LCMScheduler(**lcm_config)
        # Copy the attributes over
        scheduler.__class__ = LCMScheduler
        scheduler.__dict__.update(new_scheduler.__dict__)
        logger.info("Scheduler switched to LCMScheduler.")
    except Exception as exc:
        logger.warning("Could not switch scheduler to LCM: %s. Using default LCMScheduler.", exc)
        scheduler.__class__ = LCMScheduler
        new_scheduler = LCMScheduler()
        scheduler.__dict__.update(new_scheduler.__dict__)
