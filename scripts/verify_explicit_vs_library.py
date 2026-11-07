"""
scripts/verify_explicit_vs_library.py — M1 Verification Script.

Runs the diffusers StableDiffusionPipeline (library reference) and our
explicit pipeline (prompt_encoder -> denoiser -> decoder) with the exact same
prompt, seed, scheduler, and step count.

Reports:
  - Max and mean absolute latent difference
  - Max and mean absolute pixel difference
  - Saves both output images side-by-side to results/m1_verification/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

from t2i.config import load_config
from t2i.pipeline.decoder import Decoder, load_decoder
from t2i.pipeline.denoiser import Denoiser, load_denoiser
from t2i.pipeline.prompt_encoder import PromptEncoder, load_prompt_encoder

logger = logging.getLogger(__name__)


def run_verification(
    prompt: str = "a photograph of an astronaut riding a horse",
    seed: int = 42,
    steps: int = 4,
    resolution: tuple[int, int] = (512, 512),
    output_dir: Path = Path("results/m1_verification"),
    device: str = "cpu",
    use_pruned_unet: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    dtype = torch.float32

    logger.info("Running M1 verification on %s (dtype=%s, steps=%d, seed=%d)", device, dtype, steps, seed)

    # 1. Run Library Reference Pipeline
    model_source = "models/stable-diffusion-v1-5" if Path("models/stable-diffusion-v1-5").exists() else "runwayml/stable-diffusion-v1-5"
    logger.info("Loading diffusers reference pipeline from %s...", model_source)

    custom_unet = None
    if use_pruned_unet:
        from t2i.optimize.prune import load_pruned_unet
        logger.info("Loading pruned U-Net (bk-sdm-small) for memory-safe verification...")
        custom_unet = load_pruned_unet("bk-sdm-small", device=device_obj, dtype=dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_source,
        unet=custom_unet,
        feature_extractor=None,
        safety_checker=None,
        torch_dtype=dtype,
        local_files_only=Path(model_source).exists(),
    ).to(device_obj)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    generator_ref = torch.Generator(device=device_obj).manual_seed(seed)
    t0 = time.perf_counter()
    ref_output = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=7.5,
        generator=generator_ref,
        output_type="latent",
    )
    ref_time = time.perf_counter() - t0
    ref_latents = ref_output.images

    # Extract VAE and components, then delete the library pipeline to reclaim RAM
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    scheduler_cfg = dict(pipe.scheduler.config)

    # 2. Run Explicit Pipeline
    logger.info("Running explicit pipeline...")
    prompt_encoder = PromptEncoder(tokenizer, text_encoder, device=device_obj)
    denoiser = Denoiser(unet, "ddim", num_inference_steps=steps, device=device_obj, scheduler_config=scheduler_cfg)

    t0 = time.perf_counter()
    enc_res = prompt_encoder.encode_prompt(prompt, negative_prompt="", guidance_scale=7.5)
    generator_exp = torch.Generator(device=device_obj).manual_seed(seed)
    latents_init = Denoiser.initialize_latents(
        batch_size=1,
        latent_channels=4,
        latent_height=resolution[1] // 8,
        latent_width=resolution[0] // 8,
        dtype=dtype,
        device=device_obj,
        seed=seed,
        scheduler=denoiser.scheduler,
        generator=generator_exp,
    )
    denoise_res = denoiser.denoise(latents_init, enc_res.embeddings, guidance_scale=7.5, seed=seed)
    exp_time = time.perf_counter() - t0

    # 3. Compute Latent Differences
    latent_diff = np.abs(ref_latents.cpu().numpy() - denoise_res.latents.cpu().numpy())
    max_latent_diff = float(latent_diff.max())
    mean_latent_diff = float(latent_diff.mean())

    # Reclaim denoiser & encoder memory before VAE decode
    del pipe, ref_output, custom_unet, prompt_encoder, denoiser, unet, text_encoder, enc_res, latents_init
    import gc
    gc.collect()

    decoder = Decoder(vae, device=device_obj, enable_tiling=True)
    ref_image_np = decoder.decode(ref_latents)
    exp_image_np = decoder.decode(denoise_res.latents)

    # 4. Compute Pixel Differences
    pixel_diff = np.abs(ref_image_np.astype(np.float32) - exp_image_np.astype(np.float32))
    max_pixel_diff = float(pixel_diff.max())
    mean_pixel_diff = float(pixel_diff.mean())

    # Save images
    Image.fromarray(ref_image_np).save(output_dir / "reference_library.png")
    Image.fromarray(exp_image_np).save(output_dir / "explicit_pipeline.png")

    report = {
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "resolution": list(resolution),
        "device": str(device),
        "library_pipeline_latency_s": round(ref_time, 3),
        "explicit_pipeline_latency_s": round(exp_time, 3),
        "max_latent_difference": round(max_latent_diff, 6),
        "mean_latent_difference": round(mean_latent_diff, 6),
        "max_pixel_difference_0_to_255": round(max_pixel_diff, 4),
        "mean_pixel_difference_0_to_255": round(mean_pixel_diff, 4),
        "images_match_within_tolerance": mean_pixel_diff < 5.0,
    }

    with open(output_dir / "verification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Verification Complete: max_diff=%.2f, mean_diff=%.4f", max_pixel_diff, mean_pixel_diff)
    return report


def main():
    p = argparse.ArgumentParser(description="M1 Pipeline verification.")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    rep = run_verification(steps=args.steps, seed=args.seed, device=args.device)
    print("\n--- M1 Verification Report ---")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
