"""
t2i.pipeline.generate — End-to-end generation orchestrator.

Loads models from the Hub (or cache), runs:
  1. Prompt encoding (CLIP)
  2. Latent initialisation + denoising (U-Net + scheduler)
  3. VAE decoding
  4. Optional classical post-processing (CV modules)
  5. Quality reporting
  6. Metrics logging to a JSON file under results/

Returns a GenerationResult that is fully self-describing (seed, config hash,
per-stage timings, peak memory, truncation flag, fallback log).

CLI usage::

    python -m t2i.pipeline.generate \\
        --config configs/optimized.yaml \\
        --prompt "a red bicycle to the left of a blue bench" \\
        --seed 1234 \\
        --output results/

"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch
from PIL import Image

from t2i.config import T2IConfig, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Full record of one generation run."""

    image: np.ndarray  # [H, W, 3] uint8 RGB
    seed: int
    config_hash: str
    prompt: str
    negative_prompt: str

    # Per-stage wall-clock times in seconds
    time_encode_s: float
    time_denoise_s: float
    time_decode_s: float
    time_postprocess_s: float

    # Steps actually executed (may be < config.steps if early-stopped)
    steps_executed: int
    early_stopped: bool

    # Peak RSS in MiB (CPU) or GPU memory in MiB
    peak_memory_mib: float
    memory_type: str  # "cpu_rss" | "gpu_vram"

    # Truncation metadata from the prompt encoder
    prompt_truncated: bool
    prompt_dropped_tokens: int

    # Optional quality report from cv.quality
    quality_report: Optional[dict] = None

    # Optional image before post-processing (for before/after comparison)
    image_before_postprocess: Optional[np.ndarray] = None

    # Fallback log entries (from adaptive/fallback.py)
    fallback_log: list[dict] = field(default_factory=list)

    # Hardware / software context recorded at run time
    hardware_info: dict = field(default_factory=dict)

    # Config as a dict for the results file
    config_dict: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Peak memory helpers
# ---------------------------------------------------------------------------


def _reset_peak_memory(device: torch.device) -> None:
    """Reset the peak-memory counter before a run."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _read_peak_memory(device: torch.device) -> tuple[float, str]:
    """Return (peak_mib, label)."""
    if device.type == "cuda":
        mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        return mib, "gpu_vram"
    else:
        mib = psutil.Process().memory_info().rss / (1024**2)
        return mib, "cpu_rss"


def _hardware_info() -> dict:
    """Collect hardware and software versions for reproducibility."""
    import torch

    gpu_name = "none"
    gpu_vram_gb = 0.0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return {
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": gpu_name,
        "gpu_vram_gb": round(gpu_vram_gb, 2),
        "os": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "diffusers_version": _safe_import_version("diffusers"),
        "transformers_version": _safe_import_version("transformers"),
    }


def _safe_import_version(pkg: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(pkg)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------


class Generator:
    """
    Loads all model components and runs the generation pipeline.

    Parameters
    ----------
    config:
        Validated T2IConfig.
    models_dir:
        Directory where downloaded model weights are cached.
    results_dir:
        Directory where JSON logs are written.
    device:
        Override device (default: cuda if available, else cpu).
    """

    def __init__(
        self,
        config: T2IConfig,
        models_dir: Path = Path("models"),
        results_dir: Path = Path("results"),
        device: Optional[str] = None,
    ) -> None:
        self.config = config
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dtype = self._resolve_dtype()
        logger.info("Generator: device=%s dtype=%s", self.device, self.dtype)

        # Lazy-loaded model components
        self._prompt_encoder = None
        self._denoiser = None
        self._decoder = None

    def _resolve_dtype(self) -> torch.dtype:
        """Map config precision to a torch dtype."""
        if self.config.precision in ("fp16",) and self.device.type == "cuda":
            return torch.float16
        return torch.float32

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_prompt_encoder(self):
        if self._prompt_encoder is None:
            from t2i.pipeline.prompt_encoder import load_prompt_encoder

            model_id = self._resolve_model_id()
            self._prompt_encoder = load_prompt_encoder(
                model_id, self.config.model.revision, self.device, self.dtype
            )
        return self._prompt_encoder

    def _get_denoiser(self):
        if self._denoiser is None:
            from t2i.pipeline.denoiser import load_denoiser

            model_id = self._resolve_model_id()
            local_pruned = self.models_dir / "bk-sdm-small"
            pruned_id = str(local_pruned) if local_pruned.exists() else None
            self._denoiser = load_denoiser(
                model_id=model_id,
                revision=self.config.model.revision,
                device=self.device,
                dtype=self.dtype,
                scheduler_name=self.config.scheduler,
                num_inference_steps=self.config.steps,
                variant=self.config.variant,
                unet_model_id=pruned_id,
            )
            if self.config.lcm_lora:
                from t2i.optimize.lcm_lora import attach_lcm_lora

                local_lcm = self.models_dir / "lcm-lora-sdv1-5"
                lora_path = (
                    str(local_lcm)
                    if (local_lcm / "pytorch_lora_weights.safetensors").exists()
                    else "latent-consistency/lcm-lora-sdv1-5"
                )
                attach_lcm_lora(self._denoiser.unet, self._denoiser.scheduler, lora_id=lora_path)
            if self.config.channels_last and self.device.type == "cuda":
                self._denoiser.unet = self._denoiser.unet.to(memory_format=torch.channels_last)
        return self._denoiser

    def _get_decoder(self):
        if self._decoder is None:
            from t2i.pipeline.decoder import load_decoder

            model_id = self._resolve_model_id()
            self._decoder = load_decoder(
                model_id,
                self.config.model.revision,
                self.device,
                self.dtype,
                self.config.vae_tiling,
            )
        return self._decoder

    def _resolve_model_id(self) -> str:
        local_sd = self.models_dir / "stable-diffusion-v1-5"
        if local_sd.exists() and (local_sd / "model_index.json").exists():
            return str(local_sd)
        return self.config.model.base

    def _load_models(self) -> None:
        """Lazy-load pipeline components."""
        self._get_prompt_encoder()
        self._get_denoiser()
        self._get_decoder()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        step_callback=None,
    ) -> GenerationResult:
        """
        Run the full pipeline for one prompt and return a GenerationResult.

        Parameters
        ----------
        prompt:
            User text prompt.
        negative_prompt:
            Negative prompt (optional).
        seed:
            Override the base seed. Defaults to config.seed_base.
        step_callback:
            Optional denoiser step callback (for adaptive early stopping).
        """
        effective_seed = seed if seed is not None else self.config.seed_base

        # Set global seeds for reproducibility
        import random

        random.seed(effective_seed)
        np.random.seed(effective_seed)
        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_seed)

        _reset_peak_memory(self.device)
        hw_info = _hardware_info()

        # --- Stage 1: Prompt encoding ---
        t0 = time.perf_counter()
        encoder = self._get_prompt_encoder()
        enc_result = encoder.encode_prompt(prompt, negative_prompt, self.config.guidance_scale)
        time_encode = time.perf_counter() - t0
        logger.debug("Encode: %.3f s  truncated=%s", time_encode, enc_result.truncated)
        embeddings = enc_result.embeddings
        prompt_truncated = enc_result.truncated
        prompt_dropped_tokens = enc_result.dropped_tokens

        del encoder, enc_result
        del self._prompt_encoder
        self._prompt_encoder = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Stage 2: Latent init + denoising ---
        from t2i.pipeline.denoiser import Denoiser

        denoiser = self._get_denoiser()

        # Initial latents are drawn from the global RNG (already seeded above),
        # matching the diffusers reference pipeline which uses the global RNG
        # when no explicit generator is passed.
        latents = Denoiser.initialize_latents(
            batch_size=1,
            latent_channels=4,
            latent_height=self.config.latent_height,
            latent_width=self.config.latent_width,
            dtype=self.dtype,
            device=self.device,
            seed=effective_seed,
            scheduler=denoiser.scheduler,
            generator=None,
        )

        t0 = time.perf_counter()
        denoise_result = denoiser.denoise(
            latents,
            embeddings,
            self.config.guidance_scale,
            effective_seed,
            step_callback=step_callback,
        )
        time_denoise = time.perf_counter() - t0
        logger.debug("Denoise: %.3f s  steps=%d", time_denoise, denoise_result.steps_executed)

        denoised_latents = denoise_result.latents
        steps_executed = denoise_result.steps_executed
        early_stopped = denoise_result.early_stopped

        del denoiser, latents, embeddings, denoise_result
        del self._denoiser
        self._denoiser = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Stage 3: Decoding ---
        t0 = time.perf_counter()
        decoder = self._get_decoder()
        image_np = decoder.decode(denoised_latents)
        time_decode = time.perf_counter() - t0
        logger.debug("Decode: %.3f s", time_decode)

        del decoder, denoised_latents
        del self._decoder
        self._decoder = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Stage 4: Optional classical post-processing ---
        image_before_pp = None
        time_postprocess = 0.0
        quality_report = None

        if self.config.classical_postprocess:
            image_before_pp = image_np.copy()
            t0 = time.perf_counter()
            image_np, quality_report = self._apply_postprocessing(image_np)
            time_postprocess = time.perf_counter() - t0
            logger.debug("Post-process: %.3f s", time_postprocess)
        else:
            # Compute quality report without post-processing
            from t2i.cv.quality import compute_quality_report

            quality_report = asdict(compute_quality_report(image_np))

        peak_mib, mem_type = _read_peak_memory(self.device)

        result = GenerationResult(
            image=image_np,
            seed=effective_seed,
            config_hash=self.config.config_hash(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            time_encode_s=round(time_encode, 4),
            time_denoise_s=round(time_denoise, 4),
            time_decode_s=round(time_decode, 4),
            time_postprocess_s=round(time_postprocess, 4),
            steps_executed=steps_executed,
            early_stopped=early_stopped,
            peak_memory_mib=round(peak_mib, 2),
            memory_type=mem_type,
            prompt_truncated=prompt_truncated,
            prompt_dropped_tokens=prompt_dropped_tokens,
            quality_report=quality_report,
            image_before_postprocess=image_before_pp,
            hardware_info=hw_info,
            config_dict=self.config.model_dump(),
        )
        return result

    def _apply_postprocessing(self, image_np: np.ndarray) -> tuple[np.ndarray, dict]:
        """Apply all configured CV post-processing steps and return the result + report."""
        import dataclasses
        from t2i.cv.color import apply_color_correction
        from t2i.cv.edge_filters import apply_edge_filter
        from t2i.cv.artifacts import detect_and_correct_artifacts
        from t2i.cv.quality import compute_quality_report

        cv_cfg = self.config.cv

        # Colour correction
        image_np = apply_color_correction(image_np, cv_cfg.color)

        # Edge-aware filtering
        image_np = apply_edge_filter(image_np, cv_cfg.edge)

        # Artifact detection + correction
        image_np, artifact_report = detect_and_correct_artifacts(image_np, cv_cfg.artifacts)

        # Quality metrics on the final image
        qr = compute_quality_report(image_np)
        qr_dict = dataclasses.asdict(qr)
        qr_dict["artifact_report"] = artifact_report
        return image_np, qr_dict

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def save_result(self, result: GenerationResult, run_id: str) -> Path:
        """
        Save the image (PNG) and metadata (JSON) to results/<run_id>/.

        Parameters
        ----------
        result:
            The GenerationResult to save.
        run_id:
            Unique run identifier (e.g. timestamp + config hash).

        Returns
        -------
        Path
            Directory where files were saved.
        """
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        img_path = run_dir / "image.png"
        Image.fromarray(result.image).save(img_path)

        if result.image_before_postprocess is not None:
            Image.fromarray(result.image_before_postprocess).save(run_dir / "image_before_pp.png")

        # Build metadata dict (exclude raw numpy arrays)
        meta = {
            "run_id": run_id,
            "seed": result.seed,
            "config_hash": result.config_hash,
            "prompt": result.prompt,
            "negative_prompt": result.negative_prompt,
            "timings": {
                "encode_s": result.time_encode_s,
                "denoise_s": result.time_denoise_s,
                "decode_s": result.time_decode_s,
                "postprocess_s": result.time_postprocess_s,
                "total_s": round(
                    result.time_encode_s
                    + result.time_denoise_s
                    + result.time_decode_s
                    + result.time_postprocess_s,
                    4,
                ),
            },
            "steps_executed": result.steps_executed,
            "early_stopped": result.early_stopped,
            "peak_memory_mib": result.peak_memory_mib,
            "memory_type": result.memory_type,
            "prompt_truncated": result.prompt_truncated,
            "prompt_dropped_tokens": result.prompt_dropped_tokens,
            "quality_report": result.quality_report,
            "fallback_log": result.fallback_log,
            "hardware_info": result.hardware_info,
            "config": result.config_dict,
        }
        json_path = run_dir / "metadata.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

        logger.info("Results saved to %s", run_dir)
        return run_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate an image with the t2i pipeline.")
    p.add_argument("--config", required=True, help="Path to a YAML config file.")
    p.add_argument("--prompt", required=True, help="Text prompt.")
    p.add_argument("--negative-prompt", default="", help="Negative prompt.")
    p.add_argument("--seed", type=int, default=None, help="Override seed.")
    p.add_argument("--output", default="results", help="Results directory.")
    p.add_argument("--device", default=None, help="Override device (cpu|cuda).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    """CLI entry point."""
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    gen = Generator(cfg, results_dir=Path(args.output), device=args.device)

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{cfg.config_hash()}"
    logger.info("Run ID: %s", run_id)

    result = gen.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )

    out_dir = gen.save_result(result, run_id)
    total = (
        result.time_encode_s
        + result.time_denoise_s
        + result.time_decode_s
        + result.time_postprocess_s
    )
    print(
        f"\n[OK] Generated in {total:.2f}s  "
        f"(encode={result.time_encode_s:.2f}s "
        f"denoise={result.time_denoise_s:.2f}s "
        f"decode={result.time_decode_s:.2f}s "
        f"pp={result.time_postprocess_s:.2f}s)\n"
        f"  Peak memory: {result.peak_memory_mib:.1f} MiB ({result.memory_type})\n"
        f"  Saved to: {out_dir}"
    )


if __name__ == "__main__":
    main()
