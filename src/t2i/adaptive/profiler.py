"""
t2i.adaptive.profiler — Builds the adaptive profile table.

Enumerates candidate configurations, runs timed benchmarks for each, and writes
the results to ``configs/adaptive_profiles.yaml``.

This file is the source of truth for the budget controller; latency is never
estimated — it is always measured by this script.

Usage::

    python -m t2i.adaptive.profiler \\
        --calibration-prompts data/prompts/calibration.jsonl \\
        --output configs/adaptive_profiles.yaml \\
        --warmup 2 \\
        --runs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import psutil
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ProfileEntry:
    """One benchmarked configuration entry for the profile table."""

    config_id: str
    precision: str
    scheduler: str
    steps: int
    resolution: list[int]
    variant: str
    lcm_lora: bool
    quantization: str
    attention: str
    classical_postprocess: bool
    median_latency_s: float
    p90_latency_s: float
    peak_memory_mb: float
    clip_score_mean: float
    model_size_mb: float
    hardware_label: str


def _hardware_label() -> str:
    """Return a short label for the current hardware."""
    cpu = psutil.cpu_count(logical=False)
    return f"CPU_{cpu}cores"


def _candidate_configs() -> list[dict]:
    """
    Enumerate configurations to profile.

    Returns a list of dicts describing each candidate, covering the
    experiment matrix dimensions: precision × steps × resolution × variant ×
    classical_postprocess.
    """
    precisions = ["fp32", "fp16"]
    schedulers_steps = [
        ("ddim", 50),
        ("ddim", 25),
        ("lcm", 4),
        ("lcm", 8),
    ]
    resolutions = [[512, 512], [384, 384]]
    variants = ["full_unet"]  # "block_pruned" added when weights are available
    postprocess = [False, True]

    candidates = []
    for (sched, steps), res, prec, var, pp in product(
        schedulers_steps, resolutions, precisions, variants, postprocess
    ):
        lcm_lora = sched == "lcm"
        config_id = f"{prec}_{sched}_s{steps}_r{res[0]}_{var}" + ("_pp" if pp else "")
        candidates.append(
            dict(
                config_id=config_id,
                precision=prec,
                scheduler=sched,
                steps=steps,
                resolution=res,
                variant=var,
                lcm_lora=lcm_lora,
                quantization="none",
                attention="sdpa",
                classical_postprocess=pp,
            )
        )
    return candidates


def benchmark_config(
    config_dict: dict,
    calibration_prompts: list[str],
    warmup_runs: int = 2,
    measured_runs: int = 10,
    clip_score_fn=None,
    models_cache_dir: Optional[Path] = None,
) -> Optional[ProfileEntry]:
    """
    Benchmark one configuration and return a ProfileEntry.

    Parameters
    ----------
    config_dict:
        Dict describing the configuration (output of ``_candidate_configs``).
    calibration_prompts:
        List of text prompts to use for benchmarking.
    warmup_runs, measured_runs:
        Number of warm-up and measured runs.
    clip_score_fn:
        Optional callable(image, prompt) → float for quick CLIPScore estimation.
    models_cache_dir:
        Directory for cached model weights.

    Returns
    -------
    ProfileEntry or None if the configuration fails (e.g. OOM).
    """
    from t2i.config import T2IConfig, ModelConfig

    # Build a minimal T2IConfig for this candidate
    try:
        cfg = T2IConfig(
            name=config_dict["config_id"],
            model=ModelConfig(
                base="runwayml/stable-diffusion-v1-5",
                revision="main",
            ),
            precision=config_dict["precision"],
            scheduler=config_dict["scheduler"],
            steps=config_dict["steps"],
            guidance_scale=1.0 if config_dict["lcm_lora"] else 7.5,
            resolution=config_dict["resolution"],
            variant=config_dict["variant"],
            lcm_lora=config_dict["lcm_lora"],
            quantization=config_dict["quantization"],
            attention=config_dict["attention"],
            vae_tiling=False,
            channels_last=False,
            classical_postprocess=config_dict["classical_postprocess"],
            seed_base=1234,
        )
    except Exception as exc:
        logger.warning("Config creation failed for %s: %s", config_dict["config_id"], exc)
        return None

    from t2i.pipeline.generate import Generator

    gen = Generator(cfg, models_dir=models_cache_dir or Path("models"))

    latencies: list[float] = []
    peak_mems: list[float] = []
    clip_scores: list[float] = []

    all_runs = warmup_runs + measured_runs
    prompts = (calibration_prompts * ((all_runs // len(calibration_prompts)) + 1))[:all_runs]

    for i, prompt in enumerate(prompts):
        try:
            t_start = time.perf_counter()
            result = gen.generate(prompt=prompt, seed=1234 + i)
            elapsed = time.perf_counter() - t_start
        except Exception as exc:
            logger.warning("Run %d failed for %s: %s", i, config_dict["config_id"], exc)
            return None

        if i >= warmup_runs:
            latencies.append(elapsed)
            peak_mems.append(result.peak_memory_mib)
            if clip_score_fn is not None:
                clip_scores.append(clip_score_fn(result.image, prompt))

    import statistics

    median_lat = statistics.median(latencies)
    p90_lat = sorted(latencies)[int(0.9 * len(latencies))]
    peak_mem = max(peak_mems)
    clip_mean = statistics.mean(clip_scores) if clip_scores else 0.0

    # Estimate model size (prompt encoder + U-Net + VAE)
    model_size = 0.0
    try:
        from t2i.eval.size import measure_component_sizes

        sizes = measure_component_sizes(Path("models"))
        model_size = sizes.get("total_mb", 0.0)
    except Exception:
        pass

    return ProfileEntry(
        config_id=config_dict["config_id"],
        precision=config_dict["precision"],
        scheduler=config_dict["scheduler"],
        steps=config_dict["steps"],
        resolution=config_dict["resolution"],
        variant=config_dict["variant"],
        lcm_lora=config_dict["lcm_lora"],
        quantization=config_dict["quantization"],
        attention=config_dict["attention"],
        classical_postprocess=config_dict["classical_postprocess"],
        median_latency_s=round(median_lat, 3),
        p90_latency_s=round(p90_lat, 3),
        peak_memory_mb=round(peak_mem, 1),
        clip_score_mean=round(clip_mean, 4),
        model_size_mb=round(model_size, 1),
        hardware_label=_hardware_label(),
    )


def run_profiler(
    calibration_prompts_path: Path,
    output_path: Path,
    warmup: int = 2,
    runs: int = 10,
    models_cache_dir: Optional[Path] = None,
) -> None:
    """
    Run the full profiling sweep and write ``adaptive_profiles.yaml``.

    Parameters
    ----------
    calibration_prompts_path:
        Path to ``data/prompts/calibration.jsonl``.
    output_path:
        Path to write the profile table.
    warmup, runs:
        Warm-up and measured run counts per configuration.
    models_cache_dir:
        Directory for cached model weights.
    """
    with open(calibration_prompts_path) as fh:
        cal_prompts = [json.loads(line)["prompt"] for line in fh if line.strip()]

    candidates = _candidate_configs()
    logger.info("Profiling %d configurations…", len(candidates))

    profiles: list[dict] = []
    for cand in candidates:
        logger.info("  Benchmarking: %s", cand["config_id"])
        entry = benchmark_config(
            cand,
            calibration_prompts=cal_prompts,
            warmup_runs=warmup,
            measured_runs=runs,
            models_cache_dir=models_cache_dir,
        )
        if entry is not None:
            profiles.append(asdict(entry))
        else:
            logger.warning("  Skipped (failed): %s", cand["config_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        yaml.safe_dump({"profiles": profiles}, fh, default_flow_style=False)

    logger.info("Profile table written to %s (%d entries)", output_path, len(profiles))


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Build the adaptive profile table.")
    p.add_argument("--calibration-prompts", required=True, type=Path)
    p.add_argument("--output", default="configs/adaptive_profiles.yaml", type=Path)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    run_profiler(args.calibration_prompts, args.output, args.warmup, args.runs, args.models_dir)


if __name__ == "__main__":
    main()
