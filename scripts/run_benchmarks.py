#!/usr/bin/env python3
"""
scripts/run_benchmarks.py — Run the full evaluation harness from one command.

Runs the baseline through the complete harness: latency, memory, CLIP score,
and size. Writes a results file to results/benchmark_<run_id>/.

Usage::

    python scripts/run_benchmarks.py \\
        --config configs/baseline.yaml \\
        --prompts data/prompts/simple.jsonl \\
        --output results/ \\
        --runs 5

For the full M5 experiment matrix, run with the full prompt sets and more runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the evaluation harness.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--prompts", required=True, type=Path)
    p.add_argument("--max-prompts", type=int, default=None,
                   help="Maximum number of prompts to evaluate (for reduced benchmark runs).")
    p.add_argument("--output", default=Path("results"), type=Path)
    p.add_argument("--runs", type=int, default=5,
                   help="Number of measured runs per prompt.")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--clip-model", default="ViT-B-32")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)

    from t2i.config import load_config
    from t2i.pipeline.generate import Generator
    from t2i.eval.clip_score import CLIPScorer
    from t2i.eval.size import measure_component_sizes
    from t2i.eval.latency import measure_latency
    import numpy as np

    cfg = load_config(args.config)
    run_id = f"bench_{time.strftime('%Y%m%d_%H%M%S')}_{cfg.config_hash()}"
    out_dir = args.output / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run ID: %s", run_id)

    # Load prompts
    with open(args.prompts) as fh:
        prompts_data = [json.loads(line) for line in fh if line.strip()]
    prompts = [d["prompt"] for d in prompts_data]
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompts)

    # Build generator
    gen = Generator(cfg, results_dir=out_dir, device=args.device)

    # --- Latency measurement and image generation ---
    logger.info("Measuring latency and generating images…")
    images = []
    peak_mems = []
    all_times = []
    from PIL import Image as PILImg
    import statistics
    import psutil

    # Warm-up runs (if requested)
    for w in range(args.warmup):
        logger.info("Warm-up run %d/%d…", w + 1, args.warmup)
        gen.generate(prompt=prompts[0], seed=cfg.seed_base + 9990 + w)

    # Measured runs on prompt 0 for official latency
    for r in range(args.runs):
        logger.info("Latency run %d/%d…", r + 1, args.runs)
        t_start = time.perf_counter()
        result = gen.generate(prompt=prompts[0], seed=cfg.seed_base + r)
        elapsed = time.perf_counter() - t_start
        all_times.append(elapsed)
        if r == 0:
            images.append(result.image)
            peak_mems.append(result.peak_memory_mib)
            PILImg.fromarray(result.image).save(out_dir / "image_000.png")

    # Remaining prompts for CLIP scoring
    for i in range(1, len(prompts)):
        prompt = prompts[i]
        logger.info("Generating prompt %d/%d: %s…", i + 1, len(prompts), prompt[:40])
        t_start = time.perf_counter()
        result = gen.generate(prompt=prompt, seed=cfg.seed_base + i)
        elapsed = time.perf_counter() - t_start
        all_times.append(elapsed)
        images.append(result.image)
        peak_mems.append(result.peak_memory_mib)
        PILImg.fromarray(result.image).save(out_dir / f"image_{i:03d}.png")

    sorted_times = sorted(all_times)
    p90_idx = max(0, int(0.9 * len(sorted_times)) - 1)
    latency_result = {
        "median_s": round(statistics.median(all_times), 4),
        "p90_s": round(sorted_times[p90_idx], 4),
        "mean_s": round(statistics.mean(all_times), 4),
        "std_s": round(statistics.stdev(all_times) if len(all_times) > 1 else 0.0, 4),
        "runs": len(all_times),
        "warmup_runs": args.warmup,
        "all_times_s": [round(t, 4) for t in all_times],
        "hardware": {
            "label": f"NVIDIA GeForce GTX 1050 (CUDA, {gen.device})" if "cuda" in str(gen.device) else "PROXY (CPU, 4 threads)",
            "device": str(gen.device),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        },
    }
    logger.info("Latency: median=%.2fs p90=%.2fs", latency_result["median_s"], latency_result["p90_s"])

    # --- CLIP scoring ---
    logger.info("Computing CLIP scores…")
    scorer = CLIPScorer(model_name=args.clip_model, device=args.device or "cpu")
    clip_result = scorer.score_batch(images, prompts)
    logger.info("CLIPScore: mean=%.4f std=%.4f", clip_result["mean"], clip_result["std"])

    # --- Model sizes ---
    logger.info("Measuring model sizes…")
    size_result = measure_component_sizes(Path("models"))

    # --- Compile results ---
    results = {
        "run_id": run_id,
        "config_name": cfg.name,
        "config_hash": cfg.config_hash(),
        "config": cfg.model_dump(),
        "latency": latency_result,
        "peak_memory_mib": round(max(peak_mems), 2) if peak_mems else 0.0,
        "clip_score": clip_result,
        "model_size": size_result,
        "n_prompts": len(prompts),
    }

    results_path = out_dir / "benchmark_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    # Update results index
    index_path = args.output / "index.csv"
    import csv
    file_exists = index_path.exists()
    with open(index_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "run_id", "config_name", "date", "git_commit", "hardware", "path"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "run_id": run_id,
            "config_name": cfg.name,
            "date": time.strftime("%Y-%m-%d"),
            "git_commit": "unknown",
            "hardware": latency_result["hardware"].get("label", str(latency_result["hardware"])),
            "path": str(results_path),
        })

    print(f"\n[OK] Benchmark complete.")
    print(f"  Latency: median={latency_result['median_s']:.2f}s  p90={latency_result['p90_s']:.2f}s")
    print(f"  CLIPScore: {clip_result['mean']:.4f} +/- {clip_result['std']:.4f}")
    print(f"  Model size: {size_result['total_mb']:.1f} MB")
    print(f"  Results: {results_path}")


if __name__ == "__main__":
    main()
