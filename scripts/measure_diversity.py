#!/usr/bin/env python3
"""
scripts/measure_diversity.py — Measure LPIPS diversity across seeds for low-step configurations.
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np

from t2i.config import load_config
from t2i.pipeline.generate import Generator
from t2i.eval.diversity import compute_lpips_diversity

def main():
    cfg = load_config("configs/optimized.yaml")
    prompt = "a photograph of an astronaut riding a horse on the moon"
    seeds = [101, 202, 303, 404]
    
    out_dir = Path("results/diversity_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    gen = Generator(cfg, results_dir=out_dir, device="cuda")
    
    images = []
    print(f"Generating {len(seeds)} images across different seeds for prompt: '{prompt}'...")
    for idx, seed in enumerate(seeds):
        res = gen.generate(prompt=prompt, seed=seed)
        images.append(res.image)
        Image.fromarray(res.image).save(out_dir / f"seed_{seed}.png")
        total_time = res.time_encode_s + res.time_denoise_s + res.time_decode_s + res.time_postprocess_s
        print(f"  [Seed {seed}] Generated in {total_time:.2f}s (denoise: {res.time_denoise_s:.2f}s)")
        
    print("\nComputing pairwise LPIPS diversity...")
    diversity_results = compute_lpips_diversity(images, device="cuda")
    print(f"  Mean LPIPS: {diversity_results['mean_lpips']:.4f} +/- {diversity_results['std_lpips']:.4f}")
    print(f"  Pairs evaluated: {diversity_results['n_pairs']}")
    print(f"  All pair scores: {diversity_results['all_scores']}")
    
    out_file = Path("results/diversity_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "prompt": prompt,
            "seeds": seeds,
            "config": "configs/optimized.yaml",
            "results": diversity_results
        }, f, indent=2)
    print(f"Diversity report saved to {out_file}")

if __name__ == "__main__":
    main()
