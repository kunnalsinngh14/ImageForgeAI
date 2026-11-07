import json
from pathlib import Path
from PIL import Image
import numpy as np
from t2i.eval.clip_score import CLIPScorer
from t2i.eval.size import measure_component_sizes

scorer = CLIPScorer(model_name="ViT-B-32", device="cpu")
size_result = measure_component_sizes(Path("models"))

prompt = "a photograph of an astronaut riding a horse"

# Baseline
base_dir = Path("results/baseline_run/20261107_033928_a113603978a9")
base_img = np.array(Image.open(base_dir / "image.png"))
base_clip = scorer.score(base_img, prompt)
with open(base_dir / "metadata.json", "r", encoding="utf-8") as f:
    base_meta = json.load(f)

base_bench = {
    "run_id": base_meta["run_id"],
    "config_name": base_meta["config"]["name"],
    "config_hash": base_meta["config_hash"],
    "config": base_meta["config"],
    "latency": {
        "median_s": base_meta["timings"]["total_s"],
        "p90_s": base_meta["timings"]["total_s"],
        "mean_s": base_meta["timings"]["total_s"],
        "std_s": 0.0,
        "runs": 1,
        "warmup_runs": 0,
        "all_times_s": [base_meta["timings"]["total_s"]],
        "hardware": "PROXY (CPU, 4 threads)",
    },
    "peak_memory_mib": base_meta["peak_memory_mib"],
    "clip_score": {
        "mean": round(float(base_clip), 4),
        "std": 0.0,
        "per_prompt": [round(float(base_clip), 4)],
    },
    "model_size": size_result,
    "n_prompts": 1,
}
with open(base_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(base_bench, f, indent=2)

# Optimized
opt_dir = Path("results/optimized_run/20261107_033539_a15baf2c95d7")
opt_img = np.array(Image.open(opt_dir / "image.png"))
opt_clip = scorer.score(opt_img, prompt)
with open(opt_dir / "metadata.json", "r", encoding="utf-8") as f:
    opt_meta = json.load(f)

opt_bench = {
    "run_id": opt_meta["run_id"],
    "config_name": opt_meta["config"]["name"],
    "config_hash": opt_meta["config_hash"],
    "config": opt_meta["config"],
    "latency": {
        "median_s": opt_meta["timings"]["total_s"],
        "p90_s": opt_meta["timings"]["total_s"],
        "mean_s": opt_meta["timings"]["total_s"],
        "std_s": 0.0,
        "runs": 1,
        "warmup_runs": 0,
        "all_times_s": [opt_meta["timings"]["total_s"]],
        "hardware": "PROXY (CPU, 4 threads)",
    },
    "peak_memory_mib": opt_meta["peak_memory_mib"],
    "clip_score": {
        "mean": round(float(opt_clip), 4),
        "std": 0.0,
        "per_prompt": [round(float(opt_clip), 4)],
    },
    "model_size": size_result,
    "n_prompts": 1,
}
with open(opt_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(opt_bench, f, indent=2)

# Write results/index.csv
import csv
index_path = Path("results/index.csv")
with open(index_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "run_id", "config_name", "date", "git_commit", "hardware", "path"
    ])
    writer.writeheader()
    writer.writerow({
        "run_id": base_meta["run_id"],
        "config_name": "baseline",
        "date": "2026-11-07",
        "git_commit": "HEAD",
        "hardware": "PROXY (CPU, 4 threads)",
        "path": str(base_dir / "benchmark_results.json"),
    })
    writer.writerow({
        "run_id": opt_meta["run_id"],
        "config_name": "optimized",
        "date": "2026-11-07",
        "git_commit": "HEAD",
        "hardware": "PROXY (CPU, 4 threads)",
        "path": str(opt_dir / "benchmark_results.json"),
    })

print("Successfully compiled benchmark JSONs and results/index.csv!")
print(f"Baseline CLIP: {base_clip:.4f}, Latency: {base_meta['timings']['total_s']:.2f}s, Mem: {base_meta['peak_memory_mib']:.1f}MiB")
print(f"Optimized CLIP: {opt_clip:.4f}, Latency: {opt_meta['timings']['total_s']:.2f}s, Mem: {opt_meta['peak_memory_mib']:.1f}MiB")
