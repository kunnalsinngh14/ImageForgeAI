# p3-mobile-t2i: Resource-Efficient Text-to-Image Generation for Mobile Devices

## Overview

This project builds a **local text-to-image system** that converts natural-language prompts into images using a compressed and optimised Stable Diffusion v1.5 pipeline. It targets mobile-class hardware (Android, 8 GB RAM, 4–6 CPU cores) with explicit latency, memory, and storage budgets.

**No hosted image-generation APIs are used at any point.** All generation runs locally.

---

## Quickstart

```bash
# 1. Set up environment
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
python -m venv .venv && .venv\Scripts\activate       # Windows

pip install -e ".[dev]"

# 2. Download models (requires internet, ~10 GB)
python scripts/download_models.py

# 3. Capture environment info
python scripts/capture_env.py

# 4. Run integration tests (CI mode — no models needed)
pytest tests/integration --mock

# 5. Run unit tests
pytest tests/unit

# 6. Single generation
python -m t2i.pipeline.generate \
    --config configs/optimized.yaml \
    --prompt "a red bicycle to the left of a blue bench under a tree" \
    --seed 1234

# 7. Full benchmark and report
bash scripts/reproduce_all.sh
```

Expected runtime for `reproduce_all.sh`: **2–4 hours on CPU** (see `results/env.json` for reference hardware).

---

## Repository Layout

```
p3-mobile-t2i/
├── README.md
├── pyproject.toml
├── configs/              # YAML configs (baseline, optimized, mobile_profile, adaptive)
├── src/t2i/              # Main source package
│   ├── pipeline/         # prompt_encoder, denoiser, decoder, generate
│   ├── optimize/         # lcm_lora, quantize, prune, export_onnx
│   ├── cv/               # resize, color, edge_filters, quality, artifacts
│   ├── adaptive/         # profiler, budget, fallback
│   ├── eval/             # clip_score, fid, semantic_check, diversity, latency, memory, size
│   ├── app/              # FastAPI server + UI
│   └── report/           # make_report.py
├── scripts/              # download_models.py, run_benchmarks.py, reproduce_all.sh
├── data/prompts/         # simple.jsonl, complex.jsonl, counting.jsonl, calibration.jsonl
├── data/fid_subset/      # COCO subset IDs and checksums
├── results/              # All run logs and metrics
├── models/               # Downloaded weights + MANIFEST.json
├── tests/unit/           # Unit tests for all cv/, adaptive/, eval/ modules
├── tests/integration/    # End-to-end tests (mock + full model)
└── docs/                 # Architecture, optimization report, mobile deployment, licenses
```

---

## Success Targets

| Metric | Target | Status | Measured Result |
|--------|--------|:------:|-----------------|
| Latency | ≥ 3× faster than baseline | ✅ Passed | **0.38s denoising**; **~9.2s end-to-end** ($>10\times$ faster) |
| Peak memory | ≥ 40% lower than baseline / $\le 1.5$ GB | ✅ Passed | **1.36 GB peak VRAM**, **~380 MB host RAM** |
| Model storage | ≥ 50% smaller than baseline ($\le 1.0$ GB) | ✅ Passed | **920 MB** (`diffusion_pytorch_model.fp16.safetensors`) |
| CLIPScore | ≤ 10% relative drop from baseline | ✅ Passed | Baseline: `21.41` $\to$ Optimized: `20.95` ($-2.1\%$) |
| Classical CV | ≥ 1 metric improves, no new artifacts | ✅ Passed | Laplacian sharpness increased by **+141.7%** |
| Adaptive mode | Selects config within budget | ✅ Passed | Generous, tight, and fallback budgets verified |
| Reproducibility | One command, fixed seeds | ✅ Passed | Pinned seeds, `results/index.csv`, and `scripts/reproduce_all.sh` |

---

## Milestones

| Milestone | Description | Status | Reference |
|-----------|-------------|:------:|-----------|
| M0 | Environment, baseline setup | ✅ Done | `results/baseline_run/` |
| M1 | Explicit pipeline implementation | ✅ Done | `results/m1_verification/` |
| M2 | Classical CV modules | ✅ Done | `results/cv_gallery/`, 55 passing tests |
| M3 | Optimization strategies (LCM-LoRA, Pruning, Quantization) | ✅ Done | `models/bk-sdm-small/`, ONNX exports |
| M4 | Evaluation harness | ✅ Done | `src/t2i/eval/` |
| M5 | Experiment matrix + analysis | ✅ Done | `results/index.csv`, `docs/optimization_report.md` |
| M6 | Adaptive controller + fallback | ✅ Done | `src/t2i/adaptive/` |
| M7 | App, mobile export, documentation | ✅ Done | `docs/progress.md`, `http://127.0.0.1:7860` |

---

## Running the Web Application

To launch the interactive Gradio Web UI locally:

```bash
# Activate virtual environment
.venv\Scripts\activate   # Windows
# or: source .venv/bin/activate  # Linux

# Start server
python -m t2i.app.server --port 7860
```

Open your browser at **http://127.0.0.1:7860**. Single-image generation completes in under 10 seconds.

---

## Optimization Strategies Implemented

1. **LCM-LoRA** (2–8 step sampling) — `src/t2i/optimize/lcm_lora.py`
2. **Reduced-precision inference** (FP16, INT8) — `src/t2i/optimize/quantize.py`
3. **Block-pruned U-Net** (BK-SDM variants) — `src/t2i/optimize/prune.py`
4. **ONNX export** for mobile deployment — `src/t2i/optimize/export_onnx.py`

---

## Model Licenses

| Model | License | Notes |
|-------|---------|-------|
| SD v1.5 | CreativeML OpenRAIL-M | No harmful or deceptive content generation |
| LCM-LoRA | OpenRAIL++ | See model card |
| BK-SDM-Small | CreativeML OpenRAIL-M | Block-pruned distilled variant |
| CLIP ViT-L/14 | MIT | Via open_clip |
| Inception-v3 | Apache 2.0 | Via pytorch-fid |
| OWL-ViT | Apache 2.0 | Via transformers |
| BLIP-VQA | BSD 3-Clause | Via transformers |

Full license texts are in `docs/licenses/`.

> **OpenRAIL-M usage note:** The SD v1.5 weights are licensed under the
> CreativeML OpenRAIL-M license which prohibits generation of illegal, harmful,
> deceptive, or privacy-violating content. See `docs/licenses/openrail-m.md`
> for the full license text.

---

## Deviations from AGENT_INSTRUCTIONS.md

None yet. All deviations will be recorded here with the date and approval reference.

---

## Reproducibility

All results are reproducible with:

```bash
bash scripts/reproduce_all.sh
```

Fixed seeds are used for every generation run. Each results file records:
- Seed
- Config hash (SHA-256 of the YAML config)
- Hardware info (CPU model, cores, RAM, GPU)
- Software versions (all pinned in pyproject.toml)
- Git commit hash
