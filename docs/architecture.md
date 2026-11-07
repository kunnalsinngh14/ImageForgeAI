# Architecture

Resource-Efficient Text-to-Image Generation for Mobile Devices (P3).

## Overview

The system is a local text-to-image pipeline built on Stable Diffusion v1.5.
All generation runs on-device (or on a CPU proxy that mimics a mobile device).
No hosted image-generation API is called at any point.

```
prompt ──► [Prompt Encoder (CLIP)] ──► text embeddings [1|2, 77, 768]
                 │
                 ▼
latents [1,4,64,64] ──► [Denoiser (U-Net + scheduler)] ──► denoised latents
                 │
                 ▼
                 ┌──► [Decoder (VAE)] ──► uint8 RGB [512,512,3]
                 │
                 ▼
         [Classical CV post-processing] ──► final image
                 │
                 ▼
         [Quality report + metrics logging]
```

## Components

### `src/t2i/pipeline/`

| Module | Responsibility |
|--------|----------------|
| `prompt_encoder.py` | Tokenises the prompt, runs the CLIP text encoder, detects truncation, and concatenates uncond/cond embeddings for classifier-free guidance (CFG). |
| `denoiser.py` | Initialises latents, runs the reverse-diffusion loop (DDIM, DPM-Solver++, or LCM), applies CFG explicitly, and supports an early-stop callback. |
| `decoder.py` | Scales latents by the VAE factor (0.18215), decodes, and converts to uint8 RGB. Supports VAE tiling. |
| `generate.py` | Orchestrates encode → denoise → decode → post-process → quality report → logging. |

### `src/t2i/optimize/`

| Module | Strategy |
|--------|----------|
| `lcm_lora.py` | LCM-LoRA adapter for 2–8 step sampling; switches the scheduler to `LCMScheduler`. |
| `quantize.py` | FP16 conversion and INT8 quantization (dynamic via `torch.ao`, static via ONNX Runtime). |
| `prune.py` | Block-pruned distilled U-Net (BK-SDM variants) loader and compatibility checker. |
| `export_onnx.py` | ONNX export of the U-Net and VAE decoder with numerical validation against PyTorch. |

### `src/t2i/cv/`

Classical post-processing, each a pure function toggled by config:

| Module | Steps |
|--------|-------|
| `resize.py` | Lanczos downscale, bicubic upscale, adaptive resolution, unsharp mask. |
| `color.py` | CLAHE on LAB L channel, gray-world / percentile white balance, vibrance with oversaturation clamp. |
| `edge_filters.py` | Guided filter (bilateral fallback), detail enhancement, edge-preservation score. |
| `quality.py` | Laplacian sharpness, colourfulness (Hasler & Süsstrunk), RMS contrast, entropy, BRISQUE/NIQE. |
| `artifacts.py` | FFT checkerboard detection, over-smoothed / over-saturated masks, targeted median/NLM correction. |

### `src/t2i/adaptive/`

| Module | Responsibility |
|--------|----------------|
| `profiler.py` | Benchmarks candidate configurations and writes `configs/adaptive_profiles.yaml`. |
| `budget.py` | Selects a configuration from the profile table given latency/memory budgets and a quality preference. |
| `fallback.py` | Degradation chain: reduce resolution → smaller model → fewer steps → structured error. |

### `src/t2i/eval/`

`clip_score.py`, `semantic_check.py`, `fid.py`, `diversity.py`, `latency.py`, `memory.py`, `size.py`.

### `src/t2i/app/`

FastAPI server (`server.py`) with a minimal HTML UI in `ui/`.

### `src/t2i/report/`

`make_report.py` reads raw JSON logs from `results/` and produces CSV/Markdown
tables and Pareto plots. It never reads numbers from memory or from another document.

## Configuration

All behaviour is driven by YAML configs in `configs/`. `src/t2i/config.py` loads
and validates them with Pydantic; unknown keys raise a `ValidationError`. Every
results file stores the resolved config, its SHA-256 hash, the git commit, and
the hardware info.

## Determinism

- Seeds are mandatory and recorded in every result.
- Per-image seed = `seed_base + prompt_index`.
- Python, NumPy, and PyTorch seeds are set before every generation.