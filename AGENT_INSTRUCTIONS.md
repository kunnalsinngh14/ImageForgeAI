# AGENT INSTRUCTIONS: Resource-Efficient Text-to-Image Generation for Mobile Devices (P3)

**Read this entire file before writing any code.**
This file is the complete specification for building, optimizing, evaluating, and documenting a local text-to-image system. Follow it in milestone order. Do not skip milestones, and do not start a milestone until the previous one's exit criterion is met and reported.

---

## Table of Contents

1. Mission and Non-Negotiable Rules
2. Success Criteria
3. Background and Technical Direction
4. Environment and Hardware
5. Models, Weights, and Licenses
6. Repository Layout
7. Configuration System
8. Module Specifications
   - 8.1 Deep learning pipeline (prompt encoder, denoiser, decoder)
   - 8.2 Optimization strategies
   - 8.3 Classical computer vision
   - 8.4 Adaptive generation (budget controller and profiler)
   - 8.5 Resource-aware fallback
   - 8.6 Evaluation harness
   - 8.7 Application
   - 8.8 Reporting
9. Datasets and Prompt Sets
10. Experiment Matrix
11. Milestones M0 to M7 (tasks and exit criteria)
12. Mobile Deployment Path
13. Testing Requirements
14. Reporting Requirements for the Final Report
15. Acceptance Checklist (Definition of Done)
16. Risks and Mitigations
17. Working Rules, Progress Log, and Stop Points

---

## 1. Mission and Non-Negotiable Rules

### 1.1 Mission
Build a local text-to-image system that converts natural-language prompts into images using a compressed and optimized diffusion pipeline. The system must reduce compute, memory, storage, and latency compared with a standard baseline, while keeping prompt alignment and visual quality within stated tolerances.

### 1.2 Non-negotiable rules
1. **No hosted image-generation APIs.** Do not call OpenAI, Stability cloud, Replicate, Hugging Face Inference API, or any external generation endpoint, at any point, including tests and fallbacks. Generation must run locally. A grep for external generation endpoints in `src/` must return nothing.
2. **Do not fabricate results.** Every number in every report must come from a run logged under `results/`. If a target is missed, report the measured value and explain why.
3. **Do not copy published numbers as baseline.** The baseline must be measured by the same harness as the optimized configurations.
4. **Every optimization is a config toggle** so its effect can be isolated and measured.
5. **Prefer explicit code over library magic** in the core pipeline. The prompt encoding, denoising loop, and decoding must be written out so each step can be explained in code comments. High-level `pipe(...)` calls may be used only as a reference for verification.
6. **Ask before deviating** from the model choice, the reference mobile profile, or the FID dataset. Record every approved deviation in the README.
7. **Never present a desktop or GPU result as a mobile result.** Label every benchmark with its hardware and profile.
8. **Seeds are mandatory.** Every generation and every evaluation run records its seed.

### 1.3 Non-goals
- Training a text-to-image model from scratch.
- Calling external hosted APIs for any purpose related to generation.
- A production, app-store-ready mobile app. A reference mobile export and a measured mobile-proxy benchmark are required; a polished store build is not.
- Video generation, or image editing beyond the post-processing listed in Section 8.3.

---

## 2. Success Criteria

Summary targets. Each is checked in Section 15.

| Area | Target |
|------|--------|
| Latency | Optimized end-to-end pipeline at least **3x faster** than baseline at the chosen operating point, same hardware. |
| Peak memory | At least **40% lower** than baseline. |
| Model storage | At least **50% smaller** than baseline (sum of deployed weight files). |
| CLIPScore | No more than **10% relative drop** from baseline. |
| FID | No more than **25% relative degradation** from baseline on the evaluation subset. Larger drops must be explained in the report. |
| Classical CV | At least one quality metric (sharpness, contrast, or artifact score) measurably improves, with no new artifacts. |
| Adaptive mode | Selects a configuration that fits a stated latency or memory budget, using the profile table. |
| Fallback | Keeps the app working on constrained settings and logs every fallback step. |
| Reproducibility | All main results reproducible with one command, fixed seeds, and recorded hardware. |

---

## 3. Background and Technical Direction

### 3.1 Base architecture: Stable Diffusion v1.5
- **Text encoder:** CLIP ViT-L/14. Produces 77 token embeddings of dimension 768.
- **Denoiser:** U-Net operating in latent space (about 860M parameters).
- **Decoder:** VAE that maps 4×64×64 latents to 3×512×512 pixels.
- **Scheduler:** DDIM or DPM-Solver++ for the baseline (25 to 50 steps).
- **Guidance:** Classifier-free guidance (CFG). Baseline guidance scale is 7.5.

### 3.2 Optimization strategies
You must implement **at least two**. This project requires **three**:

1. **Reduced sampling steps via distillation.** LCM-LoRA on SD v1.5 (2 to 8 steps). Optionally compare with a distilled checkpoint (SD-Turbo-style) if license and weights permit.
2. **Reduced-precision inference.** FP16 for all modules first. Then INT8 weight quantization of the U-Net and text encoder via ONNX Runtime quantization or `torch.ao`. Compare all precisions.
3. **Architecture compression.** Evaluate a block-pruned distilled U-Net (BK-SDM variants) against the full U-Net at matched step counts.

Optional fourth strategy: latent-space optimization, such as a lower-resolution latent path (384×384) with classical upscaling in post-processing.

### 3.3 Reference mobile profile
Define and document this profile in `configs/mobile_profile.yaml`:
- Target device class: Android, 8 GB RAM, 4 to 6 CPU cores, optional NNAPI or GPU delegate.
- Model working-memory budget: **1.5 GB**.
- Model storage budget: **1 GB**.
- Latency target: **under 10 seconds** per 512×512 image. Stretch goal: **under 5 seconds**.

**Physical device caveat.** A device may not be available. Then:
1. Export the optimized U-Net and VAE to ONNX (and TFLite or Core ML if feasible). Verify the exports run.
2. Benchmark on the development CPU, restricted to the reference thread count (for example, 4 threads). Optionally cap memory via `ulimit -v`, `cgroups`, or Docker `--memory`.
3. If an Android device or emulator is available, run the exported model there and report those numbers separately.

---

## 4. Environment and Hardware

### 4.1 Software
- Python 3.10 or 3.11.
- PyTorch (CUDA build if a GPU is available, CPU build otherwise).
- `diffusers`, `transformers`, `accelerate`, `safetensors`.
- `optimum` and `onnxruntime` (export and quantization). `torch.ao` for dynamic quantization.
- Do **not** rely on `bitsandbytes` for CPU-only targets.
- OpenCV with contrib modules (guided filter, quality modules), NumPy, scikit-image, Pillow.
- `open_clip_torch` for CLIP scoring, `torchmetrics` or `pytorch-fid` for FID, `pyiqa` for BRISQUE or NIQE, `psutil` for memory.
- Owl-ViT or GroundingDINO (via `transformers`) and/or BLIP-VQA for semantic checks.
- FastAPI with a minimal HTML front end, or Gradio.
- pandas, matplotlib, Jinja2 (or plain Markdown templates).
- pytest.

Pin every dependency in `pyproject.toml` with exact versions. Record the lock file.

### 4.2 Hardware
- **Minimum for development:** 16 GB RAM. GPU with 8 GB VRAM is recommended for the GPU baseline.
- **Required:** CPU-only runs for the mobile-proxy benchmark.
- Every result file must record: CPU model, core count, thread count used, RAM, GPU (if any), OS, Python version, and all library versions.

### 4.3 Determinism
- Set seeds for Python, NumPy, and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`).
- Use a per-image seed derived from a base seed and the prompt index so runs are reproducible.
- Record the seed in every output JSON.

---

## 5. Models, Weights, and Licenses

| Component | Source | License to record |
|-----------|--------|-------------------|
| SD v1.5 weights | Hugging Face Hub (runwayml or equivalent mirror) | CreativeML OpenRAIL-M |
| LCM-LoRA (SD v1.5) | Hugging Face Hub | Check and record the license on the model card |
| Block-pruned U-Net (BK-SDM variant) | Hugging Face Hub | Check and record |
| CLIP ViT-L/14 | open_clip / OpenAI weights | Record |
| Inception-v3 (FID) | torchmetrics / pytorch-fid | Record |
| OWL-ViT / BLIP-VQA | Hugging Face Hub | Record |

Rules:
- `scripts/download_models.py` downloads every model, verifies SHA-256 checksums, and writes `models/MANIFEST.json` with source URL, revision or commit hash, checksum, and license.
- Copy each license text or model-card license reference into `docs/licenses/`.
- Pin model revisions (commit hashes), never just model names.
- Include usage notes for OpenRAIL-M in the README (for example, restrictions on harmful or deceptive use).

---

## 6. Repository Layout

Create exactly this structure:

```
p3-mobile-t2i/
├── README.md
├── pyproject.toml
├── configs/
│   ├── baseline.yaml
│   ├── optimized.yaml
│   ├── adaptive_profiles.yaml
│   └── mobile_profile.yaml
├── src/t2i/
│   ├── pipeline/
│   │   ├── prompt_encoder.py
│   │   ├── denoiser.py
│   │   ├── decoder.py
│   │   └── generate.py
│   ├── optimize/
│   │   ├── lcm_lora.py
│   │   ├── quantize.py
│   │   ├── prune.py
│   │   └── export_onnx.py
│   ├── cv/
│   │   ├── resize.py
│   │   ├── color.py
│   │   ├── edge_filters.py
│   │   ├── quality.py
│   │   └── artifacts.py
│   ├── adaptive/
│   │   ├── budget.py
│   │   ├── profiler.py
│   │   └── fallback.py
│   ├── eval/
│   │   ├── clip_score.py
│   │   ├── semantic_check.py
│   │   ├── fid.py
│   │   ├── diversity.py
│   │   ├── latency.py
│   │   ├── memory.py
│   │   └── size.py
│   ├── app/
│   │   ├── server.py
│   │   └── ui/
│   └── report/
│       └── make_report.py
├── scripts/
│   ├── download_models.py
│   ├── run_benchmarks.py
│   └── reproduce_all.sh
├── data/
│   ├── prompts/
│   │   ├── simple.jsonl
│   │   ├── complex.jsonl
│   │   └── counting.jsonl
│   └── fid_subset/
│       ├── subset_ids.txt
│       └── SHA256SUMS
├── results/
├── models/
│   └── MANIFEST.json
├── tests/
└── docs/
    ├── architecture.md
    ├── optimization_report.md
    ├── mobile_deployment.md
    ├── progress.md
    └── licenses/
```

Rules:
- Each module has one responsibility, typed function signatures (Python type hints), and a docstring.
- Each module in `cv/`, `adaptive/`, and `eval/` has a unit test in `tests/`.
- No hard-coded constants for model choice, step counts, resolutions, or thresholds. Everything comes from YAML.

---

## 7. Configuration System

All behavior is controlled by YAML configs. The code loads a config, validates it (use Pydantic or dataclasses with validation), and writes the fully resolved config into every results file.

### 7.1 Example: `configs/baseline.yaml`

```yaml
name: baseline
model:
  base: runwayml/stable-diffusion-v1-5   # pin revision in MANIFEST.json
  revision: "<commit-hash>"
precision: fp32
scheduler: ddim                 # or dpm_solver_pp
steps: 50
guidance_scale: 7.5
resolution: [512, 512]
variant: full_unet
lcm_lora: false
quantization: none
attention: default              # no fused or slicing optimizations
vae_tiling: false
classical_postprocess: false
seed_base: 1234
```

### 7.2 Example: `configs/optimized.yaml`

```yaml
name: optimized
model:
  base: runwayml/stable-diffusion-v1-5
  revision: "<commit-hash>"
precision: fp16
scheduler: lcm                  # LCM scheduler when lcm_lora is enabled
steps: 4                        # chosen from the steps sweep, documented
guidance_scale: 1.0             # LCM typically uses low or no CFG; verify in sweep
resolution: [512, 512]
variant: full_unet              # or block_pruned
lcm_lora: true
quantization: none              # or int8_unet
attention: sdpa                 # fused scaled_dot_product_attention
vae_tiling: false
channels_last: true
classical_postprocess: true
cv:
  resize: { mode: none }
  color: { clahe_clip: 2.0, clahe_tile: [8, 8], white_balance: gray_world, vibrance: 0.15 }
  edge: { guided_radius: 8, guided_eps: 0.01 }
  artifacts: { fft_peak_threshold: 6.0, correct: true }
seed_base: 1234
```

### 7.3 Config rules
- Every config field must be used by the code. Unknown keys cause an error.
- Each results file stores the resolved config, git commit hash, and hardware info.

---

## 8. Module Specifications

### 8.1 Deep learning pipeline (mandatory explicit implementation)

#### 8.1.1 `pipeline/prompt_encoder.py`
Responsibilities:
- Tokenize the prompt and negative prompt with the CLIP tokenizer (max length 77).
- Run the CLIP text encoder to get conditional and unconditional embeddings.
- Support long prompts: detect truncation, return a warning flag and the number of dropped tokens, and log the truncation.
- Implement classifier-free guidance by concatenating unconditional and conditional embeddings for the batch.
- Document in comments what each tensor shape is.

#### 8.1.2 `pipeline/denoiser.py`
Responsibilities:
- Initialize latents: `randn` with the seed, shape `[1, 4, H/8, W/8]`, scaled by the scheduler's init noise sigma.
- Set up the scheduler (DDIM, DPM-Solver++, or LCM) with the configured step count.
- For each timestep:
  1. Duplicate latents for CFG.
  2. Scale the model input using the scheduler.
  3. Call the U-Net with the timestep and embeddings.
  4. Split the output into unconditional and conditional predictions.
  5. Apply guidance: `noise = uncond + guidance_scale * (cond - uncond)` (skip when guidance is disabled, as in LCM with guidance 1.0).
  6. Step the scheduler.
- Support step control: a callback allows the adaptive controller to stop early and return the current latent.
- Run under `torch.inference_mode()`.

#### 8.1.3 `pipeline/decoder.py`
Responsibilities:
- Scale latents by the VAE scaling factor (0.18215 for SD v1.5), then decode.
- Convert the output from `[-1, 1]` to `uint8` RGB.
- Support VAE tiling for memory reduction, toggled by config.

#### 8.1.4 `pipeline/generate.py`
Responsibilities:
- Orchestrate: encode → denoise → decode → optional classical post-processing → quality report → metrics logging.
- Return an object containing: image, seed, config hash, timings per stage (encode, denoise, decode, post-process), peak memory, quality report, truncation flag, and fallback log.
- Verification requirement (Milestone M1): for a fixed seed, the explicit pipeline must match the reference library pipeline output within a documented floating-point tolerance (report max and mean absolute pixel difference).

### 8.2 Optimization strategies (`src/t2i/optimize/`)

#### 8.2.1 `lcm_lora.py`
- Load the LCM-LoRA weights and merge or attach them to the U-Net.
- Switch the scheduler to LCM.
- Expose the step count and guidance as config values.

#### 8.2.2 `quantize.py`
- FP16 conversion for U-Net, text encoder, and VAE. Keep any module that degrades quality in FP32 or FP16 and document it.
- INT8 weight quantization of the U-Net and text encoder:
  - Option A: `torch.ao.quantization.quantize_dynamic` for linear layers (quick, CPU).
  - Option B: ONNX Runtime static quantization with a calibration set of 100 to 200 prompts (preferred for quality). Calibration prompts must be separate from the evaluation prompts.
- Record the quantized model size and the quality delta against FP16.

#### 8.2.3 `prune.py`
- Load the block-pruned distilled U-Net variant with the same interface as the full U-Net.
- Verify it accepts the same inputs and returns the same output shape.
- Compare against the full U-Net at matched step counts.

#### 8.2.4 `export_onnx.py`
- Export U-Net and VAE decoder (and text encoder if feasible) to ONNX with fixed input shapes and documented opset.
- Validate numerically against PyTorch: run the same latent and timestep inputs through both and report max absolute error and mean absolute error on the outputs.
- Run the exported graph through ONNX Runtime on CPU with the reference thread count.
- Export is required for the mobile path (Section 12).

### 8.3 Classical computer vision (`src/t2i/cv/`)

Implement each module as a pure function with a config toggle and a unit test. Keep the original image so before/after comparisons are possible.

#### 8.3.1 `resize.py`: resizing and resolution optimization
- Downscale with Lanczos. Upscale with bicubic.
- Adaptive resolution selector: generate at 384×384 or 512×512, then upscale to the target size when the budget is tight.
- Optional unsharp mask after upscaling (tunable radius and amount).

#### 8.3.2 `color.py`: colour correction and enhancement
- Convert RGB to LAB. Apply CLAHE to the L channel (configurable clip limit and tile grid). Convert back to RGB.
- White balance: gray-world or percentile-based (configurable).
- Vibrance or saturation adjustment with a clamp to prevent oversaturation (for example, cap the saturation increase and check the clipped-pixel fraction).

#### 8.3.3 `edge_filters.py`: edge-aware post-processing
- Guided filter for edge-preserving denoising (OpenCV `ximgproc.guidedFilter`). Fall back to a bilateral filter if the contrib module is unavailable.
- Detail enhancement: use the guided-filter base layer, add back a scaled detail layer.
- Edge-preservation score: Sobel or Canny edge-map overlap between original and processed images (report as a number in `[0, 1]`).

#### 8.3.4 `quality.py`: image quality assessment
Compute **at least three** of:
- Laplacian-variance sharpness.
- Colourfulness (Hasler and Süsstrunk).
- RMS contrast.
- Entropy (histogram-based).
- No-reference metric: BRISQUE or NIQE (via `pyiqa`).

Expose a `QualityReport` dataclass with all fields, attached to each generated image.

#### 8.3.5 `artifacts.py`: artifact detection and correction
- Checkerboard detection: 2D FFT of the luminance channel. Find periodic peaks above a threshold relative to the spectral median. Report peak count and strength.
- Over-smoothing detection: regions with local variance below a threshold over large areas. Produce a mask.
- Over-saturation detection: regions where any RGB channel is clipped (0 or 255). Produce a mask.
- Correction: apply a median or non-local-means filter **only inside the flagged mask**, then re-run detection.
- Report artifact counts before and after correction in the QualityReport.

Classical CV acceptance: post-processing must improve at least one quality metric on the artifact-check set, and must not increase the artifact score.

### 8.4 Adaptive generation (`src/t2i/adaptive/`)

#### 8.4.1 `profiler.py`: builds the profile table
- Enumerate candidate configurations: precision × steps × resolution × model variant (and classical post-processing on/off).
- For each configuration, run the benchmark on the target hardware (CPU proxy and GPU if used) over the calibration prompt set.
- Record for each: median latency, p90 latency, peak memory, CLIPScore mean, and model size.
- Write `configs/adaptive_profiles.yaml`. This file is the source of truth. Do not estimate latency.

#### 8.4.2 `budget.py`: budget controller
- Inputs: latency budget (seconds), memory budget (MB), quality preference (`fast`, `balanced`, `quality`).
- Selection rule: among configurations whose profiled latency and memory fit the budget, choose the one with the highest CLIPScore. For `fast`, minimize latency among feasible configurations. For `balanced`, maximize CLIPScore per second of latency among feasible configurations. Document the rule in code.
- If no configuration fits, return the infeasible result and trigger the fallback (Section 8.5).

#### 8.4.3 Progressive option (stretch goal)
- Run a few early steps, decode a low-resolution preview, and stop early if the budget is at risk.
- Report the quality cost of early stopping.

### 8.5 Resource-aware fallback (`src/t2i/adaptive/fallback.py`)

Fallback chain, applied in order when memory or latency is at risk:
1. Reduce resolution (512 → 384) and upscale classically.
2. Switch to the smallest model variant (block-pruned U-Net, INT8 where available).
3. Reduce steps to the minimum acceptable count (for example, 2 to 4 with LCM-LoRA).
4. If all else fails, return a structured error with the reason and the last successful configuration. Never crash the process.

Memory-pressure check:
- Before each generation, read available system memory via `psutil`.
- If free memory is below the threshold in config, skip the upgrade path and start at the fallback level that fits.
- Log every fallback step, with timestamp, reason, and configuration.

### 8.6 Evaluation harness (`src/t2i/eval/`)

#### 8.6.1 `clip_score.py`
- Cosine similarity between CLIP image and text embeddings. Use ViT-L/14 preferred. If ViT-B/32 is used, document it and do not compare it against ViT-L/14 numbers.
- Clip the embedding similarity to the convention used (document the scaling, such as `100 × cos` or raw cosine), and use the same convention everywhere.
- Report mean and standard deviation over the prompt set.

#### 8.6.2 `semantic_check.py`
- Structured checks on complex and counting prompts using an open-vocabulary detector (OWL-ViT or GroundingDINO) or VQA (BLIP-VQA).
- Score three categories separately: object presence, attributes (colours), and spatial relations. Counting and negation prompts are scored on the expected count and the absence of excluded objects.
- Each prompt has expected entities in `data/prompts/*.jsonl`. Output a per-prompt score file.

#### 8.6.3 `fid.py`
- Inception-v3 features on generated and reference images.
- Use at least **1,000** generated images. Report the sample size.
- Report bootstrap confidence intervals.
- Also report KID or CMMD if feasible.
- Sanity check: FID between two disjoint real-image sets of equal size must be near zero. If it is not, stop and fix the harness.

#### 8.6.4 `diversity.py`
- LPIPS diversity score across seeds for a fixed prompt (report mean pairwise LPIPS).
- Use this to measure the diversity cost of low-step and distilled configurations.

#### 8.6.5 `latency.py`
- Wall-clock time for the full pipeline (encode → denoise → decode → post-process).
- 3 warm-up runs, then at least 20 measured runs at fixed settings.
- Report median and p90. Record hardware, thread count, and precision.

#### 8.6.6 `memory.py`
- GPU: `torch.cuda.max_memory_allocated` (reset before each run).
- CPU and ONNX: peak RSS sampled with `psutil` at a fixed interval, or `/usr/bin/time -v`.
- Report model-only and full-pipeline memory separately.

#### 8.6.7 `size.py`
- Sum on-disk sizes of all deployed weight files per component (text encoder, U-Net, VAE) and total.
- For INT8 and ONNX variants, report the exported artifact size.

### 8.7 Application (`src/t2i/app/`)

- Start with `server.py` (FastAPI) or Gradio.
- Inputs: prompt, negative prompt, seed, resolution, step count, mode (`baseline`, `optimized`, `adaptive`), latency budget, memory budget, quality preference, classical post-processing toggle.
- Outputs: generated image, before/after post-processing comparison when enabled, quality report, latency, peak memory, fallback log (if any).
- Fully local. No external calls.
- The `ui/` folder contains a minimal front end. Keep it simple.

### 8.8 Reporting (`src/t2i/report/make_report.py`)
- Read raw JSON logs in `results/`. Never read numbers from memory or from another document.
- Generate: summary tables (CSV and Markdown), Pareto plots (CLIPScore and FID vs. latency and memory), steps sweep plots, precision comparison plots, degradation analysis plots.
- Every table and plot records the run IDs it was built from.

---

## 9. Datasets and Prompt Sets

### 9.1 FID evaluation set
- Subset of **1,000 to 5,000** captions from MS-COCO 2014 validation.
- Reference images: the matching real COCO validation images.
- Save the subset IDs to `data/fid_subset/subset_ids.txt` with `SHA256SUMS` for all files.
- Use fixed seeds for generation.

### 9.2 Prompt sets (`data/prompts/*.jsonl`)
Each line is one JSON object with these fields: `id`, `category`, `prompt`, `negative_prompt` (optional), `expected_entities` (list), `expected_attributes` (list), `expected_relations` (list), `expected_counts` (object), `excluded_entities` (list).

| File | Count | Content |
|------|-------|---------|
| `simple.jsonl` | 50 | Single object, single attribute. |
| `complex.jsonl` | 100 | Multiple objects, attributes, and spatial relations (for example, "a red bicycle to the left of a blue bench under a tree"). |
| `counting.jsonl` | 30 | Counting and negation (for example, "three apples on a table, no bananas"). |
| `calibration.jsonl` | 100 to 200 | Separate from all evaluation sets. Used only for INT8 calibration and adaptive profiling. |

Rules:
- Evaluation prompts and calibration prompts must not overlap. Check this with a script.
- Measure the truncation rate at 77 tokens for each set and report results on truncated and non-truncated subsets separately.

### 9.3 Artifact-check set
- At least 30 images (generated and/or real) for testing classical post-processing and artifact detection. Include some known-artifact examples (for example, deliberately upscaled or quantized outputs).

---

## 10. Experiment Matrix

Run the following with fixed seeds and the same prompt sets. Each run writes a JSON file to `results/`.

| Experiment | Variables | Metrics |
|------------|-----------|---------|
| Steps sweep | Steps: 1, 2, 4, 8, 16, 25, 50. Run for baseline and optimized. | CLIPScore, FID, semantic accuracy, latency |
| Precision study | FP32, FP16, INT8 weights (with calibration) | CLIPScore, FID, latency, memory, size |
| Variant comparison | Full U-Net vs. block-pruned U-Net at matched steps | CLIPScore, FID, latency, size |
| Classical post-processing | On vs. off | QualityReport metrics, artifact counts, CLIPScore |
| Diversity | Low-step configs across seeds | LPIPS diversity |
| Degradation study | For each compression setting, plot metrics against steps and precision | Breakpoint identification with example images |

Rules:
- The steps sweep and precision study use the full evaluation prompt sets only at Milestone M5. Use reduced sets for development iterations and label them clearly as reduced.
- Keep an index file `results/index.csv` listing every run: run ID, config name, date, git commit, hardware, and path to raw log.

---

## 11. Milestones (Tasks and Exit Criteria)

Execute in order. At the end of each milestone:
1. Commit a runnable state.
2. Append a status note to `docs/progress.md`.
3. Report the exit-criterion status to the user.
4. Wait for approval before starting the next milestone, unless the user has told you to proceed continuously.

---

### M0: Environment and Baseline (Days 1 to 2)

**Tasks**
1. Create the repository layout (Section 6) and `pyproject.toml` with pinned dependencies.
2. Write `scripts/download_models.py` and generate `models/MANIFEST.json`.
3. Record hardware and software info in `results/env.json` (script that captures it).
4. Run the unmodified SD v1.5 baseline with `configs/baseline.yaml` at 512×512, 50 steps, one prompt.
5. Save the image, the seed, and the latency measurement to `results/`.

**Exit criterion:** The baseline runs end-to-end and produces a logged result with the env file.

---

### M1: Core Pipeline (Days 3 to 5)

**Tasks**
1. Implement `prompt_encoder.py`, `denoiser.py`, `decoder.py`, and `generate.py` explicitly (Section 8.1).
2. Implement CFG explicitly.
3. Write a verification script that runs the library reference pipeline and the explicit pipeline with the same seed, prompt, and config.
4. Report max and mean absolute pixel difference, and the latent difference.
5. Add unit tests for the tokenizer truncation flag and the scheduler step count.

**Exit criterion:** Explicit pipeline output matches the reference output within a documented tolerance for a fixed seed.

---

### M2: Classical CV Modules (Days 6 to 8)

**Tasks**
1. Implement all modules in Section 8.3 with unit tests.
2. Build a before/after gallery for 10 images (saved in `results/cv_gallery/`).
3. Compute QualityReport metrics before and after each module.
4. Run artifact detection on the artifact-check set and record counts before and after correction.

**Exit criterion:** Each module has passing tests and a measured effect on at least one metric, logged in results.

---

### M3: Optimization (Days 9 to 13)

**Tasks**
1. Implement LCM-LoRA (step reduction), FP16, INT8 quantization, and the block-pruned variant (Section 8.2).
2. Export the U-Net and VAE decoder to ONNX (Section 8.2.4).
3. Validate ONNX against PyTorch: report max and mean absolute error on latents and outputs.
4. Measure each optimization individually against the baseline on a reduced prompt set: latency, memory, size, and CLIPScore.

**Exit criterion:** At least three optimizations are implemented, each is individually measured, and ONNX export is validated numerically.

---

### M4: Evaluation Harness (Days 14 to 16)

**Tasks**
1. Implement CLIP scoring, semantic checks, FID, diversity, latency, memory, and size (Section 8.6).
2. Run the sanity checks:
   - Baseline CLIPScore falls in the expected range for SD v1.5 with ViT-L/14 (roughly 0.25 to 0.35 on the cosine scale). If it falls outside, investigate before continuing.
   - FID between two disjoint real-image sets is near zero.
3. Run the baseline through the full harness to establish reference numbers.
4. Make `scripts/run_benchmarks.py` run the harness from one command.

**Exit criterion:** One command produces a complete baseline results file, and both sanity checks pass.

---

### M5: Experiment Matrix and Analysis (Days 17 to 20)

**Tasks**
1. Run the full experiment matrix (Section 10) on the full evaluation sets.
2. Produce Pareto plots (CLIPScore and FID vs. latency and memory) with the operating point marked.
3. Write the degradation analysis: where quality drops with fewer steps and lower precision, with example images at each breakpoint.
4. Select the **operating point** (the configuration used as "optimized" in the final report) and record the reason.

**Exit criterion:** All tables and plots are generated from raw logs by `make_report.py`, and the operating point is chosen and justified.

---

### M6: Adaptive Controller and Fallback (Days 21 to 23)

**Tasks**
1. Build the profile table with `profiler.py` and write `configs/adaptive_profiles.yaml`.
2. Implement `budget.py` and `fallback.py` (Sections 8.4 and 8.5).
3. Test three budget scenarios:
   - **Generous:** budget comfortably above the best configuration.
   - **Tight:** budget that forces a smaller configuration.
   - **Infeasible:** budget no configuration can meet, triggering fallback.
4. Simulate a memory cap (for example, `ulimit -v` or Docker `--memory`) and confirm the fallback chain triggers and produces a logged valid image or a clear error.

**Exit criterion:** Each scenario selects a configuration that meets its budget, or triggers a documented fallback. The memory-capped run is logged.

---

### M7: App, Mobile Export, and Documentation (Days 24 to 27)

**Tasks**
1. Build the app (Section 8.7) with mode selection, budget inputs, and the output gallery with quality reports.
2. Produce the mobile export: ONNX (required), and TFLite or Core ML if feasible.
3. Run the mobile-proxy benchmark (Section 12).
4. Write `docs/architecture.md`, `docs/optimization_report.md`, `docs/mobile_deployment.md`, and the README.
5. Make `scripts/reproduce_all.sh` run from a clean clone and regenerate the report.
6. Verify a clean-clone reproduction on the target machine and record the runtime.

**Exit criterion:** `reproduce_all.sh` runs from a clean clone and regenerates the report. Documentation is complete.

---

## 12. Mobile Deployment Path

1. **ONNX export (required).** Export U-Net and VAE decoder with fixed shapes. Validate numerically against PyTorch.
2. **Mobile-proxy benchmark (required).**
   - Run the ONNX models with ONNX Runtime on CPU, limited to the reference thread count.
   - Optionally cap memory.
   - Report median and p90 latency per 512×512 image, peak RSS, and model storage.
   - Compare against the reference mobile profile budgets (1.5 GB working memory, 1 GB storage, under 10 s latency).
3. **TFLite or Core ML (if feasible).** Convert the ONNX graph using the appropriate converter. If conversion fails, document the failure, the converter version, and the blocking operator.
4. **Device or emulator run (if available).** Run the exported model on the device or emulator. Report these numbers separately from the proxy numbers and state the device model, OS, and delegate (CPU, NNAPI, or GPU).
5. **`docs/mobile_deployment.md`** must state: the profile used, what was measured, what was not measured, the caveats, and the steps a developer would take to run the model on a device.

Labeling rule: every mobile number carries one of these labels: `PROXY (CPU, N threads)` or `DEVICE (model, OS, delegate)`.

---

## 13. Testing Requirements

### 13.1 Unit tests (required)
- Every module in `src/t2i/cv/`, `src/t2i/adaptive/`, and `src/t2i/eval/` has tests.
- Tests for the pipeline: tokenizer truncation, scheduler step count, CFG shape handling, latent scaling.
- Tests for the budget controller: each budget scenario selects the expected configuration from a fixture profile table.
- Tests for the fallback chain: each level is reached when the corresponding condition is simulated.

### 13.2 Integration test (required)
- Runs the full pipeline end-to-end in reduced-step (for example, 2 steps), reduced-resolution (for example, 256×256) mode.
- Must run in CI without the full model download. Use a small or mock pipeline mode (a mock denoiser that returns deterministic tensors is acceptable for CI).

### 13.3 Regression tests (recommended)
- Fixed-seed golden image checks at reduced settings, with a tolerance.

### 13.4 Commands
- `pytest tests/unit` for unit tests.
- `pytest tests/integration --mock` for CI mode.
- `pytest tests/integration` (full, with models) for local validation.

---

## 14. Reporting Requirements for the Final Report

`docs/optimization_report.md` must contain, in this order:

1. **Summary table.** Baseline vs. optimized vs. adaptive: latency (median and p90), peak memory, model size, CLIPScore (mean ± std), FID (with sample size and CI), and semantic accuracy for simple, complex, and counting prompts. Each row includes hardware and run IDs.
2. **Optimization breakdown.** Individual effect of each strategy: steps, precision, pruning, classical post-processing, and the combined effect.
3. **Quality vs. cost Pareto plot.** Operating point marked.
4. **Degradation analysis.** Where quality drops with fewer steps and lower precision. Example images at each breakpoint.
5. **Adaptive controller results.** The three budget scenarios, the selected configuration for each, and whether each budget was met.
6. **Fallback demonstration.** Logs and images from the memory-capped run.
7. **Diversity results.** LPIPS diversity for the low-step configurations.
8. **Limitations.** Mobile-proxy caveat, FID sample size, metric limitations, truncation effects, and anything not verified.
9. **Reproduction instructions.** Exact commands, expected runtimes, and the hardware used.

Writing rules:
- Every number traces to a file in `results/`. Cite the run ID next to each number.
- Where a target is missed, state the measured value, the target, and the likely cause.
- Do not use the words "significant" or "proves" without a statistical test or a clear sample-size statement.

---

## 15. Acceptance Checklist (Definition of Done)

### Functional
- [ ] The app generates images from prompts locally, with no hosted API calls.
- [ ] Baseline, optimized, and adaptive modes run and log metrics.
- [ ] All classical CV modules run and are toggled by config.
- [ ] Fallback chain triggers under a simulated memory cap and produces a logged valid image or a clear error.

### Performance (same hardware, measured by the harness)
- [ ] Optimized end-to-end latency is at least 3x lower than baseline at the chosen operating point.
- [ ] Peak memory is at least 40% lower than baseline.
- [ ] Total model size is at least 50% smaller than baseline.
- [ ] Mobile-proxy benchmark is reported, with thread count and memory cap stated.

### Quality
- [ ] CLIPScore is within 10% relative of baseline at the operating point.
- [ ] FID is within 25% relative of baseline, or the report explains the gap and shows where degradation starts.
- [ ] Semantic consistency is reported separately for simple, complex, and counting prompts.
- [ ] Classical post-processing improves at least one quality metric with no new artifacts on the artifact-check set.

### Reproducibility and documentation
- [ ] Fixed seeds are used everywhere and recorded in every result.
- [ ] Hardware, software versions, and model checksums are recorded in each results file.
- [ ] Every number in the report traces to a raw log file.
- [ ] README lets a new developer reproduce the main results, with the time required stated.
- [ ] Licenses are recorded in `docs/licenses/`.

### Testing
- [ ] Unit tests exist for every module in `cv/`, `adaptive/`, and `eval/`.
- [ ] Integration test runs end-to-end in reduced mode in CI.

---

## 16. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| No physical mobile device | Mobile numbers may be unverifiable | CPU proxy with thread and memory caps. Label all results. Use a device or emulator only if available. |
| INT8 breaks image quality | Large quality loss | Start with FP16. Use calibration data. Keep VAE in FP16 if needed. |
| LCM-LoRA at very low steps reduces diversity | Lower diversity and higher FID | Measure LPIPS diversity. Report the trade-off. |
| FID unstable at small N | Misleading comparisons | At least 1,000 images, fixed subsets, bootstrap confidence intervals. |
| Prompts truncated at 77 tokens | Missed objects in complex prompts | Measure truncation rate. Report truncated and non-truncated subsets separately. |
| License restrictions | Compliance risk | Record each license. Include license files and usage notes. |
| Long CPU build time | Schedule overrun | Use reduced prompt sets for development. Run the full matrix only at M5. |
| Silent metric errors | Invalid conclusions | Run the sanity checks in M4 (baseline CLIPScore range, real-vs-real FID near zero). |
| Model revisions change | Irreproducible results | Pin commit hashes in MANIFEST.json and configs. |
| Evaluation and calibration prompts overlap | Inflated quality results | Automated overlap check in CI. |

---

## 17. Working Rules, Progress Log, and Stop Points

### 17.1 Progress log
Maintain `docs/progress.md` with one entry per work session:
- Date
- Milestone and task
- Decisions made and why
- Deviations from this file (with user approval)
- Open issues
- Next steps

### 17.2 Stop points
Stop and report to the user at:
- The end of every milestone (with exit-criterion status).
- Any proposed change to the model choice, the mobile profile, the FID dataset, or the success criteria.
- Any target that is missed by more than the tolerance in Section 2.
- Any blocked step (for example, a model that cannot be downloaded, a missing device, or a license issue).

### 17.3 Communication rules
- Report measured values with their units, sample sizes, and run IDs.
- If a result is surprising, check the harness first (sanity checks, seeds, config hash) before reporting it as a finding.
- Be explicit about what was not verified.

### 17.4 Code quality rules
- Type hints on all public functions.
- Docstrings on all modules and public functions.
- No dead code, and no commented-out experiments left in the main branch.
- Format with `ruff` and `black`. Lint must pass before each milestone is marked complete.

---

## Appendix A: Quick Start Commands (target state)

```bash
# Set up
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_models.py

# Quick smoke test
pytest tests/integration --mock

# Single generation
python -m t2i.pipeline.generate --config configs/optimized.yaml --prompt "a red bicycle to the left of a blue bench under a tree" --seed 1234

# Full benchmark and report
bash scripts/reproduce_all.sh
```

## Appendix B: Glossary

- **CFG:** Classifier-free guidance. Combines unconditional and conditional noise predictions to strengthen prompt adherence.
- **CLIPScore:** Cosine similarity between CLIP image and text embeddings, used for text-image alignment.
- **FID:** Fréchet Inception Distance. Compares distributions of Inception features for generated and reference images.
- **LCM-LoRA:** Latent Consistency Model distillation delivered as a LoRA adapter, enabling few-step sampling.
- **LPIPS:** Learned Perceptual Image Patch Similarity. Used here for diversity across seeds.
- **Operating point:** The configuration chosen as "optimized" for the final comparison.
- **Profile table:** Benchmarked latency, memory, and quality for each candidate configuration, used by the adaptive controller.
- **Proxy benchmark:** A measurement on a development machine restricted to mobile-like resources, used when a physical device is unavailable.
