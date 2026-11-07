# Mobile Deployment Path & Proxy Benchmark Specification

## 1. Overview & Reference Mobile Profile

This document outlines the deployment strategy, model export pathways, and benchmark results for running the optimized text-to-image pipeline on mobile-class target devices.

The target profile is defined in `configs/mobile_profile.yaml`:
- **Target Device Class:** Android (Mid-to-High Tier), 8 GB RAM, 4–6 CPU cores, optional NNAPI / GPU delegates.
- **Model Working-Memory Budget:** ≤ 1.5 GB (1536 MiB).
- **Model Storage Budget:** ≤ 1.0 GB (1024 MiB).
- **Latency Target:** Under 10 seconds per 512×512 image (Stretch Goal: < 5 seconds).

---

## 2. Model Export Strategy

### 2.1 ONNX Export (`src/t2i/optimize/export_onnx.py`)
- **U-Net:** Exported with static input shapes:
  - Latent input: `[2, 4, 64, 64]` (CFG batch = 2) or `[1, 4, 64, 64]` (LCM CFG-free)
  - Timestep: `[1]` (int64)
  - Context embeddings: `[2, 77, 768]` or `[1, 77, 768]`
  - Output noise prediction: `[2, 4, 64, 64]`
  - Opset: 17
- **VAE Decoder:** Exported with static input `[1, 4, 64, 64]` and output `[1, 3, 512, 512]`.
- **Numerical Validation:** Validated against PyTorch output with max absolute difference and mean absolute error reported.

### 2.2 Quantization (`src/t2i/optimize/quantize.py`)
- Weight quantization to INT8 via ONNX Runtime dynamic/static quantization.
- Parameter memory footprint reduction: ~4x compared to FP32, ~2x compared to FP16.

---

## 3. Mobile-Proxy Benchmark Setup

In the absence of a tethered physical Android device, the evaluation harness executes a **CPU Proxy Benchmark**:
- **Hardware Profile:** Host CPU constrained to reference thread count (4 threads).
- **Runtime:** ONNX Runtime CPU Execution Provider with `intra_op_num_threads = 4`.
- **Memory Tracking:** Continuous RSS sampling using `psutil`.

### Labeling Convention
Per Section 12:
- Proxy results: `PROXY (CPU, 4 threads)`
- Physical hardware results: `DEVICE (<model>, <OS>, <delegate>)`

---

## 4. Developer Deployment Guide (Steps to Run on Physical Device)

To deploy the exported artifacts to an Android device using ONNX Runtime Mobile or TFLite:

1. **Export the models:**
   ```bash
   python -m t2i.optimize.export_onnx --unet-out models/unet.onnx --vae-out models/vae.onnx
   ```
2. **Quantize the ONNX models to INT8:**
   ```bash
   python -c "from t2i.optimize.quantize import quantize_onnx_int8; from pathlib import Path; quantize_onnx_int8(Path('models/unet.onnx'))"
   ```
3. **Package with Android App (ONNX Runtime Mobile):**
   - Add `com.microsoft.onnxruntime:onnxruntime-android` to `app/build.gradle`.
   - Place `unet_int8.onnx` and `vae.onnx` in the Android `assets/` directory.
   - Initialize `OrtSession` with `OrtSession.SessionOptions().setIntraOpNumThreads(4)` and enable the NNAPI execution provider.
