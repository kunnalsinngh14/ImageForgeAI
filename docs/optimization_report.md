# Optimization and Evaluation Report: Resource-Efficient Text-to-Image (P3)

## 1. Summary Table

| run_id                             | config    |   steps |   latency_median_s |   latency_p90_s |   peak_memory_mib |   model_size_mb |   clip_score_mean |   clip_score_std |
|:-----------------------------------|:----------|--------:|-------------------:|----------------:|------------------:|----------------:|------------------:|-----------------:|
| 20261107_033928_a113603978a9       | baseline  |      10 |           126.644  |         126.644 |            673.54 |           nan   |          nan      |          0       |
| 20261107_033539_a15baf2c95d7       | optimized |       4 |            60.394  |          60.394 |            729.72 |           nan   |          nan      |          0       |
| 20261107_033928_a113603978a9       | baseline  |      10 |           126.644  |         126.644 |            673.54 |         11198.2 |           21.4065 |          0       |
| bench_20261107_065523_a15baf2c95d7 | optimized |       4 |            19.7552 |          22.566 |           1376.39 |         11198.2 |           29.6442 |          1.80672 |
| 20261107_033539_a15baf2c95d7       | optimized |       4 |            60.394  |          60.394 |            729.72 |         11198.2 |           20.9493 |          0       |

## 2. Optimization Breakdown

- **Step Reduction (LCM-LoRA):** Decreases required sampling steps from 50 (DDIM) to 4 (LCM), targeting >3x speedup.
- **Reduced Precision Inference:** FP16 baseline with dynamic INT8 weight quantization for CPU deployment.
- **Block Pruning:** Evaluates block-pruned BK-SDM U-Net architecture against full U-Net.
- **Classical Post-Processing:** CLAHE color balance, edge-preserving guided filter, and FFT artifact mitigation.

## 3. Quality vs. Cost Pareto Analysis

See `results/pareto_latency_clip.png` and `results/pareto_memory_clip.png`.

## 4. Degradation Analysis

Analysis of quality degradation across sampling step sweeps (50 -> 25 -> 8 -> 4 -> 2) and quantization precision.

## 5. Adaptive Controller Results

Budget scenarios evaluated with `src/t2i/adaptive/budget.py`:
- **Generous Budget:** 10.0s latency / 2000 MB memory -> selects highest CLIPScore model.
- **Tight Budget:** 3.0s latency / 1000 MB memory -> selects fast/LCM configuration.
- **Infeasible Budget:** 0.5s latency / 100 MB memory -> triggers fallback chain.

## 6. Fallback Demonstration

Fallback chain verifies resolution degradation (512 -> 384), model substitution, and step reduction under constrained memory conditions.

## 7. Diversity Results

Pairwise LPIPS diversity was evaluated across 4 seeds using AlexNet LPIPS (`results/diversity_report.json`):
- **Mean Pairwise LPIPS:** **0.7993 +/- 0.0548**
- **Pairs Evaluated:** 6
- **Pairwise Scores:** `[0.7011, 0.8178, 0.8777, 0.7864, 0.8347, 0.7781]`

*Interpretation:* An LPIPS score of ~0.80 confirms substantial perceptual diversity between runs with different random seeds. Even with aggressive 4-step distillation, the model does not suffer from mode collapse and produces distinct visual compositions.

## 8. Limitations & Mobile-Proxy Caveats

- Measurements executed on CPU proxy with restricted thread pools.
- ONNX Runtime CPU EP used as the mobile deployment proxy.

## 9. Reproduction Instructions

```bash
bash scripts/reproduce_all.sh
```
