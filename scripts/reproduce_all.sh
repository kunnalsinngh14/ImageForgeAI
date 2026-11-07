#!/usr/bin/env bash
# scripts/reproduce_all.sh — Reproduce the main results from a clean clone.
#
# Usage:
#   bash scripts/reproduce_all.sh
#
# Expected runtime: ~2–4 hours on CPU (see docs/optimization_report.md).
# Hardware used for reference: see results/env.json.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== t2i Reproduce All ==="
echo "Working directory: $REPO_ROOT"
echo "Start time: $(date)"

# 1. Set up environment
echo ""
echo "--- Step 1: Environment setup ---"
python -m venv .venv 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
pip install -e ".[dev]" --quiet

# 2. Capture environment
echo ""
echo "--- Step 2: Capture environment ---"
python scripts/capture_env.py --output results/env.json

# 3. Download models
echo ""
echo "--- Step 3: Download models ---"
python scripts/download_models.py --models-dir models

# 4. Run integration tests (mock mode — fast, no model needed)
echo ""
echo "--- Step 4: Integration tests (mock) ---"
pytest tests/integration --mock -v

# 5. Unit tests
echo ""
echo "--- Step 5: Unit tests ---"
pytest tests/unit -v

# 6. Baseline benchmark (reduced set for speed)
echo ""
echo "--- Step 6: Baseline benchmark ---"
python scripts/run_benchmarks.py \
    --config configs/baseline.yaml \
    --prompts data/prompts/simple.jsonl \
    --output results/ \
    --runs 3 \
    --warmup 1

# 7. Optimized benchmark
echo ""
echo "--- Step 7: Optimized benchmark ---"
python scripts/run_benchmarks.py \
    --config configs/optimized.yaml \
    --prompts data/prompts/simple.jsonl \
    --output results/ \
    --runs 3 \
    --warmup 1

# 8. Generate report
echo ""
echo "--- Step 8: Generate report ---"
python -m t2i.report.make_report \
    --results-dir results/ \
    --output docs/optimization_report.md

echo ""
echo "=== Done. End time: $(date) ==="
echo "Results are in results/ and docs/optimization_report.md"
