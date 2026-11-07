"""Tests for t2i.adaptive.budget module."""
import pytest
from t2i.adaptive.budget import ProfileEntry, select_config, BudgetResult


def _make_profile(config_id, latency, memory, clip_score) -> ProfileEntry:
    return ProfileEntry(
        config_id=config_id,
        precision="fp16",
        scheduler="lcm",
        steps=4,
        resolution=[512, 512],
        variant="full_unet",
        lcm_lora=True,
        quantization="none",
        attention="sdpa",
        classical_postprocess=False,
        median_latency_s=latency,
        p90_latency_s=latency * 1.2,
        peak_memory_mb=memory,
        clip_score_mean=clip_score,
        model_size_mb=1000.0,
        hardware_label="CPU_4cores",
    )


PROFILES = [
    _make_profile("fast_small", latency=2.0, memory=800, clip_score=0.28),
    _make_profile("balanced_mid", latency=5.0, memory=1200, clip_score=0.31),
    _make_profile("quality_large", latency=9.0, memory=1500, clip_score=0.33),
    _make_profile("slow_expensive", latency=15.0, memory=2000, clip_score=0.35),
]


def test_fast_mode_selects_lowest_latency():
    result = select_config(PROFILES, latency_budget_s=10.0, memory_budget_mb=2000, quality_preference="fast")
    assert result.selected is not None
    assert result.selected.config_id == "fast_small"


def test_quality_mode_selects_highest_clip():
    result = select_config(PROFILES, latency_budget_s=10.0, memory_budget_mb=2000, quality_preference="quality")
    assert result.selected is not None
    assert result.selected.config_id == "quality_large"


def test_balanced_mode_selects_best_efficiency():
    result = select_config(PROFILES, latency_budget_s=10.0, memory_budget_mb=2000, quality_preference="balanced")
    assert result.selected is not None
    # Efficiency = clip_score / latency
    # fast_small: 0.28/2.0=0.14, balanced_mid: 0.31/5.0=0.062, quality_large: 0.33/9.0=0.037
    assert result.selected.config_id == "fast_small"


def test_tight_budget_excludes_slow_configs():
    result = select_config(PROFILES, latency_budget_s=3.0, memory_budget_mb=1000, quality_preference="balanced")
    assert result.selected is not None
    assert result.selected.config_id == "fast_small"


def test_infeasible_budget_returns_none():
    result = select_config(PROFILES, latency_budget_s=0.5, memory_budget_mb=100, quality_preference="fast")
    assert result.selected is None
    assert result.infeasible is True
    assert result.all_feasible == []


def test_memory_constraint_filters_configs():
    result = select_config(PROFILES, latency_budget_s=20.0, memory_budget_mb=900, quality_preference="quality")
    # Only fast_small fits (800 MB)
    assert result.selected is not None
    assert result.selected.config_id == "fast_small"
