"""
t2i.adaptive.budget — Budget-aware configuration selector.

Selects the best configuration from the adaptive profile table given latency
and memory budgets and a quality preference.

Selection rules (documented as required by the spec):
- ``fast``:    among feasible configurations, minimise median latency.
- ``balanced``:among feasible, maximise CLIPScore / median_latency_s.
- ``quality``: among feasible, maximise CLIPScore.

If no configuration is feasible, returns None and sets ``infeasible=True``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import yaml

logger = logging.getLogger(__name__)

QualityPreference = Literal["fast", "balanced", "quality"]


@dataclass
class ProfileEntry:
    """One row in the adaptive_profiles.yaml profile table."""

    config_id: str
    precision: str
    scheduler: str
    steps: int
    resolution: list[int]
    variant: str
    lcm_lora: bool
    quantization: str
    attention: str
    classical_postprocess: bool
    median_latency_s: float
    p90_latency_s: float
    peak_memory_mb: float
    clip_score_mean: float
    model_size_mb: float
    hardware_label: str


@dataclass
class BudgetResult:
    """Return value of :func:`select_config`."""

    selected: Optional[ProfileEntry]
    infeasible: bool
    reason: str  # human-readable selection rationale
    all_feasible: list[ProfileEntry]  # for debugging


def load_profiles(profiles_path: str) -> list[ProfileEntry]:
    """
    Load the adaptive profile table from YAML.

    Parameters
    ----------
    profiles_path:
        Path to ``configs/adaptive_profiles.yaml``.

    Returns
    -------
    list[ProfileEntry]
        One entry per benchmarked configuration.
    """
    with open(profiles_path, "r") as fh:
        raw = yaml.safe_load(fh)

    entries = []
    for row in raw.get("profiles", []):
        entries.append(ProfileEntry(**row))
    return entries


def select_config(
    profiles: list[ProfileEntry],
    latency_budget_s: float,
    memory_budget_mb: float,
    quality_preference: QualityPreference = "balanced",
) -> BudgetResult:
    """
    Select the best configuration that fits within the stated budgets.

    Parameters
    ----------
    profiles:
        List of profiled configurations (from adaptive_profiles.yaml).
    latency_budget_s:
        Maximum acceptable median latency in seconds.
    memory_budget_mb:
        Maximum acceptable peak memory in MiB.
    quality_preference:
        Selection criterion among feasible configurations.

    Returns
    -------
    BudgetResult
        Selected configuration (or None if infeasible).
    """
    # Filter to configurations that fit both budgets
    feasible = [
        p
        for p in profiles
        if p.median_latency_s <= latency_budget_s and p.peak_memory_mb <= memory_budget_mb
    ]

    if not feasible:
        logger.warning(
            "No configuration fits latency=%.1fs / memory=%.0fMB budget.",
            latency_budget_s,
            memory_budget_mb,
        )
        return BudgetResult(
            selected=None,
            infeasible=True,
            reason=(
                f"No configuration fits latency≤{latency_budget_s}s "
                f"and memory≤{memory_budget_mb}MB."
            ),
            all_feasible=[],
        )

    # Apply selection rule based on quality_preference
    if quality_preference == "fast":
        # Minimise median latency
        chosen = min(feasible, key=lambda p: p.median_latency_s)
        reason = (
            f"'fast' mode: chose config with lowest median latency ({chosen.median_latency_s:.2f}s)"
        )

    elif quality_preference == "quality":
        # Maximise CLIPScore among feasible
        chosen = max(feasible, key=lambda p: p.clip_score_mean)
        reason = (
            f"'quality' mode: chose config with highest CLIPScore ({chosen.clip_score_mean:.4f})"
        )

    else:  # "balanced"
        # Maximise CLIPScore per second (efficiency)
        def efficiency(p: ProfileEntry) -> float:
            return p.clip_score_mean / max(p.median_latency_s, 1e-6)

        chosen = max(feasible, key=efficiency)
        eff = efficiency(chosen)
        reason = (
            f"'balanced' mode: chose config with highest CLIPScore/s "
            f"({chosen.clip_score_mean:.4f} / {chosen.median_latency_s:.2f}s = {eff:.4f})"
        )

    logger.info("Budget controller selected: %s — %s", chosen.config_id, reason)
    return BudgetResult(
        selected=chosen,
        infeasible=False,
        reason=reason,
        all_feasible=feasible,
    )
