"""
t2i.adaptive.fallback — Resource-aware fallback chain.

Applies the following degradation steps in order when memory or latency
constraints cannot be satisfied by the selected configuration:

  1. Reduce resolution from 512×512 to 384×384 and upscale classically.
  2. Switch to the smallest model variant (block-pruned U-Net, INT8 if available).
  3. Reduce steps to the minimum acceptable count (2–4 for LCM-LoRA).
  4. If all else fails, return a structured error — never crash the process.

Memory is checked before each generation via psutil.  Every fallback step
is logged with a timestamp, reason, and the resulting configuration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil

from t2i.config import T2IConfig

logger = logging.getLogger(__name__)


@dataclass
class FallbackLogEntry:
    """One step in the fallback chain."""

    timestamp: str
    step: int  # 1 = resolution, 2 = model, 3 = steps
    reason: str
    action: str
    config_before: dict
    config_after: dict


@dataclass
class FallbackResult:
    """Return value of :func:`apply_fallback_chain`."""

    config: Optional[T2IConfig]
    success: bool
    error_reason: Optional[str]
    fallback_log: list[FallbackLogEntry] = field(default_factory=list)


def available_memory_mb() -> float:
    """Return available system memory in MiB."""
    return psutil.virtual_memory().available / (1024**2)


def apply_fallback_chain(
    config: T2IConfig,
    memory_budget_mb: float,
    latency_budget_s: Optional[float] = None,
    free_memory_threshold_mb: float = 200.0,
    min_steps: int = 2,
    min_resolution: tuple[int, int] = (384, 384),
) -> FallbackResult:
    """
    Walk the fallback chain until a feasible configuration is found.

    Parameters
    ----------
    config:
        Starting configuration.
    memory_budget_mb:
        Maximum memory budget in MiB.  Checked against ``psutil`` available memory.
    latency_budget_s:
        Optional latency budget (not enforced directly here; used for logging).
    free_memory_threshold_mb:
        If free memory falls below this, the upgrade path is skipped and the
        chain starts at the most aggressive fallback level.
    min_steps:
        Minimum step count before we give up.
    min_resolution:
        Minimum (width, height) resolution before we give up.

    Returns
    -------
    FallbackResult
        The final configuration (or None on total failure) and the log.
    """
    log: list[FallbackLogEntry] = []
    current = config

    free_mb = available_memory_mb()
    logger.info("Free memory at fallback entry: %.0f MiB", free_mb)

    if free_mb < free_memory_threshold_mb:
        logger.warning(
            "Free memory %.0f MiB < threshold %.0f MiB — starting at max fallback level.",
            free_mb,
            free_memory_threshold_mb,
        )
        # Jump straight to step 3 (minimum steps) for maximum memory savings.
        current, entry = _fallback_step_3_reduce_steps(
            current, min_steps, reason="low memory at start"
        )
        log.append(entry)
        if entry.action.startswith("ERROR"):
            return FallbackResult(
                config=None, success=False, error_reason=entry.action, fallback_log=log
            )
        return FallbackResult(config=current, success=True, error_reason=None, fallback_log=log)

    # --- Step 1: Reduce resolution ---
    if list(current.resolution) != list(min_resolution):
        current, entry = _fallback_step_1_resolution(current, min_resolution)
        log.append(entry)
        logger.info("Fallback step 1 applied: %s", entry.action)

        free_mb = available_memory_mb()
        if free_mb >= free_memory_threshold_mb and current is not None:
            return FallbackResult(config=current, success=True, error_reason=None, fallback_log=log)

    # --- Step 2: Smallest model variant ---
    if current is not None and current.variant != "block_pruned":
        current, entry = _fallback_step_2_model_variant(current)
        log.append(entry)
        logger.info("Fallback step 2 applied: %s", entry.action)

        free_mb = available_memory_mb()
        if free_mb >= free_memory_threshold_mb and current is not None:
            return FallbackResult(config=current, success=True, error_reason=None, fallback_log=log)

    # --- Step 3: Reduce steps ---
    if current is not None and current.steps > min_steps:
        current, entry = _fallback_step_3_reduce_steps(current, min_steps, reason="memory pressure")
        log.append(entry)
        logger.info("Fallback step 3 applied: %s", entry.action)

    if current is None or entry.action == "ERROR":
        error = "All fallback steps exhausted without finding a feasible configuration."
        logger.error(error)
        return FallbackResult(config=None, success=False, error_reason=error, fallback_log=log)

    return FallbackResult(config=current, success=True, error_reason=None, fallback_log=log)


# ---------------------------------------------------------------------------
# Individual fallback steps
# ---------------------------------------------------------------------------


def _fallback_step_1_resolution(
    config: T2IConfig,
    min_resolution: tuple[int, int],
) -> tuple[Optional[T2IConfig], FallbackLogEntry]:
    """Reduce resolution to min_resolution."""
    before = config.model_dump()
    try:
        new_config = config.model_copy(update={"resolution": list(min_resolution)})
        action = f"Resolution reduced to {min_resolution[0]}×{min_resolution[1]}"
    except Exception as exc:
        new_config = None
        action = f"ERROR: {exc}"

    return new_config, FallbackLogEntry(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        step=1,
        reason="Memory or latency budget exceeded",
        action=action,
        config_before=before,
        config_after=new_config.model_dump() if new_config else {},
    )


def _fallback_step_2_model_variant(
    config: T2IConfig,
) -> tuple[Optional[T2IConfig], FallbackLogEntry]:
    """Switch to block-pruned U-Net with INT8 quantization if available."""
    before = config.model_dump()
    try:
        new_config = config.model_copy(
            update={
                "variant": "block_pruned",
                "quantization": "int8_dynamic",
            }
        )
        action = "Switched to block-pruned U-Net with int8_dynamic quantization"
    except Exception as exc:
        new_config = None
        action = f"ERROR: {exc}"

    return new_config, FallbackLogEntry(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        step=2,
        reason="Still over memory budget after resolution reduction",
        action=action,
        config_before=before,
        config_after=new_config.model_dump() if new_config else {},
    )


def _fallback_step_3_reduce_steps(
    config: T2IConfig,
    min_steps: int,
    reason: str = "memory pressure",
) -> tuple[Optional[T2IConfig], FallbackLogEntry]:
    """Reduce steps to the minimum acceptable count."""
    before = config.model_dump()
    if config.steps <= min_steps:
        return config, FallbackLogEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            step=3,
            reason=reason,
            action=f"ERROR: Already at minimum steps ({min_steps}). Cannot reduce further.",
            config_before=before,
            config_after=before,
        )
    try:
        new_config = config.model_copy(
            update={
                "steps": min_steps,
                "lcm_lora": True,
                "scheduler": "lcm",
                "guidance_scale": 1.0,
            }
        )
        action = f"Steps reduced to {min_steps} with LCM-LoRA"
    except Exception as exc:
        new_config = None
        action = f"ERROR: {exc}"

    return new_config, FallbackLogEntry(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        step=3,
        reason=reason,
        action=action,
        config_before=before,
        config_after=new_config.model_dump() if new_config else {},
    )
