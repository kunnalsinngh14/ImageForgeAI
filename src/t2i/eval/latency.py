"""
t2i.eval.latency — Wall-clock latency measurement.

Measures end-to-end pipeline latency over multiple runs with warm-up.
Reports median and p90 latency, and records hardware / thread context.

Protocol:
  - 3 warm-up runs (not included in measurements).
  - At least 20 measured runs at fixed settings.
  - Fixed seed per run (seed_base + run_index).
"""

from __future__ import annotations

import logging
import statistics
import time

import psutil

logger = logging.getLogger(__name__)


def measure_latency(
    generator,
    prompt: str,
    negative_prompt: str = "",
    seed_base: int = 1234,
    warmup_runs: int = 3,
    measured_runs: int = 20,
) -> dict:
    """
    Measure end-to-end generation latency.

    Parameters
    ----------
    generator:
        A ``t2i.pipeline.generate.Generator`` instance, fully loaded.
    prompt:
        Fixed prompt used for all runs.
    negative_prompt:
        Fixed negative prompt.
    seed_base:
        Each run uses seed = seed_base + run_index.
    warmup_runs:
        Number of runs to discard as warm-up.
    measured_runs:
        Number of runs to include in statistics.

    Returns
    -------
    dict
        ``{median_s, p90_s, runs, warmup_runs, all_times_s, hardware}``
    """
    times: list[float] = []
    total_runs = warmup_runs + measured_runs

    for i in range(total_runs):
        t_start = time.perf_counter()
        generator.generate(prompt=prompt, negative_prompt=negative_prompt, seed=seed_base + i)
        elapsed = time.perf_counter() - t_start

        if i >= warmup_runs:
            times.append(elapsed)
            logger.debug("Run %d/%d: %.3f s", i - warmup_runs + 1, measured_runs, elapsed)
        else:
            logger.debug("Warm-up run %d/%d: %.3f s", i + 1, warmup_runs, elapsed)

    sorted_times = sorted(times)
    p90_idx = max(0, int(0.9 * len(sorted_times)) - 1)

    return {
        "median_s": round(statistics.median(times), 4),
        "p90_s": round(sorted_times[p90_idx], 4),
        "mean_s": round(statistics.mean(times), 4),
        "std_s": round(statistics.stdev(times) if len(times) > 1 else 0.0, 4),
        "runs": measured_runs,
        "warmup_runs": warmup_runs,
        "all_times_s": [round(t, 4) for t in times],
        "hardware": {
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        },
    }
