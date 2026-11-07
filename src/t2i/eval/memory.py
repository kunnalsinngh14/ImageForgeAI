"""
t2i.eval.memory — Peak memory measurement for CPU and GPU pipelines.

GPU: torch.cuda.max_memory_allocated (reset before each run).
CPU / ONNX: peak RSS sampled via psutil at a fixed interval during generation.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import psutil
import torch

logger = logging.getLogger(__name__)


def measure_peak_gpu_memory_mib(
    fn: Callable,
    device: torch.device,
) -> tuple[float, any]:
    """
    Run ``fn()`` and return (peak_gpu_memory_mib, fn_return_value).

    Resets the peak memory counter before the call.
    """
    if device.type != "cuda":
        raise RuntimeError("measure_peak_gpu_memory_mib requires a CUDA device.")
    torch.cuda.reset_peak_memory_stats(device)
    result = fn()
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    return peak_mib, result


class RSSPoller:
    """
    Polls process RSS (Resident Set Size) at a fixed interval in a background thread.
    Records the peak value.

    Usage::

        with RSSPoller(interval_ms=50) as poller:
            do_work()
        peak_mib = poller.peak_mib
    """

    def __init__(self, interval_ms: int = 50) -> None:
        self._interval_s = interval_ms / 1000.0
        self._proc = psutil.Process()
        self._peak: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _poll(self) -> None:
        while self._running:
            try:
                rss_mib = self._proc.memory_info().rss / (1024**2)
                if rss_mib > self._peak:
                    self._peak = rss_mib
            except Exception:
                pass
            time.sleep(self._interval_s)

    @property
    def peak_mib(self) -> float:
        return self._peak

    def __enter__(self) -> "RSSPoller":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


def measure_peak_cpu_memory_mib(fn: Callable, interval_ms: int = 50) -> tuple[float, any]:
    """
    Run ``fn()`` while polling RSS and return (peak_rss_mib, fn_return_value).

    Parameters
    ----------
    fn:
        Callable to execute (no arguments).
    interval_ms:
        Polling interval in milliseconds.
    """
    with RSSPoller(interval_ms=interval_ms) as poller:
        result = fn()
    return poller.peak_mib, result
