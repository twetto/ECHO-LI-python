"""
Loop timing utility for profiling the EqVIO pipeline.

Port of: LoopTimer.h / LoopTimer.cpp
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List


class LoopTimer:
    """Hierarchical timer for profiling pipeline stages.

    Usage:
        timer = LoopTimer()
        timer.start("propagation")
        ...
        timer.stop("propagation")
        timer.start("features")
        ...
        timer.stop("features")
        timer.end_loop()
        print(timer.summary())
    """

    def __init__(self):
        self._starts: Dict[str, float] = {}
        self._accum: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._loop_count: int = 0
        self._order: List[str] = []

    def start(self, name: str):
        self._starts[name] = time.perf_counter()

    def stop(self, name: str):
        if name in self._starts:
            elapsed = time.perf_counter() - self._starts.pop(name)
            self._accum[name] += elapsed
            self._counts[name] += 1
            if name not in self._order:
                self._order.append(name)

    def end_loop(self):
        self._loop_count += 1

    def summary(self) -> str:
        lines = [
            f"{'Stage':<30s} {'Total (s)':>10s} {'Mean (ms)':>10s} {'Count':>8s} {'%':>6s}"
        ]
        lines.append("-" * 66)

        total_time = sum(self._accum.values())

        for name in self._order:
            t = self._accum[name]
            n = self._counts[name]
            mean_ms = 1000 * t / n if n > 0 else 0
            pct = 100 * t / total_time if total_time > 0 else 0
            lines.append(
                f"{name:<30s} {t:>10.3f} {mean_ms:>10.2f} {n:>8d} {pct:>5.1f}%"
            )

        lines.append("-" * 66)
        lines.append(
            f"{'TOTAL':<30s} {total_time:>10.3f} {'':>10s} {self._loop_count:>8d}"
        )

        return "\n".join(lines)

    def reset(self):
        self._starts.clear()
        self._accum.clear()
        self._counts.clear()
        self._loop_count = 0
        self._order.clear()
