"""
Metrics tracking: per-epoch aggregators, latency, throughput, accuracy, loss.
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional


class MetricAggregator:
    """Running average aggregator for scalar metrics."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    def update(self, metrics: Dict[str, float], count: int = 1) -> None:
        for key, value in metrics.items():
            self._sums[key] += value * count
            self._counts[key] += count

    def average(self, reset: bool = True) -> Dict[str, float]:
        result = {}
        for key in self._sums:
            if self._counts[key] > 0:
                result[key] = self._sums[key] / self._counts[key]
        if reset:
            self.reset()
        return result

    def peek(self) -> Dict[str, float]:
        """Compute averages without resetting."""
        result = {}
        for key in self._sums:
            if self._counts[key] > 0:
                result[key] = self._sums[key] / self._counts[key]
        return result


class MetricsTracker:
    """
    Per-batch metrics accumulator with timing.

    Usage:
        tracker = MetricsTracker()
        for batch in loader:
            tracker.tick()               # start timer
            loss = ...
            acc = compute_accuracy(...)
            tracker.update(loss=loss.item(), acc=acc)
            tracker.tock()               # stop batch timer
        epoch_metrics = tracker.epoch_averages()
    """

    def __init__(self):
        self._batch_start: Optional[float] = None
        self._batch_times: List[float] = []
        self._aggregator = MetricAggregator()
        self._num_batches = 0

    def tick(self) -> None:
        """Mark start of a batch."""
        self._batch_start = time.perf_counter()

    def tock(self) -> None:
        """Mark end of a batch and record batch time."""
        if self._batch_start is not None:
            elapsed = time.perf_counter() - self._batch_start
            self._batch_times.append(elapsed)

    def update(self, **metrics: float) -> None:
        """Accumulate scalar metrics for this batch."""
        self._aggregator.update(metrics)
        self._num_batches += 1

    def epoch_averages(self) -> Dict[str, float]:
        """
        Compute averages for the current epoch.
        Returns dict {loss: ..., acc: ..., batch_time: ..., throughput: ...}
        and resets internal aggregator.
        """
        avg_metrics = self._aggregator.average(reset=True)

        if self._batch_times:
            avg_batch_time = sum(self._batch_times) / len(self._batch_times)
            avg_metrics["batch_time_ms"] = round(avg_batch_time * 1000, 3)
            avg_metrics["throughput"] = round(1.0 / avg_batch_time, 1)  # batches/sec
            self._batch_times.clear()

        return avg_metrics

    @property
    def num_batches(self) -> int:
        return self._num_batches

    def reset(self) -> None:
        self._batch_times.clear()
        self._aggregator.reset()
        self._num_batches = 0
