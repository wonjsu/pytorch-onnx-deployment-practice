"""Dependency-light statistics shared by the benchmark and CPU tests."""

from __future__ import annotations

import math
import statistics
from typing import Iterable


def percentile(values: Iterable[float], percentage: float) -> float:
    """Return a linearly interpolated percentile (NumPy's default convention)."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("statistics require at least one value")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: Iterable[float]) -> dict[str, float | int]:
    """Describe samples; standard deviation is population standard deviation."""
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("statistics require at least one value")
    return {
        "count": len(samples),
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "standard_deviation": statistics.pstdev(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "p90": percentile(samples, 90),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
    }


def aggregate_rounds(rounds: list[dict], fields: Iterable[str]) -> dict:
    """Exclude discarded rounds and return pooled and round-mean statistics."""
    included = [item for item in rounds if not item["discarded"]]
    if not included:
        raise ValueError("no non-discarded rounds")
    result = {}
    for field in fields:
        pooled = [row[field] for item in included for row in item["raw"]]
        means = [statistics.fmean(row[field] for row in item["raw"]) for item in included]
        result[field] = {"all_iterations": describe(pooled), "round_means": describe(means)}
    return result
