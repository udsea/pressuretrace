"""Lightweight numeric helpers for reporting and analysis."""

from __future__ import annotations

from collections.abc import Sequence


def safe_divide(numerator: float, denominator: float) -> float:
    """Return a stable division result, falling back to 0.0 for empty denominators."""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def safe_mean(values: Sequence[float]) -> float | None:
    """Compute a mean for a non-empty sequence."""

    if not values:
        return None
    return sum(values) / len(values)
