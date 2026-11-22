"""Small statistics for the pilot report: no scipy in this environment, so written here and tested."""

from __future__ import annotations

import math
from collections.abc import Sequence


def ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks; ties get the mean of the ranks they span (as in Spearman's definition)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = mean_rank
        i = j + 1
    return out


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson correlation; None when undefined (fewer than 2 pairs or a constant series)."""
    if len(x) != len(y):
        raise ValueError("series of different lengths")
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman's rho: Pearson correlation of the ranks."""
    return pearson(ranks(x), ranks(y))


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, ``q`` in [0, 100]."""
    if not values:
        raise ValueError("no values")
    data = sorted(values)
    pos = (len(data) - 1) * q / 100
    low = math.floor(pos)
    high = min(low + 1, len(data) - 1)
    return data[low] + (data[high] - data[low]) * (pos - low)
