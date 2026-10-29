"""Projection-view ordering used by the live CT scan (kept free of heavy imports)."""

from __future__ import annotations

from typing import List

import numpy as np

GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def acquisition_order(n_views: int) -> np.ndarray:
    """
    Golden-angle-style permutation of ``range(n_views)``.

    The k-th acquired view is the unused index closest to frac(k * 0.618...) * n_views, so any
    prefix of the order is spread over the full angular range.
    """
    if n_views < 1:
        raise ValueError("n_views must be >= 1")
    free = list(range(n_views))
    order: List[int] = []
    for k in range(n_views):
        target = ((k * GOLDEN_RATIO_CONJUGATE) % 1.0) * n_views
        best = min(free, key=lambda i: min(abs(i - target), n_views - abs(i - target)))
        free.remove(best)
        order.append(best)
    return np.asarray(order, dtype=int)
