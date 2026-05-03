"""Shared stereo feature matching + Theil-Sen regression.

Extracted from StereoAligner so SmartOverlapAnalyzer can reuse the same
matching pipeline. No behaviour change versus the original private methods
in stereo_align.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def theil_sen(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Theil-Sen robust linear fit: y = slope*x + intercept.

    Returns (slope, intercept, rms_residual). For n <= 200 uses all unique
    pairs; above that, samples 50_000 random pairs to keep runtime bounded.
    Tolerates ~29 % gross outliers.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    min_sep = 30.0  # minimum x-separation for a useful slope pair

    if n <= 200:
        ii, jj = np.triu_indices(n, k=1)
        dx = x[jj] - x[ii]
        valid = np.abs(dx) > min_sep
        if valid.sum() < 3:
            slope = 0.0
        else:
            slopes = (y[jj[valid]] - y[ii[valid]]) / dx[valid]
            slope = float(np.median(slopes))
    else:
        rng = np.random.default_rng(42)
        n_pairs = min(50_000, n * (n - 1) // 2)
        ai = rng.integers(0, n, n_pairs)
        bi = rng.integers(0, n, n_pairs)
        keep = ai != bi
        ai, bi = ai[keep], bi[keep]
        dx = x[bi] - x[ai]
        valid = np.abs(dx) > min_sep
        if valid.sum() < 3:
            slope = 0.0
        else:
            slopes = (y[bi[valid]] - y[ai[valid]]) / dx[valid]
            slope = float(np.median(slopes))

    intercepts = y - slope * x
    intercept = float(np.median(intercepts))
    residuals = y - (intercept + slope * x)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return slope, intercept, rms
