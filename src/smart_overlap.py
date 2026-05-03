"""Smart overlap calibration - diagnostic stereo-pair pattern matching.

Pure-Python core (no Qt). Two modes:
  * chessboard - uses physical_grid_calibration.detect_grid for perfect correspondence
  * live       - uses StereoFeatureMatcher for SIFT-based correspondence

The analyzer returns OverlapMetrics; the renderer draws coloured numbered
markers and connecting threads onto a side-by-side BGR frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class OverlapPair:
    index: int
    color: Tuple[int, int, int]            # BGR
    left_xy: Tuple[float, float]
    right_xy: Tuple[float, float]


@dataclass(frozen=True)
class OverlapMetrics:
    mode: str                               # "chessboard" | "live"
    pairs: list                             # list[OverlapPair]
    vert_dy_px: float
    rotation_deg: float
    zoom_ratio: Optional[float]             # None when not enough data
    n_inliers: int
    n_requested: int
    align_ok: bool
    zoom_ok: bool


def compute_align_ok(
    vert_dy_px: float,
    rotation_deg: float,
    n_inliers: int,
    max_vert_dy_px: float,
    max_rotation_deg: float,
    min_pairs_for_metrics: int,
) -> bool:
    return (
        n_inliers >= min_pairs_for_metrics
        and abs(vert_dy_px) <= max_vert_dy_px
        and abs(rotation_deg) <= max_rotation_deg
    )


def compute_zoom_ok(
    zoom_ratio: Optional[float],
    n_inliers: int,
    max_zoom_ratio_err: float,
    min_pairs_for_metrics: int,
) -> bool:
    if zoom_ratio is None:
        return False
    return (
        n_inliers >= min_pairs_for_metrics
        and abs(zoom_ratio - 1.0) <= max_zoom_ratio_err
    )
