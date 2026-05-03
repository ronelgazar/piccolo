"""Smart overlap calibration - diagnostic stereo-pair pattern matching.

Pure-Python core (no Qt). Two modes:
  * chessboard - uses physical_grid_calibration.detect_grid for perfect correspondence
  * live       - uses StereoFeatureMatcher for SIFT-based correspondence

The analyzer returns OverlapMetrics; the renderer draws coloured numbered
markers and connecting threads onto a side-by-side BGR frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .physical_grid_calibration import GridDetection
# Imported lazily inside the function to keep top-level imports light:
#   from .physical_cal import _detect_grid_scaled, estimate_square_px


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


def _detect_grid(img, inner_cols=9, inner_rows=6, max_dim=420):
    """Wrapper around physical_grid_calibration.detect_grid that downscales for speed."""
    from .physical_grid_calibration import detect_grid
    h, w = img.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return detect_grid(img, inner_cols=inner_cols, inner_rows=inner_rows, exhaustive=False)
    scale = max_dim / largest
    small = _resize_for_detect(img, scale)
    detected = detect_grid(small, inner_cols=inner_cols, inner_rows=inner_rows, exhaustive=False)
    if detected is None:
        return None
    corners = detected.corners / scale
    center = (detected.center[0] / scale, detected.center[1] / scale)
    return GridDetection(corners=corners, center=center)


def _resize_for_detect(img, scale):
    import cv2
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def find_chessboard_pairs(
    eye_l: np.ndarray,
    eye_r: np.ndarray,
    pair_count: int,
    inner_cols: int = 9,
    inner_rows: int = 6,
):
    """Returns (pairs, zoom_ratio) for the chessboard mode.

    pairs is a list of OverlapPair (without color/index assigned - those
    come from the analyzer's stability tracking).
    zoom_ratio is right_square_px / left_square_px, or None if either eye
    failed grid detection.
    """
    from .physical_grid_calibration import estimate_square_px

    det_l = _detect_grid(eye_l, inner_cols=inner_cols, inner_rows=inner_rows)
    det_r = _detect_grid(eye_r, inner_cols=inner_cols, inner_rows=inner_rows)
    if det_l is None or det_r is None:
        return [], None

    n_corners = det_l.corners.shape[0]
    grid = max(1, int(math.ceil(math.sqrt(pair_count))))
    rows = inner_rows
    cols = inner_cols
    chosen: list[int] = []
    for gr in range(grid):
        for gc in range(grid):
            if len(chosen) >= pair_count:
                break
            r0 = int(gr * rows / grid)
            r1 = max(r0 + 1, int((gr + 1) * rows / grid))
            c0 = int(gc * cols / grid)
            c1 = max(c0 + 1, int((gc + 1) * cols / grid))
            r_centre = (r0 + r1) // 2
            c_centre = (c0 + c1) // 2
            idx = r_centre * cols + c_centre
            if 0 <= idx < n_corners and idx not in chosen:
                chosen.append(idx)
    chosen = chosen[:pair_count]

    pairs = [
        OverlapPair(
            index=-1,                               # filled in by analyzer
            color=(255, 255, 255),                  # filled in by analyzer
            left_xy=(float(det_l.corners[i, 0]), float(det_l.corners[i, 1])),
            right_xy=(float(det_r.corners[i, 0]), float(det_r.corners[i, 1])),
        )
        for i in chosen
    ]

    sq_l = estimate_square_px(det_l, inner_cols=inner_cols, inner_rows=inner_rows)
    sq_r = estimate_square_px(det_r, inner_cols=inner_cols, inner_rows=inner_rows)
    if sq_l and sq_r:
        zoom_ratio = float(sq_r) / float(sq_l)
    else:
        zoom_ratio = None
    return pairs, zoom_ratio


def find_live_pairs(
    eye_l: np.ndarray,
    eye_r: np.ndarray,
    matcher,
    pair_count: int,
    min_inliers: int = 4,
):
    """Returns (pairs, zoom_ratio) for the live mode.

    `matcher` is a StereoFeatureMatcher (passed in to avoid re-creating SIFT).
    `zoom_ratio` is computed from the median pairwise-distance ratio between
    inlier points, or None if there aren't at least 2 pairs.
    """
    import cv2
    gray_l = cv2.cvtColor(eye_l, cv2.COLOR_BGR2GRAY) if eye_l.ndim == 3 else eye_l
    gray_r = cv2.cvtColor(eye_r, cv2.COLOR_BGR2GRAY) if eye_r.ndim == 3 else eye_r

    result = matcher.match(gray_l, gray_r)
    n = len(result.pts_l)
    if n < min_inliers:
        return [], None

    h, w = eye_l.shape[:2]
    chosen = _sample_well_distributed(result.pts_l, pair_count, frame_w=w, frame_h=h)
    pairs = [
        OverlapPair(
            index=-1,
            color=(255, 255, 255),
            left_xy=(float(result.pts_l[i, 0]), float(result.pts_l[i, 1])),
            right_xy=(float(result.pts_r[i, 0]), float(result.pts_r[i, 1])),
        )
        for i in chosen
    ]
    zoom_ratio = _zoom_from_pair_distances(result.pts_l, result.pts_r)
    return pairs, zoom_ratio


def _sample_well_distributed(pts: np.ndarray, k: int, frame_w: int, frame_h: int) -> list[int]:
    if len(pts) == 0:
        return []
    grid = max(1, int(math.ceil(math.sqrt(k))))
    cell_w = frame_w / grid
    cell_h = frame_h / grid
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(pts):
        gx = min(int(x / cell_w), grid - 1)
        gy = min(int(y / cell_h), grid - 1)
        buckets.setdefault((gx, gy), []).append(i)
    chosen: list[int] = []
    # Prefer one per cell first
    for indices in buckets.values():
        chosen.append(indices[0])
        if len(chosen) >= k:
            break
    # Top up from leftover indices if we still need more
    if len(chosen) < k:
        leftover = [i for indices in buckets.values() for i in indices[1:]]
        chosen.extend(leftover[: k - len(chosen)])
    return chosen[:k]


def _zoom_from_pair_distances(pts_l: np.ndarray, pts_r: np.ndarray) -> Optional[float]:
    n = len(pts_l)
    if n < 2:
        return None
    ratios: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dl = float(np.hypot(pts_l[j, 0] - pts_l[i, 0], pts_l[j, 1] - pts_l[i, 1]))
            dr = float(np.hypot(pts_r[j, 0] - pts_r[i, 0], pts_r[j, 1] - pts_r[i, 1]))
            if dl > 5.0:
                ratios.append(dr / dl)
    if not ratios:
        return None
    return float(np.median(ratios))
