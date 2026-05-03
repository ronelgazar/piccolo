"""Smart overlap calibration - diagnostic stereo-pair pattern matching.

Pure-Python core (no Qt). Two modes:
  * chessboard - uses physical_grid_calibration.detect_grid for perfect correspondence
  * live       - uses StereoFeatureMatcher for SIFT-based correspondence

The analyzer returns OverlapMetrics; the renderer draws coloured numbered
markers and connecting threads onto a side-by-side BGR frame.
"""
from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .physical_grid_calibration import GridDetection
from .stereo_matching import theil_sen


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


_PALETTE_SIZE = 20  # max user-tunable pair count is 20
_PAIR_POSITION_SMOOTHING = 0.75


def _build_palette(n: int) -> list[Tuple[int, int, int]]:
    """Generate n visually distinct BGR colours via HSV spacing."""
    out = []
    for i in range(n):
        h = (i / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        out.append((int(b * 255), int(g * 255), int(r * 255)))
    return out


class SmartOverlapAnalyzer:
    """Diagnostic stereo overlap analysis. Holds previous-frame pairs for stability."""

    def __init__(
        self,
        max_vert_dy_px: float,
        max_rotation_deg: float,
        max_zoom_ratio_err: float,
        min_pairs_for_metrics: int,
        pair_stability_tol_px: float,
        matcher,                                    # StereoFeatureMatcher | None
    ):
        self.max_vert_dy_px = float(max_vert_dy_px)
        self.max_rotation_deg = float(max_rotation_deg)
        self.max_zoom_ratio_err = float(max_zoom_ratio_err)
        self.min_pairs_for_metrics = int(min_pairs_for_metrics)
        self.pair_stability_tol_px = float(pair_stability_tol_px)
        self.matcher = matcher
        self._palette = _build_palette(_PALETTE_SIZE)
        self._previous: list[OverlapPair] = []

    def reset(self) -> None:
        self._previous = []

    def analyze(
        self,
        eye_l: np.ndarray,
        eye_r: np.ndarray,
        mode: str,
        pair_count: int,
    ) -> OverlapMetrics:
        if mode == "chessboard":
            raw, zoom_ratio = find_chessboard_pairs(eye_l, eye_r, pair_count)
        elif mode == "live":
            if self.matcher is None:
                raise RuntimeError("live mode requires a matcher")
            raw, zoom_ratio = find_live_pairs(eye_l, eye_r, self.matcher, pair_count,
                                              min_inliers=self.min_pairs_for_metrics)
        else:
            raise ValueError(f"unknown mode: {mode}")

        if not raw:
            self._previous = []
            return OverlapMetrics(
                mode=mode, pairs=[], vert_dy_px=0.0, rotation_deg=0.0,
                zoom_ratio=zoom_ratio, n_inliers=0, n_requested=pair_count,
                align_ok=False, zoom_ok=False,
            )

        pairs = self._assign_colors_and_indices(raw)
        self._previous = pairs

        # Metrics
        h_w_cx = self._frame_cx(eye_l)
        x_centered = np.array([p.left_xy[0] - h_w_cx for p in pairs])
        dy = np.array([p.right_xy[1] - p.left_xy[1] for p in pairs])
        slope, dy_off, _ = theil_sen(x_centered, dy)
        vert_dy = float(dy_off)
        rotation_deg = float(math.degrees(slope))

        align_ok = compute_align_ok(
            vert_dy, rotation_deg, len(pairs),
            self.max_vert_dy_px, self.max_rotation_deg, self.min_pairs_for_metrics,
        )
        zoom_ok = compute_zoom_ok(
            zoom_ratio, len(pairs),
            self.max_zoom_ratio_err, self.min_pairs_for_metrics,
        )
        return OverlapMetrics(
            mode=mode, pairs=pairs, vert_dy_px=vert_dy, rotation_deg=rotation_deg,
            zoom_ratio=zoom_ratio, n_inliers=len(pairs), n_requested=pair_count,
            align_ok=align_ok, zoom_ok=zoom_ok,
        )

    # --- internals ---------------------------------------------------------

    @staticmethod
    def _frame_cx(eye_l: np.ndarray) -> float:
        return eye_l.shape[1] / 2.0

    @staticmethod
    def _blend_xy(prev_xy: tuple[float, float], new_xy: tuple[float, float]) -> tuple[float, float]:
        alpha = _PAIR_POSITION_SMOOTHING
        return (
            alpha * prev_xy[0] + (1.0 - alpha) * new_xy[0],
            alpha * prev_xy[1] + (1.0 - alpha) * new_xy[1],
        )

    def _assign_colors_and_indices(self, raw: list[OverlapPair]) -> list[OverlapPair]:
        """Inherit colour/index from the closest previous pair within tolerance.

        Unmatched new pairs receive fresh palette slots not currently in use.
        Matched pairs are position-smoothed to avoid jittery visual threads.
        """
        used_indices: set[int] = set()
        out: list[OverlapPair] = []
        tol = self.pair_stability_tol_px

        # First pass: try to match each new pair to a previous pair
        matched: list[Optional[OverlapPair]] = [None] * len(raw)
        available_prev = list(self._previous)
        for new_i, new_p in enumerate(raw):
            best_j = -1
            best_dist = tol
            for j, prev_p in enumerate(available_prev):
                if prev_p is None:
                    continue
                d = math.hypot(new_p.left_xy[0] - prev_p.left_xy[0],
                               new_p.left_xy[1] - prev_p.left_xy[1])
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j >= 0:
                prev_p = available_prev[best_j]
                available_prev[best_j] = None  # consume
                matched[new_i] = OverlapPair(
                    index=prev_p.index, color=prev_p.color,
                    left_xy=self._blend_xy(prev_p.left_xy, new_p.left_xy),
                    right_xy=self._blend_xy(prev_p.right_xy, new_p.right_xy),
                )
                used_indices.add(prev_p.index)

        # Second pass: assign fresh palette slots to unmatched pairs
        free_indices = [i for i in range(len(self._palette)) if i not in used_indices]
        free_iter = iter(free_indices)
        for new_i, new_p in enumerate(raw):
            if matched[new_i] is not None:
                out.append(matched[new_i])
                continue
            try:
                fresh = next(free_iter)
            except StopIteration:
                fresh = new_i % len(self._palette)
            out.append(OverlapPair(
                index=fresh, color=self._palette[fresh],
                left_xy=new_p.left_xy, right_xy=new_p.right_xy,
            ))
        return out


def render_overlay(sbs: np.ndarray, metrics: OverlapMetrics) -> np.ndarray:
    """Draw coloured numbered markers + connecting threads onto the SBS frame.

    `sbs` is a (H, 2W, 3) BGR image with left half + right half concatenated
    horizontally. Markers are drawn at each pair's left_xy (in left half) and
    right_xy (offset by W into the right half). Threads connect them.
    """
    import cv2
    out = sbs.copy()
    if out.ndim != 3 or out.shape[2] != 3:
        return out
    h, total_w, _ = out.shape
    eye_w = total_w // 2

    # Eye-gap divider
    cv2.line(out, (eye_w, 0), (eye_w, h), (60, 60, 60), 1, cv2.LINE_AA)

    for p in metrics.pairs:
        lx = int(round(p.left_xy[0]))
        ly = int(round(p.left_xy[1]))
        rx = int(round(p.right_xy[0])) + eye_w
        ry = int(round(p.right_xy[1]))
        # Marker circle and number per eye
        cv2.circle(out, (lx, ly), 9, p.color, 2, cv2.LINE_AA)
        cv2.circle(out, (rx, ry), 9, p.color, 2, cv2.LINE_AA)
        cv2.putText(out, str(p.index + 1), (lx - 4, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, p.color, 1, cv2.LINE_AA)
        cv2.putText(out, str(p.index + 1), (rx - 4, ry + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, p.color, 1, cv2.LINE_AA)
        # Connecting thread
        cv2.line(out, (lx, ly), (rx, ry), p.color, 1, cv2.LINE_AA)
    return out
