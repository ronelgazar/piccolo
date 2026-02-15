"""Stereo rectification — epipolar-constrained vertical alignment.

Surgery cameras are mounted on a rigid stereo baseline, but mechanical
tolerances, vibration and cable pull cause the two views to drift apart
vertically or rotate slightly.  This module automatically detects and
corrects those errors so the surgeon sees a comfortable, strain-free
stereo image.

Algorithm (v3 – Theil-Sen epipolar regression)
-----------------------------------------------
The critical insight is that in a stereo pair, **horizontal** displacement
varies with depth (parallax) but **vertical** displacement depends only on
camera misalignment and should be spatially consistent.  The previous
algorithm (v2) fit a global affine transform via ``estimateAffinePartial2D``
which was confused by the depth-varying horizontal disparity, producing
noisy rotation and offset estimates.

The new pipeline:

1. **CLAHE** adaptive histogram equalisation on both frames — dramatically
   improves feature detection in low-contrast surgical tissue.

2. **SIFT** detection + computation at ½ scale for speed.

3. **Cross-checked FLANN matching** — match L→R *and* R→L, keep only
   mutual best-matches.  This eliminates most false matches before any
   geometric filtering.

4. **Grid-based spatial distribution** — bin matches into a 6×6 grid and
   cap each cell, preventing high-texture regions (e.g. printed text on
   a surgical drape) from dominating the estimate.

5. **Fundamental-matrix RANSAC** — outlier rejection via epipolar geometry.

6. **Theil-Sen robust regression** on the inlier vertical residuals::

       Δy_i = offset + θ · (x_i − cx)

   where ``Δy_i = y_right − y_left`` for match *i*.  This model directly
   captures the physical misalignment (rotation θ plus constant vertical
   shift) without being confused by horizontal disparity.  Theil-Sen is
   a non-parametric estimator robust to up to 29 % gross outliers.

7. **Quality metric** — RMS of the post-fit vertical residuals.  Values
   below 1 px indicate sub-pixel-accurate calibration.

8. **Adaptive timing** — rapid (0.5 s) updates during initial convergence,
   slower (``interval_sec``) once converged (RMS < 1.5 px).

9. **Phase-correlation fallback** when the scene is too homogeneous for
   feature matching.

10. Per-frame warp with **overlap mask** to prevent stereo edge artefacts
    (unchanged from v2).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import AlignmentCfg


# ------------------------------------------------------------------
# Alignment result
# ------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Stores the latest alignment estimate."""
    dy: float = 0.0          # vertical offset (px) – right relative to left
    dtheta: float = 0.0      # rotation diff (rad) – right relative to left
    n_matches: int = 0        # inlier match count
    confidence: float = 0.0   # inliers / total good matches
    timestamp: float = 0.0
    method: str = "none"      # "epipolar" | "phase" | "none"
    rms_residual: float = float("inf")  # post-fit vertical residual (px)


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------

class StereoAligner:
    """Stereo vertical-alignment via Theil-Sen epipolar regression."""

    # Spatial-distribution grid (matches per cell are capped)
    _GRID = 6

    def __init__(self, cfg: AlignmentCfg, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.frame_w = frame_w
        self.frame_h = frame_h

        # --- CLAHE for adaptive histogram equalisation ---
        # Dramatically improves SIFT detection in low-contrast surgery
        # tissue that the raw camera image would otherwise miss.
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # --- SIFT detector ---
        self._sift = cv2.SIFT_create(nfeatures=cfg.max_features)

        # --- FLANN matcher (KD-tree) ---
        index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
        search_params = dict(checks=80)
        self._matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Current alignment state
        self.result = AlignmentResult()

        # Smoothed correction values
        self._smooth_dy: float = 0.0
        self._smooth_dtheta: float = 0.0

        # Pre-computed warp matrices (2 × 3 affine)
        self._warp_l: Optional[np.ndarray] = None
        self._warp_r: Optional[np.ndarray] = None

        # Pre-allocated destination buffers for warpAffine
        self._dst_l = np.empty((frame_h, frame_w, 3), dtype=np.uint8)
        self._dst_r = np.empty((frame_h, frame_w, 3), dtype=np.uint8)

        # Overlap mask – valid region where BOTH warped images have content.
        self._overlap_mask: Optional[np.ndarray] = None
        self._overlap_mask_3ch: Optional[np.ndarray] = None

        # Timing & convergence
        self._last_update: float = 0.0
        self._enabled = cfg.enabled
        self._update_count: int = 0
        self._converged: bool = False
        self._quality: float = float("inf")  # RMS vertical residual

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool):
        self._enabled = val
        if not val:
            self._warp_l = None
            self._warp_r = None
            self._overlap_mask = None
            self._overlap_mask_3ch = None
            self._smooth_dy = 0.0
            self._smooth_dtheta = 0.0
            self._converged = False
            self._quality = float("inf")
            self._update_count = 0
            self.result = AlignmentResult()

    @property
    def has_correction(self) -> bool:
        return self._warp_l is not None

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def quality(self) -> float:
        """RMS vertical residual in pixels (lower = better)."""
        return self._quality

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def needs_update(self) -> bool:
        """Return True when it's time to re-estimate alignment."""
        if not self._enabled:
            return False
        # Rapid updates (0.5 s) until converged, then normal interval
        rapid = not self._converged and self._update_count < 20
        interval = 0.5 if rapid else self.cfg.interval_sec
        return (time.monotonic() - self._last_update) >= interval

    def update(self, frame_l: np.ndarray, frame_r: np.ndarray) -> AlignmentResult:
        """Re-estimate alignment from the current frame pair.

        Pipeline:
          1. SIFT + cross-check FLANN + F-matrix + Theil-Sen regression
          2. Phase-correlation fallback (textureless scenes)
        """
        self._last_update = time.monotonic()
        self._update_count += 1

        # --- Down-scale for speed ---
        s = self.cfg.detection_scale
        if s < 1.0:
            small_l = cv2.resize(frame_l, None, fx=s, fy=s,
                                 interpolation=cv2.INTER_AREA)
            small_r = cv2.resize(frame_r, None, fx=s, fy=s,
                                 interpolation=cv2.INTER_AREA)
        else:
            small_l, small_r = frame_l, frame_r

        # --- CLAHE on greyscale ---
        gray_l = cv2.cvtColor(small_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(small_r, cv2.COLOR_BGR2GRAY)
        gray_l = self._clahe.apply(gray_l)
        gray_r = self._clahe.apply(gray_r)

        # --- Primary: Epipolar regression ---
        result = self._epipolar_align(gray_l, gray_r, scale=s)

        # --- Fallback: Phase correlation ---
        if result is None:
            result = self._phase_align(gray_l, gray_r, scale=s)

        if result is not None:
            self._quality = result.rms_residual
            self._apply_result(result)
            # Check convergence — real cameras with minor lens distortion
            # differences will always have some vertical residual even when
            # perfectly aligned, so 3.0 px is a realistic threshold.
            if result.rms_residual < 3.0 and self._update_count >= 5:
                self._converged = True

        return self.result

    def warp_pair(
        self,
        frame_l: np.ndarray,
        frame_r: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the cached alignment correction to a frame pair.

        Called every frame.  Uses pre-allocated destination buffers.
        """
        if not self._enabled or self._warp_l is None:
            return frame_l, frame_r

        h, w = frame_l.shape[:2]
        size = (w, h)

        if self._dst_l.shape[:2] != (h, w):
            self._dst_l = np.empty_like(frame_l)
            self._dst_r = np.empty_like(frame_r)

        cv2.warpAffine(frame_l, self._warp_l, size, dst=self._dst_l,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0))
        cv2.warpAffine(frame_r, self._warp_r, size, dst=self._dst_r,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0))

        if self._overlap_mask_3ch is not None:
            cv2.bitwise_and(self._dst_l, self._overlap_mask_3ch, dst=self._dst_l)
            cv2.bitwise_and(self._dst_r, self._overlap_mask_3ch, dst=self._dst_r)

        return self._dst_l, self._dst_r

    def force_update(self):
        """Force the next ``needs_update()`` call to return True."""
        self._last_update = 0.0

    def reset(self):
        """Clear all alignment state."""
        self._smooth_dy = 0.0
        self._smooth_dtheta = 0.0
        self._warp_l = None
        self._warp_r = None
        self._overlap_mask = None
        self._overlap_mask_3ch = None
        self.result = AlignmentResult()
        self._last_update = 0.0
        self._update_count = 0
        self._converged = False
        self._quality = float("inf")

    # ------------------------------------------------------------------
    # Epipolar regression alignment  (primary method)
    # ------------------------------------------------------------------

    def _epipolar_align(
        self, gray_l: np.ndarray, gray_r: np.ndarray, scale: float
    ) -> Optional[AlignmentResult]:
        """SIFT → cross-check FLANN → F-matrix → Theil-Sen regression.

        Instead of fitting a global affine transform (which is confused
        by depth-varying horizontal disparity), we regress the vertical
        component of each match against x-position:

            Δy_i = offset + θ · (x_i − cx)

        Theil-Sen gives a robust estimate of both parameters.
        """
        kp_l, des_l = self._sift.detectAndCompute(gray_l, None)
        kp_r, des_r = self._sift.detectAndCompute(gray_r, None)

        if des_l is None or des_r is None:
            return None
        if len(kp_l) < 10 or len(kp_r) < 10:
            return None

        # --- Cross-checked matching ---
        matches = self._cross_check_match(des_l, des_r)
        if len(matches) < self.cfg.min_matches:
            # Fall back to one-way matching if cross-check too strict
            matches = self._one_way_match(des_l, des_r)
            if len(matches) < self.cfg.min_matches:
                return None

        inv_s = 1.0 / scale
        pts_l = np.float32([kp_l[m[0]].pt for m in matches]) * inv_s
        pts_r = np.float32([kp_r[m[1]].pt for m in matches]) * inv_s
        n_after_ratio = len(matches)

        # --- Stereo spatial filter ---
        dx = pts_r[:, 0] - pts_l[:, 0]
        dy = pts_r[:, 1] - pts_l[:, 1]
        max_horiz = self.frame_w * 0.4
        spatial_ok = (
            (dx > -max_horiz) & (dx < max_horiz) &
            (np.abs(dy) < self.cfg.max_correction_px * 1.5)
        )
        pts_l = pts_l[spatial_ok]
        pts_r = pts_r[spatial_ok]

        if len(pts_l) < self.cfg.min_matches:
            return None

        # --- Grid-based spatial distribution ---
        pts_l, pts_r = self._enforce_distribution(pts_l, pts_r)
        if len(pts_l) < self.cfg.min_matches:
            return None

        # --- Fundamental-matrix RANSAC ---
        if len(pts_l) >= 8:
            try:
                F, mask_f = cv2.findFundamentalMat(
                    pts_l, pts_r,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=self.cfg.ransac_thresh,
                    confidence=0.999,
                )
            except cv2.error:
                F, mask_f = None, None

            if F is not None and mask_f is not None:
                inlier_mask = mask_f.ravel() == 1
                if inlier_mask.sum() >= self.cfg.min_matches:
                    pts_l = pts_l[inlier_mask]
                    pts_r = pts_r[inlier_mask]

        # --- Theil-Sen regression ---
        return self._epipolar_regression(pts_l, pts_r, n_after_ratio)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _cross_check_match(
        self, des_l: np.ndarray, des_r: np.ndarray
    ) -> list:
        """L→R and R→L matching; keep only mutual best matches.

        Cross-checking eliminates most wrong matches that pass the ratio
        test in one direction but fail the symmetric check.
        """
        try:
            raw_lr = self._matcher.knnMatch(des_l, des_r, k=2)
            raw_rl = self._matcher.knnMatch(des_r, des_l, k=2)
        except cv2.error:
            return []

        ratio = self.cfg.match_ratio

        good_lr: dict[int, int] = {}
        for pair in raw_lr:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good_lr[m.queryIdx] = m.trainIdx

        good_rl: dict[int, int] = {}
        for pair in raw_rl:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good_rl[m.queryIdx] = m.trainIdx

        # Keep only mutual matches
        mutual = []
        for l_idx, r_idx in good_lr.items():
            if good_rl.get(r_idx) == l_idx:
                mutual.append((l_idx, r_idx))

        return mutual

    def _one_way_match(
        self, des_l: np.ndarray, des_r: np.ndarray
    ) -> list:
        """Standard one-way ratio-tested matching (less strict fallback)."""
        try:
            raw = self._matcher.knnMatch(des_l, des_r, k=2)
        except cv2.error:
            return []

        matches = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.cfg.match_ratio * n.distance:
                    matches.append((m.queryIdx, m.trainIdx))
        return matches

    # ------------------------------------------------------------------
    # Spatial distribution
    # ------------------------------------------------------------------

    def _enforce_distribution(
        self, pts_l: np.ndarray, pts_r: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cap matches per grid cell so no image region dominates.

        A high-texture region (surgical drape pattern, instrument label)
        can produce hundreds of matches in a small area, overwhelming the
        regression.  By capping per-cell contributions, every part of the
        frame gets equal influence on the correction estimate.
        """
        g = self._GRID
        cell_w = self.frame_w / g
        cell_h = self.frame_h / g

        buckets: dict[tuple, list] = {}
        for i in range(len(pts_l)):
            gx = min(int(pts_l[i, 0] / cell_w), g - 1)
            gy = min(int(pts_l[i, 1] / cell_h), g - 1)
            buckets.setdefault((gx, gy), []).append(i)

        max_per = max(6, len(pts_l) // (g * g) + 1)
        sel = []
        for indices in buckets.values():
            sel.extend(indices[:max_per])

        idx = np.array(sel)
        return pts_l[idx], pts_r[idx]

    # ------------------------------------------------------------------
    # Theil-Sen robust regression
    # ------------------------------------------------------------------

    def _epipolar_regression(
        self,
        pts_l: np.ndarray,
        pts_r: np.ndarray,
        n_total: int,
    ) -> Optional[AlignmentResult]:
        """Estimate Δy and θ via robust regression of vertical residuals.

        Model::

            Δy_i = offset + θ · (x_i − cx)

        where ``Δy_i = y_right − y_left`` for match *i*.

        **Why this works better than affine estimation:**
        In a stereo pair the *horizontal* displacement varies with depth
        (parallax), but the *vertical* displacement depends only on the
        camera misalignment.  By regressing *only* the vertical component,
        we are immune to depth variation — unlike ``estimateAffinePartial2D``
        which fits rotation + scale + translation to *both* axes and gets
        confused by the varying disparity.
        """
        n = len(pts_l)
        if n < 4:
            return None

        cx = self.frame_w / 2.0
        vert_diff = pts_r[:, 1] - pts_l[:, 1]
        x_centered = pts_l[:, 0] - cx

        theta, offset, rms = self._theil_sen(x_centered, vert_diff)

        # Clamp
        max_dy = self.cfg.max_correction_px
        max_dt = math.radians(self.cfg.max_correction_deg)
        offset = float(np.clip(offset, -max_dy, max_dy))
        theta = float(np.clip(theta, -max_dt, max_dt))

        return AlignmentResult(
            dy=offset,
            dtheta=theta,
            n_matches=n,
            confidence=n / max(n_total, 1),
            timestamp=time.monotonic(),
            method="epipolar",
            rms_residual=rms,
        )

    @staticmethod
    def _theil_sen(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Theil-Sen robust linear fit: ``y = slope·x + intercept``.

        The Theil-Sen estimator computes the **median** of all pairwise
        slopes, followed by the **median** of the resulting intercepts.
        It tolerates up to ≈29 % gross outliers — far more robust than
        ordinary least-squares and comparably robust to RANSAC, but
        deterministic and parameter-free.

        For large *n* the all-pairs computation is O(n²); we switch to
        random sub-sampling above 200 points to keep runtime bounded.

        Returns ``(slope, intercept, rms_residual)``.
        """
        n = len(x)
        min_sep = 30.0  # minimum x-separation for a useful slope pair

        if n <= 200:
            # --- Exact: all unique pairs (i < j) via vectorised indices ---
            ii, jj = np.triu_indices(n, k=1)
            dx = x[jj] - x[ii]
            valid = np.abs(dx) > min_sep
            if valid.sum() < 3:
                slope = 0.0          # not enough x-spread → assume no rotation
            else:
                slopes = (y[jj[valid]] - y[ii[valid]]) / dx[valid]
                slope = float(np.median(slopes))
        else:
            # --- Sub-sampled: random pairs ---
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
                slope = float(np.median(slopes[valid]))

        intercepts = y - slope * x
        intercept = float(np.median(intercepts))

        residuals = y - (intercept + slope * x)
        rms = float(np.sqrt(np.mean(residuals ** 2)))

        return slope, intercept, rms

    # ------------------------------------------------------------------
    # Phase-correlation fallback
    # ------------------------------------------------------------------

    def _phase_align(
        self, gray_l: np.ndarray, gray_r: np.ndarray, scale: float
    ) -> Optional[AlignmentResult]:
        """Sub-pixel translation estimate via Fourier shift theorem.

        Works when the scene is too homogeneous for feature matching.
        Only the vertical component of the shift is used; horizontal is
        stereo disparity and must NOT be corrected.
        """
        inv_s = 1.0 / scale

        try:
            shift, response = cv2.phaseCorrelate(
                gray_l.astype(np.float64),
                gray_r.astype(np.float64),
            )
        except cv2.error:
            return None

        _, dy = shift
        dy *= inv_s

        if abs(dy) > self.cfg.max_correction_px:
            return None
        if response < 0.15:
            return None

        return AlignmentResult(
            dy=dy,
            dtheta=0.0,
            n_matches=0,
            confidence=min(response, 1.0),
            timestamp=time.monotonic(),
            method="phase",
            rms_residual=float("inf"),  # no match-level residual available
        )

    # ------------------------------------------------------------------
    # Smoothing + warp matrix construction
    # ------------------------------------------------------------------

    def _apply_result(self, result: AlignmentResult):
        """Adaptive smoothing + warp matrix rebuild.

        Uses lighter smoothing during initial convergence (to reach the
        correct correction quickly) and heavier smoothing once converged
        (to maintain stability).
        """
        base_a = self.cfg.smoothing
        if not self._converged:
            # Use full base smoothing even during convergence — halving
            # it caused oscillation with noisy real-camera estimates.
            a = base_a
        else:
            # Heavy smoothing once converged — real-camera estimates
            # fluctuate ±1-2px frame-to-frame; we want the displayed
            # correction to be rock-stable.
            a = min(base_a * 2.5, 0.85)

        self._smooth_dy = a * self._smooth_dy + (1 - a) * result.dy
        self._smooth_dtheta = a * self._smooth_dtheta + (1 - a) * result.dtheta

        self.result = AlignmentResult(
            dy=self._smooth_dy,
            dtheta=self._smooth_dtheta,
            n_matches=result.n_matches,
            confidence=result.confidence,
            timestamp=result.timestamp,
            method=result.method,
            rms_residual=result.rms_residual,
        )
        self._build_warp_matrices()

    def _build_warp_matrices(self):
        """Build affine matrices that split the correction between both eyes.

        Left eye receives +½ of the correction, right eye receives −½,
        so neither image shifts by more than half the total error.
        """
        cx = self.frame_w / 2.0
        cy = self.frame_h / 2.0

        half_dy = self._smooth_dy / 2.0
        half_theta = self._smooth_dtheta / 2.0

        self._warp_l = self._rotation_matrix(
            cx, cy, angle=half_theta, ty=half_dy)
        self._warp_r = self._rotation_matrix(
            cx, cy, angle=-half_theta, ty=-half_dy)

        self._compute_overlap_mask()

    def _compute_overlap_mask(self):
        """Precompute the region where BOTH warped images have valid pixels."""
        h, w = self.frame_h, self.frame_w
        ones = np.full((h, w), 255, dtype=np.uint8)

        mask_l = cv2.warpAffine(
            ones, self._warp_l, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_r = cv2.warpAffine(
            ones, self._warp_r, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        overlap = cv2.bitwise_and(mask_l, mask_r)
        kernel = np.ones((5, 5), np.uint8)
        self._overlap_mask = cv2.erode(overlap, kernel, iterations=1)
        self._overlap_mask_3ch = cv2.cvtColor(
            self._overlap_mask, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _rotation_matrix(
        cx: float, cy: float, angle: float, ty: float
    ) -> np.ndarray:
        """2 × 3 affine: rotate by *angle* around (cx, cy) then
        translate vertically by *ty* pixels."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        tx = cx * (1 - cos_a) + cy * sin_a
        ty_rot = cy * (1 - cos_a) - cx * sin_a
        return np.float32([
            [cos_a, -sin_a, tx],
            [sin_a,  cos_a, ty_rot + ty],
        ])
