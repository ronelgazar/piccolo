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

10. Per-frame warp. Optional overlap masking can black pixels that are not
    valid in both eyes, but it is disabled by default to preserve full FOV.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import AlignmentCfg
from .stereo_matching import StereoFeatureMatcher, theil_sen


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

    def __init__(self, cfg: AlignmentCfg, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.frame_w = frame_w
        self.frame_h = frame_h

        # --- Shared feature matcher (CLAHE + SIFT + FLANN + distribution + RANSAC) ---
        self._matcher = StereoFeatureMatcher(
            max_features=cfg.max_features,
            match_ratio=cfg.match_ratio,
            ransac_thresh=cfg.ransac_thresh,
            frame_w=frame_w,
            frame_h=frame_h,
        )

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

        # --- Greyscale (CLAHE moved into StereoFeatureMatcher) ---
        gray_l = cv2.cvtColor(small_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(small_r, cv2.COLOR_BGR2GRAY)

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

        if self.cfg.mask_overlap and self._overlap_mask_3ch is not None:
            cv2.bitwise_and(self._dst_l, self._overlap_mask_3ch, dst=self._dst_l)
            cv2.bitwise_and(self._dst_r, self._overlap_mask_3ch, dst=self._dst_r)

        return self._dst_l, self._dst_r

    def warp_pair_gpu(
        self,
        gpu_l: "cv2.cuda_GpuMat",
        gpu_r: "cv2.cuda_GpuMat",
        out_l: "cv2.cuda_GpuMat",
        out_r: "cv2.cuda_GpuMat",
    ) -> bool:
        """GPU equivalent of `warp_pair`.

        Writes the warped frames into `out_l` / `out_r`. Returns True if a
        warp was applied; False if the aligner is disabled or has no
        correction yet (in that case callers should fall back to a copy).
        """
        if not self._enabled or self._warp_l is None:
            return False

        size = gpu_l.size()  # (width, height)
        cv2.cuda.warpAffine(
            gpu_l, self._warp_l, size, out_l, cv2.INTER_LINEAR,
            cv2.BORDER_CONSTANT, (0, 0, 0, 0),
        )
        cv2.cuda.warpAffine(
            gpu_r, self._warp_r, size, out_r, cv2.INTER_LINEAR,
            cv2.BORDER_CONSTANT, (0, 0, 0, 0),
        )

        if self.cfg.mask_overlap and self._overlap_mask_3ch is not None:
            # Mask overlap stays CPU-side for now; uploading the mask each
            # frame would defeat the purpose. Phase 1 leaves mask_overlap
            # off (it's already false by default in config).
            pass

        return True

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
        result = self._matcher.match(gray_l, gray_r)
        if len(result.pts_l) < self.cfg.min_matches:
            return None
        inv_s = 1.0 / scale
        pts_l = result.pts_l * inv_s
        pts_r = result.pts_r * inv_s

        dx = pts_r[:, 0] - pts_l[:, 0]
        dy = pts_r[:, 1] - pts_l[:, 1]
        max_horiz = self.frame_w * 0.4
        spatial_ok = (
            (dx > -max_horiz) & (dx < max_horiz)
            & (np.abs(dy) < self.cfg.max_correction_px * 1.5)
        )
        pts_l = pts_l[spatial_ok]
        pts_r = pts_r[spatial_ok]
        if len(pts_l) < self.cfg.min_matches:
            return None
        return self._epipolar_regression(pts_l, pts_r, len(pts_l))

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

        theta, offset, rms = theil_sen(x_centered, vert_diff)

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

    def adjust_alignment(self, delta_dy: float, delta_dtheta: float):
        """Adjust alignment parameters interactively."""
        if not self._enabled:
            return

        self._smooth_dy += delta_dy
        self._smooth_dtheta += delta_dtheta

        # Update warp matrices based on new parameters
        self._warp_l, self._warp_r = self._compute_warp_matrices(self._smooth_dy, self._smooth_dtheta)

    def _compute_warp_matrices(self, dy: float, dtheta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute warp matrices for the given alignment parameters."""
        # Placeholder implementation for computing warp matrices
        # Replace with actual computation logic
        warp_matrix_l = np.eye(2, 3, dtype=np.float32)
        warp_matrix_r = np.eye(2, 3, dtype=np.float32)
        return warp_matrix_l, warp_matrix_r

    def adjust_alignment_for_zoom(self, zoom_level: float):
        """Adjust alignment parameters dynamically based on zoom level."""
        # Adjust smoothed correction values based on zoom level
        self._smooth_dy *= zoom_level
        self._smooth_dtheta *= zoom_level

        # Recompute warp matrices for alignment
        if self._warp_l is not None and self._warp_r is not None:
            scale_matrix = np.array([
                [zoom_level, 0, 0],
                [0, zoom_level, 0]
            ])
            self._warp_l = scale_matrix @ self._warp_l
            self._warp_r = scale_matrix @ self._warp_r

        print(f"Alignment dynamically adjusted for zoom level: {zoom_level}")

    def adjust_for_zoom(self, zoom_level: float):
        """Adjust alignment parameters dynamically based on zoom level."""
        # Example: Scale the smoothing factors based on zoom level
        self._smooth_dy *= zoom_level
        self._smooth_dtheta *= zoom_level

        # Recompute overlap mask for the new zoom level
        scale_matrix = np.array([
            [zoom_level, 0, 0],
            [0, zoom_level, 0]
        ])
        self._warp_l = scale_matrix @ self._warp_l if self._warp_l is not None else None
        self._warp_r = scale_matrix @ self._warp_r if self._warp_r is not None else None

        if self._warp_l is not None and self._warp_r is not None:
            self._overlap_mask = cv2.warpAffine(
                np.ones((self.frame_h, self.frame_w), dtype=np.uint8),
                self._warp_l, (self.frame_w, self.frame_h)
            ) & cv2.warpAffine(
                np.ones((self.frame_h, self.frame_w), dtype=np.uint8),
                self._warp_r, (self.frame_w, self.frame_h)
            )

        print(f"Alignment adjusted for zoom level: {zoom_level}")

