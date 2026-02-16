"""Stereo image processor – zoom, convergence & composition.

This module takes a pair of camera frames (left, right) and produces:
1. A side-by-side stereo frame for the surgeon's goggles (HDMI output).
2. Individual processed eye frames (for the calibration overlay or viewer
   stream).

Zoom algorithm
--------------
"Natural" zoom for stereo means each eye's image is cropped symmetrically
from its own camera centre, then up-scaled to fill the display half.

*   At ``zoom = 1.0`` the full camera frame is shown.
*   At ``zoom = 2.0`` only the centre 50 % of each frame is shown (2× mag).

Because the crop is always centred on each camera's optical axis, the
relative disparity between objects at a given depth scales linearly with
zoom – exactly what a real optical magnification would do.  This prevents
the images from "colliding" toward the centre of the display.

Convergence
-----------
An optional horizontal pixel offset shifts the crop window of each eye
inward (positive ``base_offset``) or outward.  When ``auto_adjust`` is
enabled, the effective offset is ``base_offset / zoom`` so that the
convergence plane stays at the same real-world distance regardless of
magnification.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import StereoCfg


class StereoProcessor:
    """Stateful stereo processor that tracks the current zoom level and
    convergence offset."""

    def __init__(self, cfg: StereoCfg, eye_width: int, eye_height: int):
        self.cfg = cfg
        self.eye_w = eye_width    # per-eye output width  (usually 960)
        self.eye_h = eye_height   # per-eye output height (usually 1080)

        # Mutable state
        self.zoom: float = cfg.zoom.min
        self.base_offset: int = cfg.convergence.base_offset

        # Joint zoom center (horizontal and vertical, percent 0-100)
        self.joint_zoom_center = 50
        self.joint_zoom_center_y = 50

        # Pre-allocated output buffers (avoids np.hstack allocation every frame)
        self._sbs = np.empty((eye_height, eye_width * 2, 3), dtype=np.uint8)
        self._eye_l = self._sbs[:, :eye_width]       # view into left half
        self._eye_r = self._sbs[:, eye_width:]        # view into right half

    # ------------------------------------------------------------------
    # Public helpers to drive from the input handler
    # ------------------------------------------------------------------

    def zoom_in(self):
        self.zoom = min(self.zoom + self.cfg.zoom.step, self.cfg.zoom.max)

    def zoom_out(self):
        self.zoom = max(self.zoom - self.cfg.zoom.step, self.cfg.zoom.min)

    def converge_in(self):
        self.base_offset += self.cfg.convergence.step

    def converge_out(self):
        self.base_offset -= self.cfg.convergence.step

    def reset(self):
        self.zoom = self.cfg.zoom.min
        self.base_offset = self.cfg.convergence.base_offset

    def set_joint_zoom_center(self, center: int):
        """Set the joint zoom center as a percentage (0-100)."""
        self.joint_zoom_center = max(0, min(100, center))

    def set_joint_zoom_center_y(self, center_y: int):
        """Set the joint zoom vertical center as a percentage (0-100)."""
        self.joint_zoom_center_y = max(0, min(100, center_y))

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    @property
    def effective_offset(self) -> float:
        if self.cfg.convergence.auto_adjust:
            return self.base_offset / self.zoom
        return float(self.base_offset)

    def process_eye(self, frame: np.ndarray, side: str, dst: np.ndarray | None = None) -> np.ndarray:
        """Crop + resize one eye's frame based on current zoom and
        convergence.

        Parameters
        ----------
        frame : np.ndarray  – raw camera frame (BGR).
        side  : ``"left"`` or ``"right"``.
        dst   : optional pre-allocated output array to resize into.
        """
        h, w = frame.shape[:2]
        # ROI dimensions at current zoom
        roi_w = int(w / self.zoom)
        roi_h = int(h / self.zoom)

        # Centre of the crop (camera optical centre + convergence shift)
        cx, cy = w // 2, h // 2
        offset = int(round(self.effective_offset))
        if side == "left":
            cx += offset   # shift crop right → convergent
        else:
            cx -= offset   # shift crop left  → convergent

        # Compute crop rectangle, clamped to frame bounds
        x1 = max(cx - roi_w // 2, 0)
        y1 = max(cy - roi_h // 2, 0)
        x2 = min(x1 + roi_w, w)
        y2 = min(y1 + roi_h, h)
        x1 = max(x2 - roi_w, 0)  # re-clamp after right/bottom clamp
        y1 = max(y2 - roi_h, 0)

        crop = frame[y1:y2, x1:x2]

        # Fast path: if the crop already matches the output size
        # (common at zoom=1 when camera res == display res), skip resize.
        ch, cw = crop.shape[:2]
        if cw == self.eye_w and ch == self.eye_h:
            if dst is not None:
                dst[:] = crop
                return dst
            return crop.copy()

        # Resize to eye output resolution (into dst if provided)
        if dst is not None:
            cv2.resize(crop, (self.eye_w, self.eye_h),
                       dst=dst, interpolation=cv2.INTER_LINEAR)
            return dst
        return cv2.resize(crop, (self.eye_w, self.eye_h),
                          interpolation=cv2.INTER_LINEAR)

    def compose_sbs(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return a side-by-side frame (width = 2 × eye_w).

        If *left* and *right* are already views into ``_sbs``, this is
        effectively a no-op that just returns the pre-allocated buffer.
        """
        if left.base is self._sbs and right.base is self._sbs:
            return self._sbs
        self._eye_l[:] = left
        self._eye_r[:] = right
        return self._sbs

    def process_pair(self, frame_l: np.ndarray, frame_r: np.ndarray):
        """Convenience: process both eyes and return ``(eye_l, eye_r, sbs)``.

        Writes directly into the pre-allocated SBS buffer to avoid
        per-frame allocation.
        """
        eye_l = self.process_eye(frame_l, "left", dst=self._eye_l)
        eye_r = self.process_eye(frame_r, "right", dst=self._eye_r)
        return eye_l, eye_r, self._sbs

    def process_pair_joint_zoom(self, frame_l: np.ndarray, frame_r: np.ndarray):
        """Process both eyes using a joint zoom center (horizontal and vertical percent)."""
        h, w = frame_l.shape[:2]
        roi_w = int(w / self.zoom)
        roi_h = int(h / self.zoom)
        # Center as percent (0-100)
        cx = int(w * (getattr(self, 'joint_zoom_center', 50) / 100.0))
        cy = int(h * (getattr(self, 'joint_zoom_center_y', 50) / 100.0))
        # Clamp crop
        x1 = max(cx - roi_w // 2, 0)
        y1 = max(cy - roi_h // 2, 0)
        x2 = min(x1 + roi_w, w)
        y2 = min(y1 + roi_h, h)
        x1 = max(x2 - roi_w, 0)
        y1 = max(y2 - roi_h, 0)
        crop_l = frame_l[y1:y2, x1:x2]
        crop_r = frame_r[y1:y2, x1:x2]
        eye_l = cv2.resize(crop_l, (self.eye_w, self.eye_h), interpolation=cv2.INTER_LINEAR)
        eye_r = cv2.resize(crop_r, (self.eye_w, self.eye_h), interpolation=cv2.INTER_LINEAR)
        self._eye_l[:] = eye_l
        self._eye_r[:] = eye_r
        return self._eye_l, self._eye_r, self._sbs
