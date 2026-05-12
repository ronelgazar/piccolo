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
import time

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
        self._full_fov_sbs = np.empty_like(self._sbs)
        self._full_fov_eye_l = self._full_fov_sbs[:, :eye_width]
        self._full_fov_eye_r = self._full_fov_sbs[:, eye_width:]

    # ------------------------------------------------------------------
    # Public helpers to drive from the input handler
    # ------------------------------------------------------------------

    def zoom_in(self):
        prev_zoom = self.zoom
        self.zoom = min(self.zoom + self.cfg.zoom.step, self.cfg.zoom.max)
        # Check alignment match count after zoom
        if hasattr(self, 'aligner') and hasattr(self.aligner, 'result'):
            min_matches = getattr(self.cfg.alignment, 'min_matches', 12)
            n_matches = getattr(self.aligner.result, 'n_matches', min_matches)
            if n_matches < min_matches:
                # Too few matches, revert zoom
                self.zoom = prev_zoom
                print(f"[Zoom] Prevented excessive zoom: only {n_matches} matches (min {min_matches})")

    def zoom_out(self):
        self.zoom = max(self.zoom - self.cfg.zoom.step, self.cfg.zoom.min)

    def reset_zoom(self):
        """Reset digital zoom to no magnification.

        Internally, 1.0 means user-facing zoom 0: the full camera frame is
        shown with no crop. A literal 0.0 zoom would make the crop math invalid.
        """
        self.zoom = self.cfg.zoom.min

    def converge_in(self):
        prev_offset = self.base_offset
        self.base_offset += self.cfg.convergence.step
        # Check alignment match count after convergence
        if hasattr(self, 'aligner') and hasattr(self.aligner, 'result'):
            min_matches = getattr(self.cfg.alignment, 'min_matches', 12)
            n_matches = getattr(self.aligner.result, 'n_matches', min_matches)
            if n_matches < min_matches:
                # Too few matches, revert convergence
                self.base_offset = prev_offset
                print(f"[Convergence] Prevented excessive convergence: only {n_matches} matches (min {min_matches})")

    def converge_out(self):
        prev_offset = self.base_offset
        self.base_offset -= self.cfg.convergence.step
        # Check alignment match count after convergence
        if hasattr(self, 'aligner') and hasattr(self.aligner, 'result'):
            min_matches = getattr(self.cfg.alignment, 'min_matches', 12)
            n_matches = getattr(self.aligner.result, 'n_matches', min_matches)
            if n_matches < min_matches:
                # Too few matches, revert convergence
                self.base_offset = prev_offset
                print(f"[Convergence] Prevented excessive convergence: only {n_matches} matches (min {min_matches})")

    def reset(self):
        self.reset_zoom()
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
        # ROI dimensions at current zoom.  In "full" mode we keep the 16:9
        # camera field of view and let the SBS display path map it into one
        # eye.  In "crop" mode we preserve pixel geometry inside the SBS eye,
        # but this necessarily discards horizontal field of view.
        roi_w = int(w / self.zoom)
        roi_h = int(h / self.zoom)
        roi_w, roi_h = self._adjust_roi_aspect(roi_w, roi_h)

        # Centre of the crop (joint zoom center + convergence shift).
        # Joint center comes from pedal/UI controls in percent [0,100].
        cx = int(w * (self.joint_zoom_center / 100.0))
        cy = int(h * (self.joint_zoom_center_y / 100.0))
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

        return self._resize_to_eye(crop, dst)

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

    def process_pair_full_fov(self, frame_l: np.ndarray, frame_r: np.ndarray) -> np.ndarray:
        """Compose a full-FOV SBS frame for calibration views.

        This intentionally ignores live zoom, convergence, and joint-center
        crop. Overlay/physical calibration should start from the complete
        camera view so users do not see a cropped slave just because normal
        live viewing is zoomed in.
        """
        self._resize_to_eye(frame_l, self._full_fov_eye_l, mode="fit")
        self._resize_to_eye(frame_r, self._full_fov_eye_r, mode="fit")
        return self._full_fov_sbs

    def process_pair_full_fov_sbs_anamorphic(self, frame_l: np.ndarray, frame_r: np.ndarray) -> np.ndarray:
        """Compose full-FOV SBS pre-compensated for Goovis side-by-side stretch.

        Goovis/SBS playback stretches each half-frame horizontally into a full
        eye. To keep a 4:3 camera image square after that hardware stretch, the
        source half must be horizontally compressed before it is sent.
        """
        self._resize_to_eye(frame_l, self._full_fov_eye_l, mode="sbs_fit")
        self._resize_to_eye(frame_r, self._full_fov_eye_r, mode="sbs_fit")
        return self._full_fov_sbs
    def process_pair_full_fov_centered(self, frame_l: np.ndarray, frame_r: np.ndarray) -> np.ndarray:
        """Compose full-FOV SBS with each camera image centered unscaled.

        This is intended for manual Goovis calibration. It avoids resampling
        blur and aspect surprises by placing the camera frame at native pixel
        size when it fits inside the SBS half. Larger frames are fit down with
        aspect preserved.
        """
        self._center_to_eye(frame_l, self._full_fov_eye_l)
        self._center_to_eye(frame_r, self._full_fov_eye_r)
        return self._full_fov_sbs


    def process_pair_joint_zoom(self, frame_l: np.ndarray, frame_r: np.ndarray):
        """Process both eyes using a joint zoom center (horizontal and vertical percent)."""
        h, w = frame_l.shape[:2]
        roi_w = int(w / self.zoom)
        roi_h = int(h / self.zoom)
        roi_w, roi_h = self._adjust_roi_aspect(roi_w, roi_h)
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
        self._resize_to_eye(crop_l, self._eye_l)
        self._resize_to_eye(crop_r, self._eye_r)
        return self._eye_l, self._eye_r, self._sbs

    def _adjust_roi_aspect(self, roi_w: int, roi_h: int) -> tuple[int, int]:
        if getattr(self.cfg, "aspect_mode", "fit") != "crop":
            return roi_w, roi_h
        out_aspect = self.eye_w / self.eye_h
        roi_aspect = roi_w / roi_h
        if roi_aspect > out_aspect:
            roi_w = max(1, int(round(roi_h * out_aspect)))
        elif roi_aspect < out_aspect:
            roi_h = max(1, int(round(roi_w / out_aspect)))
        return roi_w, roi_h

    def _center_to_eye(self, img: np.ndarray, dst: np.ndarray | None = None) -> np.ndarray:
        h, w = img.shape[:2]
        out = dst if dst is not None else np.zeros((self.eye_h, self.eye_w, 3), dtype=img.dtype)
        out.fill(0)
        if w <= self.eye_w and h <= self.eye_h:
            fit_w, fit_h = w, h
        else:
            scale = min(self.eye_w / max(1, w), self.eye_h / max(1, h))
            fit_w = max(1, min(self.eye_w, int(round(w * scale))))
            fit_h = max(1, min(self.eye_h, int(round(h * scale))))
        x0 = (self.eye_w - fit_w) // 2
        y0 = (self.eye_h - fit_h) // 2
        roi = out[y0:y0 + fit_h, x0:x0 + fit_w]
        if fit_w == w and fit_h == h:
            roi[:] = img
        else:
            cv2.resize(img, (fit_w, fit_h), dst=roi, interpolation=cv2.INTER_AREA)
        return out
    def _resize_to_eye(self, img: np.ndarray, dst: np.ndarray | None = None, mode: str | None = None) -> np.ndarray:
        mode = mode or getattr(self.cfg, "aspect_mode", "fit")
        h, w = img.shape[:2]
        if mode not in ("fit", "sbs_fit"):
            if w == self.eye_w and h == self.eye_h:
                if dst is not None:
                    dst[:] = img
                    return dst
                return img.copy()
            if dst is not None:
                cv2.resize(img, (self.eye_w, self.eye_h), dst=dst, interpolation=cv2.INTER_LINEAR)
                return dst
            return cv2.resize(img, (self.eye_w, self.eye_h), interpolation=cv2.INTER_LINEAR)

        if mode == "sbs_fit":
            display_eye_w = self.eye_w * 2
            display_eye_h = self.eye_h
            img_aspect = w / max(1, h)
            display_aspect = display_eye_w / max(1, display_eye_h)
            if display_aspect > img_aspect:
                display_fit_h = display_eye_h
                display_fit_w = int(round(display_fit_h * img_aspect))
            else:
                display_fit_w = display_eye_w
                display_fit_h = int(round(display_fit_w / img_aspect))
            fit_w = max(1, min(self.eye_w, int(round(display_fit_w / 2.0))))
            fit_h = max(1, min(self.eye_h, display_fit_h))
        else:
            scale = min(self.eye_w / max(1, w), self.eye_h / max(1, h))
            fit_w = max(1, min(self.eye_w, int(round(w * scale))))
            fit_h = max(1, min(self.eye_h, int(round(h * scale))))
        out = dst if dst is not None else np.zeros((self.eye_h, self.eye_w, 3), dtype=img.dtype)
        out.fill(0)
        x0 = (self.eye_w - fit_w) // 2
        y0 = (self.eye_h - fit_h) // 2
        roi = out[y0:y0 + fit_h, x0:x0 + fit_w]
        if fit_w == w and fit_h == h:
            roi[:] = img
        else:
            cv2.resize(img, (fit_w, fit_h), dst=roi, interpolation=cv2.INTER_LINEAR)
        return out

    def smooth_zoom_transition(self, target_zoom: float, steps: int = 10):
        """Smoothly transition to the target zoom level in defined steps."""
        step_size = (target_zoom - self.zoom) / steps
        for _ in range(steps):
            self.zoom += step_size
            time.sleep(0.05)  # Small delay for smooth transition
        self.zoom = target_zoom
        print(f"Zoom smoothly transitioned to: {self.zoom}")
