"""Manual per-eye calibration for surgeon stereo setup.

The surgeon needs to independently verify and adjust each eye's image
before enabling stereo fusion.  Automatic alignment corrects vertical
misalignment and rotation, but the *horizontal* position of each eye
may also need manual tweaking (e.g. inter-pupillary distance mismatch
with the headset optics).

Calibration flow
----------------
1. **Left eye** – only the left eye image is shown (right is blacked
   out).  The surgeon uses left/right keys to nudge the left image
   horizontally until it is centred.

2. **Right eye** – same for the right eye (left is blacked out).

3. **Fuse** – both eyes shown together with crosshairs.  The surgeon
   confirms comfortable fusion.

4. Press next / toggle -> exits calibration and triggers a fresh
   auto-alignment scan.

At each step the user explicitly presses a **next** key (N or web
button) to advance — there is no timer.  This lets the surgeon take
as long as needed to relax each eye and position the image.

The per-eye horizontal nudge offsets persist after calibration exits
and are applied every frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import CalibrationCfg


class CalibrationOverlay:
    """Interactive per-eye calibration with manual horizontal nudge."""

    # Nudge step size in pixels per key press / click
    NUDGE_STEP: int = 2

    def __init__(self, cfg: CalibrationCfg):
        self.cfg = cfg
        self.active = False

        # Phase: "left" | "right" | "fuse" | inactive (active=False)
        self._phase: str = "left"

        # Persistent per-eye horizontal offsets (survive calibration exit)
        self.nudge_left: int = 0    # positive = shift image rightward
        self.nudge_right: int = 0
        # Persistent per-eye vertical offsets (positive = shift image downward)
        self.nudge_left_y: int = 0
        self.nudge_right_y: int = 0
        # Persistent per-eye scale compensation. 100 means unchanged.
        self.scale_left_pct: int = 100
        self.scale_right_pct: int = 100

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def toggle(self):
        """Toggle calibration mode on / off."""
        if self.active:
            self.active = False
        else:
            self.active = True
            self._phase = "left"

    def next_phase(self):
        """Advance to the next calibration phase.

        left -> right -> fuse -> done (active=False).
        """
        if not self.active:
            return
        if self._phase == "left":
            self._phase = "right"
        elif self._phase == "right":
            self._phase = "fuse"
        elif self._phase == "fuse":
            self.active = False

    @property
    def phase(self) -> str:
        """Current calibration phase name."""
        return self._phase if self.active else "off"

    # ------------------------------------------------------------------
    # Nudge controls
    # ------------------------------------------------------------------

    def nudge_current_left(self):
        """Shift the currently-active eye's image to the left."""
        if not self.active:
            return
        if self._phase == "left":
            self.nudge_left -= self.NUDGE_STEP
        elif self._phase == "right":
            self.nudge_right -= self.NUDGE_STEP

    def nudge_current_right(self):
        """Shift the currently-active eye's image to the right."""
        if not self.active:
            return
        if self._phase == "left":
            self.nudge_left += self.NUDGE_STEP
        elif self._phase == "right":
            self.nudge_right += self.NUDGE_STEP

    def reset_nudge(self):
        """Reset all nudge offsets (X and Y) to zero."""
        self.nudge_left = 0
        self.nudge_right = 0
        self.nudge_left_y = 0
        self.nudge_right_y = 0
        self.scale_left_pct = 100
        self.scale_right_pct = 100

    def set_nudge_y(self, eye: str, value: int) -> None:
        """Set the vertical nudge offset (in pixels) for one eye.

        Positive shifts image down; negative shifts up.  Used by the
        web UI to correct physical camera height misalignment.
        """
        if eye == "left":
            self.nudge_left_y = int(value)
        elif eye == "right":
            self.nudge_right_y = int(value)

    # ------------------------------------------------------------------
    # Per-frame application
    # ------------------------------------------------------------------

    @staticmethod
    def _roll_and_zero(img: np.ndarray, shift: int, axis: int) -> np.ndarray:
        """Shift ``img`` by ``shift`` along ``axis`` without wrapping pixels."""
        if shift == 0:
            return img
        out = np.zeros_like(img)
        if axis == 0:
            h = img.shape[0]
            if abs(shift) >= h:
                return out
            if shift > 0:
                out[shift:, :] = img[:h - shift, :]
            else:
                out[:h + shift, :] = img[-shift:, :]
        else:
            w = img.shape[1]
            if abs(shift) >= w:
                return out
            if shift > 0:
                out[:, shift:] = img[:, :w - shift]
            else:
                out[:, :w + shift] = img[:, -shift:]
        return out

    def apply_nudge(self, eye_l: np.ndarray, eye_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply persistent per-eye nudge offsets (both X and Y) to both frames.

        Called every frame (even outside calibration) so the surgeon's
        manual corrections are always active.  Uses ``np.roll`` for
        speed; border pixels are zeroed to avoid wrap artefacts.
        """
        eye_l = self._scale_eye(eye_l, self.scale_left_pct)
        eye_r = self._scale_eye(eye_r, self.scale_right_pct)
        eye_l = self._roll_and_zero(eye_l, self.nudge_left, axis=1)
        eye_r = self._roll_and_zero(eye_r, self.nudge_right, axis=1)
        eye_l = self._roll_and_zero(eye_l, self.nudge_left_y, axis=0)
        eye_r = self._roll_and_zero(eye_r, self.nudge_right_y, axis=0)
        return eye_l, eye_r

    @staticmethod
    def _scale_eye(img: np.ndarray, scale_pct: int) -> np.ndarray:
        """Center-scale one eye, preserving the output frame size."""
        if scale_pct == 100:
            return img
        h, w = img.shape[:2]
        scale = max(0.05, float(scale_pct) / 100.0)
        scaled_w = max(1, int(round(w * scale)))
        scaled_h = max(1, int(round(h * scale)))
        resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        out = np.zeros_like(img)

        if scaled_w >= w:
            src_x0 = (scaled_w - w) // 2
            dst_x0 = 0
            copy_w = w
        else:
            src_x0 = 0
            dst_x0 = (w - scaled_w) // 2
            copy_w = scaled_w

        if scaled_h >= h:
            src_y0 = (scaled_h - h) // 2
            dst_y0 = 0
            copy_h = h
        else:
            src_y0 = 0
            dst_y0 = (h - scaled_h) // 2
            copy_h = scaled_h

        out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = resized[
            src_y0:src_y0 + copy_h,
            src_x0:src_x0 + copy_w,
        ]
        return out

    def apply(self, eye_l: np.ndarray, eye_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Draw calibration overlay and blank the inactive eye.

        Only active during calibration mode.

        Returns the (possibly modified) ``(eye_l, eye_r)`` tuple.
        """
        if not self.active:
            return eye_l, eye_r

        if self._phase == "left":
            eye_r[:] = 0
            self._draw_crosshair(eye_l)
            self._draw_phase_label(eye_l, "LEFT EYE", self.nudge_left)

        elif self._phase == "right":
            eye_l[:] = 0
            self._draw_crosshair(eye_r)
            self._draw_phase_label(eye_r, "RIGHT EYE", self.nudge_right)

        elif self._phase == "fuse":
            self._draw_crosshair(eye_l)
            self._draw_crosshair(eye_r)
            self._draw_fuse_label(eye_l)
            self._draw_fuse_label(eye_r)

        return eye_l, eye_r

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_crosshair(self, img: np.ndarray, alpha: float = 1.0):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        size = self.cfg.crosshair_size
        color = tuple(int(c * alpha) for c in self.cfg.crosshair_color)
        thick = self.cfg.crosshair_thickness

        cv2.line(img, (cx - size, cy), (cx + size, cy), color, thick)
        cv2.line(img, (cx, cy - size), (cx, cy + size), color, thick)
        cv2.circle(img, (cx, cy), 4, color, -1)

    @staticmethod
    def _draw_phase_label(img: np.ndarray, label: str, nudge: int):
        """Draw phase label and nudge offset on the image."""
        h, w = img.shape[:2]
        cv2.putText(img, label, (w // 2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
                    cv2.LINE_AA)
        nudge_txt = f"Nudge: {nudge:+d}px   (arrow keys to adjust)"
        cv2.putText(img, nudge_txt, (w // 2 - 180, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1,
                    cv2.LINE_AA)
        cv2.putText(img, "Press N for next eye", (w // 2 - 130, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1,
                    cv2.LINE_AA)

    @staticmethod
    def _draw_fuse_label(img: np.ndarray):
        h, w = img.shape[:2]
        cv2.putText(img, "FUSE - Both Eyes", (w // 2 - 130, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(img, "Press N to finish", (w // 2 - 110, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1,
                    cv2.LINE_AA)
