"""Physical camera calibration — pattern rendering and session logic."""
from __future__ import annotations

import math

import cv2
import numpy as np

PHASES: tuple[str, ...] = ("focus", "scale", "horizontal", "rotation")


class PhysicalCalSession:
    """Phase state machine + per-eye sharpness metric."""

    def __init__(self):
        self._idx = 0

    @property
    def phase(self) -> str:
        return PHASES[self._idx]

    @property
    def phase_index(self) -> int:
        return self._idx

    @property
    def total_phases(self) -> int:
        return len(PHASES)

    def next_phase(self) -> bool:
        """Advance to next phase. Returns True when all phases are done."""
        if self._idx < len(PHASES) - 1:
            self._idx += 1
            return False
        return True

    def prev_phase(self) -> None:
        if self._idx > 0:
            self._idx -= 1

    @staticmethod
    def sharpness(img: np.ndarray) -> float:
        """Laplacian variance of central 200×200 px ROI. Higher = sharper."""
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        roi = img[max(0, cy - 100):cy + 100, max(0, cx - 100):cx + 100]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else roi
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class PatternRenderer:
    """Draws phase-specific test patterns onto BGR images in-place."""

    def render_focus(self, img: np.ndarray, sharpness: float) -> None:
        """Siemens star (36-spoke radial wheel) + sharpness score."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        radius = min(h, w) // 4
        for i in range(36):
            angle = i * math.pi / 36
            x1 = int(cx + radius * math.cos(angle))
            y1 = int(cy + radius * math.sin(angle))
            x2 = int(cx - radius * math.cos(angle))
            y2 = int(cy - radius * math.sin(angle))
            color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
            cv2.line(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, "FOCUS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"sharp: {sharpness:.0f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

    def render_scale(self, img: np.ndarray) -> None:
        """Concentric circles at 10 %, 25 %, 50 % of frame height."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        color = (200, 200, 0)
        for pct in [10, 25, 50]:
            r = int(h * pct / 100)
            cv2.circle(img, (cx, cy), r, color, 1, cv2.LINE_AA)
            cv2.putText(img, f"{pct}%", (cx + r + 4, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(img, "SCALE  (match circle sizes visually)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    def render_horizontal(self, img: np.ndarray, dy: float | None) -> None:
        """Horizontal grid lines + centre crosshair + vertical-offset readout."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        color = (0, 200, 255)
        for pct in [25, 50, 75]:
            y = int(h * pct / 100)
            cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx - 40, cy), (cx + 40, cy), (0, 255, 0), 2)
        cv2.line(img, (cx, cy - 40), (cx, cy + 40), (0, 255, 0), 2)
        cv2.putText(img, "HORIZONTAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        dy_txt = f"Vertical offset: {dy:+.1f} px" if dy is not None else "Vertical offset: --"
        cv2.putText(img, dy_txt, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)

    def render_rotation(self, img: np.ndarray, dtheta_deg: float | None) -> None:
        """Diagonal reference lines + spirit-level arc + rotation readout."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        line_color = (255, 100, 0)
        cv2.line(img, (0, 0), (w, h), line_color, 1, cv2.LINE_AA)
        cv2.line(img, (w, 0), (0, h), line_color, 1, cv2.LINE_AA)
        arc_r = min(w, h) // 6
        arc_cx, arc_cy = cx, cy + arc_r + 20
        cv2.ellipse(img, (arc_cx, arc_cy), (arc_r, arc_r), 0, 180, 360,
                    (180, 180, 180), 1, cv2.LINE_AA)
        if dtheta_deg is not None:
            clamped = max(-5.0, min(5.0, dtheta_deg))
            dot_angle_rad = math.radians(270 + clamped * 18)
            dot_x = int(arc_cx + arc_r * math.cos(dot_angle_rad))
            dot_y = int(arc_cy + arc_r * math.sin(dot_angle_rad))
            cv2.circle(img, (dot_x, dot_y), 8, (0, 255, 0), -1)
        cv2.putText(img, "ROTATION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2, cv2.LINE_AA)
        rot_txt = (f"Rotation: {dtheta_deg:+.2f} deg"
                   if dtheta_deg is not None else "Rotation: --")
        cv2.putText(img, rot_txt, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_color, 1, cv2.LINE_AA)
