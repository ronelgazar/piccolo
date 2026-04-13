"""Physical camera calibration — pattern rendering and session logic."""
from __future__ import annotations

import math
import cv2
import numpy as np

PHASES = ["focus", "scale", "horizontal", "rotation"]


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
    """Draws phase-specific test patterns onto BGR images in-place.

    Placeholder — render methods added in Task 2.
    """
