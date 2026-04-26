"""Background physical-calibration analysis for the Calibration tab."""
from __future__ import annotations

import threading

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..physical_cal import GridPairMetrics, PhysicalCalSession, PatternRenderer
from .qt_helpers import ndarray_to_qimage


class PhysicalCalResult:
    def __init__(
        self,
        image,
        phase: str,
        metrics: GridPairMetrics,
        sharp_l: float,
        sharp_r: float,
        focus_ok: bool,
        best_l: float,
        best_r: float,
    ):
        self.image = image
        self.phase = phase
        self.metrics = metrics
        self.sharp_l = sharp_l
        self.sharp_r = sharp_r
        self.focus_ok = focus_ok
        self.best_l = best_l
        self.best_r = best_r


class PhysicalCalWorker(QThread):
    """Processes at most one latest physical-calibration frame at a time."""

    result_ready = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._condition = threading.Condition()
        self._pending: tuple[np.ndarray, str, dict] | None = None
        self._stopping = False
        self._renderer = PatternRenderer()
        self._session = PhysicalCalSession()

    def submit(self, sbs: np.ndarray, phase: str, status: dict) -> None:
        """Queue the latest SBS frame, replacing any older unprocessed frame."""
        with self._condition:
            self._pending = (sbs.copy(), phase, dict(status))
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()
        self.wait(2000)

    def run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                sbs, phase, status = self._pending
                self._pending = None

            try:
                self.result_ready.emit(self._process(sbs, phase, status))
            except Exception:
                # Keep the display path alive even if one calibration frame fails.
                continue

    def _process(self, sbs: np.ndarray, phase: str, status: dict) -> PhysicalCalResult:
        eye_w = sbs.shape[1] // 2
        fl = sbs[:, :eye_w]
        fr = sbs[:, eye_w:]
        eye_l = fl.copy()
        eye_r = fr.copy()

        metrics = PhysicalCalSession.grid_pair_metrics(fl, fr)
        sharp_l = metrics.left.sharpness if metrics.left.sharpness is not None else PhysicalCalSession.sharpness(fl)
        sharp_r = metrics.right.sharpness if metrics.right.sharpness is not None else PhysicalCalSession.sharpness(fr)
        focus_ok, best_l, best_r = self._session.update_focus_peak(sharp_l, sharp_r)

        if phase == "brightness":
            ok = metrics.brightness_ok()
            self._renderer.render_brightness(eye_l, metrics.left, ok=ok)
            self._renderer.render_brightness(eye_r, metrics.right, ok=ok)
        elif phase == "focus":
            self._renderer.render_grid_focus(eye_l, metrics.left, ok=focus_ok)
            self._renderer.render_grid_focus(eye_r, metrics.right, ok=focus_ok)
        elif phase == "scale":
            ok = metrics.zoom_ok()
            self._renderer.render_scale(eye_l, metrics.left, ok=ok)
            self._renderer.render_scale(eye_r, metrics.right, ok=ok)
        elif phase == "horizontal":
            ok = metrics.vertical_ok()
            self._renderer.render_horizontal(eye_l, metrics.vertical_delta_px, ok=ok)
            self._renderer.render_horizontal(eye_r, metrics.vertical_delta_px, ok=ok)
        elif phase == "rotation":
            ok = metrics.rotation_ok()
            self._renderer.render_rotation(eye_l, metrics.rotation_delta_deg, ok=ok)
            self._renderer.render_rotation(eye_r, metrics.rotation_delta_deg, ok=ok)

        self._renderer.render_pair_hud(eye_l, phase, metrics, side="left")
        self._renderer.render_pair_hud(eye_r, phase, metrics, side="right")

        return PhysicalCalResult(
            image=ndarray_to_qimage(cv2.hconcat([eye_l, eye_r])),
            phase=phase,
            metrics=metrics,
            sharp_l=sharp_l,
            sharp_r=sharp_r,
            focus_ok=focus_ok,
            best_l=best_l,
            best_r=best_r,
        )
