"""Background analysis for the Smart overlap calibration panel."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..smart_overlap import (
    OverlapMetrics,
    OverlapPair,
    SmartOverlapAnalyzer,
    compute_align_ok,
    compute_zoom_ok,
    render_overlay,
)
from .qt_helpers import ndarray_to_qimage


@dataclass
class SmartOverlapResult:
    image: object              # QImage
    metrics: OverlapMetrics


class SmartOverlapWorker(QThread):
    """Processes at most one latest SBS frame at a time and emits a rendered image + metrics."""

    result_ready = pyqtSignal(object)   # SmartOverlapResult
    _MAX_ANALYSIS_EYE_DIM = 480

    def __init__(self, analyzer: SmartOverlapAnalyzer, parent=None):
        super().__init__(parent)
        self._analyzer = analyzer
        self._condition = threading.Condition()
        self._pending: Optional[tuple[np.ndarray, str, int]] = None
        self._stopping = False

    def submit(self, sbs: np.ndarray, mode: str, pair_count: int) -> None:
        with self._condition:
            self._pending = (sbs.copy(), mode, int(pair_count))
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()
        self.wait(2000)

    def reset_state(self) -> None:
        """Called when the surgeon presses Stop - clears stability tracking."""
        self._analyzer.reset()

    def run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                sbs, mode, pair_count = self._pending
                self._pending = None
            try:
                self.result_ready.emit(self._process(sbs, mode, pair_count))
            except Exception:
                continue

    def _process(self, sbs: np.ndarray, mode: str, pair_count: int) -> SmartOverlapResult:
        analysis_sbs, inv_scale = self._analysis_frame(sbs)
        eye_w = analysis_sbs.shape[1] // 2
        eye_l = analysis_sbs[:, :eye_w]
        eye_r = analysis_sbs[:, eye_w:]
        metrics = self._analyzer.analyze(eye_l, eye_r, mode=mode, pair_count=pair_count)
        if inv_scale != 1.0:
            metrics = self._scale_metrics(metrics, inv_scale)
        rendered = render_overlay(sbs, metrics)
        return SmartOverlapResult(
            image=ndarray_to_qimage(rendered),
            metrics=metrics,
        )

    def _analysis_frame(self, sbs: np.ndarray) -> tuple[np.ndarray, float]:
        eye_w = sbs.shape[1] // 2
        eye_h = sbs.shape[0]
        largest = max(eye_w, eye_h)
        if largest <= self._MAX_ANALYSIS_EYE_DIM:
            return sbs, 1.0
        scale = self._MAX_ANALYSIS_EYE_DIM / float(largest)
        resized = cv2.resize(
            sbs,
            (max(2, int(round(sbs.shape[1] * scale))), max(1, int(round(sbs.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        # Keep the side-by-side frame split exactly even after rounding.
        if resized.shape[1] % 2:
            resized = resized[:, :-1]
        return resized, 1.0 / scale

    def _scale_metrics(self, metrics: OverlapMetrics, scale: float) -> OverlapMetrics:
        pairs = [
            OverlapPair(
                index=p.index,
                color=p.color,
                left_xy=(p.left_xy[0] * scale, p.left_xy[1] * scale),
                right_xy=(p.right_xy[0] * scale, p.right_xy[1] * scale),
            )
            for p in metrics.pairs
        ]
        vert_dy_px = metrics.vert_dy_px * scale
        align_ok = compute_align_ok(
            vert_dy_px,
            metrics.rotation_deg,
            metrics.n_inliers,
            self._analyzer.max_vert_dy_px,
            self._analyzer.max_rotation_deg,
            self._analyzer.min_pairs_for_metrics,
        )
        zoom_ok = compute_zoom_ok(
            metrics.zoom_ratio,
            metrics.n_inliers,
            self._analyzer.max_zoom_ratio_err,
            self._analyzer.min_pairs_for_metrics,
        )
        return OverlapMetrics(
            mode=metrics.mode,
            pairs=pairs,
            vert_dy_px=vert_dy_px,
            rotation_deg=metrics.rotation_deg,
            zoom_ratio=metrics.zoom_ratio,
            n_inliers=metrics.n_inliers,
            n_requested=metrics.n_requested,
            align_ok=align_ok,
            zoom_ok=zoom_ok,
        )
