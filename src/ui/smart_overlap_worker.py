"""Background analysis for the Smart overlap calibration panel."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..smart_overlap import OverlapMetrics, SmartOverlapAnalyzer, render_overlay
from .qt_helpers import ndarray_to_qimage


@dataclass
class SmartOverlapResult:
    image: object              # QImage
    metrics: OverlapMetrics


class SmartOverlapWorker(QThread):
    """Processes at most one latest SBS frame at a time and emits a rendered image + metrics."""

    result_ready = pyqtSignal(object)   # SmartOverlapResult

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
        eye_w = sbs.shape[1] // 2
        eye_l = sbs[:, :eye_w]
        eye_r = sbs[:, eye_w:]
        metrics = self._analyzer.analyze(eye_l, eye_r, mode=mode, pair_count=pair_count)
        rendered = render_overlay(sbs, metrics)
        return SmartOverlapResult(
            image=ndarray_to_qimage(rendered),
            metrics=metrics,
        )
