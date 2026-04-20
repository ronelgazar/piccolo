"""Calibration tab: nudge sliders + physical-cal wizard (wizard in Task 8)."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox,
)

from ..config_state import save_calibration_state


class CalibrationTab(QWidget):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        root = QVBoxLayout(self)
        root.addWidget(self._make_nudge_group())
        root.addWidget(self._make_reset_row())
        root.addStretch(1)

    def _make_nudge_group(self) -> QGroupBox:
        box = QGroupBox("Per-eye pixel nudge", self)
        lay = QVBoxLayout(box)
        if self.worker is None:
            lay.addWidget(QLabel("(unavailable — worker not started)"))
            self.sld_lx = self.sld_rx = self.sld_ly = self.sld_ry = None
            return box
        st = self.worker.cfg.calibration_state
        self.sld_lx = self._add_slider(lay, "Left eye X",  st.nudge_left_x,  self._on_lx)
        self.sld_rx = self._add_slider(lay, "Right eye X", st.nudge_right_x, self._on_rx)
        self.sld_ly = self._add_slider(lay, "Left eye Y",  st.nudge_left_y,  self._on_ly)
        self.sld_ry = self._add_slider(lay, "Right eye Y", st.nudge_right_y, self._on_ry)
        return box

    def _add_slider(self, parent_layout, label: str, initial: int, cb) -> QSlider:
        row = QHBoxLayout()
        lbl = QLabel(f"{label}: {initial}px")
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setMinimum(-300)
        sld.setMaximum(300)
        sld.setValue(initial)
        sld.valueChanged.connect(lambda v, l=lbl, t=label: l.setText(f"{t}: {v}px"))
        sld.sliderReleased.connect(lambda s=sld, cb=cb: cb(s.value()))
        row.addWidget(lbl, stretch=1)
        row.addWidget(sld, stretch=3)
        parent_layout.addLayout(row)
        return sld

    def _make_reset_row(self) -> QWidget:
        w = QWidget(self)
        lay = QHBoxLayout(w)
        btn = QPushButton("Reset all nudges")
        btn.clicked.connect(self._reset)
        lay.addWidget(btn)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------

    def _on_lx(self, v: int) -> None:
        self._set("nudge_left_x",  v, lambda: setattr(self.worker.calibration, "nudge_left",  v))

    def _on_rx(self, v: int) -> None:
        self._set("nudge_right_x", v, lambda: setattr(self.worker.calibration, "nudge_right", v))

    def _on_ly(self, v: int) -> None:
        self._set("nudge_left_y",  v, lambda: setattr(self.worker.calibration, "nudge_left_y",  v))

    def _on_ry(self, v: int) -> None:
        self._set("nudge_right_y", v, lambda: setattr(self.worker.calibration, "nudge_right_y", v))

    def _set(self, attr: str, value: int, apply_runtime) -> None:
        setattr(self.worker.cfg.calibration_state, attr, value)
        apply_runtime()
        save_calibration_state(self.worker.cfg)

    def _reset(self) -> None:
        if self.worker is None or self.sld_lx is None:
            return
        for s in (self.sld_lx, self.sld_rx, self.sld_ly, self.sld_ry):
            s.setValue(0)
        self.worker.calibration.reset_nudge()
        st = self.worker.cfg.calibration_state
        st.nudge_left_x = st.nudge_right_x = st.nudge_left_y = st.nudge_right_y = 0
        save_calibration_state(self.worker.cfg)
