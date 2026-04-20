"""Calibration tab: nudge sliders + physical-cal wizard."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox,
)

from ..config_state import save_calibration_state
from ..physical_cal import PhysicalCalSession, PatternRenderer
from .qt_helpers import ndarray_to_qimage
from .video_widget import VideoWidget


class CalibrationTab(QWidget):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.session = PhysicalCalSession()
        self.renderer = PatternRenderer()
        self._latest_status: dict = {}

        root = QVBoxLayout(self)
        root.addWidget(self._make_nudge_group())
        root.addWidget(self._make_reset_row())
        root.addWidget(self._make_wizard_group(), stretch=1)

        if worker is not None:
            worker.frame_ready.connect(self._on_frame_for_wizard)
            worker.status_tick.connect(self._on_status_for_wizard)

    # ------------------ Nudge sliders ----------------------------------

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

    # ------------------ Wizard -----------------------------------------

    def _make_wizard_group(self) -> QGroupBox:
        box = QGroupBox("Physical calibration wizard", self)
        lay = QVBoxLayout(box)
        self.wizard_preview = VideoWidget(box)
        lay.addWidget(self.wizard_preview, stretch=1)
        btns = QHBoxLayout()
        self.wizard_phase_lbl = QLabel("Phase: Focus (1/4)")
        self.wizard_readout_lbl = QLabel("")
        self.wizard_readout_lbl.setStyleSheet("font-family: monospace;")
        self.btn_prev = QPushButton("← Prev")
        self.btn_next = QPushButton("Next →")
        self.btn_prev.clicked.connect(self._wizard_prev)
        self.btn_next.clicked.connect(self._wizard_next)
        btns.addWidget(self.btn_prev)
        btns.addWidget(self.btn_next)
        btns.addSpacing(20)
        btns.addWidget(self.wizard_phase_lbl)
        btns.addStretch(1)
        btns.addWidget(self.wizard_readout_lbl)
        lay.addLayout(btns)
        return box

    @pyqtSlot(dict)
    def _on_status_for_wizard(self, st: dict) -> None:
        self._latest_status = st

    @pyqtSlot(object)
    def _on_frame_for_wizard(self, _qimg: QImage) -> None:
        if self.worker is None:
            return
        cam_l = self.worker.cam_l
        cam_r = self.worker.cam_r
        if cam_l is None or cam_r is None:
            return
        fl = cam_l.read_no_copy()
        fr = cam_r.read_no_copy()
        if fl is None or fr is None:
            return
        if self.worker.cfg.cameras.left.flip_180:
            fl = cv2.rotate(fl, cv2.ROTATE_180)
        if self.worker.cfg.cameras.right.flip_180:
            fr = cv2.rotate(fr, cv2.ROTATE_180)
        eye_l = fl.copy()
        eye_r = fr.copy()
        phase = self.session.phase
        dy = self._latest_status.get("dy")
        dtheta_deg = self._latest_status.get("dtheta_deg")
        sharp_l = PhysicalCalSession.sharpness(fl)
        sharp_r = PhysicalCalSession.sharpness(fr)
        if phase == "focus":
            self.renderer.render_focus(eye_l, sharp_l)
            self.renderer.render_focus(eye_r, sharp_r)
        elif phase == "scale":
            self.renderer.render_scale(eye_l)
            self.renderer.render_scale(eye_r)
        elif phase == "horizontal":
            self.renderer.render_horizontal(eye_l, dy)
            self.renderer.render_horizontal(eye_r, dy)
        elif phase == "rotation":
            self.renderer.render_rotation(eye_l, dtheta_deg)
            self.renderer.render_rotation(eye_r, dtheta_deg)
        sbs = np.concatenate([eye_l, eye_r], axis=1)
        self.wizard_preview.set_frame(ndarray_to_qimage(sbs))
        self._update_wizard_readout(sharp_l, sharp_r, dy, dtheta_deg)

    def _update_wizard_readout(self, sl, sr, dy, dt):
        phase = self.session.phase
        self.wizard_phase_lbl.setText(
            f"Phase: {phase.capitalize()} ({self.session.phase_index + 1}/{self.session.total_phases})"
        )
        if phase == "focus":
            self.wizard_readout_lbl.setText(f"sharp L={sl:.0f}  R={sr:.0f}")
        elif phase == "horizontal":
            self.wizard_readout_lbl.setText(f"dy={dy:+.1f}px" if dy is not None else "dy=--")
        elif phase == "rotation":
            self.wizard_readout_lbl.setText(f"rot={dt:+.2f}°" if dt is not None else "rot=--")
        else:
            self.wizard_readout_lbl.setText("")

    def _wizard_next(self) -> None:
        self.session.next_phase()

    def _wizard_prev(self) -> None:
        self.session.prev_phase()
