"""Calibration tab: nudge sliders + physical-cal wizard."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QImage, QKeyEvent, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox,
    QComboBox, QSpinBox,
)

from ..config_state import save_calibration_state
from ..physical_cal import PhysicalCalSession, PatternRenderer
from .qt_helpers import ndarray_to_qimage
from .video_widget import VideoWidget


class CalibrationTab(QWidget):
    overlay_mode_changed = pyqtSignal(bool)
    overlay_frame_ready = pyqtSignal(object)  # QImage

    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.session = PhysicalCalSession()
        self.renderer = PatternRenderer()
        self._latest_status: dict = {}
        self._overlay_active = False
        self._overlay_master = "left"
        self._overlay_slave_x = 0
        self._overlay_slave_y = 0
        self._overlay_flash_on = True
        self._overlay_last_sbs: np.ndarray | None = None
        self._overlay_flash_timer = QTimer(self)
        self._overlay_flash_timer.timeout.connect(self._toggle_overlay_flash)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._overlay_shortcuts: list[QShortcut] = []
        self._init_overlay_shortcuts()

        root = QVBoxLayout(self)
        root.addWidget(self._make_nudge_group())
        root.addWidget(self._make_reset_row())
        root.addWidget(self._make_overlay_cal_group())
        root.addWidget(self._make_wizard_group(), stretch=1)

        self._last_wizard_render_t: float = 0.0
        if worker is not None:
            worker.sbs_frame_ready.connect(self._on_frame_for_wizard)
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

    # ------------------ Overlay/manual calibration --------------------

    def _make_overlay_cal_group(self) -> QGroupBox:
        box = QGroupBox("Overlay manual calibration", self)
        lay = QVBoxLayout(box)
        self.overlay_preview = VideoWidget(box)
        lay.addWidget(self.overlay_preview, stretch=1)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Master eye:"))
        self.master_combo = QComboBox(box)
        self.master_combo.addItems(["Left (slave: Right)", "Right (slave: Left)"])
        self.master_combo.currentIndexChanged.connect(self._on_master_changed)
        controls.addWidget(self.master_combo)

        controls.addSpacing(16)
        controls.addWidget(QLabel("Flash every (ms):"))
        self.flash_ms = QSpinBox(box)
        self.flash_ms.setRange(300, 4000)
        self.flash_ms.setSingleStep(100)
        self.flash_ms.setValue(1000)
        self.flash_ms.valueChanged.connect(self._on_flash_interval_changed)
        controls.addWidget(self.flash_ms)

        controls.addSpacing(16)
        controls.addWidget(QLabel("Arrow step (px):"))
        self.step_px = QSpinBox(box)
        self.step_px.setRange(1, 20)
        self.step_px.setValue(2)
        controls.addWidget(self.step_px)
        controls.addStretch(1)
        lay.addLayout(controls)

        buttons = QHBoxLayout()
        self.btn_overlay_start = QPushButton("Start overlay calibration")
        self.btn_overlay_start.clicked.connect(self._toggle_overlay_mode)
        self.btn_overlay_save = QPushButton("Save to sliders")
        self.btn_overlay_save.clicked.connect(self._save_overlay_calibration)
        self.btn_overlay_cancel = QPushButton("Cancel")
        self.btn_overlay_cancel.clicked.connect(self._cancel_overlay_mode)
        self.btn_overlay_save.setEnabled(False)
        self.btn_overlay_cancel.setEnabled(False)
        buttons.addWidget(self.btn_overlay_start)
        buttons.addWidget(self.btn_overlay_save)
        buttons.addWidget(self.btn_overlay_cancel)
        buttons.addStretch(1)
        self.overlay_hint_lbl = QLabel(
            "Press Start, then use arrow keys to move the slave image over the master."
        )
        self.overlay_hint_lbl.setStyleSheet("font-family: monospace;")
        buttons.addWidget(self.overlay_hint_lbl)
        lay.addLayout(buttons)
        return box

    def _toggle_overlay_mode(self) -> None:
        if self.worker is None:
            return
        if self._overlay_active:
            self._cancel_overlay_mode()
            return
        self._overlay_active = True
        self._overlay_flash_on = True
        self._overlay_last_sbs = None
        self._load_overlay_slave_from_state()
        self._overlay_flash_timer.start(self.flash_ms.value())
        self.btn_overlay_start.setText("Stop overlay calibration")
        self.btn_overlay_save.setEnabled(True)
        self.btn_overlay_cancel.setEnabled(True)
        self.overlay_hint_lbl.setText(self._overlay_status_text())
        self.overlay_mode_changed.emit(True)
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        self._refresh_overlay_preview()

    def _cancel_overlay_mode(self) -> None:
        self._overlay_active = False
        self._overlay_flash_timer.stop()
        self.btn_overlay_start.setText("Start overlay calibration")
        self.btn_overlay_save.setEnabled(False)
        self.btn_overlay_cancel.setEnabled(False)
        self.overlay_mode_changed.emit(False)
        self.overlay_hint_lbl.setText(
            "Press Start, then use arrow keys to move the slave image over the master."
        )

    def _save_overlay_calibration(self) -> None:
        if not self._overlay_active or self.worker is None:
            return
        slave = self._slave_eye()
        if slave == "left":
            self._set("nudge_left_x", self._overlay_slave_x,
                      lambda: setattr(self.worker.calibration, "nudge_left", self._overlay_slave_x))
            self._set("nudge_left_y", self._overlay_slave_y,
                      lambda: setattr(self.worker.calibration, "nudge_left_y", self._overlay_slave_y))
            if self.sld_lx is not None:
                self.sld_lx.setValue(self._overlay_slave_x)
            if self.sld_ly is not None:
                self.sld_ly.setValue(self._overlay_slave_y)
        else:
            self._set("nudge_right_x", self._overlay_slave_x,
                      lambda: setattr(self.worker.calibration, "nudge_right", self._overlay_slave_x))
            self._set("nudge_right_y", self._overlay_slave_y,
                      lambda: setattr(self.worker.calibration, "nudge_right_y", self._overlay_slave_y))
            if self.sld_rx is not None:
                self.sld_rx.setValue(self._overlay_slave_x)
            if self.sld_ry is not None:
                self.sld_ry.setValue(self._overlay_slave_y)
        self._cancel_overlay_mode()

    def _on_master_changed(self, idx: int) -> None:
        self._overlay_master = "left" if idx == 0 else "right"
        self._load_overlay_slave_from_state()
        if self._overlay_active:
            self.overlay_hint_lbl.setText(self._overlay_status_text())

    def _on_flash_interval_changed(self, ms: int) -> None:
        if self._overlay_flash_timer.isActive():
            self._overlay_flash_timer.start(ms)

    def _toggle_overlay_flash(self) -> None:
        self._overlay_flash_on = not self._overlay_flash_on
        self._refresh_overlay_preview()

    def _load_overlay_slave_from_state(self) -> None:
        if self.worker is None:
            return
        st = self.worker.cfg.calibration_state
        if self._slave_eye() == "left":
            self._overlay_slave_x = st.nudge_left_x
            self._overlay_slave_y = st.nudge_left_y
        else:
            self._overlay_slave_x = st.nudge_right_x
            self._overlay_slave_y = st.nudge_right_y

    def _slave_eye(self) -> str:
        return "right" if self._overlay_master == "left" else "left"

    def _overlay_status_text(self) -> str:
        slave = self._slave_eye().upper()
        return (
            f"Master={self._overlay_master.upper()}  Slave={slave}  "
            f"offset=({self._overlay_slave_x:+d},{self._overlay_slave_y:+d}) px"
        )

    @staticmethod
    def _shift_eye(img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
        if shift_x == 0 and shift_y == 0:
            return img.copy()
        out = img.copy()
        if shift_x != 0:
            out = np.roll(out, shift_x, axis=1)
            if shift_x > 0:
                out[:, :shift_x] = 0
            else:
                out[:, shift_x:] = 0
        if shift_y != 0:
            out = np.roll(out, shift_y, axis=0)
            if shift_y > 0:
                out[:shift_y, :] = 0
            else:
                out[shift_y:, :] = 0
        return out

    def _render_overlay_preview(self, sbs: np.ndarray) -> None:
        if sbs.ndim != 3 or sbs.shape[2] != 3:
            return
        eye_w = sbs.shape[1] // 2
        left = sbs[:, :eye_w].copy()
        right = sbs[:, eye_w:].copy()
        st = self.worker.cfg.calibration_state if self.worker is not None else None

        if self._overlay_master == "left":
            base_x = st.nudge_right_x if st is not None else 0
            base_y = st.nudge_right_y if st is not None else 0
            slave_shifted = self._shift_eye(right, self._overlay_slave_x - base_x, self._overlay_slave_y - base_y)
            out_left = left
            out_right = slave_shifted if self._overlay_flash_on else np.zeros_like(slave_shifted)
        else:
            base_x = st.nudge_left_x if st is not None else 0
            base_y = st.nudge_left_y if st is not None else 0
            slave_shifted = self._shift_eye(left, self._overlay_slave_x - base_x, self._overlay_slave_y - base_y)
            out_left = slave_shifted if self._overlay_flash_on else np.zeros_like(slave_shifted)
            out_right = right

        composite = np.concatenate([out_left, out_right], axis=1)

        h, w = composite.shape[:2]
        # Draw a crosshair in each eye so alignment is easy while preserving native eye layout.
        for cx in (eye_w // 2, eye_w + eye_w // 2):
            cv2.line(composite, (cx - 40, h // 2), (cx + 40, h // 2), (255, 255, 255), 1)
            cv2.line(composite, (cx, h // 2 - 40), (cx, h // 2 + 40), (255, 255, 255), 1)
        cv2.putText(
            composite,
            f"{self._overlay_status_text()}  flash={'ON' if self._overlay_flash_on else 'OFF'}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        qimg = ndarray_to_qimage(composite)
        self.overlay_preview.set_frame(qimg)
        self.overlay_frame_ready.emit(qimg)

    def _init_overlay_shortcuts(self) -> None:
        def bind(key: str, handler):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(handler)
            self._overlay_shortcuts.append(sc)

        bind("Left", lambda: self._overlay_move(-self.step_px.value(), 0))
        bind("Right", lambda: self._overlay_move(self.step_px.value(), 0))
        bind("Up", lambda: self._overlay_move(0, -self.step_px.value()))
        bind("Down", lambda: self._overlay_move(0, self.step_px.value()))
        bind("Return", self._save_overlay_calibration)
        bind("Enter", self._save_overlay_calibration)
        bind("Escape", self._cancel_overlay_mode)

    def _overlay_move(self, dx: int, dy: int) -> None:
        if not self._overlay_active:
            return
        self._overlay_slave_x += dx
        self._overlay_slave_y += dy
        self.overlay_hint_lbl.setText(self._overlay_status_text())
        self._refresh_overlay_preview()

    def _refresh_overlay_preview(self) -> None:
        if self._overlay_last_sbs is None or not self._overlay_active:
            return
        self._render_overlay_preview(self._overlay_last_sbs)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not self._overlay_active:
            super().keyPressEvent(event)
            return
        step = self.step_px.value()
        handled = True
        if event.key() == Qt.Key.Key_Left:
            self._overlay_slave_x -= step
        elif event.key() == Qt.Key.Key_Right:
            self._overlay_slave_x += step
        elif event.key() == Qt.Key.Key_Up:
            self._overlay_slave_y -= step
        elif event.key() == Qt.Key.Key_Down:
            self._overlay_slave_y += step
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._save_overlay_calibration()
        elif event.key() == Qt.Key.Key_Escape:
            self._cancel_overlay_mode()
        else:
            handled = False

        if handled:
            self.overlay_hint_lbl.setText(self._overlay_status_text())
            self._refresh_overlay_preview()
            event.accept()
            return
        super().keyPressEvent(event)

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
    def _on_frame_for_wizard(self, sbs) -> None:
        # Skip work entirely if this tab isn't visible — avoids wasting CPU
        # re-processing camera frames while the user is on Live or Settings.
        if not self.isVisible():
            return
        # Throttle wizard re-render to ~10 FPS (calibration UI doesn't need
        # smooth video, but pattern overlay drawing + sharpness is heavy).
        import time as _t
        now = _t.perf_counter()
        if now - self._last_wizard_render_t < 0.1:
            return
        self._last_wizard_render_t = now
        if self._overlay_active:
            # sbs is the worker's reused buffer; copy so subsequent ticks
            # don't overwrite the pixels while we're rendering.
            self._overlay_last_sbs = sbs.copy()
            self._render_overlay_preview(self._overlay_last_sbs)
            return
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
        self.wizard_preview.set_sbs_frame(sbs)
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
