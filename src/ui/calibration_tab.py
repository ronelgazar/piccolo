"""Calibration tab: nudge sliders + Smart overlap calibration."""
from __future__ import annotations

import time

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox,
    QComboBox, QSpinBox,
)

from ..config_state import save_calibration_state
from ..smart_overlap import OverlapMetrics, SmartOverlapAnalyzer
from ..stereo_matching import StereoFeatureMatcher
from .qt_helpers import ndarray_to_qimage
from .smart_overlap_worker import SmartOverlapResult, SmartOverlapWorker
from .video_widget import VideoWidget


class CalibrationTab(QWidget):
    overlay_mode_changed = pyqtSignal(bool)
    overlay_frame_ready = pyqtSignal(object)  # QImage
    smart_overlap_mode_changed = pyqtSignal(bool)
    smart_overlap_frame_ready = pyqtSignal(object)  # QImage
    OVERLAY_SLAVE_OPACITY = 0.70

    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        cfg_so = worker.cfg.smart_overlap if worker is not None else None
        state = worker.cfg.calibration_state if worker is not None else None

        eye_w = worker.cfg.cameras.left.width if worker is not None else 1920
        eye_h = worker.cfg.cameras.left.height if worker is not None else 1080

        matcher = StereoFeatureMatcher(
            max_features=500, match_ratio=0.75, ransac_thresh=2.0,
            frame_w=eye_w, frame_h=eye_h,
        )
        analyzer = SmartOverlapAnalyzer(
            max_vert_dy_px=cfg_so.max_vert_dy_px if cfg_so else 5.0,
            max_rotation_deg=cfg_so.max_rotation_deg if cfg_so else 0.5,
            max_zoom_ratio_err=cfg_so.max_zoom_ratio_err if cfg_so else 0.02,
            min_pairs_for_metrics=cfg_so.min_pairs_for_metrics if cfg_so else 4,
            pair_stability_tol_px=cfg_so.pair_stability_tol_px if cfg_so else 30.0,
            matcher=matcher,
        )
        self.smart_overlap_worker = SmartOverlapWorker(analyzer, self)
        self.smart_overlap_worker.result_ready.connect(self._on_smart_overlap_result)
        self.smart_overlap_worker.start()
        self._smart_active = False
        self._smart_mode = (state.smart_overlap_mode if state else "chessboard")
        self._smart_pair_count = (state.smart_overlap_pair_count if state else 8)
        self._latest_metrics: OverlapMetrics | None = None
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
        root.addWidget(self._make_smart_overlap_group(), stretch=1)

        self._last_wizard_render_t: float = 0.0
        if worker is not None:
            worker.sbs_frame_ready.connect(self._on_frame_for_smart_overlap)

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
        self.sld_lscale = self._add_slider(lay, "Left scale", st.scale_left_pct, self._on_lscale, lo=80, hi=120, unit="%")
        self.sld_rscale = self._add_slider(lay, "Right scale", st.scale_right_pct, self._on_rscale, lo=80, hi=120, unit="%")
        return box

    def _add_slider(self, parent_layout, label: str, initial: int, cb, lo: int = -300, hi: int = 300, unit: str = "px") -> QSlider:
        row = QHBoxLayout()
        lbl = QLabel(f"{label}: {initial}{unit}")
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setMinimum(lo)
        sld.setMaximum(hi)
        sld.setValue(initial)
        sld.valueChanged.connect(lambda v, l=lbl, t=label, u=unit: l.setText(f"{t}: {v}{u}"))
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

    def _on_lscale(self, v: int) -> None:
        self._set("scale_left_pct", v, lambda: setattr(self.worker.calibration, "scale_left_pct", v))

    def _on_rscale(self, v: int) -> None:
        self._set("scale_right_pct", v, lambda: setattr(self.worker.calibration, "scale_right_pct", v))

    def _set(self, attr: str, value: int, apply_runtime) -> None:
        setattr(self.worker.cfg.calibration_state, attr, value)
        apply_runtime()
        save_calibration_state(self.worker.cfg)

    def _reset(self) -> None:
        if self.worker is None or self.sld_lx is None:
            return
        for s in (self.sld_lx, self.sld_rx, self.sld_ly, self.sld_ry):
            s.setValue(0)
        for s in (self.sld_lscale, self.sld_rscale):
            s.setValue(100)
        self.worker.calibration.reset_nudge()
        st = self.worker.cfg.calibration_state
        st.nudge_left_x = st.nudge_right_x = st.nudge_left_y = st.nudge_right_y = 0
        st.scale_left_pct = st.scale_right_pct = 100
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
        if self._smart_active:
            self._stop_smart_mode()
        self._overlay_active = True
        self._update_worker_raw_rate()
        self._overlay_flash_on = True
        self._overlay_last_sbs = None
        self._reset_overlay_slave_offset()
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
        self._update_worker_raw_rate()
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
        self._reset_overlay_slave_offset()
        if self._overlay_active:
            self.overlay_hint_lbl.setText(self._overlay_status_text())

    def _on_flash_interval_changed(self, ms: int) -> None:
        if self._overlay_flash_timer.isActive():
            self._overlay_flash_timer.start(ms)

    def _toggle_overlay_flash(self) -> None:
        self._overlay_flash_on = not self._overlay_flash_on
        self._refresh_overlay_preview()

    def _reset_overlay_slave_offset(self) -> None:
        # Overlay calibration uses the pre-nudge full-FOV frame. Starting from
        # the saved nudge would immediately shift/crop the slave image when the
        # previous offset is large, making recalibration hard to reason about.
        self._overlay_slave_x = 0
        self._overlay_slave_y = 0

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
        out = np.zeros_like(img)
        h, w = img.shape[:2]
        src_x0 = max(0, -shift_x)
        src_x1 = min(w, w - shift_x)
        dst_x0 = max(0, shift_x)
        dst_x1 = min(w, w + shift_x)
        src_y0 = max(0, -shift_y)
        src_y1 = min(h, h - shift_y)
        dst_y0 = max(0, shift_y)
        dst_y1 = min(h, h + shift_y)
        if src_x0 >= src_x1 or src_y0 >= src_y1:
            return out
        out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
        return out

    @classmethod
    def _slave_visible_image(cls, img: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(img, alpha=cls.OVERLAY_SLAVE_OPACITY, beta=0)

    def _render_overlay_preview(self, sbs: np.ndarray) -> None:
        if sbs.ndim != 3 or sbs.shape[2] != 3:
            return
        eye_w = sbs.shape[1] // 2
        left = sbs[:, :eye_w].copy()
        right = sbs[:, eye_w:].copy()

        if self._overlay_master == "left":
            slave_shifted = self._shift_eye(right, self._overlay_slave_x, self._overlay_slave_y)
            composite_eye = self._overlay_composite(left, slave_shifted)
        else:
            slave_shifted = self._shift_eye(left, self._overlay_slave_x, self._overlay_slave_y)
            composite_eye = self._overlay_composite(right, slave_shifted)

        composite = np.concatenate([composite_eye.copy(), composite_eye.copy()], axis=1)

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

    def _overlay_composite(self, master: np.ndarray, slave: np.ndarray) -> np.ndarray:
        out = master.copy()
        if self._overlay_flash_on:
            slave_visible = self._slave_visible_image(slave)
            mask = np.any(slave > 0, axis=2)
            out[mask] = cv2.addWeighted(
                master[mask],
                1.0 - self.OVERLAY_SLAVE_OPACITY,
                slave_visible[mask],
                self.OVERLAY_SLAVE_OPACITY,
                0,
            )
        return out

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

    # ------------------ Smart overlap calibration ----------------------

    def _make_smart_overlap_group(self) -> QGroupBox:
        box = QGroupBox("Smart overlap calibration", self)
        lay = QVBoxLayout(box)
        self.smart_preview = VideoWidget(box)
        lay.addWidget(self.smart_preview, stretch=1)

        readouts = QHBoxLayout()
        self.lbl_vert = QLabel("Vert offset: --")
        self.lbl_rot = QLabel("Rotation: --")
        self.lbl_zoom = QLabel("Zoom ratio: --")
        self.lbl_pairs = QLabel("Match pairs: 0 / 0")
        for w in (self.lbl_vert, self.lbl_rot, self.lbl_zoom, self.lbl_pairs):
            w.setStyleSheet("font-family: monospace;")
            readouts.addWidget(w)
        readouts.addStretch(1)
        lay.addLayout(readouts)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox(box)
        self.mode_combo.addItems(["Chessboard", "Live scene"])
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(0 if self._smart_mode == "chessboard" else 1)
        self.mode_combo.blockSignals(False)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self.mode_combo)

        controls.addSpacing(12)
        controls.addWidget(QLabel("Pairs:"))
        self.pair_spin = QSpinBox(box)
        self.pair_spin.setRange(4, 20)
        self.pair_spin.blockSignals(True)
        self.pair_spin.setValue(self._smart_pair_count)
        self.pair_spin.blockSignals(False)
        self.pair_spin.valueChanged.connect(self._on_pair_count_changed)
        controls.addWidget(self.pair_spin)

        controls.addSpacing(12)
        self.lbl_align_badge = QLabel("ALIGN --")
        self.lbl_zoom_badge = QLabel("ZOOM --")
        for w in (self.lbl_align_badge, self.lbl_zoom_badge):
            w.setStyleSheet("font-family: monospace; padding: 2px 6px; border-radius: 3px;")
            controls.addWidget(w)

        controls.addStretch(1)
        self.btn_smart_start = QPushButton("Start")
        self.btn_smart_start.clicked.connect(self._toggle_smart_mode)
        controls.addWidget(self.btn_smart_start)

        self.btn_apply_scale = QPushButton("Apply detected scale")
        self.btn_apply_scale.setEnabled(False)
        self.btn_apply_scale.clicked.connect(self._apply_detected_scale)
        controls.addWidget(self.btn_apply_scale)

        lay.addLayout(controls)
        return box

    @pyqtSlot(object)
    def _on_frame_for_smart_overlap(self, sbs) -> None:
        if not self.isVisible():
            return
        now = time.perf_counter()
        interval = 0.1 if self._overlay_active else (0.2 if self._smart_active else 1.0)
        if now - self._last_wizard_render_t < interval:
            return
        self._last_wizard_render_t = now
        if self._overlay_active:
            self._overlay_last_sbs = sbs.copy()
            self._render_overlay_preview(self._overlay_last_sbs)
            return
        if not self._smart_active:
            return
        if sbs is None or sbs.ndim != 3 or sbs.shape[2] != 3:
            return
        self.smart_overlap_worker.submit(sbs, self._smart_mode, self._smart_pair_count)

    @pyqtSlot(object)
    def _on_smart_overlap_result(self, result: SmartOverlapResult) -> None:
        self._latest_metrics = result.metrics
        self.smart_preview.set_frame(result.image)
        if self._smart_active:
            self.smart_overlap_frame_ready.emit(result.image)
        self._update_smart_readouts(result.metrics)

    def _update_smart_readouts(self, m: OverlapMetrics) -> None:
        min_pairs = self.worker.cfg.smart_overlap.min_pairs_for_metrics if self.worker else 4
        if m.n_inliers < min_pairs:
            self.lbl_vert.setText("Vert offset: --")
            self.lbl_rot.setText("Rotation: --")
            self.lbl_zoom.setText("Zoom ratio: --")
        else:
            self.lbl_vert.setText(f"Vert offset: {m.vert_dy_px:+.1f} px")
            self.lbl_rot.setText(f"Rotation: {m.rotation_deg:+.2f} deg")
            zr = "--" if m.zoom_ratio is None else f"{m.zoom_ratio:.3f}"
            self.lbl_zoom.setText(f"Zoom ratio: {zr}")
        self.lbl_pairs.setText(f"Match pairs: {m.n_inliers} / {m.n_requested}")

        self._set_badge(self.lbl_align_badge, "ALIGN", m.align_ok, neutral=m.n_inliers == 0)
        self._set_badge(self.lbl_zoom_badge, "ZOOM", m.zoom_ok, neutral=m.zoom_ratio is None)
        self.btn_apply_scale.setEnabled(
            self._smart_active and m.zoom_ratio is not None and m.n_inliers >= min_pairs
        )

    @staticmethod
    def _set_badge(label: QLabel, prefix: str, ok: bool, neutral: bool) -> None:
        if neutral:
            label.setText(f"{prefix} --")
            label.setStyleSheet("font-family: monospace; padding: 2px 6px; border-radius: 3px;"
                                "background: #2a2a2a; color: #aaa;")
            return
        if ok:
            label.setText(f"{prefix} OK")
            label.setStyleSheet("font-family: monospace; padding: 2px 6px; border-radius: 3px;"
                                "background: #2d6a3d; color: #fff;")
        else:
            label.setText(f"{prefix} ADJUST")
            label.setStyleSheet("font-family: monospace; padding: 2px 6px; border-radius: 3px;"
                                "background: #8a4a1a; color: #fff;")

    def stop_background_work(self) -> None:
        self._stop_smart_mode()
        self.smart_overlap_worker.stop()

    def _toggle_smart_mode(self) -> None:
        if self.worker is None:
            return
        if self._smart_active:
            self._stop_smart_mode()
        else:
            self._start_smart_mode()

    def _start_smart_mode(self) -> None:
        if self._overlay_active:
            self._cancel_overlay_mode()
        self._smart_active = True
        self.btn_smart_start.setText("Stop")
        self.smart_overlap_mode_changed.emit(True)
        self._update_worker_raw_rate()

    def _stop_smart_mode(self) -> None:
        if not self._smart_active:
            return
        self._smart_active = False
        self.btn_smart_start.setText("Start")
        self.smart_overlap_worker.reset_state()
        self.btn_apply_scale.setEnabled(False)
        self.smart_overlap_mode_changed.emit(False)
        self._update_worker_raw_rate()

    def _on_mode_changed(self, idx: int) -> None:
        self._smart_mode = "chessboard" if idx == 0 else "live"
        self.smart_overlap_worker.reset_state()
        if self.worker is not None:
            self.worker.cfg.calibration_state.smart_overlap_mode = self._smart_mode
            save_calibration_state(self.worker.cfg)

    def _on_pair_count_changed(self, value: int) -> None:
        self._smart_pair_count = int(value)
        self.smart_overlap_worker.reset_state()
        if self.worker is not None:
            self.worker.cfg.calibration_state.smart_overlap_pair_count = int(value)
            save_calibration_state(self.worker.cfg)

    def _update_worker_raw_rate(self) -> None:
        if self.worker is None:
            return
        if self._overlay_active:
            self.worker.raw_frame_interval = 0.1
        elif self._smart_active:
            self.worker.raw_frame_interval = 0.2
        else:
            self.worker.raw_frame_interval = 1.0

    def _apply_detected_scale(self) -> None:
        metrics = self._latest_metrics
        if self.worker is None or metrics is None or metrics.zoom_ratio is None:
            return
        right_scale = int(round(100.0 / metrics.zoom_ratio))
        right_scale = max(80, min(120, right_scale))
        self.sld_lscale.setValue(100)
        self.sld_rscale.setValue(right_scale)
        self._on_lscale(100)
        self._on_rscale(right_scale)
