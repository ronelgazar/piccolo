"""Settings tab: cameras, display, stereo, pedals, config file groups."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from dataclasses import asdict

import yaml
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QLabel, QLineEdit, QPushButton,
    QMessageBox, QFileDialog,
)

from ..config_state import save_calibration_state, _default_config_path


class SettingsTab(QWidget):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        root = QVBoxLayout(self)
        if worker is None:
            root.addWidget(QLabel("(Settings unavailable — worker not started)"))
            root.addStretch(1)
            return
        root.addWidget(self._make_cameras_group())
        root.addWidget(self._make_display_group())
        root.addWidget(self._make_stereo_group())
        root.addWidget(self._make_depth_group())
        root.addWidget(self._make_performance_group())
        root.addWidget(self._make_pedals_group())
        root.addWidget(self._make_config_group())
        root.addStretch(1)

    # ------------------ Cameras ----------------------------------------

    def _make_cameras_group(self) -> QGroupBox:
        box = QGroupBox("Cameras (requires app restart to apply)", self)
        form = QFormLayout(box)
        c = self.worker.cfg.cameras
        form.addRow(QLabel("Left index"),  self._spinbox(c.left.index, 0, 10,
                                                          lambda v: setattr(c.left, "index", v)))
        form.addRow(QLabel("Right index"), self._spinbox(c.right.index, 0, 10,
                                                          lambda v: setattr(c.right, "index", v)))
        cb_flip_l = QCheckBox("Flip Left 180°")
        cb_flip_l.setChecked(c.left.flip_180)
        cb_flip_l.stateChanged.connect(
            lambda s: setattr(c.left, "flip_180", s == Qt.CheckState.Checked.value))
        form.addRow(cb_flip_l)
        cb_flip_r = QCheckBox("Flip Right 180°")
        cb_flip_r.setChecked(c.right.flip_180)
        cb_flip_r.stateChanged.connect(
            lambda s: setattr(c.right, "flip_180", s == Qt.CheckState.Checked.value))
        form.addRow(cb_flip_r)
        return box

    # ------------------ Display ----------------------------------------

    def _make_display_group(self) -> QGroupBox:
        box = QGroupBox("Goovis Display", self)
        form = QFormLayout(box)
        self.cmb_monitor = QComboBox()
        screens = QGuiApplication.screens()
        for i, s in enumerate(screens):
            g = s.geometry()
            self.cmb_monitor.addItem(f"[{i}] {s.name()}  {g.width()}x{g.height()}", i)
        form.addRow(QLabel("Monitor"), self.cmb_monitor)
        self.cmb_monitor.currentIndexChanged.connect(
            lambda i: setattr(self.worker.cfg.display, "monitor", self.cmb_monitor.itemData(i)))
        return box

    # ------------------ Stereo -----------------------------------------

    def _make_stereo_group(self) -> QGroupBox:
        box = QGroupBox("Stereo", self)
        form = QFormLayout(box)
        s = self.worker.cfg.stereo
        # Convergence slider
        row = QHBoxLayout()
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setMinimum(-200); sld.setMaximum(200)
        sld.setValue(s.convergence.base_offset)
        lbl = QLabel(f"{s.convergence.base_offset}px")
        sld.valueChanged.connect(lambda v, l=lbl: l.setText(f"{v}px"))
        sld.sliderReleased.connect(lambda: self._save_conv(sld.value()))
        row.addWidget(sld); row.addWidget(lbl)
        wrap = QWidget(); wrap.setLayout(row)
        form.addRow(QLabel("Convergence offset"), wrap)
        # Auto-alignment
        cb = QCheckBox("Auto alignment")
        cb.setChecked(s.alignment.enabled)
        cb.stateChanged.connect(
            lambda v: setattr(self.worker.aligner, "enabled", v == Qt.CheckState.Checked.value))
        form.addRow(cb)
        # Zoom step
        self.cmb_aspect = QComboBox()
        self.cmb_aspect.addItem("Fit full FOV (no stretch)", "fit")
        self.cmb_aspect.addItem("Full camera width (stretch)", "full")
        self.cmb_aspect.addItem("Geometry-correct crop", "crop")
        idx = self.cmb_aspect.findData(getattr(s, "aspect_mode", "fit"))
        self.cmb_aspect.setCurrentIndex(max(0, idx))
        self.cmb_aspect.currentIndexChanged.connect(
            lambda i: setattr(s, "aspect_mode", self.cmb_aspect.itemData(i)))
        form.addRow(QLabel("Per-eye aspect"), self.cmb_aspect)
        form.addRow(QLabel("Zoom step"),
                    self._doublebox(s.zoom.step, 0.001, 1.0, 0.01,
                                    lambda v: setattr(s.zoom, "step", v)))
        form.addRow(QLabel("Zoom tick (ms)"),
                    self._spinbox(s.zoom.tick_ms, 1, 1000,
                                  lambda v: setattr(s.zoom, "tick_ms", v)))
        return box

    def _make_performance_group(self) -> QGroupBox:
        box = QGroupBox("Performance", self)
        form = QFormLayout(box)
        perf = self.worker.cfg.performance
        cb_low = QCheckBox("Low-latency mode")
        cb_low.setChecked(getattr(perf, 'low_latency_mode', False))
        cb_low.stateChanged.connect(lambda s: setattr(self.worker.cfg.performance, 'low_latency_mode', s == Qt.CheckState.Checked.value))
        form.addRow(cb_low)

        cb_gpu = QCheckBox("Use GPU for depth (if available)")
        cb_gpu.setChecked(getattr(perf, 'use_gpu_for_depth', False))
        cb_gpu.stateChanged.connect(lambda s: setattr(self.worker.cfg.performance, 'use_gpu_for_depth', s == Qt.CheckState.Checked.value))
        form.addRow(cb_gpu)

        return box

    def _make_depth_group(self) -> QGroupBox:
        box = QGroupBox("Depth calibration", self)
        form = QFormLayout(box)
        d = self.worker.cfg.stereo_calibration
        form.addRow(QLabel("Focal length (mm)"), self._doublebox(d.focal_length_mm, 1.0, 50.0, 0.1, lambda v: setattr(d, "focal_length_mm", v)))
        form.addRow(QLabel("Sensor width (mm)"), self._doublebox(d.sensor_width_mm, 1.0, 20.0, 0.01, lambda v: setattr(d, "sensor_width_mm", v)))
        form.addRow(QLabel("Sensor height (mm)"), self._doublebox(d.sensor_height_mm, 1.0, 20.0, 0.01, lambda v: setattr(d, "sensor_height_mm", v)))
        form.addRow(QLabel("Baseline (mm)"), self._doublebox(d.baseline_mm, 1.0, 300.0, 0.1, lambda v: setattr(d, "baseline_mm", v)))
        form.addRow(QLabel("Derived focal px"), self._doublebox(d.focal_length_px, 0.0, 10000.0, 1.0, lambda v: setattr(d, "focal_length_px", v)))
        form.addRow(QLabel("Ruler near (mm)"), self._doublebox(d.depth_ruler_near_mm, 1.0, 5000.0, 1.0, lambda v: setattr(d, "depth_ruler_near_mm", v)))
        form.addRow(QLabel("Ruler far (mm)"), self._doublebox(d.depth_ruler_far_mm, 2.0, 10000.0, 1.0, lambda v: setattr(d, "depth_ruler_far_mm", v)))
        form.addRow(QLabel("Depth downscale"), self._doublebox(d.depth_downscale, 0.1, 1.0, 0.05, lambda v: setattr(d, "depth_downscale", v)))
        form.addRow(QLabel("Num disparities"), self._spinbox(d.num_disparities, 16, 256, lambda v: setattr(d, "num_disparities", v)))
        form.addRow(QLabel("Block size"), self._spinbox(d.block_size, 5, 51, lambda v: setattr(d, "block_size", v | 1)))
        return box

    # ------------------ Helpers ----------------------------------------

    def _spinbox(self, value: int, lo: int, hi: int, cb) -> QSpinBox:
        w = QSpinBox(); w.setRange(lo, hi); w.setValue(value)
        w.valueChanged.connect(cb)
        return w

    def _doublebox(self, value: float, lo: float, hi: float, step: float, cb) -> QDoubleSpinBox:
        w = QDoubleSpinBox(); w.setRange(lo, hi); w.setSingleStep(step); w.setDecimals(3); w.setValue(value)
        w.valueChanged.connect(cb)
        return w

    def _save_conv(self, value: int) -> None:
        self.worker.cfg.stereo.convergence.base_offset = value
        self.worker.processor.base_offset = value
        self.worker.cfg.calibration_state.convergence_offset = value
        save_calibration_state(self.worker.cfg)

    # ------------------ Pedals -----------------------------------------

    def _make_pedals_group(self) -> QGroupBox:
        box = QGroupBox("Pedals", self)
        form = QFormLayout(box)
        ctl = self.worker.cfg.controls

        cb_enable = QCheckBox("Pedal input enabled")
        cb_enable.setChecked(True)
        form.addRow(cb_enable)

        self.ed_key_a = QLineEdit(ctl.pedal_key_a); self.ed_key_a.setMaxLength(1)
        self.ed_key_b = QLineEdit(ctl.pedal_key_b); self.ed_key_b.setMaxLength(1)
        self.ed_key_c = QLineEdit(ctl.pedal_key_c); self.ed_key_c.setMaxLength(1)
        for ed, attr in ((self.ed_key_a, "pedal_key_a"),
                          (self.ed_key_b, "pedal_key_b"),
                          (self.ed_key_c, "pedal_key_c")):
            ed.editingFinished.connect(
                lambda e=ed, a=attr: self._set_pedal_key(a, e.text()))
        form.addRow(QLabel("Pedal A key"), self.ed_key_a)
        form.addRow(QLabel("Pedal B key"), self.ed_key_b)
        form.addRow(QLabel("Pedal C key"), self.ed_key_c)

        modes = ["zoom", "side", "updown", "none"]
        self.cmb_mode_a = QComboBox(); self.cmb_mode_a.addItems(modes); self.cmb_mode_a.setCurrentText(ctl.pedal_mode_a)
        self.cmb_mode_b = QComboBox(); self.cmb_mode_b.addItems(modes); self.cmb_mode_b.setCurrentText(ctl.pedal_mode_b)
        self.cmb_mode_c = QComboBox(); self.cmb_mode_c.addItems(modes); self.cmb_mode_c.setCurrentText(ctl.pedal_mode_c)
        self.cmb_mode_a.currentTextChanged.connect(lambda t: setattr(ctl, "pedal_mode_a", t))
        self.cmb_mode_b.currentTextChanged.connect(lambda t: setattr(ctl, "pedal_mode_b", t))
        self.cmb_mode_c.currentTextChanged.connect(lambda t: setattr(ctl, "pedal_mode_c", t))
        form.addRow(QLabel("Pedal A mode"), self.cmb_mode_a)
        form.addRow(QLabel("Pedal B mode"), self.cmb_mode_b)
        form.addRow(QLabel("Pedal C mode"), self.cmb_mode_c)

        form.addRow(QLabel("Long-press repeat (ms)"),
                    self._spinbox(ctl.pedal_repeat_ms, 1, 1000,
                                  lambda v: setattr(ctl, "pedal_repeat_ms", v)))

        self.lbl_live_mode = QLabel("Pedal mode: OFF")
        self.lbl_live_mode.setStyleSheet("font-family: monospace;")
        form.addRow(self.lbl_live_mode)
        self.worker.status_tick.connect(self._on_status_mode)
        return box

    def _on_status_mode(self, st: dict) -> None:
        mode = st.get("pedal_mode")
        mode_names = {"a": "ZOOM", "b": "SIDE", "c": "UP/DOWN"}
        self.lbl_live_mode.setText(f"Pedal mode: {mode_names.get(mode, 'OFF')}")

    def _set_pedal_key(self, attr: str, text: str) -> None:
        if len(text) != 1:
            return
        setattr(self.worker.cfg.controls, attr, text.lower())

    # ------------------ Config file ------------------------------------

    def _make_config_group(self) -> QGroupBox:
        box = QGroupBox("Config file", self)
        row = QHBoxLayout(box)
        btn_load = QPushButton("Load…")
        btn_save = QPushButton("Save")
        btn_reset = QPushButton("Reset to defaults")
        btn_load.clicked.connect(self._load_config)
        btn_save.clicked.connect(self._save_config)
        btn_reset.clicked.connect(self._reset_config)
        row.addWidget(btn_load); row.addWidget(btn_save); row.addWidget(btn_reset); row.addStretch(1)
        return box

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load config", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        QMessageBox.information(self, "Restart required",
                                 f"Config loaded from {path}.\nRestart the app for all changes to take effect.")

    def _save_config(self) -> None:
        cfg = self.worker.cfg
        raw = self._cfg_to_dict(cfg)
        path = _default_config_path()
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(raw, fh, sort_keys=False)
        QMessageBox.information(self, "Saved", f"Wrote {path}")

    def _reset_config(self) -> None:
        if QMessageBox.question(self, "Reset config", "Overwrite config.yaml with defaults?") \
                != QMessageBox.StandardButton.Yes:
            return
        from ..config import PiccoloCfg
        defaults = PiccoloCfg()
        raw = self._cfg_to_dict(defaults)
        path = _default_config_path()
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(raw, fh, sort_keys=False)
        QMessageBox.information(self, "Reset", f"Wrote defaults to {path}.\nRestart to apply.")

    @staticmethod
    def _cfg_to_dict(cfg) -> dict:
        return {
            "display": asdict(cfg.display),
            "cameras": {
                "backend": cfg.cameras.backend,
                "left":  asdict(cfg.cameras.left),
                "right": asdict(cfg.cameras.right),
                "test_mode": cfg.cameras.test_mode,
            },
            "stereo": {
                "zoom": asdict(cfg.stereo.zoom),
                "convergence": asdict(cfg.stereo.convergence),
                "alignment": asdict(cfg.stereo.alignment),
                "aspect_mode": cfg.stereo.aspect_mode,
            },
            "calibration": asdict(cfg.calibration),
            "calibration_state": asdict(cfg.calibration_state),
            "stereo_calibration": asdict(cfg.stereo_calibration),
            "performance": asdict(cfg.performance),
            "controls": asdict(cfg.controls),
        }
