"""Settings tab: cameras, display, stereo, pedals, config file groups."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QLabel,
)

from ..config_state import save_calibration_state


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
        form.addRow(QLabel("Zoom step"),
                    self._doublebox(s.zoom.step, 0.001, 1.0, 0.01,
                                    lambda v: setattr(s.zoom, "step", v)))
        form.addRow(QLabel("Zoom tick (ms)"),
                    self._spinbox(s.zoom.tick_ms, 1, 1000,
                                  lambda v: setattr(s.zoom, "tick_ms", v)))
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
