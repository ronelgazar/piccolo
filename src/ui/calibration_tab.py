"""Calibration tab (stub — populated in Task 7)."""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout


class CalibrationTab(QWidget):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Calibration tab — coming soon"))
