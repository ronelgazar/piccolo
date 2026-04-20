"""Settings tab (stub — populated in Tasks 9 and 10)."""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout


class SettingsTab(QWidget):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Settings tab — coming soon"))
