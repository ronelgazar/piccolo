"""Live tab (stub — populated in Task 5)."""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout


class LiveTab(QWidget):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Live tab — preview coming soon"))
