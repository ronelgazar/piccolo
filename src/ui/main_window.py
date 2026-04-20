"""Main Qt window: tabbed assistant UI."""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow, QTabWidget

from ..config import PiccoloCfg
from .live_tab import LiveTab
from .calibration_tab import CalibrationTab
from .settings_tab import SettingsTab


class MainWindow(QMainWindow):
    def __init__(self, cfg: PiccoloCfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Piccolo")
        self.resize(1280, 800)

        # Worker is None until Task 4; tab stubs tolerate None.
        self.worker = None

        tabs = QTabWidget(self)
        self.live_tab = LiveTab(self.worker, self)
        self.calibration_tab = CalibrationTab(self.worker, self)
        self.settings_tab = SettingsTab(self.worker, self)
        tabs.addTab(self.live_tab, "Live")
        tabs.addTab(self.calibration_tab, "Calibration")
        tabs.addTab(self.settings_tab, "Settings")
        self.setCentralWidget(tabs)
