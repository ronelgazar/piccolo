"""Borderless full-screen stereo output for the Goovis headset."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication, QScreen
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from ..config import DisplayCfg
from .video_widget import VideoWidget


class GoovisWindow(QWidget):
    def __init__(self, cfg: DisplayCfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.video = VideoWidget(self)
        lay.addWidget(self.video)

    def show_on_goovis(self) -> bool:
        """Position on the detected Goovis monitor and show full-screen.

        Returns True if a matching screen was found and the window shown.
        """
        screen = _find_goovis_screen(self.cfg)
        if screen is None:
            return False
        geom = screen.geometry()
        self.move(geom.x(), geom.y())
        self.resize(geom.width(), geom.height())
        self.showFullScreen()
        return True


def _find_goovis_screen(cfg: DisplayCfg) -> QScreen | None:
    screens = QGuiApplication.screens()
    if isinstance(cfg.monitor, int) and 0 <= cfg.monitor < len(screens):
        return screens[cfg.monitor]
    for s in screens:
        name = s.name().upper()
        if "GOOVIS" in name or "NED" in name:
            return s
    primary = QGuiApplication.primaryScreen()
    for s in screens:
        if s is not primary and s.geometry().width() == 1920 and s.geometry().height() == 1080:
            return s
    return None
