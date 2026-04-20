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
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.video = VideoWidget(self)
        lay.addWidget(self.video)

    def show_on_goovis(self) -> bool:
        """Position on the detected Goovis monitor and show full-screen.

        Returns True if a matching screen was found and the window shown.
        """
        _log_screens()
        screen = _find_goovis_screen(self.cfg)
        if screen is None:
            print("[goovis] No secondary monitor detected — headset output disabled.")
            return False
        geom = screen.geometry()
        print(f"[goovis] Using monitor: {screen.name()!r} {geom.width()}x{geom.height()} @ ({geom.x()},{geom.y()})")
        self.setGeometry(geom)
        self.show()
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        return True


def _log_screens() -> None:
    screens = QGuiApplication.screens()
    primary = QGuiApplication.primaryScreen()
    print(f"[goovis] Detected {len(screens)} monitor(s):")
    for i, s in enumerate(screens):
        g = s.geometry()
        is_primary = " (primary)" if s is primary else ""
        print(f"  [{i}] {s.name()!r}  {g.width()}x{g.height()} @ ({g.x()},{g.y()}){is_primary}")


def _find_goovis_screen(cfg: DisplayCfg) -> QScreen | None:
    screens = QGuiApplication.screens()
    # Explicit integer monitor index wins
    if isinstance(cfg.monitor, int) and 0 <= cfg.monitor < len(screens):
        return screens[cfg.monitor]
    # Name-based auto-detect (preferred)
    for s in screens:
        name = s.name().upper()
        if "GOOVIS" in name or "NED" in name:
            return s
    # Fallback: any non-primary display
    primary = QGuiApplication.primaryScreen()
    for s in screens:
        if s is not primary:
            return s
    return None
