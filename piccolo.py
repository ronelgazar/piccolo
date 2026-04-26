"""Piccolo — stereoscopic surgery display (Qt desktop app)."""
from __future__ import annotations

import sys
import os

import cv2
from PyQt6.QtWidgets import QApplication

from src.config import load_config
from src.ui.main_window import MainWindow


def main() -> int:
    # Keep OpenCV from oversubscribing the whole machine. The app already has
    # camera and pipeline threads, so a small OpenCV pool is smoother under load.
    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, min(2, (os.cpu_count() or 2) // 2)))

    cfg = load_config()
    app = QApplication(sys.argv)
    window = MainWindow(cfg)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
