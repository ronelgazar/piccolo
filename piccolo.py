"""Piccolo — stereoscopic surgery display (Qt desktop app)."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from src.config import load_config
from src.ui.main_window import MainWindow


def main() -> int:
    cfg = load_config()
    app = QApplication(sys.argv)
    window = MainWindow(cfg)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
