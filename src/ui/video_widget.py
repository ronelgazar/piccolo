"""QLabel-based widget that displays QImage frames with auto-scaling."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class VideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 225)
        self.setStyleSheet("background-color: black;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._last: QImage | None = None

    def set_frame(self, image: QImage) -> None:
        self._last = image
        self._rescale()

    def resizeEvent(self, event) -> None:
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self) -> None:
        if self._last is None:
            return
        scaled = self._last.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(scaled))
