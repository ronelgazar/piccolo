"""QLabel-based widget that displays QImage frames with auto-scaling."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy
from .qt_helpers import ndarray_to_qimage


class VideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 225)
        self.setStyleSheet("background-color: black;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._last: QImage | None = None

    def set_sbs_frame(self, sbs_frame: np.ndarray) -> None:
        """Set the video frame from a side-by-side BGR ndarray."""
        self.set_frame(ndarray_to_qimage(sbs_frame))

    def set_frame(self, image: QImage) -> None:
        self._last = image
        if self.isVisible():
            self._rescale()

    def resizeEvent(self, event) -> None:
        self._rescale()
        super().resizeEvent(event)

    def showEvent(self, event) -> None:
        self._rescale()
        super().showEvent(event)

    def _rescale(self) -> None:
        if self._last is None or not self.isVisible():
            return
        if self._last.size() == self.size():
            self.setPixmap(QPixmap.fromImage(self._last))
            return
        scaled = self._last.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.setPixmap(QPixmap.fromImage(scaled))
