"""Translucent drawing canvas for freehand annotations."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QMouseEvent
from PyQt6.QtWidgets import QWidget


class AnnotationCanvas(QWidget):
    stroke_finished = pyqtSignal(list)  # list of (x, y) in widget coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._points: list = []
        self._strokes: list[list] = []

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._points = [event.position()]
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._points:
            self._points.append(event.position())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._points:
            self._strokes.append(self._points)
            self.stroke_finished.emit([(p.x(), p.y()) for p in self._points])
            self._points = []
            self.update()

    def paintEvent(self, _) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(255, 200, 0), 3))
        for stroke in self._strokes:
            for i in range(1, len(stroke)):
                painter.drawLine(stroke[i - 1], stroke[i])
        if self._points:
            for i in range(1, len(self._points)):
                painter.drawLine(self._points[i - 1], self._points[i])

    def clear(self) -> None:
        self._strokes.clear()
        self._points.clear()
        self.update()

    def undo(self) -> None:
        if self._strokes:
            self._strokes.pop()
            self.update()
