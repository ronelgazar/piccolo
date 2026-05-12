"""QOpenGLWidget that renders an SBS BGR frame as a textured quad."""
from __future__ import annotations

import numpy as np
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

try:
    from OpenGL import GL
    _GL_AVAILABLE = True
except ImportError:
    GL = None
    _GL_AVAILABLE = False


class GLDisplayWidget(QOpenGLWidget):
    """Minimal GL display widget with synchronous texture upload."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self._texture: int = 0
        self._tex_w: int = 0
        self._tex_h: int = 0
        if not _GL_AVAILABLE:
            print("[GLDisplayWidget] PyOpenGL not installed; widget will be inert")

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """Set the next BGR frame to render."""
        self._frame = frame_bgr
        self.update()

    def initializeGL(self) -> None:
        if not _GL_AVAILABLE:
            return
        self._texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)

    def resizeGL(self, w: int, h: int) -> None:
        if not _GL_AVAILABLE:
            return
        GL.glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        if not _GL_AVAILABLE or self._frame is None:
            return

        h, w = self._frame.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        if (w, h) != (self._tex_w, self._tex_h):
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGB,
                w,
                h,
                0,
                GL.GL_BGR,
                GL.GL_UNSIGNED_BYTE,
                self._frame,
            )
            self._tex_w, self._tex_h = w, h
        else:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D,
                0,
                0,
                0,
                w,
                h,
                GL.GL_BGR,
                GL.GL_UNSIGNED_BYTE,
                self._frame,
            )

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBegin(GL.GL_QUADS)
        for u, v, x, y in (
            (0.0, 1.0, -1.0, -1.0),
            (1.0, 1.0, 1.0, -1.0),
            (1.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, -1.0, 1.0),
        ):
            GL.glTexCoord2f(u, v)
            GL.glVertex2f(x, y)
        GL.glEnd()
        GL.glDisable(GL.GL_TEXTURE_2D)
