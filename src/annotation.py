"""Annotation overlay for surgical marking and teaching.

Annotations are drawn by an assistant on the web UI and optionally
sent to the Goovis display so the surgeon sees them in stereo.

Each annotation is a dict with:
    type     – "freehand" | "line" | "arrow" | "circle" | "rect" | "text"
    color    – [B, G, R] (0-255)
    width    – line thickness in pixels
    points   – list of [x, y] (normalised 0-1 relative to single-eye frame)
    text     – (text type only) the string to draw

The web UI works in normalised coordinates so annotations remain
resolution-independent and apply correctly to both eyes of the SBS
buffer.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import cv2
import numpy as np


class AnnotationOverlay:
    """Thread-safe annotation store + renderer."""

    def __init__(self):
        self._annotations: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        # When True, annotations are rendered onto the Goovis display
        self.show_on_screen: bool = False

        # Cross-eye disparity correction (pixels in output eye frame).
        # When an annotation drawn on one eye is rendered on the *other*
        # eye, all x-coordinates are shifted by this many pixels.
        # Positive = shift rightward when going left→right eye.
        # Adjustable via the web UI slider.
        self.disparity_offset: int = 0

    # ------------------------------------------------------------------
    # Mutation (called from Flask thread)
    # ------------------------------------------------------------------

    def add(self, annotation: dict[str, Any]):
        with self._lock:
            self._annotations.append(annotation)

    def undo(self):
        with self._lock:
            if self._annotations:
                self._annotations.pop()

    def clear(self):
        with self._lock:
            self._annotations.clear()

    def get_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._annotations)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._annotations)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, eye_l: np.ndarray, eye_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Draw all annotations onto both eye frames (mutates in place).

        Only called when ``show_on_screen`` is True.
        Coordinates are normalised 0-1 → scaled to the actual eye size.
        When an annotation was drawn on one eye, the cross-eye copy is
        shifted horizontally by ``disparity_offset`` pixels so that
        markings align with the same feature in both views.
        """
        if not self.show_on_screen:
            return eye_l, eye_r

        with self._lock:
            annotations = list(self._annotations)

        if not annotations:
            return eye_l, eye_r

        disp = self.disparity_offset

        for ann in annotations:
            source = ann.get("source_eye", "left")
            if source == "left":
                # Drawn on left → render as-is on left, shifted on right
                self._draw_one(eye_l, ann, x_offset_px=0)
                self._draw_one(eye_r, ann, x_offset_px=-disp)
            else:
                # Drawn on right → render as-is on right, shifted on left
                self._draw_one(eye_r, ann, x_offset_px=0)
                self._draw_one(eye_l, ann, x_offset_px=disp)

        return eye_l, eye_r

    # ------------------------------------------------------------------
    # Per-annotation drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_one(img: np.ndarray, ann: dict, x_offset_px: int = 0):
        h, w = img.shape[:2]
        color = tuple(ann.get("color", [0, 255, 0]))
        thick = ann.get("width", 2)
        atype = ann.get("type", "freehand")
        pts = ann.get("points", [])

        def px(pt):
            """Normalised [0-1] → pixel coords, with optional x offset."""
            return int(pt[0] * w) + x_offset_px, int(pt[1] * h)

        if atype == "freehand" and len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(img, px(pts[i]), px(pts[i + 1]), color, thick, cv2.LINE_AA)

        elif atype == "line" and len(pts) >= 2:
            cv2.line(img, px(pts[0]), px(pts[-1]), color, thick, cv2.LINE_AA)

        elif atype == "arrow" and len(pts) >= 2:
            cv2.arrowedLine(img, px(pts[0]), px(pts[-1]), color, thick,
                            cv2.LINE_AA, tipLength=0.03)

        elif atype == "circle" and len(pts) >= 2:
            cx, cy = px(pts[0])
            ex, ey = px(pts[-1])
            radius = int(math.hypot(ex - cx, ey - cy))
            cv2.circle(img, (cx, cy), radius, color, thick, cv2.LINE_AA)

        elif atype == "rect" and len(pts) >= 2:
            cv2.rectangle(img, px(pts[0]), px(pts[-1]), color, thick, cv2.LINE_AA)

        elif atype == "text" and len(pts) >= 1:
            text = ann.get("text", "")
            if text:
                pos = px(pts[0])
                font_scale = ann.get("font_scale", 0.8)
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, color, thick, cv2.LINE_AA)
