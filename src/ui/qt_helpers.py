"""Qt helpers — BGR ndarray → QImage conversion."""
from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QImage


def ndarray_to_qimage(bgr: np.ndarray) -> QImage:
    """Convert a BGR HxWx3 uint8 ndarray to a QImage.

    The returned QImage owns a copy of the pixel data; the source
    ndarray can be freed immediately.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError(f"Expected HxWx3 uint8 BGR, got {bgr.shape} {bgr.dtype}")
    h, w, _ = bgr.shape
    src = np.ascontiguousarray(bgr)
    bgr_format = getattr(QImage.Format, "Format_BGR888", None)
    if bgr_format is not None:
        # Zero-copy QImage view for lower latency. Keep ndarray alive by
        # attaching it to the QImage instance.
        qimg = QImage(src.data, w, h, w * 3, bgr_format)
        qimg._ndarray_ref = src  # type: ignore[attr-defined]
        return qimg

    import cv2
    rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    qimg._ndarray_ref = rgb  # type: ignore[attr-defined]
    return qimg
