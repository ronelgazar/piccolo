"""Glass-to-glass latency watermark.

Renders the current monotonic clock (ms) into a high-contrast block in the
top-left of the SBS frame. Pair with an external high-speed camera filming
both a system clock and the headset to estimate glass-to-glass latency:

    latency_ms = camera_clock_at_shutter_ms - watermark_value_ms
"""
from __future__ import annotations

import cv2
import numpy as np


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.7
_THICKNESS = 2
_BG_PAD = 4


def draw_timestamp_watermark(sbs: np.ndarray, timestamp_ms: float) -> np.ndarray:
    """Draw a timestamp watermark in place and return ``sbs``."""
    text = f"{timestamp_ms:.1f}"
    (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS)
    x, y = 6, 6 + th
    cv2.rectangle(
        sbs,
        (x - _BG_PAD, y - th - _BG_PAD),
        (x + tw + _BG_PAD, y + baseline + _BG_PAD),
        (255, 255, 255),
        thickness=-1,
    )
    cv2.putText(
        sbs,
        text,
        (x, y),
        _FONT,
        _FONT_SCALE,
        (0, 0, 0),
        _THICKNESS,
        cv2.LINE_AA,
    )
    return sbs
