"""Latency watermark draws a readable timestamp on top-left."""
from __future__ import annotations

import numpy as np


def test_latency_watermark_draws_text_in_top_left():
    from src.ui.latency_watermark import draw_timestamp_watermark

    sbs = np.full((100, 200, 3), 200, dtype=np.uint8)
    before = sbs.copy()
    out = draw_timestamp_watermark(sbs, timestamp_ms=12345.678)
    assert out is sbs
    assert not np.array_equal(out[0:30, 0:120], before[0:30, 0:120])
    assert np.array_equal(out[50:, 150:], before[50:, 150:])


def test_latency_watermark_returns_same_shape_and_dtype():
    from src.ui.latency_watermark import draw_timestamp_watermark

    sbs = np.zeros((60, 120, 3), dtype=np.uint8)
    out = draw_timestamp_watermark(sbs, timestamp_ms=0.0)
    assert out.shape == sbs.shape
    assert out.dtype == sbs.dtype
