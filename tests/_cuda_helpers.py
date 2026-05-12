"""Shared helpers for GPU-pipeline tests."""
from __future__ import annotations

import cv2
import pytest


def skip_if_no_cuda() -> None:
    """Skip the calling test if CUDA is not usable on this machine."""
    if not hasattr(cv2, "cuda"):
        pytest.skip("cv2.cuda module not available")
    if not hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
        pytest.skip("cv2.cuda.getCudaEnabledDeviceCount not available")
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() <= 0:
            pytest.skip("No CUDA-enabled devices")
    except cv2.error as exc:
        pytest.skip(f"CUDA query failed: {exc}")
