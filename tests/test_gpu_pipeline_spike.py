"""One-off spike: verify cv2.cuda_GpuMat ROI write semantics.

Phase 1 of the GPU pipeline relies on writing into ROI views of a single
SBS GpuMat. This test fails fast if that idiom is broken on this OpenCV
build, so we know to use a different SBS compose strategy.
"""
from __future__ import annotations

import cv2
import numpy as np

from tests._cuda_helpers import skip_if_no_cuda


def test_gpu_mat_upload_download_roundtrip():
    skip_if_no_cuda()
    src = np.full((10, 10, 3), 42, dtype=np.uint8)
    gpu = cv2.cuda_GpuMat()
    gpu.upload(src)
    out = gpu.download()
    assert out.shape == src.shape
    assert (out == 42).all()


def test_gpu_mat_roi_write_visible_in_parent():
    skip_if_no_cuda()
    parent = cv2.cuda_GpuMat(20, 20, cv2.CV_8UC3)
    parent.upload(np.zeros((20, 20, 3), dtype=np.uint8))

    # Try to obtain a 10x20 ROI covering the right half.
    roi = cv2.cuda_GpuMat(parent, (10, 0, 10, 20))  # (x, y, width, height)
    fill = np.full((20, 10, 3), 99, dtype=np.uint8)
    roi.upload(fill)

    out = parent.download()
    assert (out[:, :10] == 0).all(), "left half should be untouched"
    assert (out[:, 10:] == 99).all(), "right half should reflect ROI write"


def test_gpu_resize_into_preallocated_destination():
    skip_if_no_cuda()
    src = cv2.cuda_GpuMat()
    src.upload(np.full((40, 40, 3), 7, dtype=np.uint8))
    dst = cv2.cuda_GpuMat(20, 20, cv2.CV_8UC3)
    cv2.cuda.resize(src, (20, 20), dst=dst, interpolation=cv2.INTER_AREA)
    out = dst.download()
    assert out.shape == (20, 20, 3)
    assert (out == 7).all()
