"""GPU fill-holes-cross matches CPU output exactly."""
from __future__ import annotations

import numpy as np

from tests._cuda_helpers import skip_if_no_cuda


def _cpu_fill_holes(left: np.ndarray, right: np.ndarray):
    # Mirror of pipeline_worker._fill_holes_cross
    if left.shape != right.shape:
        return left, right
    mask_l = np.all(left == 0, axis=2)
    mask_r = np.all(right == 0, axis=2)
    copy_to_l = mask_l & (~mask_r)
    if np.any(copy_to_l):
        left[copy_to_l] = right[copy_to_l]
    copy_to_r = mask_r & (~mask_l)
    if np.any(copy_to_r):
        right[copy_to_r] = left[copy_to_r]
    return left, right


def _make_pair_with_borders():
    rng = np.random.default_rng(0)
    left = rng.integers(1, 256, size=(60, 80, 3), dtype=np.uint8)
    right = rng.integers(1, 256, size=(60, 80, 3), dtype=np.uint8)
    # Carve a 10-px black left border in `left` and 10-px black right border in `right`.
    left[:, :10] = 0
    right[:, -10:] = 0
    return left, right


def test_fill_holes_gpu_matches_cpu():
    skip_if_no_cuda()
    from src.gpu_pipeline import fill_holes_cross_gpu

    left, right = _make_pair_with_borders()
    cpu_left, cpu_right = _cpu_fill_holes(left.copy(), right.copy())

    import cv2
    gpu_l = cv2.cuda_GpuMat()
    gpu_r = cv2.cuda_GpuMat()
    gpu_l.upload(left)
    gpu_r.upload(right)
    fill_holes_cross_gpu(gpu_l, gpu_r)

    out_l = gpu_l.download()
    out_r = gpu_r.download()

    assert np.array_equal(out_l, cpu_left), "left mismatch"
    assert np.array_equal(out_r, cpu_right), "right mismatch"


def test_fill_holes_gpu_no_borders_is_noop():
    skip_if_no_cuda()
    import cv2
    from src.gpu_pipeline import fill_holes_cross_gpu

    rng = np.random.default_rng(1)
    left = rng.integers(1, 256, size=(40, 50, 3), dtype=np.uint8)
    right = rng.integers(1, 256, size=(40, 50, 3), dtype=np.uint8)
    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(left)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(right)
    fill_holes_cross_gpu(gpu_l, gpu_r)
    assert np.array_equal(gpu_l.download(), left)
    assert np.array_equal(gpu_r.download(), right)


def test_fill_holes_gpu_split_returns_gpu_mat_channels():
    """Guard: cv2.cuda.split must return GpuMat channels, not host ndarrays.

    If split returned ndarrays on this build (as cv2.cuda.merge does),
    fill_holes_cross_gpu would silently break because compareWithScalar
    would receive host arrays. This test fails fast if that ever changes.
    """
    skip_if_no_cuda()
    import cv2
    rng = np.random.default_rng(11)
    frame = rng.integers(1, 256, size=(20, 30, 3), dtype=np.uint8)
    gpu = cv2.cuda_GpuMat()
    gpu.upload(frame)
    channels = cv2.cuda.split(gpu)
    assert len(channels) == 3, f"expected 3 channels, got {len(channels)}"
    for i, ch in enumerate(channels):
        assert isinstance(ch, cv2.cuda_GpuMat), (
            f"channel {i} is {type(ch).__name__}, expected cv2.cuda_GpuMat"
        )


def test_fill_holes_gpu_overlapping_zeros_stay_zero():
    """Where both eyes are zero at the same pixel, neither copy fires.

    The CPU reference leaves such pixels unchanged. Verify GPU does the same.
    """
    skip_if_no_cuda()
    import cv2
    from src.gpu_pipeline import fill_holes_cross_gpu

    rng = np.random.default_rng(12)
    left = rng.integers(1, 256, size=(40, 50, 3), dtype=np.uint8)
    right = rng.integers(1, 256, size=(40, 50, 3), dtype=np.uint8)
    # Carve overlapping zero columns at x=10..14 in both eyes.
    left[:, 10:15] = 0
    right[:, 10:15] = 0

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(left)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(right)
    fill_holes_cross_gpu(gpu_l, gpu_r)

    out_l = gpu_l.download()
    out_r = gpu_r.download()

    assert (out_l[:, 10:15] == 0).all(), "overlapping zero pixels in L must stay zero"
    assert (out_r[:, 10:15] == 0).all(), "overlapping zero pixels in R must stay zero"


def test_fill_holes_gpu_mismatched_size_is_noop():
    """Calling with mismatched sizes must be a documented no-op.

    The function's first line guards against this; the guard is part of
    the contract that Task 7's GpuPipeline orchestrator depends on.
    """
    skip_if_no_cuda()
    import cv2
    from src.gpu_pipeline import fill_holes_cross_gpu

    rng = np.random.default_rng(13)
    left = rng.integers(1, 256, size=(40, 50, 3), dtype=np.uint8)
    right = rng.integers(1, 256, size=(30, 40, 3), dtype=np.uint8)  # different size

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(left)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(right)
    fill_holes_cross_gpu(gpu_l, gpu_r)

    # Neither buffer should have been touched.
    assert np.array_equal(gpu_l.download(), left)
    assert np.array_equal(gpu_r.download(), right)
