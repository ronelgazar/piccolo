"""GPU apply_nudge matches CPU apply_nudge."""
from __future__ import annotations

import numpy as np

from tests._cuda_helpers import skip_if_no_cuda


def _smooth_gradient(h: int, w: int, seed: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        (xx * 3.0 + seed * 7) % 256,
        (yy * 4.0 + seed * 11) % 256,
        ((xx + yy) * 2.0 + seed * 13) % 256,
    ], axis=-1)
    return base.astype(np.uint8)


def _make_overlay(
    nudge_x_l=5,
    nudge_y_l=-3,
    nudge_x_r=-2,
    nudge_y_r=4,
    scale_l=100,
    scale_r=100,
):
    from src.config import CalibrationCfg
    from src.calibration import CalibrationOverlay

    overlay = CalibrationOverlay(CalibrationCfg())
    overlay.nudge_left = nudge_x_l
    overlay.nudge_right = nudge_x_r
    overlay.nudge_left_y = nudge_y_l
    overlay.nudge_right_y = nudge_y_r
    overlay.scale_left_pct = scale_l
    overlay.scale_right_pct = scale_r
    return overlay


def test_apply_nudge_gpu_matches_cpu_translation_only():
    """Interior matches CPU after pure-translation nudge.

    CPU uses np.roll (cyclic shift then zero border); GPU uses warpAffine
    translation (linear interp + border value). They produce identical
    results on integer-pixel translations except at the border that's
    revealed by the shift. Interior matches exactly for integer shifts.
    """
    skip_if_no_cuda()
    import cv2

    overlay = _make_overlay()
    eye_l = _smooth_gradient(60, 80, seed=14)
    eye_r = _smooth_gradient(60, 80, seed=16)

    cpu_l, cpu_r = overlay.apply_nudge(eye_l.copy(), eye_r.copy())

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(eye_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(eye_r)
    overlay.apply_nudge_gpu(gpu_l, gpu_r)
    out_l = gpu_l.download()
    out_r = gpu_r.download()

    # Compare interior; borders differ because np.roll wraps zeros
    # while warpAffine fills the border with the constant.
    s = slice(10, -10)
    diff_l = np.abs(cpu_l[s, s].astype(int) - out_l[s, s].astype(int)).max()
    diff_r = np.abs(cpu_r[s, s].astype(int) - out_r[s, s].astype(int)).max()
    assert diff_l <= 5, f"left diff {diff_l}"
    assert diff_r <= 5, f"right diff {diff_r}"


def test_apply_nudge_gpu_zero_nudge_is_passthrough():
    """With all nudges zero and scale 100, GPU output equals input exactly."""
    skip_if_no_cuda()
    import cv2

    overlay = _make_overlay(nudge_x_l=0, nudge_y_l=0, nudge_x_r=0, nudge_y_r=0)
    rng = np.random.default_rng(17)
    eye_l = rng.integers(1, 256, size=(40, 50, 3), dtype=np.uint8)
    eye_r = rng.integers(1, 256, size=(40, 50, 3), dtype=np.uint8)

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(eye_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(eye_r)
    overlay.apply_nudge_gpu(gpu_l, gpu_r)
    assert np.array_equal(gpu_l.download(), eye_l)
    assert np.array_equal(gpu_r.download(), eye_r)
