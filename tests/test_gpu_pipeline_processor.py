"""GPU process_pair matches CPU process_pair output."""
from __future__ import annotations

import numpy as np

from tests._cuda_helpers import skip_if_no_cuda


def _make_processor(eye_w: int = 100, eye_h: int = 100):
    from src.config import StereoCfg
    from src.stereo_processor import StereoProcessor

    cfg = StereoCfg()
    cfg.zoom.min = 1.0
    cfg.zoom.max = 5.0
    cfg.zoom.step = 0.1
    cfg.convergence.base_offset = 0
    cfg.convergence.auto_adjust = False
    return StereoProcessor(cfg, eye_width=eye_w, eye_height=eye_h)


def _smooth_gradient(h: int, w: int, seed: int) -> np.ndarray:
    """Camera-like smooth gradient (no per-pixel noise).

    Same rationale as test_gpu_pipeline_align: random noise amplifies sub-pixel
    bilinear-resampler diffs between CPU and GPU into large per-value diffs.
    Real camera frames are spatially correlated, so the per-pixel diff is small
    even when the samplers disagree at the sub-pixel level.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        (xx * 3.0 + seed * 7) % 256,
        (yy * 4.0 + seed * 11) % 256,
        ((xx + yy) * 2.0 + seed * 13) % 256,
    ], axis=-1)
    return base.astype(np.uint8)


def test_process_pair_gpu_matches_cpu_zoom_1x():
    skip_if_no_cuda()
    import cv2

    p_cpu = _make_processor()
    p_gpu = _make_processor()

    frame_l = _smooth_gradient(80, 80, seed=4)
    frame_r = _smooth_gradient(80, 80, seed=6)

    _, _, sbs_cpu = p_cpu.process_pair(frame_l, frame_r)

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(frame_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(frame_r)
    gpu_sbs = cv2.cuda_GpuMat(p_gpu.eye_h, p_gpu.eye_w * 2, cv2.CV_8UC3)
    p_gpu.process_pair_gpu(gpu_l, gpu_r, gpu_sbs)
    sbs_gpu = gpu_sbs.download()

    # Compare interior (10 px from each edge) — same border-effect mitigation
    # as the align tests. Tolerance covers bilinear resampler differences.
    s = slice(10, -10)
    # SBS is two side-by-side halves; check each independently.
    left_cpu  = sbs_cpu[s, 10:p_cpu.eye_w-10]
    left_gpu  = sbs_gpu[s, 10:p_gpu.eye_w-10]
    right_cpu = sbs_cpu[s, p_cpu.eye_w+10:-10]
    right_gpu = sbs_gpu[s, p_gpu.eye_w+10:-10]
    diff_l = np.abs(left_cpu.astype(int) - left_gpu.astype(int)).max()
    diff_r = np.abs(right_cpu.astype(int) - right_gpu.astype(int)).max()
    assert diff_l <= 5, f"left interior diff was {diff_l}"
    assert diff_r <= 5, f"right interior diff was {diff_r}"


def test_process_pair_gpu_matches_cpu_zoom_2x():
    skip_if_no_cuda()
    import cv2

    p_cpu = _make_processor()
    p_gpu = _make_processor()
    p_cpu.zoom = 2.0
    p_gpu.zoom = 2.0

    frame_l = _smooth_gradient(80, 80, seed=8)
    frame_r = _smooth_gradient(80, 80, seed=10)

    _, _, sbs_cpu = p_cpu.process_pair(frame_l, frame_r)

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(frame_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(frame_r)
    gpu_sbs = cv2.cuda_GpuMat(p_gpu.eye_h, p_gpu.eye_w * 2, cv2.CV_8UC3)
    p_gpu.process_pair_gpu(gpu_l, gpu_r, gpu_sbs)
    sbs_gpu = gpu_sbs.download()

    s = slice(10, -10)
    left_cpu  = sbs_cpu[s, 10:p_cpu.eye_w-10]
    left_gpu  = sbs_gpu[s, 10:p_gpu.eye_w-10]
    right_cpu = sbs_cpu[s, p_cpu.eye_w+10:-10]
    right_gpu = sbs_gpu[s, p_gpu.eye_w+10:-10]
    diff_l = np.abs(left_cpu.astype(int) - left_gpu.astype(int)).max()
    diff_r = np.abs(right_cpu.astype(int) - right_gpu.astype(int)).max()
    assert diff_l <= 5, f"left interior diff at 2x zoom was {diff_l}"
    assert diff_r <= 5, f"right interior diff at 2x zoom was {diff_r}"


def test_process_pair_gpu_writes_to_provided_sbs():
    """The gpu_sbs argument must be the destination written to."""
    skip_if_no_cuda()
    import cv2

    p = _make_processor()
    frame_l = _smooth_gradient(80, 80, seed=11)
    frame_r = _smooth_gradient(80, 80, seed=13)

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(frame_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(frame_r)
    gpu_sbs = cv2.cuda_GpuMat()
    gpu_sbs.upload(np.full((p.eye_h, p.eye_w * 2, 3), 99, dtype=np.uint8))

    p.process_pair_gpu(gpu_l, gpu_r, gpu_sbs)
    out = gpu_sbs.download()

    # The sentinel 99 should be gone at least in the interior of each eye.
    assert (out[10:-10, 10:p.eye_w-10] != 99).any()
    assert (out[10:-10, p.eye_w+10:-10] != 99).any()
