"""Performance regression for the equivalent CPU and GPU hot paths."""
from __future__ import annotations

import time

import numpy as np
import pytest

from tests._cuda_helpers import skip_if_no_cuda


def _fill_holes_cross_cpu(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror PipelineWorker._fill_holes_cross without constructing a QThread."""
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


@pytest.mark.perf
def test_gpu_pipeline_keeps_pace_with_cpu_hot_path_at_640x480():
    skip_if_no_cuda()
    from src.config import AlignmentCfg, CalibrationCfg, StereoCfg
    from src.calibration import CalibrationOverlay
    from src.gpu_pipeline import GpuPipeline
    from src.stereo_align import StereoAligner
    from src.stereo_processor import StereoProcessor

    eye_w, eye_h = 960, 1080
    frame_w, frame_h = 640, 480

    stereo_cfg = StereoCfg()
    stereo_cfg.zoom.min = 1.0
    stereo_cfg.convergence.base_offset = 0
    stereo_cfg.convergence.auto_adjust = False

    aligner = StereoAligner(AlignmentCfg(enabled=False), frame_w, frame_h)
    processor = StereoProcessor(stereo_cfg, eye_w, eye_h)
    calibration = CalibrationOverlay(CalibrationCfg())

    pipeline = GpuPipeline(aligner, processor, calibration)

    rng = np.random.default_rng(42)
    pairs = [
        (
            rng.integers(0, 256, size=(frame_h, frame_w, 3), dtype=np.uint8),
            rng.integers(0, 256, size=(frame_h, frame_w, 3), dtype=np.uint8),
        )
        for _ in range(60)
    ]
    cpu_pairs = [(fl.copy(), fr.copy()) for fl, fr in pairs]
    gpu_pairs = [(fl.copy(), fr.copy()) for fl, fr in pairs]

    for fl, fr in gpu_pairs[:5]:
        pipeline.process(fl, fr)
    for fl, fr in cpu_pairs[:5]:
        fl, fr = aligner.warp_pair(fl, fr)
        fl, fr = _fill_holes_cross_cpu(fl, fr)
        eye_l, eye_r, _ = processor.process_pair(fl, fr)
        calibration.apply_nudge(eye_l, eye_r)

    t0 = time.perf_counter()
    for fl, fr in cpu_pairs:
        fl, fr = aligner.warp_pair(fl, fr)
        fl, fr = _fill_holes_cross_cpu(fl, fr)
        eye_l, eye_r, _ = processor.process_pair(fl, fr)
        calibration.apply_nudge(eye_l, eye_r)
    cpu_ms = (time.perf_counter() - t0) / len(pairs) * 1000.0

    t0 = time.perf_counter()
    for fl, fr in gpu_pairs:
        pipeline.process(fl, fr)
    gpu_ms = (time.perf_counter() - t0) / len(pairs) * 1000.0

    print(f"\n[perf] CPU per-frame: {cpu_ms:.2f} ms")
    print(f"[perf] GPU per-frame: {gpu_ms:.2f} ms")
    assert gpu_ms <= cpu_ms + 1.0, (
        f"GPU pipeline ({gpu_ms:.2f} ms) should stay within 1 ms of the "
        f"equivalent CPU hot path ({cpu_ms:.2f} ms) at 640x480"
    )
