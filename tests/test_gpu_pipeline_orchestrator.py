"""End-to-end GpuPipeline matches CPU pipeline within tolerance."""
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


def _build_components(eye_w=100, eye_h=100, frame_w=80, frame_h=60):
    from src.config import AlignmentCfg, CalibrationCfg, StereoCfg
    from src.calibration import CalibrationOverlay
    from src.stereo_align import StereoAligner
    from src.stereo_processor import StereoProcessor

    stereo_cfg = StereoCfg()
    stereo_cfg.zoom.min = 1.0
    stereo_cfg.convergence.base_offset = 0
    stereo_cfg.convergence.auto_adjust = False

    aligner = StereoAligner(AlignmentCfg(enabled=False), frame_w, frame_h)
    processor = StereoProcessor(stereo_cfg, eye_w, eye_h)
    calibration = CalibrationOverlay(CalibrationCfg())
    calibration.nudge_left = 0
    calibration.nudge_right = 0
    return aligner, processor, calibration


def test_gpu_pipeline_end_to_end_matches_cpu_when_aligner_disabled():
    skip_if_no_cuda()
    from src.gpu_pipeline import GpuPipeline

    aligner, processor, calibration = _build_components()
    pipeline = GpuPipeline(aligner, processor, calibration)

    frame_l = _smooth_gradient(60, 80, seed=20)
    frame_r = _smooth_gradient(60, 80, seed=22)

    sbs_gpu = pipeline.process(frame_l, frame_r)

    # CPU reference path matching pipeline_worker's flow when aligner disabled.
    eye_l, eye_r, sbs_cpu = processor.process_pair(frame_l.copy(), frame_r.copy())
    eye_l, eye_r = calibration.apply_nudge(eye_l, eye_r)

    # Interior comparison only (border-zero pixels can differ).
    s = slice(10, -10)
    diff = np.abs(
        sbs_cpu[s, 10:processor.eye_w - 10].astype(int)
        - sbs_gpu[s, 10:processor.eye_w - 10].astype(int)
    ).max()
    assert diff <= 5, f"left half interior diff {diff}"
    diff_r = np.abs(
        sbs_cpu[s, processor.eye_w + 10:-10].astype(int)
        - sbs_gpu[s, processor.eye_w + 10:-10].astype(int)
    ).max()
    assert diff_r <= 5, f"right half interior diff {diff_r}"


def test_gpu_pipeline_releases_buffers_on_teardown():
    skip_if_no_cuda()
    from src.gpu_pipeline import GpuPipeline

    aligner, processor, calibration = _build_components()
    pipeline = GpuPipeline(aligner, processor, calibration)

    frame_l = _smooth_gradient(60, 80, seed=24)
    frame_r = _smooth_gradient(60, 80, seed=26)
    pipeline.process(frame_l, frame_r)
    pipeline.release()
    assert pipeline._gpu_in_l is None
    assert pipeline._gpu_sbs is None


def test_gpu_pipeline_handles_repeated_calls_without_reallocating():
    """Persistent buffers stay allocated across multiple frames."""
    skip_if_no_cuda()
    from src.gpu_pipeline import GpuPipeline

    aligner, processor, calibration = _build_components()
    pipeline = GpuPipeline(aligner, processor, calibration)

    frame_l = _smooth_gradient(60, 80, seed=28)
    frame_r = _smooth_gradient(60, 80, seed=30)
    pipeline.process(frame_l, frame_r)
    sbs_id_before = id(pipeline._gpu_sbs)
    pipeline.process(frame_l, frame_r)
    pipeline.process(frame_l, frame_r)
    sbs_id_after = id(pipeline._gpu_sbs)
    assert sbs_id_before == sbs_id_after, "buffers reallocated unnecessarily"


def test_pipeline_worker_constructs_gpu_pipeline_when_flag_set():
    skip_if_no_cuda()
    from src.config import PiccoloCfg
    from src.ui.pipeline_worker import PipelineWorker

    cfg = PiccoloCfg()
    cfg.cameras.test_mode = True
    cfg.performance.use_gpu_pipeline = True

    worker = PipelineWorker(cfg)
    assert worker._gpu_pipeline is not None, "GpuPipeline should be constructed"

    cfg2 = PiccoloCfg()
    cfg2.cameras.test_mode = True
    cfg2.performance.use_gpu_pipeline = False
    worker2 = PipelineWorker(cfg2)
    assert worker2._gpu_pipeline is None, "should be None when flag off"


def test_pipeline_worker_does_not_gate_processing_on_emit_interval():
    skip_if_no_cuda()
    import time

    from src.config import PiccoloCfg
    from src.ui.pipeline_worker import PipelineWorker

    cfg = PiccoloCfg()
    cfg.cameras.test_mode = True
    cfg.display.fps = 30
    worker = PipelineWorker(cfg)

    worker._last_emit_t = time.perf_counter()
    assert worker._can_process_now(time.perf_counter()) is True
