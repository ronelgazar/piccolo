"""Backend pipeline running in a QThread.

Reads frames from both cameras, applies flip_180, runs the StereoAligner,
processes stereo (zoom + convergence), applies calibration nudges and
overlays, and emits a QImage signal for the latest SBS frame.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QObject

from ..config import PiccoloCfg
from ..camera import CameraCapture, TestPatternCamera
from ..stereo_processor import StereoProcessor
from ..stereo_align import StereoAligner
from ..calibration import CalibrationOverlay
import math
from ..input_handler import InputHandler, Action
from ..smart_overlap import OverlapMetrics, render_overlay
from .qt_helpers import ndarray_to_qimage


class PipelineWorker(QThread):
    """Long-running backend thread."""

    sbs_frame_ready = pyqtSignal(object)        # np.ndarray — processed SBS (for overlay/wizard)
    sbs_ndarray_ready = pyqtSignal(object)      # np.ndarray — processed SBS (for GL display)
    sbs_qimage_ready = pyqtSignal(object)       # QImage — processed SBS (for display)
    status_tick = pyqtSignal(dict)              # FPS, alignment, pedal mode
    recording_frame_ready = pyqtSignal(object)  # dict with BGR ndarray copies
    error = pyqtSignal(str)
    request_restart = pyqtSignal()

    def __init__(self, cfg: PiccoloCfg, parent: QObject | None = None):
        super().__init__(parent)
        self.cfg = cfg
        self._running = False
        eye_w = cfg.display.width // 2
        eye_h = cfg.display.height
        self.processor = StereoProcessor(cfg.stereo, eye_w, eye_h)
        self.aligner = StereoAligner(
            cfg.stereo.alignment,
            cfg.cameras.left.width,
            cfg.cameras.left.height,
        )
        # Apply persisted alignment state if present
        try:
            st = cfg.calibration_state
            if hasattr(st, 'alignment_dy'):
                self.aligner._smooth_dy = float(st.alignment_dy)
            if hasattr(st, 'alignment_dtheta_deg'):
                self.aligner._smooth_dtheta = math.radians(float(st.alignment_dtheta_deg))
            try:
                self.aligner._build_warp_matrices()
            except Exception:
                # ensure next update rebuilds
                self.aligner.force_update()
        except Exception:
            pass
        self.calibration = CalibrationOverlay(cfg.calibration)
        self.input = InputHandler(self.aligner, cfg.controls)

        # Apply persisted calibration state
        st = cfg.calibration_state
        self.calibration.nudge_left = st.nudge_left_x
        self.calibration.nudge_right = st.nudge_right_x
        self.calibration.nudge_left_y = st.nudge_left_y
        self.calibration.nudge_right_y = st.nudge_right_y
        self.calibration.scale_left_pct = st.scale_left_pct
        self.calibration.scale_right_pct = st.scale_right_pct
        self.processor.base_offset = st.convergence_offset
        if hasattr(self.processor, "set_joint_zoom_center"):
            self.processor.set_joint_zoom_center(st.joint_zoom_center)
        if hasattr(self.processor, "set_joint_zoom_center_y"):
            self.processor.set_joint_zoom_center_y(st.joint_zoom_center_y)
        self.processor.reset_zoom()

        self.cam_l = None
        self.cam_r = None
        self._fps_hist: list[float] = []
        self._last_frame_ids: tuple[int, int] = (-1, -1)
        self._target_interval: float = 1.0 / max(1, int(cfg.display.fps))
        # Throttle GUI-facing emits to save CPU. GUI conversion and scaling
        # are expensive at full-HD, so respect the configured display FPS.
        self._last_emit_t: float = 0.0
        self._emit_interval: float = self._target_interval
        self.raw_frame_requested: bool = False
        self.raw_frame_interval: float = 1.0
        self.raw_frame_mode: str = "full_fov"
        self._last_raw_emit_t: float = 0.0
        self.recording_enabled: bool = False
        self.recording_mode: str = "dual"
        self.recording_single_eye: str = "left"
        self.recording_fps: int = 30
        self._last_record_emit_t: float = 0.0
        self._last_status_emit_t: float = 0.0
        self._status_interval: float = 0.2
        self._smart_overlap_active: bool = False
        self._smart_overlap_metrics: OverlapMetrics | None = None
        # Depth estimator (fast block matcher) for low-latency calibrated depth
        self._depth_matcher = None
        self._depth_matcher_params: tuple[int, int] | None = None
        self._last_depth_mm: float | None = None
        self._use_gpu_depth = getattr(cfg, "performance", None) and cfg.performance.use_gpu_for_depth
        self._low_latency_mode = getattr(cfg, "performance", None) and cfg.performance.low_latency_mode
        self._frame_index = 0
        self._perf_history = defaultdict(lambda: deque(maxlen=45))
        # GPU support flags (detect at init, cache the result)
        self._cuda_available = self._detect_cuda()
        self._cuda_caps = self._query_cuda_capabilities() if self._cuda_available else {}
        self._use_gpu_pipeline = bool(
            self._cuda_available
            and getattr(cfg, "performance", None)
            and getattr(cfg.performance, "use_gpu_pipeline", False)
        )
        if self._use_gpu_pipeline:
            from ..gpu_pipeline import GpuPipeline

            self._gpu_pipeline = GpuPipeline(
                self.aligner, self.processor, self.calibration
            )
        else:
            self._gpu_pipeline = None
        self._gpu_warp_upload_time = 0.0
        gpu_status = "GPU-ACCELERATED" if self._cuda_available else "CPU-ONLY"
        print(f"[Pipeline] Initialized in {gpu_status} mode")
        if self._cuda_available:
            print(f"[CUDA] capabilities={self._cuda_caps}")

    # ------------------------------------------------------------------

    def set_smart_overlap_overlay_active(self, active: bool) -> None:
        self._smart_overlap_active = bool(active)
        if not active:
            self._smart_overlap_metrics = None

    def set_smart_overlap_metrics(self, metrics: OverlapMetrics | None) -> None:
        self._smart_overlap_metrics = metrics

    def run(self) -> None:
        try:
            self._open_cameras()
            self._running = True
            while self._running:
                t0 = time.perf_counter()
                self._tick()
                dt = time.perf_counter() - t0
                remaining = self._target_interval - dt
                if remaining > 0:
                    self.msleep(max(1, int(remaining * 1000)))
                total_dt = time.perf_counter() - t0
                self._fps_hist.append(total_dt)
                if len(self._fps_hist) > 60:
                    self._fps_hist.pop(0)
        except Exception as exc:
            self.error.emit(f"Pipeline error: {exc}")
        finally:
            self._cleanup()

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    # ------------------------------------------------------------------

    def _open_cameras(self) -> None:
        c = self.cfg.cameras
        if c.test_mode:
            self.cam_l = TestPatternCamera(c.left.width, c.left.height, side="left", name="test-L").start()
            self.cam_r = TestPatternCamera(c.right.width, c.right.height, side="right", name="test-R").start()
            return
        self.cam_l = CameraCapture(
            c.left.index, c.left.width, c.left.height,
            fps=c.left.fps, backend=c.backend, name="cam-L",
            decode_backend=c.left.decode_backend,
        ).start()
        try:
            self.cam_r = CameraCapture(
                c.right.index, c.right.width, c.right.height,
                fps=c.right.fps, backend=c.backend, name="cam-R",
                decode_backend=c.right.decode_backend,
            ).start()
        except Exception:
            if self.cam_l is not None:
                self.cam_l.stop()
            raise

    def _tick(self) -> None:
        self._frame_index += 1
        for act in self.input.poll_actions():
            self._apply_action(act)
        if not self._running:
            return
        now = time.perf_counter()
        if not self._can_process_now(now):
            return

        t0 = time.perf_counter()
        perf: dict[str, float] = {}
        frame_l, frame_r, frame_ids = self._read_frame_pair(perf)
        if frame_l is None or frame_r is None:
            self.msleep(5)
            return
        if frame_ids == self._last_frame_ids:
            self.msleep(1)
            return
        self._last_frame_ids = frame_ids

        frame_l, frame_r = self._apply_camera_flips(frame_l, frame_r)
        self._maybe_emit_recording_frame(frame_l, frame_r, now)
        low_latency = self._low_latency_enabled()
        if self._gpu_pipeline is not None and not low_latency:
            t_warp = time.perf_counter()
            if (
                self.aligner.needs_update()
                and not self.cfg.calibration_state.alignment_locked
            ):
                self.aligner.update(frame_l, frame_r)
            sbs = self._gpu_pipeline.process(frame_l, frame_r)
            timings = self._gpu_pipeline.last_timings_ms
            perf['gpu_pipeline_host_ms'] = (time.perf_counter() - t_warp) * 1000.0
            perf['gpu_upload_ms'] = timings.get('upload_ms', 0.0)
            perf['align_warp_ms'] = timings.get('warp_ms', 0.0)
            perf['fill_ms'] = timings.get('fill_ms', 0.0)
            perf['process_nudge_ms'] = (
                timings.get('process_ms', 0.0) + timings.get('nudge_ms', 0.0)
            )
            self._last_depth_mm, perf['depth_ms'] = self._maybe_update_depth(
                frame_l, frame_r, low_latency
            )
        else:
            frame_l, frame_r, perf['align_warp_ms'] = self._maybe_update_and_warp(frame_l, frame_r, low_latency)
            # Fill any black/invalid pixels in one eye with the other eye's pixels
            try:
                t_fill = time.perf_counter()
                frame_l, frame_r = self._fill_holes_cross(frame_l, frame_r)
                perf['fill_ms'] = (time.perf_counter() - t_fill) * 1000.0
            except Exception:
                perf['fill_ms'] = 0.0
            self._last_depth_mm, perf['depth_ms'] = self._maybe_update_depth(
                frame_l, frame_r, low_latency
            )

            eye_l, eye_r, sbs, perf['process_nudge_ms'] = self._process_and_nudge(frame_l, frame_r)
            sbs[:, :self.processor.eye_w] = eye_l
            sbs[:, self.processor.eye_w:] = eye_r

        calibration_sbs = self._maybe_build_calibration_sbs(frame_l, frame_r, sbs, now)
        self._maybe_emit_calibration_sbs(calibration_sbs, sbs, now)
        perf['overlay_ms'] = self._maybe_render_overlay(sbs, low_latency)
        t_q = time.perf_counter()
        if getattr(self.cfg.performance, "latency_watermark", False):
            from .latency_watermark import draw_timestamp_watermark

            draw_timestamp_watermark(sbs, time.monotonic() * 1000.0)
        self.sbs_ndarray_ready.emit(sbs)
        self.sbs_qimage_ready.emit(ndarray_to_qimage(sbs))
        perf['qimage_ms'] = (time.perf_counter() - t_q) * 1000.0
        self._finalize_frame(perf, t0, now)

    def _low_latency_enabled(self) -> bool:
        return bool(getattr(self.cfg, "performance", None) and getattr(self.cfg.performance, "low_latency_mode", False))

    def _can_process_now(self, now: float) -> bool:
        # Phase 1: process whenever a new camera frame is available. Frame-id
        # deduplication below handles repeated reads without adding latency.
        return True

    def _read_frame_pair(self, perf: dict[str, float]) -> tuple[np.ndarray | None, np.ndarray | None, tuple[int, int]]:
        t0 = time.perf_counter()
        frame_l, id_l = self.cam_l.read_latest_no_copy() if self.cam_l else (None, -1)
        frame_r, id_r = self.cam_r.read_latest_no_copy() if self.cam_r else (None, -1)
        perf['capture_ms'] = (time.perf_counter() - t0) * 1000.0
        return frame_l, frame_r, (id_l, id_r)

    def _apply_camera_flips(self, frame_l: np.ndarray, frame_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.cfg.cameras.left.flip_180:
            frame_l = cv2.rotate(frame_l, cv2.ROTATE_180)
        if self.cfg.cameras.right.flip_180:
            frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)
        return frame_l, frame_r

    def _maybe_emit_recording_frame(self, frame_l: np.ndarray, frame_r: np.ndarray, now: float) -> None:
        if not self.recording_enabled:
            return
        interval = 1.0 / max(1, min(60, int(self.recording_fps)))
        if now - self._last_record_emit_t < interval:
            return
        mode = self.recording_mode
        if mode == "single":
            frame = frame_l if self.recording_single_eye == "left" else frame_r
            self.recording_frame_ready.emit({"left": frame, "right": None, "mode": mode})
        else:
            self.recording_frame_ready.emit({"left": frame_l, "right": frame_r, "mode": mode})
        self._last_record_emit_t = now

    def _maybe_update_and_warp(self, frame_l: np.ndarray, frame_r: np.ndarray, low_latency: bool) -> tuple[np.ndarray, np.ndarray, float]:
        start = time.perf_counter()
        if (not low_latency) and self.aligner.needs_update() and not self.cfg.calibration_state.alignment_locked:
            self.aligner.update(frame_l, frame_r)
        # In low-latency mode, skip per-frame warp to save significant CPU/GPU time.
        if low_latency:
            return frame_l, frame_r, (time.perf_counter() - start) * 1000.0

        # Fast path: if CUDA warp is available and overlap masking is off,
        # run both eye warps directly here and avoid the slower CPU path.
        use_gpu_warp = (
            self._cuda_available
            and self._cuda_caps.get("warp_affine", False)
            and not bool(getattr(self.cfg.stereo.alignment, "mask_overlap", False))
            and getattr(self.aligner, "_warp_l", None) is not None
            and getattr(self.aligner, "_warp_r", None) is not None
        )
        if use_gpu_warp:
            h, w = frame_l.shape[:2]
            warp_l = getattr(self.aligner, "_warp_l")
            warp_r = getattr(self.aligner, "_warp_r")
            warped_l = self._warp_affine_fast(frame_l, warp_l, h, w)
            warped_r = self._warp_affine_fast(frame_r, warp_r, h, w)
            return warped_l, warped_r, (time.perf_counter() - start) * 1000.0

        warped_l, warped_r = self.aligner.warp_pair(frame_l, frame_r)
        return warped_l, warped_r, (time.perf_counter() - start) * 1000.0

    def _maybe_update_depth(self, frame_l: np.ndarray, frame_r: np.ndarray, low_latency: bool) -> tuple[float | None, float]:
        start = time.perf_counter()
        # Depth does not need to update every frame; decimation gives a large FPS win.
        depth_period = 4 if low_latency else 2
        depth_mm = self._last_depth_mm
        if self._frame_index % depth_period == 0:
            depth_mm = self._compute_depth_mm(frame_l, frame_r)
        return depth_mm, (time.perf_counter() - start) * 1000.0

    def _process_and_nudge(self, frame_l: np.ndarray, frame_r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        start = time.perf_counter()
        eye_l, eye_r, sbs = self.processor.process_pair(frame_l, frame_r)
        eye_l, eye_r = self.calibration.apply_nudge(eye_l, eye_r)
        return eye_l, eye_r, sbs, (time.perf_counter() - start) * 1000.0

    def _maybe_build_calibration_sbs(self, frame_l: np.ndarray, frame_r: np.ndarray, sbs: np.ndarray, now: float) -> np.ndarray | None:
        if not (self.raw_frame_requested and now - self._last_raw_emit_t >= self.raw_frame_interval):
            return None
        if self.raw_frame_mode == "full_fov_centered":
            # Manual overlay is shown in the Goovis. Keep the camera image at
            # native size in the middle of each half to avoid stretch/smear.
            return self.processor.process_pair_full_fov_centered(frame_l, frame_r).copy()
        if self.raw_frame_mode == "full_fov_goovis":
            # Manual overlay is shown in the Goovis. SBS headsets may stretch
            # each half horizontally, so this optional mode pre-compensates.
            return self.processor.process_pair_full_fov_sbs_anamorphic(frame_l, frame_r).copy()
        if self.raw_frame_mode == "full_fov":
            # Calibration views need the full FOV and must ignore live zoom,
            # convergence, saved nudges, and per-eye scale.
            return self.processor.process_pair_full_fov(frame_l, frame_r).copy()
        return sbs.copy()

    def _maybe_emit_calibration_sbs(self, calibration_sbs: np.ndarray | None, sbs: np.ndarray, now: float) -> None:
        if calibration_sbs is None:
            return
        self.sbs_frame_ready.emit(calibration_sbs)
        self._last_raw_emit_t = now

    def _maybe_render_overlay(self, sbs: np.ndarray, low_latency: bool) -> float:
        if low_latency or self._smart_overlap_metrics is None or not self._smart_overlap_active:
            return 0.0
        start = time.perf_counter()
        rendered = render_overlay(sbs, self._smart_overlap_metrics)
        if rendered is not sbs:
            sbs[:] = rendered
        return (time.perf_counter() - start) * 1000.0

    def _finalize_frame(self, perf: dict[str, float], start_t: float, now: float) -> None:
        if now - self._last_status_emit_t >= self._status_interval:
            self._emit_status()
            self._last_status_emit_t = now
        self._last_emit_t = now
        perf['total_frame_ms'] = (time.perf_counter() - start_t) * 1000.0
        for key, value in perf.items():
            self._perf_history[key].append(value)
        self._last_perf = perf

    def _compute_depth_mm(self, frame_l: np.ndarray, frame_r: np.ndarray) -> float | None:
        """Estimate calibrated depth from median disparity in the center ROI."""
        try:
            calib = self.cfg.stereo_calibration
            focal_px = self._resolved_focal_length_px(frame_l.shape[1])
            if focal_px <= 0 or calib.baseline_mm <= 0:
                return None

            disparity = self._compute_disparity(frame_l, frame_r)
            if disparity is None:
                return None

            median_disp = self._center_disparity_median(disparity)
            if median_disp <= 0.1:
                return None
            return float((focal_px * calib.baseline_mm) / median_disp)
        except Exception:
            return None

    def _resolved_focal_length_px(self, frame_width_px: int) -> float:
        calib = self.cfg.stereo_calibration
        if calib.focal_length_px > 0:
            return float(calib.focal_length_px)
        if calib.sensor_width_mm <= 0 or calib.focal_length_mm <= 0:
            return 0.0
        return float((calib.focal_length_mm / calib.sensor_width_mm) * frame_width_px)

    def _compute_disparity(self, frame_l: np.ndarray, frame_r: np.ndarray) -> np.ndarray | None:
        downscale = self.cfg.stereo_calibration.depth_downscale
        # GPU path: keep all pre-processing on GPU and avoid extra upload/download hops.
        disparity = self._compute_disparity_gpu_from_bgr(frame_l, frame_r, downscale)
        if disparity is not None:
            return disparity

        # CPU fallback path.
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        if abs(downscale - 1.0) > 1e-6:
            gray_l = cv2.resize(gray_l, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
            gray_r = cv2.resize(gray_r, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)

        matcher = self._ensure_depth_matcher()
        if disparity is None and matcher is not None:
            disparity = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
        return disparity

    def _compute_disparity_gpu_from_bgr(self, frame_l: np.ndarray, frame_r: np.ndarray, downscale: float) -> np.ndarray | None:
        if not self._use_gpu_depth or not self._cuda_available:
            return None
        if not self._cuda_caps.get("stereo_bm", False):
            return None
        if not self._cuda_caps.get("cvt_color", False):
            return None
        try:
            gpu_l = cv2.cuda_GpuMat()
            gpu_r = cv2.cuda_GpuMat()
            gpu_l.upload(frame_l)
            gpu_r.upload(frame_r)
            gpu_l = cv2.cuda.cvtColor(gpu_l, cv2.COLOR_BGR2GRAY)
            gpu_r = cv2.cuda.cvtColor(gpu_r, cv2.COLOR_BGR2GRAY)

            if abs(downscale - 1.0) > 1e-6:
                if not self._cuda_caps.get("resize", False):
                    return None
                h, w = int(frame_l.shape[0] * downscale), int(frame_l.shape[1] * downscale)
                gpu_l = cv2.cuda.resize(gpu_l, (w, h), interpolation=cv2.INTER_AREA)
                gpu_r = cv2.cuda.resize(gpu_r, (w, h), interpolation=cv2.INTER_AREA)

            calib = self.cfg.stereo_calibration
            num_disparities = max(16, int(round(calib.num_disparities / 16.0)) * 16)
            block_size = max(5, int(calib.block_size) | 1)
            params = (num_disparities, block_size)
            if getattr(self, "_gpu_bm_params", None) != params or getattr(self, "_gpu_bm", None) is None:
                self._gpu_bm = cv2.cuda.createStereoBM(numDisparities=num_disparities, blockSize=block_size)
                self._gpu_bm_params = params

            disp = self._gpu_bm.compute(gpu_l, gpu_r)
            return disp.download().astype(np.float32) / 16.0
        except Exception:
            return None

    def _ensure_depth_matcher(self):
        calib = self.cfg.stereo_calibration
        params = (int(calib.num_disparities), int(calib.block_size))
        if self._depth_matcher is not None and self._depth_matcher_params == params:
            return self._depth_matcher
        try:
            num_disparities = max(16, int(round(params[0] / 16.0)) * 16)
            block_size = max(5, int(params[1]) | 1)
            self._depth_matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
            self._depth_matcher_params = params
            return self._depth_matcher
        except Exception:
            self._depth_matcher = None
            self._depth_matcher_params = None
            return None

    def _compute_disparity_gpu(self, gray_l: np.ndarray, gray_r: np.ndarray) -> np.ndarray | None:
        if not self._use_gpu_depth:
            return None
        if not hasattr(cv2, "cuda") or not hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
            return None
        if cv2.cuda.getCudaEnabledDeviceCount() <= 0:
            return None
        if not hasattr(cv2.cuda, "createStereoBM") or not hasattr(cv2, "cuda_GpuMat"):
            return None
        try:
            calib = self.cfg.stereo_calibration
            num_disparities = max(16, int(round(calib.num_disparities / 16.0)) * 16)
            block_size = max(5, int(calib.block_size) | 1)
            gpu_l = cv2.cuda_GpuMat()
            gpu_r = cv2.cuda_GpuMat()
            gpu_l.upload(gray_l)
            gpu_r.upload(gray_r)
            gpu_bm = cv2.cuda.createStereoBM(numDisparities=num_disparities, blockSize=block_size)
            return gpu_bm.compute(gpu_l, gpu_r).download().astype(np.float32) / 16.0
        except Exception:
            return None

    def _center_disparity_median(self, disparity: np.ndarray) -> float | None:
        h, w = disparity.shape[:2]
        cx1, cx2 = w // 4, (w * 3) // 4
        cy1, cy2 = h // 4, (h * 3) // 4
        central = disparity[cy1:cy2, cx1:cx2]
        valid = central[central > 0.5]
        if valid.size == 0:
            return None
        return float(np.median(valid)) / float(self.cfg.stereo_calibration.depth_downscale)

    def _detect_cuda(self) -> bool:
        """Check if CUDA is available for GPU acceleration."""
        try:
            if not hasattr(cv2, "cuda"):
                print("[CUDA] cv2.cuda module not found")
                return False
            if not hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
                print("[CUDA] getCudaEnabledDeviceCount not available")
                return False
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"[CUDA] CUDA devices detected: {device_count}")
            if device_count <= 0:
                print("[CUDA] No CUDA devices available")
                return False
            print("[CUDA] GPU acceleration ENABLED")
            return True
        except Exception as e:
            print(f"[CUDA] Exception during detection: {e}")
            return False

    def _query_cuda_capabilities(self) -> dict[str, bool]:
        """Cache CUDA API availability so we don't raise exceptions every frame."""
        return {
            "stereo_bm": bool(hasattr(cv2.cuda, "createStereoBM") and hasattr(cv2, "cuda_GpuMat")),
            "warp_affine": bool(hasattr(cv2.cuda, "warpAffine") and hasattr(cv2, "cuda_GpuMat")),
            "cvt_color": bool(hasattr(cv2.cuda, "cvtColor") and hasattr(cv2, "cuda_GpuMat")),
            "resize": bool(hasattr(cv2.cuda, "resize") and hasattr(cv2, "cuda_GpuMat")),
        }

    def _warp_affine_fast(self, frame: np.ndarray, warp_mat: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
        """Warp frame using GPU if available, CPU fallback. Returns warped frame same shape/type."""
        if not self._cuda_available:
            return cv2.warpAffine(frame, warp_mat, (frame_w, frame_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_out = cv2.cuda_GpuMat(frame_h, frame_w, gpu_frame.type())
            cv2.cuda.warpAffine(gpu_frame, gpu_out, warp_mat, (frame_w, frame_h), borderMode=cv2.BORDER_CONSTANT)
            return gpu_out.download()
        except Exception:
            return cv2.warpAffine(frame, warp_mat, (frame_w, frame_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _fill_holes_cross(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fill near-black (hole) pixels in one eye with pixels from the other eye.

        Strategy: CPU-first, simple mask where all channels are zero. For pixels
        where left is invalid and right is valid, copy from right into left and
        vice-versa. Keeps existing pixels when both are invalid.
        """
        try:
            if left is None or right is None:
                return left, right
            # Ensure shapes match
            if left.shape != right.shape:
                return left, right

            # Detect near-black/invalid pixels per-eye.
            # Use strict all-zero test which corresponds to warp borderValue=0.
            mask_l = np.all(left == 0, axis=2)
            mask_r = np.all(right == 0, axis=2)

            # Where left is invalid and right valid -> copy right->left
            copy_to_l = mask_l & (~mask_r)
            if np.any(copy_to_l):
                left[copy_to_l] = right[copy_to_l]

            # Where right is invalid and left valid -> copy left->right
            copy_to_r = mask_r & (~mask_l)
            if np.any(copy_to_r):
                right[copy_to_r] = left[copy_to_r]

            return left, right
        except Exception:
            return left, right

    def _cvtcolor_and_resize_fast(self, frame: np.ndarray, code: int, scale: float) -> np.ndarray:
        """Convert color and resize using GPU if available, CPU fallback."""
        if not self._cuda_available:
            result = cv2.cvtColor(frame, code)
            if abs(scale - 1.0) > 1e-6:
                result = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            return result
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, code)
            if abs(scale - 1.0) > 1e-6:
                h, w = int(frame.shape[0] * scale), int(frame.shape[1] * scale)
                gpu_resized = cv2.cuda_GpuMat(h, w, gpu_gray.type())
                cv2.cuda.resize(gpu_gray, gpu_resized, (w, h), interpolation=cv2.INTER_AREA)
                return gpu_resized.download()
            return gpu_gray.download()
        except Exception:
            result = cv2.cvtColor(frame, code)
            if abs(scale - 1.0) > 1e-6:
                result = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            return result

    def _emit_status(self) -> None:
        avg_dt = sum(self._fps_hist) / len(self._fps_hist) if self._fps_hist else 0.016
        fps = 1.0 / avg_dt if avg_dt > 0 else 0
        ar = self.aligner.result
        perf_current = getattr(self, '_last_perf', None)
        perf_median = {
            key: float(np.median(values))
            for key, values in self._perf_history.items()
            if values
        }
        status = {
            "fps": fps,
            "dy": ar.dy,
            "dtheta_deg": ar.dtheta * 57.2958,
            "aligner_converged": self.aligner.converged,
            "pedal_mode": self.input.get_pedal_mode(),
            "zoom": self.processor.zoom,
            "depth_mm": self._last_depth_mm,
            "perf": perf_current,
            "perf_median": perf_median,
        }
        # Debug output: GPU status and latency metrics for performance tuning
        try:
            gpu_status = "GPU" if self._cuda_available else "CPU"
            perf_med = status.get('perf_median', {})
            total_ms = perf_med.get('total_frame_ms', 0.0)
            cap_ms = perf_med.get('capture_ms', 0.0)
            warp_ms = perf_med.get('align_warp_ms', 0.0)
            depth_ms = perf_med.get('depth_ms', 0.0)
            proc_ms = perf_med.get('process_nudge_ms', 0.0)
            qimg_ms = perf_med.get('qimage_ms', 0.0)
            depth_val = status.get('depth_mm')
            depth_str = f"{depth_val:.1f}mm" if depth_val is not None else "None"
            print(
                f"[{gpu_status}] depth={depth_str} total={total_ms:.2f}ms "
                f"capture={cap_ms:.2f} warp={warp_ms:.2f} depth={depth_ms:.2f} process={proc_ms:.2f} qimage={qimg_ms:.2f} "
                f"fps={status.get('fps', 0):.0f}"
            )
        except Exception:
            pass
        self.status_tick.emit(status)

    def _apply_action(self, act: Action) -> None:
        if act in (Action.ZOOM_IN, Action.ZOOM_OUT):
            self._apply_zoom_action(act)
        elif act in (Action.CONVERGE_IN, Action.CONVERGE_OUT):
            self._apply_converge_action(act)
        elif act in (
            Action.PEDAL_CENTER_LEFT,
            Action.PEDAL_CENTER_RIGHT,
            Action.PEDAL_CENTER_UP,
            Action.PEDAL_CENTER_DOWN,
        ):
            self._apply_pedal_action(act)
        elif act == Action.RESET:
            self.processor.reset()
            self.aligner.reset()
            self.calibration.reset_nudge()
        elif act == Action.QUIT:
            self._running = False

    def _apply_zoom_action(self, act: Action) -> None:
        if act == Action.ZOOM_IN:
            self.processor.zoom_in()
        elif act == Action.ZOOM_OUT:
            self.processor.zoom_out()

    def _apply_converge_action(self, act: Action) -> None:
        if act == Action.CONVERGE_IN:
            self.processor.converge_in()
        elif act == Action.CONVERGE_OUT:
            self.processor.converge_out()

    def _apply_pedal_action(self, act: Action) -> None:
        if act == Action.PEDAL_CENTER_LEFT:
            if hasattr(self.processor, "joint_zoom_center"):
                self.processor.set_joint_zoom_center(self.processor.joint_zoom_center - 1)
        elif act == Action.PEDAL_CENTER_RIGHT:
            if hasattr(self.processor, "joint_zoom_center"):
                self.processor.set_joint_zoom_center(self.processor.joint_zoom_center + 1)
        elif act == Action.PEDAL_CENTER_UP:
            if hasattr(self.processor, "joint_zoom_center_y"):
                self.processor.set_joint_zoom_center_y(self.processor.joint_zoom_center_y - 1)
        elif act == Action.PEDAL_CENTER_DOWN:
            if hasattr(self.processor, "joint_zoom_center_y"):
                self.processor.set_joint_zoom_center_y(self.processor.joint_zoom_center_y + 1)

    def _cleanup(self) -> None:
        if self.cam_l:
            self.cam_l.stop()
        if self.cam_r:
            self.cam_r.stop()
