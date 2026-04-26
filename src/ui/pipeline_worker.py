"""Backend pipeline running in a QThread.

Reads frames from both cameras, applies flip_180, runs the StereoAligner,
processes stereo (zoom + convergence), applies calibration nudges and
overlays, and emits a QImage signal for the latest SBS frame.
"""
from __future__ import annotations

import time

import cv2
from PyQt6.QtCore import QThread, pyqtSignal, QObject

from ..config import PiccoloCfg
from ..camera import CameraCapture, TestPatternCamera
from ..stereo_processor import StereoProcessor
from ..stereo_align import StereoAligner
from ..calibration import CalibrationOverlay
from ..input_handler import InputHandler, Action
from .qt_helpers import ndarray_to_qimage


class PipelineWorker(QThread):
    """Long-running backend thread."""

    sbs_frame_ready = pyqtSignal(object)        # np.ndarray — processed SBS (for overlay/wizard)
    sbs_qimage_ready = pyqtSignal(object)       # QImage — processed SBS (for display)
    status_tick = pyqtSignal(dict)              # FPS, alignment, pedal mode
    error = pyqtSignal(str)

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
        self.calibration = CalibrationOverlay(cfg.calibration)
        self.input = InputHandler(self.aligner, cfg.controls)

        # Apply persisted calibration state
        st = cfg.calibration_state
        self.calibration.nudge_left = st.nudge_left_x
        self.calibration.nudge_right = st.nudge_right_x
        self.calibration.nudge_left_y = st.nudge_left_y
        self.calibration.nudge_right_y = st.nudge_right_y
        self.processor.base_offset = st.convergence_offset
        if hasattr(self.processor, "set_joint_zoom_center"):
            self.processor.set_joint_zoom_center(st.joint_zoom_center)
        if hasattr(self.processor, "set_joint_zoom_center_y"):
            self.processor.set_joint_zoom_center_y(st.joint_zoom_center_y)

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
        self._last_raw_emit_t: float = 0.0
        self._last_status_emit_t: float = 0.0
        self._status_interval: float = 0.2

    # ------------------------------------------------------------------

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
        self.cam_l = CameraCapture(c.left.index, c.left.width, c.left.height,
                                    backend=c.backend, name="cam-L").start()
        try:
            self.cam_r = CameraCapture(c.right.index, c.right.width, c.right.height,
                                        backend=c.backend, name="cam-R").start()
        except Exception:
            if self.cam_l is not None:
                self.cam_l.stop()
            raise

    def _tick(self) -> None:
        for act in self.input.poll_actions():
            self._apply_action(act)
        if not self._running:
            return

        # Throttle: skip the entire pipeline between emit windows so we don't
        # do warp + process_pair work that nothing will ever see. Big CPU win.
        now = time.perf_counter()
        if now - self._last_emit_t < self._emit_interval:
            self.msleep(2)
            return

        frame_l, id_l = self.cam_l.read_latest_no_copy() if self.cam_l else (None, -1)
        frame_r, id_r = self.cam_r.read_latest_no_copy() if self.cam_r else (None, -1)
        if frame_l is None or frame_r is None:
            self.msleep(5)
            return
        if (id_l, id_r) == self._last_frame_ids:
            self.msleep(1)
            return
        self._last_frame_ids = (id_l, id_r)

        if self.cfg.cameras.left.flip_180:
            frame_l = cv2.rotate(frame_l, cv2.ROTATE_180)
        if self.cfg.cameras.right.flip_180:
            frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)

        if self.aligner.needs_update():
            self.aligner.update(frame_l, frame_r)
        frame_l, frame_r = self.aligner.warp_pair(frame_l, frame_r)

        eye_l, eye_r, sbs = self.processor.process_pair(frame_l, frame_r)
        eye_l, eye_r = self.calibration.apply_nudge(eye_l, eye_r)
        sbs[:, :self.processor.eye_w] = eye_l
        sbs[:, self.processor.eye_w:] = eye_r

        # Single conversion in worker thread; both Live tab and Goovis
        # share the same QImage (Qt copy-on-write — cheap to fan out).
        # The ndarray emit feeds the calibration overlay system (slot
        # gates on visibility so it costs nothing when wizard tab is hidden).
        if self.raw_frame_requested and now - self._last_raw_emit_t >= self.raw_frame_interval:
            self.sbs_frame_ready.emit(sbs.copy())
            self._last_raw_emit_t = now
        self.sbs_qimage_ready.emit(ndarray_to_qimage(sbs))
        if now - self._last_status_emit_t >= self._status_interval:
            self._emit_status()
            self._last_status_emit_t = now
        self._last_emit_t = now

    def _emit_status(self) -> None:
        avg_dt = sum(self._fps_hist) / len(self._fps_hist) if self._fps_hist else 0.016
        fps = 1.0 / avg_dt if avg_dt > 0 else 0
        ar = self.aligner.result
        self.status_tick.emit({
            "fps": fps,
            "dy": ar.dy,
            "dtheta_deg": ar.dtheta * 57.2958,
            "aligner_converged": self.aligner.converged,
            "pedal_mode": self.input.get_pedal_mode(),
            "zoom": self.processor.zoom,
        })

    def _apply_action(self, act: Action) -> None:
        if act == Action.ZOOM_IN:
            self.processor.zoom_in()
        elif act == Action.ZOOM_OUT:
            self.processor.zoom_out()
        elif act == Action.CONVERGE_IN:
            self.processor.converge_in()
        elif act == Action.CONVERGE_OUT:
            self.processor.converge_out()
        elif act == Action.PEDAL_CENTER_LEFT:
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
        elif act == Action.RESET:
            self.processor.reset()
            self.aligner.reset()
            self.calibration.reset_nudge()
        elif act == Action.QUIT:
            self._running = False

    def _cleanup(self) -> None:
        if self.cam_l:
            self.cam_l.stop()
        if self.cam_r:
            self.cam_r.stop()
