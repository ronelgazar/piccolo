"""Main application – wires every component together.

Lifecycle
---------
1. Load config.
2. Open cameras (or test-pattern generators).
3. Initialise stereo processor, display, calibration, input, viewer stream.
4. Enter main loop:
   a. Poll input → translate to actions.
   b. Read latest camera frames (zero-copy).
   c. Process stereo (zoom, convergence).
   d. Apply calibration overlay (if active).
   e. Display on HDMI (Pygame).
   f. Push frames to viewer stream.
5. On quit → clean up all resources.
"""

from __future__ import annotations

import sys
import time
from collections import deque

import cv2
import numpy as np

from .config import PiccoloCfg
from .camera import CameraCapture, TestPatternCamera
from .stereo_processor import StereoProcessor
from .stereo_align import StereoAligner
from .calibration import CalibrationOverlay
from .display import StereoDisplay
from .input_handler import InputHandler, Action
from .viewer_stream import ViewerStream


class PiccoloApp:
    """Top-level application object."""

    def __init__(self, cfg: PiccoloCfg):
        self.cfg = cfg

        eye_w = cfg.display.width // 2
        eye_h = cfg.display.height

        # Components
        self.cam_l: CameraCapture | TestPatternCamera | None = None
        self.cam_r: CameraCapture | TestPatternCamera | None = None
        self.processor = StereoProcessor(cfg.stereo, eye_w, eye_h)
        self.aligner = StereoAligner(
            cfg.stereo.alignment,
            cfg.cameras.left.width,
            cfg.cameras.left.height,
        )
        self.calibration = CalibrationOverlay(cfg.calibration)
        self.display = StereoDisplay(cfg.display)
        self.input = InputHandler(cfg.controls)
        self.stream = ViewerStream(cfg.stream) if cfg.stream.enabled else None

        self._running = False

        # Timing / diagnostics
        self._fps_hist: deque[float] = deque(maxlen=60)
        self._loop_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self):
        """Blocking main loop.  Call from the main thread."""
        self._open_cameras()
        self.display.open()
        if self.stream:
            self.stream.start()
            print(f"[piccolo] Viewer stream at http://localhost:{self.cfg.stream.port}/")

        self._running = True
        print("[piccolo] Running.  Press ESC to quit, C for calibration.")
        try:
            self._main_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_cameras(self):
        ccfg = self.cfg.cameras
        if ccfg.test_mode:
            print("[piccolo] Using synthetic test patterns (no cameras).")
            self.cam_l = TestPatternCamera(ccfg.left.width, ccfg.left.height, side="left", name="test-L").start()
            self.cam_r = TestPatternCamera(ccfg.right.width, ccfg.right.height, side="right", name="test-R").start()
        else:
            print(f"[piccolo] Opening cameras (backend={ccfg.backend})…")
            self.cam_l = CameraCapture(
                ccfg.left.index, ccfg.left.width, ccfg.left.height,
                backend=ccfg.backend, name="cam-L"
            ).start()
            self.cam_r = CameraCapture(
                ccfg.right.index, ccfg.right.width, ccfg.right.height,
                backend=ccfg.backend, name="cam-R"
            ).start()

    def _main_loop(self):
        while self._running:
            t0 = time.perf_counter()

            # 1 ─ Input (keyboard + web UI)
            actions = self.input.poll()
            if self.stream:
                for cmd in self.stream.drain_commands():
                    self._handle_web_command(cmd)
            self._handle_actions(actions)
            if not self._running:
                break

            # 2 ─ Grab latest frames (zero-copy – no memcpy)
            frame_l = self.cam_l.read_no_copy()
            frame_r = self.cam_r.read_no_copy()
            if frame_l is None or frame_r is None:
                self.display.tick()
                continue

            # 2b ─ Auto-align: correct vertical misalignment & rotation
            #      between cameras (periodic re-estimation + per-frame warp)
            if self.aligner.needs_update():
                self.aligner.update(frame_l, frame_r)
            frame_l, frame_r = self.aligner.warp_pair(frame_l, frame_r)

            # 3 ─ Stereo processing (zoom + convergence) → writes into
            #     pre-allocated SBS buffer to avoid allocation every frame
            eye_l, eye_r, sbs = self.processor.process_pair(frame_l, frame_r)

            # 4a ─ Apply persistent per-eye nudge offsets (always)
            eye_l, eye_r = self.calibration.apply_nudge(eye_l, eye_r)

            # 4b ─ Calibration overlay (only when calibrating)
            if self.calibration.active:
                eye_l, eye_r = self.calibration.apply(eye_l, eye_r)

            # Write back into the SBS buffer (nudge always, overlay when active)
            if self.calibration.active or self.calibration.nudge_left != 0 or self.calibration.nudge_right != 0:
                sbs[:, :self.processor.eye_w] = eye_l
                sbs[:, self.processor.eye_w:] = eye_r

            # 5 ─ Push CLEAN frames to viewer stream BEFORE annotations
            #     so the web annotation page shows un-annotated video
            #     (the page's own canvas provides the local preview).
            if self.stream:
                self.stream.update_frame(sbs=sbs, left=eye_l, right=eye_r)

            # 6 ─ Annotation overlay (Goovis display only)
            if self.stream and self.stream.annotations.show_on_screen:
                self.stream.annotations.render(eye_l, eye_r)
                sbs[:, :self.processor.eye_w] = eye_l
                sbs[:, self.processor.eye_w:] = eye_r

            # 7 ─ Draw HUD (zoom level + FPS)
            sbs = self._draw_hud(sbs)

            # 8 ─ Display (pre-allocated surface, zero-alloc blit)
            self.display.show(sbs)
            self.display.tick()

            # 9 ─ Push status to web UI
            if self.stream:
                self._push_status()

            # Track frame timing
            dt = time.perf_counter() - t0
            self._loop_time = dt
            self._fps_hist.append(dt)

    def _handle_actions(self, actions: set):
        if Action.QUIT in actions:
            self._running = False
        if Action.ZOOM_IN in actions:
            self.processor.zoom_in()
        if Action.ZOOM_OUT in actions:
            self.processor.zoom_out()
        if Action.CONVERGE_IN in actions:
            self.processor.converge_in()
        if Action.CONVERGE_OUT in actions:
            self.processor.converge_out()
        if Action.TOGGLE_CALIBRATION in actions:
            was_active = self.calibration.active
            self.calibration.toggle()
            print(f"[piccolo] Calibration {'ON' if self.calibration.active else 'OFF'}")
            # Trigger fresh alignment when calibration exits
            if was_active and not self.calibration.active and self.aligner.enabled:
                self.aligner.force_update()
                print("[piccolo] Post-calibration alignment triggered.")
        if Action.CALIB_NEXT in actions:
            was_active = self.calibration.active
            self.calibration.next_phase()
            phase = self.calibration.phase
            print(f"[piccolo] Calibration phase → {phase}")
            # If next_phase caused exit, trigger re-alignment
            if was_active and not self.calibration.active and self.aligner.enabled:
                self.aligner.force_update()
                print("[piccolo] Post-calibration alignment triggered.")
        if Action.CALIB_NUDGE_LEFT in actions:
            self.calibration.nudge_current_left()
        if Action.CALIB_NUDGE_RIGHT in actions:
            self.calibration.nudge_current_right()
        if Action.TOGGLE_ALIGNMENT in actions:
            self.aligner.enabled = not self.aligner.enabled
            state = "ON" if self.aligner.enabled else "OFF"
            print(f"[piccolo] Auto-alignment {state}")
            if self.aligner.enabled:
                self.aligner.force_update()  # re-estimate immediately
        if Action.RESET in actions:
            self.processor.reset()
            self.aligner.reset()
            self.calibration.reset_nudge()
            print("[piccolo] Reset zoom, convergence, alignment & nudge.")

    def _handle_web_command(self, cmd: str):
        """Translate a web UI command string into an action."""
        action_map = {
            "zoom_in": Action.ZOOM_IN,
            "zoom_out": Action.ZOOM_OUT,
            "converge_in": Action.CONVERGE_IN,
            "converge_out": Action.CONVERGE_OUT,
            "toggle_calibration": Action.TOGGLE_CALIBRATION,
            "toggle_alignment": Action.TOGGLE_ALIGNMENT,
            "calib_next": Action.CALIB_NEXT,
            "calib_nudge_left": Action.CALIB_NUDGE_LEFT,
            "calib_nudge_right": Action.CALIB_NUDGE_RIGHT,
            "reset": Action.RESET,
            "quit": Action.QUIT,
        }
        action = action_map.get(cmd)
        if action:
            self._handle_actions({action})
        elif cmd == "force_align":
            self.aligner.force_update()
            print("[piccolo] Forced alignment re-scan.")
        elif cmd == "reset_nudge":
            self.calibration.reset_nudge()
            print("[piccolo] Nudge offsets reset.")

    def _push_status(self):
        """Push current state to the viewer stream for the web UI."""
        ar = self.aligner.result
        avg_dt = (sum(self._fps_hist) / len(self._fps_hist)) if self._fps_hist else 0.016
        fps = 1.0 / avg_dt if avg_dt > 0 else 0
        self.stream.update_status({
            "fps": fps,
            "loop_ms": self._loop_time * 1000,
            "zoom": self.processor.zoom,
            "convergence_offset": self.processor.base_offset,
            "alignment": {
                "enabled": self.aligner.enabled,
                "method": ar.method,
                "dy": ar.dy,
                "dtheta_deg": ar.dtheta * 57.2958,
                "n_matches": ar.n_matches,
                "confidence": ar.confidence,
                "rms_residual": ar.rms_residual,
                "converged": self.aligner.converged,
            },
            "calibration": self.calibration.active,
            "calibration_phase": self.calibration.phase,
            "nudge_left": self.calibration.nudge_left,
            "nudge_right": self.calibration.nudge_right,
            "annotations": {
                "count": self.stream.annotations.count,
                "show_on_screen": self.stream.annotations.show_on_screen,
                "disparity_offset": self.stream.annotations.disparity_offset,
            },
        })

    def _draw_hud(self, sbs: np.ndarray) -> np.ndarray:
        """Minimal heads-up display: zoom level + FPS/latency + alignment."""
        h, w = sbs.shape[:2]
        z = self.processor.zoom
        if z > 1.001:
            text = f"ZOOM {z:.1f}x"
            cv2.putText(
                sbs, text,
                (w // 2 - 60, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1, cv2.LINE_AA,
            )
        # FPS / loop-time indicator (top-left, small)
        if self._fps_hist:
            avg_dt = sum(self._fps_hist) / len(self._fps_hist)
            fps = 1.0 / avg_dt if avg_dt > 0 else 0
            ms = self._loop_time * 1000
            cv2.putText(
                sbs, f"{fps:.0f}fps  {ms:.1f}ms",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1, cv2.LINE_AA,
            )
        # Alignment status (top-right area)
        ar = self.aligner.result
        if self.aligner.enabled:
            if self.aligner.has_correction:
                dy_txt = f"dy={ar.dy:+.1f}px"
                dt_txt = f"rot={ar.dtheta * 57.2958:+.2f}deg"  # rad→deg
                rms_txt = f"rms={ar.rms_residual:.1f}px" if ar.rms_residual < 1e6 else ""
                align_txt = f"ALIGN {ar.method} {dy_txt} {dt_txt} {rms_txt} ({ar.n_matches}m)"
                color = (0, 220, 0) if self.aligner.converged else (0, 200, 200)
            else:
                align_txt = "ALIGN calibrating..."
                color = (0, 140, 140)
        else:
            align_txt = "ALIGN off"
            color = (100, 100, 100)
        cv2.putText(
            sbs, align_txt,
            (w - 420, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
        return sbs

    def _shutdown(self):
        print("[piccolo] Shutting down…")
        if self.cam_l:
            self.cam_l.stop()
        if self.cam_r:
            self.cam_r.stop()
        self.display.close()
