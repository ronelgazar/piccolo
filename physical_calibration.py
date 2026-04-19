"""Physical camera calibration tool.

Run:  python physical_calibration.py

Overlays diagnostic test patterns on live camera feed.
Outputs: PC monitor (OpenCV) + Goovis headset (Pygame).

Keys: N = next phase  P = prev phase  R = reset aligner  Q/ESC = quit
"""
from __future__ import annotations

import sys
import cv2
import numpy as np
import pygame

from src.config import load_config
from src.camera import CameraCapture, TestPatternCamera
from src.stereo_align import StereoAligner
from src.display import StereoDisplay
from src.physical_cal import PhysicalCalSession, PatternRenderer

PC_DISPLAY_WIDTH = 1280


def _make_sbs(eye_l: np.ndarray, eye_r: np.ndarray,
              target_w: int, target_h: int) -> np.ndarray:
    """Resize both eyes to (target_w//2, target_h) and concatenate SBS."""
    ew = target_w // 2
    l = cv2.resize(eye_l, (ew, target_h), interpolation=cv2.INTER_LINEAR)
    r = cv2.resize(eye_r, (ew, target_h), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([l, r], axis=1)


def _make_pc_frame(eye_l: np.ndarray, eye_r: np.ndarray) -> np.ndarray:
    """Side-by-side frame scaled to PC_DISPLAY_WIDTH."""
    sbs = np.concatenate([eye_l, eye_r], axis=1)
    h, w = sbs.shape[:2]
    new_h = int(h * PC_DISPLAY_WIDTH / w)
    return cv2.resize(sbs, (PC_DISPLAY_WIDTH, new_h), interpolation=cv2.INTER_LINEAR)


def _poll_keys() -> set[str]:
    """Poll both OpenCV and Pygame key events. Returns set of action strings."""
    actions: set[str] = set()
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        actions.add("quit")
    elif key in (ord('n'), ord('N')):
        actions.add("next")
    elif key in (ord('p'), ord('P')):
        actions.add("prev")
    elif key in (ord('r'), ord('R')):
        actions.add("reset")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            actions.add("quit")
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                actions.add("quit")
            elif event.key == pygame.K_n:
                actions.add("next")
            elif event.key == pygame.K_p:
                actions.add("prev")
            elif event.key == pygame.K_r:
                actions.add("reset")
    return actions


def _draw_phase_hud(img: np.ndarray, phase: str, idx: int, total: int) -> None:
    h = img.shape[0]
    label = (f"Phase {idx + 1}/{total}: {phase.upper()}"
             "   [N] next  [P] prev  [R] reset  [Q] quit")
    cv2.putText(img, label, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    cfg = load_config()

    if cfg.cameras.test_mode:
        print("[physcal] Using test patterns (no cameras).")
        cam_l = TestPatternCamera(
            cfg.cameras.left.width, cfg.cameras.left.height,
            side="left", name="test-L").start()
        cam_r = TestPatternCamera(
            cfg.cameras.right.width, cfg.cameras.right.height,
            side="right", name="test-R").start()
    else:
        print("[physcal] Opening cameras…")
        try:
            cam_l = CameraCapture(
                cfg.cameras.left.index, cfg.cameras.left.width,
                cfg.cameras.left.height,
                backend=cfg.cameras.backend, name="cam-L").start()
            cam_r = CameraCapture(
                cfg.cameras.right.index, cfg.cameras.right.width,
                cfg.cameras.right.height,
                backend=cfg.cameras.backend, name="cam-R").start()
        except RuntimeError as e:
            print(f"[physcal] ERROR: {e}")
            sys.exit(1)

    aligner = StereoAligner(
        cfg.stereo.alignment,
        cfg.cameras.left.width,
        cfg.cameras.left.height,
    )
    session = PhysicalCalSession()
    renderer = PatternRenderer()

    headset_ok = True
    display = StereoDisplay(cfg.display)
    try:
        display.open()
        print("[physcal] Headset display opened.")
    except Exception as e:
        print(f"[physcal] WARNING: Headset display failed ({e}). Continuing without it.")
        headset_ok = False

    cv2.namedWindow("Physical Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Physical Calibration", PC_DISPLAY_WIDTH,
                     PC_DISPLAY_WIDTH * 9 // 32)

    print(f"[physcal] Starting. Phase: {session.phase}")
    print("[physcal] Keys: N=next  P=prev  R=reset-aligner  Q/ESC=quit")

    try:
        while True:
            frame_l = cam_l.read_no_copy()
            frame_r = cam_r.read_no_copy()
            if frame_l is None or frame_r is None:
                if headset_ok:
                    display.tick()
                continue

            # Apply per-eye 180° flip if configured (matches run.py behavior)
            if cfg.cameras.left.flip_180:
                frame_l = cv2.rotate(frame_l, cv2.ROTATE_180)
            if cfg.cameras.right.flip_180:
                frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)

            if aligner.needs_update():
                aligner.update(frame_l, frame_r)
            frame_l, frame_r = aligner.warp_pair(frame_l, frame_r)

            eye_l = frame_l.copy()
            eye_r = frame_r.copy()

            ar = aligner.result
            dy = ar.dy if aligner.has_correction else None
            dtheta_deg = ar.dtheta * 57.2958 if aligner.has_correction else None
            sharp_l = session.sharpness(eye_l)
            sharp_r = session.sharpness(eye_r)

            phase = session.phase
            if phase == "focus":
                renderer.render_focus(eye_l, sharp_l)
                renderer.render_focus(eye_r, sharp_r)
            elif phase == "scale":
                renderer.render_scale(eye_l)
                renderer.render_scale(eye_r)
            elif phase == "horizontal":
                renderer.render_horizontal(eye_l, dy)
                renderer.render_horizontal(eye_r, dy)
            elif phase == "rotation":
                renderer.render_rotation(eye_l, dtheta_deg)
                renderer.render_rotation(eye_r, dtheta_deg)

            _draw_phase_hud(eye_l, phase, session.phase_index, session.total_phases)
            _draw_phase_hud(eye_r, phase, session.phase_index, session.total_phases)

            pc_frame = _make_pc_frame(eye_l, eye_r)
            cv2.imshow("Physical Calibration", pc_frame)

            if headset_ok:
                sbs = _make_sbs(eye_l, eye_r,
                                cfg.display.width, cfg.display.height)
                display.show(sbs)
                display.tick()

            actions = _poll_keys()
            if "quit" in actions:
                break
            if "next" in actions:
                done = session.next_phase()
                print(f"[physcal] Phase → {session.phase}")
                if done:
                    print("[physcal] All phases complete.")
                    break
            if "prev" in actions:
                session.prev_phase()
                print(f"[physcal] Phase → {session.phase}")
            if "reset" in actions:
                aligner.reset()
                aligner.force_update()
                print("[physcal] Aligner reset.")

    finally:
        cam_l.stop()
        cam_r.stop()
        if headset_ok:
            display.close()
        cv2.destroyAllWindows()
        print("[physcal] Done.")


if __name__ == "__main__":
    main()
