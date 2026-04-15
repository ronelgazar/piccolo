# Physical Camera Calibration Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `physical_calibration.py` — a standalone script that overlays diagnostic test patterns on both the PC monitor and the Goovis headset to guide physical camera alignment.

**Architecture:** A `PatternRenderer` class draws phase-specific overlays (focus/scale/horizontal/rotation) onto raw camera frames in-place. A `PhysicalCalSession` manages phase state and computes sharpness. The main script wires cameras, aligner, PC window, and headset display together in a single loop.

**Tech Stack:** `opencv-python`, `numpy`, `pygame` (already used by `StereoDisplay`)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/physical_cal.py` | **Create** | `PatternRenderer` + `PhysicalCalSession` |
| `physical_calibration.py` | **Create** | Entry point: cameras, aligner, loop, outputs |
| `tests/test_physical_cal.py` | **Create** | Unit tests for session logic + pattern rendering |
| `tests/conftest.py` | **Create** | `sys.path` fix so `src` is importable in tests |
| `requirements.txt` | **Modify** | Add `pytest>=8.0` and `pygame>=2.5` |

---

## Task 1: `PhysicalCalSession` — phase navigation + sharpness

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_physical_cal.py`
- Create: `src/physical_cal.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add pytest and pygame to requirements**

Edit `requirements.txt` — append at the end:
```
pygame>=2.5
pytest>=8.0
```

- [ ] **Step 2: Create `tests/conftest.py`**

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
```

- [ ] **Step 3: Write failing tests for `PhysicalCalSession`**

Create `tests/test_physical_cal.py`:
```python
import numpy as np
import cv2
import pytest
from src.physical_cal import PhysicalCalSession, PatternRenderer


class TestPhysicalCalSession:
    def test_initial_phase_is_focus(self):
        s = PhysicalCalSession()
        assert s.phase == "focus"

    def test_phase_index_starts_at_zero(self):
        s = PhysicalCalSession()
        assert s.phase_index == 0

    def test_total_phases_is_four(self):
        s = PhysicalCalSession()
        assert s.total_phases == 4

    def test_next_advances_through_all_phases(self):
        s = PhysicalCalSession()
        assert s.next_phase() is False and s.phase == "scale"
        assert s.next_phase() is False and s.phase == "horizontal"
        assert s.next_phase() is False and s.phase == "rotation"

    def test_next_on_last_phase_returns_true_and_stays(self):
        s = PhysicalCalSession()
        for _ in range(3):
            s.next_phase()
        assert s.next_phase() is True
        assert s.phase == "rotation"

    def test_prev_goes_back(self):
        s = PhysicalCalSession()
        s.next_phase()
        s.prev_phase()
        assert s.phase == "focus"

    def test_prev_on_first_phase_stays(self):
        s = PhysicalCalSession()
        s.prev_phase()
        assert s.phase == "focus"
        assert s.phase_index == 0


class TestSharpness:
    def test_sharp_image_scores_higher_than_blurred(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        for y in range(0, 400, 20):
            for x in range(0, 400, 20):
                if (x // 20 + y // 20) % 2 == 0:
                    img[y:y + 20, x:x + 20] = 255
        blurred = cv2.GaussianBlur(img, (21, 21), 0)
        assert PhysicalCalSession.sharpness(img) > PhysicalCalSession.sharpness(blurred)
```

- [ ] **Step 4: Run tests to confirm they fail**

```
cd c:/Users/ronelgazar/Documents/University/year2/Semester1/Research/SurgeryRobot/piccolo
.venv/Scripts/python -m pytest tests/test_physical_cal.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'src.physical_cal'`

- [ ] **Step 5: Create `src/physical_cal.py` with `PhysicalCalSession`**

```python
"""Physical camera calibration — pattern rendering and session logic."""
from __future__ import annotations

import math
import cv2
import numpy as np

PHASES = ["focus", "scale", "horizontal", "rotation"]


class PhysicalCalSession:
    """Phase state machine + per-eye sharpness metric."""

    def __init__(self):
        self._idx = 0

    @property
    def phase(self) -> str:
        return PHASES[self._idx]

    @property
    def phase_index(self) -> int:
        return self._idx

    @property
    def total_phases(self) -> int:
        return len(PHASES)

    def next_phase(self) -> bool:
        """Advance to next phase. Returns True when all phases are done."""
        if self._idx < len(PHASES) - 1:
            self._idx += 1
            return False
        return True

    def prev_phase(self) -> None:
        if self._idx > 0:
            self._idx -= 1

    @staticmethod
    def sharpness(img: np.ndarray) -> float:
        """Laplacian variance of central 200×200 px ROI. Higher = sharper."""
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        roi = img[max(0, cy - 100):cy + 100, max(0, cx - 100):cx + 100]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else roi
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class PatternRenderer:
    """Draws phase-specific test patterns onto BGR images in-place.

    Placeholder — render methods added in Task 2.
    """
```

- [ ] **Step 6: Run tests to confirm they pass**

```
.venv/Scripts/python -m pytest tests/test_physical_cal.py::TestPhysicalCalSession tests/test_physical_cal.py::TestSharpness -v
```

Expected: all 8 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/physical_cal.py tests/test_physical_cal.py tests/conftest.py requirements.txt
git commit -m "feat: add PhysicalCalSession phase logic and sharpness metric"
```

---

## Task 2: `PatternRenderer` — all four render methods

**Files:**
- Modify: `src/physical_cal.py`
- Modify: `tests/test_physical_cal.py`

- [ ] **Step 1: Add rendering tests to `tests/test_physical_cal.py`**

Append to the end of the file:

```python
class TestPatternRenderer:
    def _blank(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_render_focus_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        r.render_focus(img, sharpness=750.0)
        assert img.sum() > 0

    def test_render_scale_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        r.render_scale(img)
        assert img.sum() > 0

    def test_render_horizontal_with_dy_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        r.render_horizontal(img, dy=3.5)
        assert img.sum() > 0

    def test_render_horizontal_with_none_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        r.render_horizontal(img, dy=None)
        assert img.sum() > 0

    def test_render_rotation_with_angle_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        r.render_rotation(img, dtheta_deg=1.2)
        assert img.sum() > 0

    def test_render_rotation_with_none_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        r.render_rotation(img, dtheta_deg=None)
        assert img.sum() > 0
```

- [ ] **Step 2: Run new tests to confirm they fail**

```
.venv/Scripts/python -m pytest tests/test_physical_cal.py::TestPatternRenderer -v
```

Expected: FAIL — `PatternRenderer` has no render methods yet.

- [ ] **Step 3: Replace the `PatternRenderer` stub in `src/physical_cal.py` with the full implementation**

Replace the `PatternRenderer` class (everything from `class PatternRenderer:` to end of file) with:

```python
class PatternRenderer:
    """Draws phase-specific test patterns onto BGR images in-place."""

    def render_focus(self, img: np.ndarray, sharpness: float) -> None:
        """Siemens star (36-spoke radial wheel) + sharpness score."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        radius = min(h, w) // 4
        for i in range(36):
            angle = i * math.pi / 36
            x1 = int(cx + radius * math.cos(angle))
            y1 = int(cy + radius * math.sin(angle))
            x2 = int(cx - radius * math.cos(angle))
            y2 = int(cy - radius * math.sin(angle))
            color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
            cv2.line(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, "FOCUS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"sharp: {sharpness:.0f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

    def render_scale(self, img: np.ndarray) -> None:
        """Concentric circles at 10 %, 25 %, 50 % of frame height."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        color = (200, 200, 0)
        for pct in [10, 25, 50]:
            r = int(h * pct / 100)
            cv2.circle(img, (cx, cy), r, color, 1, cv2.LINE_AA)
            cv2.putText(img, f"{pct}%", (cx + r + 4, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(img, "SCALE  (match circle sizes visually)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    def render_horizontal(self, img: np.ndarray, dy: float | None) -> None:
        """Horizontal grid lines + centre crosshair + vertical-offset readout."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        color = (0, 200, 255)
        for pct in [25, 50, 75]:
            y = int(h * pct / 100)
            cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx - 40, cy), (cx + 40, cy), (0, 255, 0), 2)
        cv2.line(img, (cx, cy - 40), (cx, cy + 40), (0, 255, 0), 2)
        cv2.putText(img, "HORIZONTAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        dy_txt = f"Vertical offset: {dy:+.1f} px" if dy is not None else "Vertical offset: --"
        cv2.putText(img, dy_txt, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)

    def render_rotation(self, img: np.ndarray, dtheta_deg: float | None) -> None:
        """Diagonal reference lines + spirit-level arc + rotation readout."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        line_color = (255, 100, 0)
        cv2.line(img, (0, 0), (w, h), line_color, 1, cv2.LINE_AA)
        cv2.line(img, (w, 0), (0, h), line_color, 1, cv2.LINE_AA)
        arc_r = min(w, h) // 6
        arc_cx, arc_cy = cx, cy + arc_r + 20
        cv2.ellipse(img, (arc_cx, arc_cy), (arc_r, arc_r), 0, 180, 360,
                    (180, 180, 180), 1, cv2.LINE_AA)
        if dtheta_deg is not None:
            clamped = max(-5.0, min(5.0, dtheta_deg))
            dot_angle_rad = math.radians(270 + clamped * 18)
            dot_x = int(arc_cx + arc_r * math.cos(dot_angle_rad))
            dot_y = int(arc_cy + arc_r * math.sin(dot_angle_rad))
            cv2.circle(img, (dot_x, dot_y), 8, (0, 255, 0), -1)
        cv2.putText(img, "ROTATION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2, cv2.LINE_AA)
        rot_txt = (f"Rotation: {dtheta_deg:+.2f} deg"
                   if dtheta_deg is not None else "Rotation: --")
        cv2.putText(img, rot_txt, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_color, 1, cv2.LINE_AA)
```

- [ ] **Step 4: Run all tests**

```
.venv/Scripts/python -m pytest tests/test_physical_cal.py -v
```

Expected: all 14 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/physical_cal.py tests/test_physical_cal.py
git commit -m "feat: add PatternRenderer with focus/scale/horizontal/rotation overlays"
```

---

## Task 3: Entry point scaffold — config + cameras

**Files:**
- Create: `physical_calibration.py`

- [ ] **Step 1: Create `physical_calibration.py` with config loading and camera open/close**

```python
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
```

- [ ] **Step 2: Run a syntax check**

```
.venv/Scripts/python -c "import physical_calibration; print('OK')"
```

Expected: `OK` (no import errors)

- [ ] **Step 3: Run a smoke test with test_mode enabled in config**

Temporarily set `test_mode: true` in `config.yaml`, then:

```
.venv/Scripts/python physical_calibration.py
```

Expected: window opens showing test patterns, N/P keys cycle phases, Q quits cleanly. Restore `test_mode: false` after.

- [ ] **Step 4: Run the full test suite to confirm nothing is broken**

```
.venv/Scripts/python -m pytest tests/test_physical_cal.py -v
```

Expected: all 14 tests PASS

- [ ] **Step 5: Commit**

```bash
git add physical_calibration.py
git commit -m "feat: add physical_calibration.py standalone calibration tool"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| Standalone script at repo root | Task 3 creates `physical_calibration.py` |
| PC monitor OpenCV window, scaled to 1280 px | `_make_pc_frame` + `cv2.imshow` in Task 3 |
| Goovis headset Pygame SBS display | `StereoDisplay` + `_make_sbs` in Task 3 |
| Phase 1 — Focus: Siemens star + sharpness score | Task 2 `render_focus` |
| Phase 2 — Scale: concentric circles, visual only | Task 2 `render_scale` |
| Phase 3 — Horizontal: grid lines + crosshair + dy readout | Task 2 `render_horizontal` |
| Phase 4 — Rotation: diagonals + spirit-level arc + dtheta readout | Task 2 `render_rotation` |
| N / P / R / Q keys on both outputs | `_poll_keys` in Task 3 |
| Headset fail → continue without it | `try/except` around `display.open()` in Task 3 |
| Camera open failure → exit with message | `try/except` around `CameraCapture` in Task 3 |
| Aligner data not ready → show `--` | `dy = ... if aligner.has_correction else None` + `None` guard in renderer |

All spec requirements covered. No gaps found.

**Placeholder scan:** No TBD, TODO, or vague "handle edge cases" language present.

**Type consistency:**
- `render_focus(img, sharpness: float)` — used identically in Task 2 definition and Task 3 call site ✓
- `render_scale(img)` — consistent ✓
- `render_horizontal(img, dy: float | None)` — `dy` is `float | None` in both renderer and call site ✓
- `render_rotation(img, dtheta_deg: float | None)` — consistent ✓
- `PhysicalCalSession.sharpness(img)` — static method, called as `session.sharpness(eye_l)` ✓
- `session.phase_index`, `session.total_phases`, `session.phase` — all defined in Task 1, used in Task 3 ✓
