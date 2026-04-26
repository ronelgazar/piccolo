# Physical Camera Calibration Tool — Design Spec
**Date:** 2026-04-13  
**Status:** Approved

---

## Overview

A standalone Python script (`physical_calibration.py`) that opens both cameras, overlays diagnostic test patterns on the live feed, and sends output to two destinations simultaneously:

1. **PC monitor** — OpenCV `imshow` window (side-by-side left | right, scaled to fit)
2. **Goovis headset** — Pygame full-screen SBS display (same output path as the main app)

The surgeon wears the headset while an assistant physically adjusts the camera rig. Four sequential phases guide each alignment parameter independently.

---

## Entry Point

```
python physical_calibration.py
```

Located at the repo root alongside `run.py` and `bench.py`. Reads `config.yaml` for camera indices, resolution, and display settings.

---

## Architecture

```
physical_calibration.py   ← entry point, arg parsing, run loop
src/physical_cal.py       ← PatternRenderer + PhysicalCalSession (phase logic)
```

Reused from `src/`:
- `CameraCapture` — threaded camera grab
- `config.py` / `config.yaml` — camera indices, resolution, display settings
- `StereoAligner` — live vertical-offset and rotation measurements
- `StereoDisplay` — Pygame full-screen SBS output to the Goovis headset

---

## Phases

Four phases, advanced with **N** (next) or **P** (previous). Each phase overlays a different test pattern on the raw camera feed.

### Phase 1 — Focus

**Goal:** Each camera is sharp at the surgical working distance.

**Pattern:**
- Siemens star (radial spoke wheel, 36 spokes) centred on each eye's frame
- Fine text label ("FOCUS") below the star

**Live readout (per eye):**
- Sharpness score = Laplacian variance of a 200×200 px ROI around the star centre
- Displayed as `L sharp: 1234  R sharp: 1234`
- Higher = sharper; aim for both values to be similarly high

---

### Phase 2 — Scale

**Goal:** Both cameras see the same field of view / magnification.

**Pattern:**
- Concentric circles at 10 %, 25 %, and 50 % of frame height (radius), centred
- Each circle labelled with its percentage

**Live readout:**
- No numerical readout — `StereoAligner` models only translation + rotation, not scale.
- The concentric circles serve as a purely visual reference: the user physically adjusts zoom/aperture until the circles appear the same size in both the left and right eye views on the PC window.

---

### Phase 3 — Horizontal

**Goal:** Cameras are at the same physical height (no vertical offset).

**Pattern:**
- Full-width horizontal lines at 25 %, 50 %, and 75 % of frame height
- Centre crosshair (vertical + horizontal lines through mid-point)

**Live readout:**
- Vertical offset `dy` from `StereoAligner.result.dy`
- Displayed as `Vertical offset: +3.2 px` (aim for 0)

---

### Phase 4 — Rotation

**Goal:** Both cameras are level (no tilt relative to each other).

**Pattern:**
- Two long diagonal reference lines (top-left → bottom-right, top-right → bottom-left)
- Spirit-level arc at the bottom of the frame: a semicircle with a moving indicator dot whose position represents the measured rotation angle

**Live readout:**
- Rotation `dtheta` from `StereoAligner.result.dtheta` converted to degrees
- Displayed as `Rotation: +0.8 deg` (aim for 0)
---

## Controls

| Key | Action |
|-----|--------|
| **N** | Next phase |
| **P** | Previous phase |
| **R** | Reset `StereoAligner` (force fresh measurement) |
| **Q** / **ESC** | Quit |

---

## Display

### PC window (`cv2.imshow`)

- Single window titled `"Physical Calibration"`
- Content: left and right frames side by side, each with the phase overlay drawn
- Scaled down to fit the PC monitor (target display width ≤ 1280 px)
- Phase name and per-eye readouts shown as text at the top of the window

### Goovis headset (`StereoDisplay`)

- Full 1920×1080 SBS frame, identical overlay to the PC window
- Instantiated and shown every frame via `StereoDisplay.show()`
- The headset display runs on the monitor index specified in `config.yaml`

Both outputs are updated every iteration of the main loop (same frame).

---

## Pattern Rendering (`PatternRenderer`)

A single class in `src/physical_cal.py` with one method per phase:

```python
class PatternRenderer:
    def render_focus(self, img: np.ndarray, sharpness: float) -> None
    def render_scale(self, img: np.ndarray, scale_ratio: float) -> None
    def render_horizontal(self, img: np.ndarray, dy: float) -> None
    def render_rotation(self, img: np.ndarray, dtheta_deg: float) -> None
```

Each method draws directly onto `img` in-place. Called once per eye per frame.

---

## Session Logic (`PhysicalCalSession`)

```python
class PhysicalCalSession:
    phases = ["focus", "scale", "horizontal", "rotation"]
    current_phase: str

    def next_phase(self) -> bool   # returns True if all phases done
    def prev_phase(self) -> None
    def sharpness(self, img: np.ndarray) -> float
```

`sharpness()` computes Laplacian variance on a central 200×200 px crop.

---

## Main Loop (in `physical_calibration.py`)

```
1. Load config.yaml
2. Open left + right cameras (CameraCapture)
3. Init StereoAligner, StereoDisplay, PatternRenderer, PhysicalCalSession
4. Open Pygame display (StereoDisplay.open())
5. Open OpenCV window
6. Loop:
   a. Read frames from both cameras
   b. StereoAligner.update() if needed; StereoAligner.warp_pair()
   c. Compute per-phase metrics (sharpness, dy, dtheta, scale)
   d. PatternRenderer renders overlay onto copies of each frame
   e. Push SBS to StereoDisplay.show()
   f. Push scaled side-by-side to cv2.imshow()
   g. Poll keyboard (cv2.waitKey + Pygame events):
      - N / P / R / Q / ESC
   h. If all phases done or Q pressed: break
7. Cleanup: stop cameras, close display
```

---

## Error Handling

- If a camera fails to open: print a clear error and exit with code 1
- If `StereoAligner` has no data yet (first few seconds): show `"--"` instead of numerical readouts
- If Pygame display fails to open: continue without headset output, print a warning; PC window still works

---

## Dependencies

All already present in the project:
```
opencv-python
numpy
pygame
```

No new dependencies required.

---

## Success Criteria

1. Both camera feeds visible on PC window and Goovis headset simultaneously
2. Each phase shows a recognisably different test pattern
3. Live numerical readouts update in real-time as cameras are physically moved
4. N/P/Q keys work reliably on both outputs
5. Script exits cleanly without leaving windows or threads hanging
