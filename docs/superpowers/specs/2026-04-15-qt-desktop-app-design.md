# Qt Desktop App — Design Spec

**Date:** 2026-04-15
**Branch:** `qt-desktop-app`
**Status:** Approved

---

## Overview

Replace the current Pygame + Flask web-UI split with a single PyQt6 desktop application packaged as a Windows `.exe`. The app provides a tabbed control panel on the primary monitor and an optional borderless full-screen stereo output on the Goovis headset monitor. Pedal and keyboard input flow through the existing `InputHandler`. The backend pipeline (cameras, stereo processor, aligner, calibration, annotations) is reused unchanged.

---

## Users & Workflow

- **Surgeon** — wears the Goovis headset and uses the foot pedals during surgery. Never touches the exe mid-procedure.
- **Surgeon (pre-op) or assistant** — uses the exe for calibration, annotation drawing, and zoom-focus adjustments. May operate it during the procedure.

The exe is designed for both contexts: the **Live** tab is uncluttered enough to use during surgery; the **Calibration** and **Settings** tabs are setup-time workflows.

---

## Architecture

Three pieces run in one Python process:

1. **Main Qt window** — `QMainWindow` on the primary monitor with a `QTabWidget` holding Live, Calibration, and Settings tabs.
2. **Goovis output window** — a separate borderless `QWidget` shown full-screen on the Goovis monitor. Displays the processed stereo SBS frame. No controls.
3. **Pipeline worker** — a `QThread` that runs the existing backend pipeline (read → flip → align → process → overlay). Emits `QImage` signals every frame for the preview widget and the Goovis window.

Camera capture threads (from `src/camera.py`) continue as daemon threads. Pedal and keyboard input are polled inside the pipeline worker; actions dispatch through the existing `InputHandler` → `Action` enum.

### What gets removed

- `run.py`
- `src/viewer_stream.py` (Flask server + inline HTML)
- `src/display.py` (Pygame display)
- Dependencies: `pygame`, `flask`, `python-bidi`

### What gets reused unchanged

- `src/camera.py`, `src/stereo_processor.py`, `src/stereo_align.py`
- `src/calibration.py`, `src/input_handler.py`, `src/physical_cal.py`
- `src/annotation.py`, `src/config.py`
- `physical_calibration.py` logic — absorbed into the Calibration tab

---

## Tabs

### Tab 1 — Live (default on startup)

Center:
- **Stereo preview** — live processed frame. View toggle: `SBS` / `Left only` / `Right only` / `Anaglyph`.

Side panel:
- **Zoom-center arrow pad** — four buttons (↑ ↓ ← →) with the same 1 % step as the pedals. Useful when no pedals are attached.
- **Annotation toolbar** — freehand draw, arrow, circle, text, undo, clear. Backed by `src/annotation.py` unchanged. Disparity-offset slider for 3D annotation depth.

Bottom status strip:
- FPS, alignment status (`dy`, rotation degrees), camera connection (green/red dot per eye), pedal-mode indicator (`OFF / ZOOM / SIDE / UP-DOWN`).

### Tab 2 — Calibration (pre-op)

- **Physical calibration wizard** — the 4-phase flow from `physical_calibration.py` (focus → scale → horizontal → rotation) presented as a Next/Prev stepper with the live overlay shown in the tab's main area.
- **Per-eye nudge sliders** — X and Y for Left and Right eyes (inherits the sliders added in the prior iteration). Each with a numeric display and a "Reset" button. A master "Reset all nudges" button clears all four.
- **Aligner** — force re-scan button, auto-align on/off toggle.

### Tab 3 — Settings

Grouped sections (each a `QGroupBox`):

**Cameras**
- Per-eye: index, resolution, backend (`opencv` / `dshow` / `picamera2`), `flip_180` checkbox.
- "List devices" button runs the equivalent of `python -m src.camera --list` and shows results in a dialog.

**Goovis Display**
- Monitor dropdown (populated from `screeninfo`, auto-detected Goovis highlighted).
- Fullscreen on launch (checkbox).
- Enable output (checkbox — for bench testing without headset).
- Target FPS (spin box, bounded by camera FPS).

**Stereo**
- Convergence base offset (slider).
- Auto-alignment on/off (checkbox).
- Zoom: min / max / step / tick_ms (spin boxes).

**Pedals**
- Enable pedal input (checkbox).
- Key bindings — three fields for Pedal A / B / C. Click a field, press a key to rebind. Defaults: `a` / `b` / `c`.
- Mode assignments — three dropdowns: "Pedal A toggles ___", "Pedal B toggles ___", "Pedal C toggles ___". Options: `Zoom` / `Side-to-side` / `Up-down` / `(none)`. Defaults: A → Zoom, B → Side, C → Up/Down.
- Long-press repeat rate — slider (ms between repeats while adjust pedal is held).
- Live pedal-mode indicator.

**Config file**
- Load, Save, Reset to defaults. Save writes the full current state to `config.yaml` using its existing schema plus new pedal keys (`controls.pedal_key_a/b/c`, `controls.pedal_mode_a/b/c`, `controls.pedal_repeat_ms`).

---

## File Layout

```
piccolo.py                  # New Qt entry point
piccolo.spec                # PyInstaller config
build.bat                   # Build script
src/
  ui/                       # New — all Qt code
    __init__.py
    main_window.py          # QMainWindow with QTabWidget
    live_tab.py             # Tab 1
    calibration_tab.py      # Tab 2
    settings_tab.py         # Tab 3
    goovis_window.py        # Borderless full-screen output
    pipeline_worker.py      # QThread running the backend loop
    video_widget.py         # Reusable QLabel-based frame display
  camera.py                 # Unchanged
  stereo_processor.py       # Unchanged
  stereo_align.py           # Unchanged
  calibration.py            # Unchanged
  input_handler.py          # Minor: config-driven key rebinds
  physical_cal.py           # Unchanged
  annotation.py             # Unchanged
  config.py                 # Extended with new pedal fields
tests/
  test_physical_cal.py      # Unchanged
  test_ui_smoke.py          # New — pytest-qt smoke tests
docs/
  MANUAL_TEST.md            # New — manual QA checklist
```

---

## Tech Stack & Packaging

**Runtime:**
- PyQt6 (new), opencv-python, numpy, screeninfo, Pillow
- Removed: pygame, flask, python-bidi

**Packaging:**
- PyInstaller **one-file** build produces `piccolo.exe`.
- Spec file `piccolo.spec` committed to the repo.
- `build.bat` runs `pyinstaller piccolo.spec`.
- Target: Windows 10/11 x64 only.
- Icon + metadata baked into the exe.

**Threading:**
- Main thread: Qt event loop, all widget updates.
- Pipeline worker (`QThread`): runs the backend loop, emits `QImage` signals.
- Camera capture: existing daemon threads unchanged.
- Qt requires that widget updates happen on the main thread; the worker uses signals, not direct widget access.

---

## Error Handling

- **Camera open failure:** dialog on the Live tab ("Camera X failed to open — check USB connection"). App stays on Settings tab so indices can be fixed. No process exit.
- **Goovis monitor missing:** silent degradation — the second window is simply not opened. Status strip shows "Goovis: not detected". Console log only.
- **Pipeline thread crash:** caught at `QThread.run()`, shows error dialog, preview freezes, rest of the app remains responsive.
- **Malformed `config.yaml`:** log warning, fall back to in-memory defaults, do NOT overwrite the user's file.

---

## Testing

- **Existing backend tests** (`tests/test_physical_cal.py`, 17 tests) continue to pass headless.
- **New `tests/test_ui_smoke.py`** using `pytest-qt`:
  - Main window opens and closes cleanly.
  - Each tab is constructible.
  - Sliders and checkboxes fire the correct signals.
- **Manual test plan** (`docs/MANUAL_TEST.md`):
  - Open app → see live preview → switch between tabs → adjust nudge sliders → open calibration wizard → quit cleanly.
  - Goovis path: with headset connected, confirm full-screen output on correct monitor.
- No visual test of Goovis output — validated by the surgeon on real hardware.

---

## Migration & Rollout

1. All work lands on the `qt-desktop-app` branch. `master` keeps working with the web UI until cutover.
2. Implementation order (each its own plan task):
   1. UI scaffolding (`main_window.py`, three empty tabs, entry point).
   2. Pipeline worker + Live tab preview.
   3. Goovis output window.
   4. Calibration tab (physical-cal wizard + nudge sliders).
   5. Settings tab (cameras / display / stereo / pedals / config I/O).
   6. Pedal rebinding + config schema extension.
   7. Annotation integration.
   8. PyInstaller spec + build script.
   9. Manual test pass + README update.
3. Once merged, delete `src/viewer_stream.py`, `src/display.py`, `run.py`, and the corresponding dependencies.

---

## Success Criteria

1. `piccolo.exe` launches on Windows 10/11 x64 with no external installation required.
2. Main Qt window opens on the primary monitor; Goovis window opens full-screen on the secondary monitor when present.
3. All three tabs are functional: Live preview + annotations + zoom pad; Calibration wizard + nudge sliders; Settings with live-editable cameras/display/stereo/pedals.
4. Pedals continue to work as they do on `master`, with rebindable keys and mode assignments from Settings.
5. Closing the main window cleanly stops all threads and releases cameras.
6. All existing `tests/test_physical_cal.py` tests continue to pass.
