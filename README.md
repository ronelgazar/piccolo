# Piccolo – Stereoscopic Surgery Display

Real-time stereo-3D display system for surgical goggles — a single-exe PyQt6 desktop app with tabbed control panel and full-screen Goovis headset output.

---

## Quick Start

### 1. Install dependencies

```bash
cd piccolo
pip install -r requirements.txt
```

### 2. Configure

Edit `config.yaml` — set your camera indices, display monitor, and any per-eye calibration. To run without hardware, set `cameras.test_mode: true`.

### 3. Run from source

```bash
python piccolo.py
```

### 4. Build a standalone exe

```bash
build.bat
```

Produces `dist\piccolo.exe` (Windows 10/11 x64).

---

## Tabs

- **Live** — stereo preview, zoom-center arrow pad, freehand annotations, status strip (FPS / alignment / pedal mode).
- **Calibration** — per-eye pixel nudge sliders (X and Y) that persist to `config.yaml`, plus a 4-phase physical-calibration wizard (Focus → Scale → Horizontal → Rotation) with live overlays.
- **Settings** — cameras, Goovis display, stereo, pedal bindings, config file load/save/reset.

---

## Goovis Output

When a Goovis headset is detected as a secondary monitor (1920×1080 non-primary, or a screen whose name contains "GOOVIS" or "NED"), a borderless full-screen window automatically opens on that monitor and mirrors the processed SBS feed. No separate process needed.

---

## Pedal Controls

The app supports a three-pedal USB footswitch that sends `a` / `b` / `c` key events. Toggle semantics:

| Press | Action |
|-------|--------|
| `a` | Toggle **zoom** mode |
| `b` | Toggle **side** mode |
| `c` | Toggle **up/down** mode |

Once a mode is active, the other two pedals become adjust pedals (long-press supported):

| Mode | Other pedals do |
|------|-----------------|
| zoom | `b` = zoom in, `c` = zoom out |
| side | `a` = move left, `c` = move right |
| up/down | `a` = move up, `b` = move down |

Press the mode pedal again to toggle off.

Key bindings and mode assignments are editable in the Settings tab.

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `=` / `+` | Zoom in (hold) |
| `-` | Zoom out (hold) |
| `]` / `[` | Convergence in / out |
| `c` | Toggle calibration overlay |
| `m` | Toggle auto-alignment |
| `r` | Reset zoom, convergence, nudge |
| `Esc` | Quit |

---

## Project Structure

```
piccolo/
├── piccolo.py               # Qt entry point
├── piccolo.spec             # PyInstaller config
├── build.bat                # One-file exe build
├── config.yaml              # All tuneable parameters
├── requirements.txt
├── src/
│   ├── camera.py            # Threaded capture (OpenCV / picamera2 / test)
│   ├── stereo_processor.py  # Zoom, convergence, SBS composition
│   ├── stereo_align.py      # SIFT-based auto-alignment
│   ├── calibration.py       # Per-eye nudge (X and Y), persistent
│   ├── input_handler.py     # Framework-neutral key/pedal handler
│   ├── physical_cal.py      # 4-phase calibration wizard logic
│   ├── annotation.py        # Annotation rendering on stereo frames
│   ├── config.py            # YAML loader + typed dataclasses
│   ├── config_state.py      # Persist calibration state to config.yaml
│   └── ui/                  # Qt UI
│       ├── main_window.py   # Main QMainWindow + tabs + Goovis window
│       ├── live_tab.py
│       ├── calibration_tab.py
│       ├── settings_tab.py
│       ├── goovis_window.py
│       ├── pipeline_worker.py
│       ├── video_widget.py
│       ├── annotation_overlay_widget.py
│       └── qt_helpers.py
└── tests/
    ├── conftest.py
    ├── test_physical_cal.py
    ├── test_input_handler.py
    ├── test_config_state.py
    └── test_ui_smoke.py
```
