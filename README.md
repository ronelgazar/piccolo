
RUN COMMAND IS py -3.12 run.py    

# Piccolo – Stereoscopic Surgery Display

Real-time stereo-3D display system for surgical goggles, replacing the
earlier OBS/Lua prototype with a Python application that can run on a
Raspberry Pi or a development PC.

---

## Architecture

```
 ┌──────────┐   ┌──────────┐
 │ Camera L │   │ Camera R │   (USB / CSI – any two video sources)
 └────┬─────┘   └────┬─────┘
      │               │
      ▼               ▼
 ┌────────────────────────────────┐
 │      Threaded Camera Capture   │   src/camera.py
 │  (each camera in its own       │   - OpenCV backend (USB / capture cards)
 │   thread, latest-frame model)  │   - picamera2 backend (RPi CSI)
 └────────────┬───────────────────┘   - Test-pattern generator
              │
              ▼
 ┌────────────────────────────────┐
 │     Stereo Processor           │   src/stereo_processor.py
 │  • ROI crop → zoom             │   - Zoom centred on each camera's
 │  • Convergence offset          │     optical axis (prevents collision)
 │  • Side-by-side composition    │   - Auto convergence scaling
 └─────┬──────────────┬──────────┘
       │              │
       ▼              ▼
 ┌────────────┐ ┌──────────────┐
 │ Calibration│ │ HDMI Display │   src/display.py
 │  Overlay   │ │  (Pygame)    │   - Full-screen SBS stereo
 │ src/calib..│ │              │   - 1920×1080 → 960×1080 per eye
 └────────────┘ └──────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Viewer Stream   │   src/viewer_stream.py
              │  Flask + MJPEG   │   - http://<host>:8080/
              │  (background)    │   - SBS / Left / Right feeds
              └──────────────────┘

 Input (keyboard / future pedal) ← src/input_handler.py
 Configuration                   ← config.yaml + src/config.py
 Entry point                     ← run.py
```

---

## Quick Start

### 1. Install dependencies

```bash
cd piccolo
pip install -r requirements.txt
```

### 2. Run with test patterns (no cameras needed)

```bash
python run.py --test --windowed
```

### 3. Run with real cameras

Edit `config.yaml` to set your camera indices, then:

```bash
python run.py
```

### 4. Open the viewer stream

Browse to **http://localhost:8080/** from any device on the network.

---

## Controls

| Key      | Action                  |
|----------|-------------------------|
| `=` / `+` | Hold to zoom in        |
| `-`       | Hold to zoom out       |
| `]`       | Increase convergence   |
| `[`       | Decrease convergence   |
| `C`       | Start calibration      |
| `R`       | Reset zoom & convergence|
| `ESC`     | Quit                   |

All keys support **hold-to-repeat** for continuous adjustment, matching
the foot-pedal interaction model.

---

## How the Zoom Works (Why Images Don't Collide)

The OBS prototype scaled both scene items toward the *display* centre.
At high zoom the left and right images would physically overlap in the
middle of the screen – the "collision" problem.

Piccolo uses **per-camera ROI cropping**:

1. Each camera frame is cropped symmetrically from its own optical centre.
2. The crop is resized to fill half the display (960×1080).
3. Left and right halves are always placed side-by-side – they *can't*
   overlap.

When `auto_adjust` is enabled (the default), the convergence offset is
divided by the zoom level:

```
effective_offset = base_offset / zoom
```

This keeps the convergence plane at a constant real-world distance as the
surgeon zooms in, producing the same comfortable 3-D depth that a real
optical magnification loop would give.

---

## Calibration Sequence

Press **C** to start:

1. **Crosshair** (2 s) – green cross at the centre of each eye view.
   The surgeon fixates here.
2. **Blink phase** – left and right views alternate (the hidden eye sees
   black).  Each eye learns its view independently, avoiding binocular
   rivalry.
3. **Fuse** (3 s) – both views shown together with a pulsing crosshair.
   The surgeon confirms comfortable stereo fusion.

---

## Deploying on Raspberry Pi 5

### Hardware

| Component | Recommendation |
|-----------|---------------|
| Board     | Raspberry Pi 5 (4 GB+) |
| Cameras   | 2 × RPi Camera Module 3 (CSI) **or** 2 × USB cameras |
| Display   | HDMI SBS goggles (e.g. Goovies) |
| Pedal     | USB foot pedal (registers as keyboard HID) |

### Software Setup

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-pygame python3-opencv python3-flask python3-yaml

# For CSI cameras
sudo apt install -y python3-picamera2

# Clone & run
cd ~/piccolo
pip install -r requirements.txt
python run.py
```

In `config.yaml`, set:

```yaml
cameras:
  backend: picamera2    # use libcamera for CSI cameras
  left:
    index: 0
  right:
    index: 1
```

### Latency Optimisation

For production-grade latency (<50 ms glass-to-glass):

1. **Use GStreamer** instead of OpenCV/Pygame.  A libcamera → GStreamer →
   KMS/DRM pipeline avoids user-space copies entirely.
2. **Disable compositor** – boot directly to a console, run the app via
   DRM/KMS overlay.
3. **Pin CPU frequency** – `echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
4. **Use V4L2 directly** – `v4l2-ctl --set-ctrl=exposure_auto=1` for
   low-latency exposure.

A GStreamer example pipeline for reference:

```bash
gst-launch-1.0 \
  libcamerasrc camera-name="/base/soc/i2c0mux/i2c@1/imx708@1a" ! \
  video/x-raw,width=1920,height=1080,framerate=30/1 ! \
  queue max-size-buffers=1 leaky=downstream ! \
  kmssink connector-id=<id>
```

---

## Project Structure

```
piccolo/
├── config.yaml              # All tuneable parameters
├── requirements.txt
├── run.py                   # CLI entry point
└── src/
    ├── __init__.py
    ├── app.py               # Main loop, wires everything
    ├── camera.py            # Threaded capture (OpenCV / picamera2 / test)
    ├── stereo_processor.py  # Zoom, convergence, SBS composition
    ├── calibration.py       # Eye-fusion calibration overlay
    ├── display.py           # Pygame full-screen output
    ├── input_handler.py     # Key → Action mapping (keyboard / pedal)
    ├── viewer_stream.py     # Flask MJPEG server for external viewers
    └── config.py            # YAML loader + typed dataclasses
```

---

## Roadmap

- [ ] GPIO pedal input on RPi (two-switch: zoom in / zoom out)
- [ ] GStreamer pipeline backend for sub-50 ms latency
- [ ] Web-based annotation layer for external viewers
- [ ] Recording (H.264 to file) with timestamp overlay
- [ ] Auto-calibration using ArUco markers
- [ ] Motorised zoom lens control via serial/I²C
