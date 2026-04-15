# Needle Tracker — Design Spec
**Date:** 2026-04-12  
**Status:** Approved

---

## Overview

A standalone Python script (`needle_tracker.py`) that analyzes a microscope camera video of a surgical needle moving across a calibration grid. The needle traces the first (top) row of the grid from right to left, touching each 0.1 mm cube. The script tracks the needle tip, identifies which cell it is in, produces an annotated output video, and generates position and velocity graphs.

---

## Input

- **Video file:** `WhatsApp Video 2026-04-12 at 13.46.50 (online-video-cutter.com).mp4`  
  Located at: `c:/Users/ronelgazar/Documents/University/year2/Semester1/Research/SurgeryRobot/piccolo/WhatsApp Video 2026-04-12 at 13.46.50 (online-video-cutter.com).mp4`
- **Grid:** Regular square grid; each cell = 0.1 mm × 0.1 mm
- **Needle direction:** Right to left across the top horizontal row
- **Camera:** Microscope camera with mild shake/drift

---

## Architecture

Single Python script, three sequential phases:

### Phase 1 — Calibration
- Opens first video frame in an interactive OpenCV window
- User clicks **4 corner points** of the grid region (any order; script computes bounding box)
- Script derives:
  - Pixel-per-mm scale from known cell size (0.1 mm)
  - Top row bounding box in pixel coordinates
  - Number of cells N in the top row
  - Cell boundaries (right → left = cell 1, 2, 3 … N)
- Reference frame saved for stabilization

### Phase 2 — Tracking (per frame)
```
raw frame
  → ORB feature matching → homography warp (camera stabilization)
  → MOG2 background subtraction
  → morphological cleanup (erode + dilate)
  → blob detection (largest blob = needle tip)
  → Kalman filter predict + correct
  → needle tip (x, y) in pixels → convert to mm
  → determine current cell number
  → log frame data
```

**Kalman filter state:** `[x, y, dx, dy]` — position + velocity, 2D constant-velocity model.  
**MOG2 settings:** slow learning rate (~0.001) to tolerate gradual lighting change.  
**ORB stabilization:** matches keypoints on each frame to the reference frame; computes partial homography (translation + rotation only) to warp frame back to reference alignment.

### Phase 3 — Output
- Writes all output files (see below)
- Opens Plotly HTML graphs in the default browser

---

## Grid Logic

- After calibration, top row pixel bounds are known
- Top row divided into N cells of equal width
- Cell numbering: **cell 1 = rightmost**, cell N = leftmost (right→left traversal)
- "Needle in cell k" = needle tip X pixel falls within cell k's pixel range
- Cell entry event logged when cell number changes
- `cell_number = 0` = needle tip detected but outside the top row bounds
- `cell_number = -1` = no blob detected this frame (Kalman prediction used for position)

---

## Logged Data (per frame)

| Column | Description |
|--------|-------------|
| `frame_idx` | Frame number |
| `timestamp_sec` | Time in seconds |
| `x_pixel` | Needle tip X in pixels |
| `y_pixel` | Needle tip Y in pixels |
| `x_mm` | Needle tip X in mm (from right edge of row) |
| `delta_x_mm` | Change in X from previous frame (mm) |
| `cell_number` | Current cell (1 = rightmost, 0 = not in row) |

---

## Live Display

During tracking, an OpenCV window shows the current frame with:
- Red crosshair on the needle tip
- Blue trajectory trail (last 60 frames)
- Top row highlighted with current cell filled in green
- Text overlay: frame number, X position (mm), current cell, timestamp

---

## Output Files

All written to the same directory as the input video:

| File | Description |
|------|-------------|
| `needle_tracked.mp4` | Annotated video with tracking overlays |
| `needle_position.png` | X position (mm) vs time (s), with cell boundary markers |
| `needle_velocity.png` | Delta X (mm/frame) vs time (s), smoothed (rolling avg 5 frames) |
| `needle_position.html` | Interactive Plotly version of position graph |
| `needle_velocity.html` | Interactive Plotly version of velocity graph |
| `needle_data.csv` | Raw per-frame logged data |

---

## Dependencies

```
opencv-python
numpy
matplotlib
plotly
scipy
```

Install via: `pip install opencv-python numpy matplotlib plotly scipy`

---

## Error Handling

- If blob detection finds no moving blob in a frame: Kalman filter uses prediction only (needle position extrapolated), frame logged with `cell_number = -1`
- If ORB finds insufficient keypoints for stabilization: frame used as-is without warp
- Calibration requires exactly 4 clicks; script validates and shows preview before starting

---

## Success Criteria

1. Needle tip tracked visually through full video with minimal drift
2. Cell crossings correctly identified (right→left) across the top row
3. Position graph clearly shows step-like progression across cells
4. Velocity graph shows consistent left-directed motion with dips at hesitation points
5. Output video playable and overlays readable
