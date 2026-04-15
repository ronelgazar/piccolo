# Needle Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python script that tracks a surgical needle across a calibration grid, identifies which 0.1mm cell it occupies, and produces an annotated video plus position/velocity graphs.

**Architecture:** Single script `needle_tracker.py` with clearly separated functions/classes. Phase 1 = interactive calibration (4 clicks). Phase 2 = per-frame pipeline: ORB stabilize → MOG2 subtract → blob detect → Kalman correct → annotate + log. Phase 3 = graph generation.

**Tech Stack:** Python 3.10+, opencv-python, numpy, matplotlib, plotly, pandas, scipy (not used directly — numpy Kalman is sufficient)

---

## File Structure

```
c:/Users/ronelgazar/Documents/University/year2/Semester1/Research/SurgeryRobot/
├── needle_tracker.py          # main script (all logic)
└── tests/
    └── test_needle_tracker.py # unit tests for pure functions
```

Output files (written next to the input video in the piccolo/ folder):
```
needle_tracked.mp4
needle_position.png
needle_velocity.png
needle_position.html
needle_velocity.html
needle_data.csv
```

---

## Task 1: Install Dependencies

**Files:**
- No files created

- [ ] **Step 1: Install packages**

```bash
pip install opencv-python numpy matplotlib plotly pandas
```

Expected output: `Successfully installed ...` (no errors)

- [ ] **Step 2: Verify imports work**

```bash
python -c "import cv2, numpy, matplotlib, plotly, pandas; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Create test directory and empty test file**

```bash
mkdir -p "c:/Users/ronelgazar/Documents/University/year2/Semester1/Research/SurgeryRobot/tests"
touch "c:/Users/ronelgazar/Documents/University/year2/Semester1/Research/SurgeryRobot/tests/test_needle_tracker.py"
```

- [ ] **Step 4: Create empty needle_tracker.py with imports**

Create `c:/Users/ronelgazar/Documents/University/year2/Semester1/Research/SurgeryRobot/needle_tracker.py`:

```python
"""Needle tracker — microscope video analysis tool."""
import cv2
import csv
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from pathlib import Path
from collections import deque

CELL_SIZE_MM = 0.1   # each grid cell is 0.1 mm × 0.1 mm
TRAJECTORY_LEN = 60  # frames to keep in trajectory trail

VIDEO_PATH = (
    r"c:\Users\ronelgazar\Documents\University\year2\Semester1"
    r"\Research\SurgeryRobot\piccolo"
    r"\WhatsApp Video 2026-04-12 at 13.46.50 (online-video-cutter.com).mp4"
)
```

- [ ] **Step 5: Commit scaffold**

```bash
cd "c:/Users/ronelgazar/Documents/University/year2/Semester1/Research/SurgeryRobot"
git add needle_tracker.py tests/test_needle_tracker.py
git commit -m "feat: scaffold needle tracker script and test file"
```

---

## Task 2: Pure Grid Math Functions (TDD)

**Files:**
- Modify: `needle_tracker.py` — add 4 pure functions after the constants
- Modify: `tests/test_needle_tracker.py` — add tests

These functions have no OpenCV dependency and are fully unit-testable.

- [ ] **Step 1: Write failing tests**

In `tests/test_needle_tracker.py`:

```python
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from needle_tracker import (
    pixel_to_mm,
    cell_number_from_x,
    compute_cell_count,
    compute_pixels_per_mm,
)


def test_compute_pixels_per_mm_horizontal():
    # Two points 50px apart = 0.1mm → 500 px/mm
    result = compute_pixels_per_mm((100, 200), (150, 200))
    assert abs(result - 500.0) < 0.01


def test_compute_pixels_per_mm_uses_horizontal_distance():
    # Vertical distance should be ignored when horizontal is nonzero
    result = compute_pixels_per_mm((0, 0), (50, 999))
    assert abs(result - 500.0) < 0.01


def test_pixel_to_mm_at_right_edge():
    # At right edge: 0 mm traveled
    assert pixel_to_mm(1000, row_x_right=1000, pixels_per_mm=500) == 0.0


def test_pixel_to_mm_moved_left():
    # 500px to the left of right edge = 1.0mm
    assert abs(pixel_to_mm(500, row_x_right=1000, pixels_per_mm=500) - 1.0) < 1e-9


def test_cell_number_rightmost():
    # Needle at right edge → cell 1
    assert cell_number_from_x(x_pixel=1000, row_x_right=1000, cell_width_px=50) == 1


def test_cell_number_second_cell():
    # 60px from right edge, cell width=50px → cell 2
    assert cell_number_from_x(x_pixel=940, row_x_right=1000, cell_width_px=50) == 2


def test_cell_number_before_row():
    # To the right of the row → 0
    assert cell_number_from_x(x_pixel=1010, row_x_right=1000, cell_width_px=50) == 0


def test_compute_cell_count():
    # Row width = 500px, cell width = 50px → 10 cells
    assert compute_cell_count(row_x_left=500, row_x_right=1000, cell_width_px=50) == 10


def test_compute_cell_count_rounds():
    # 505px / 50px ≈ 10.1 → rounds to 10
    assert compute_cell_count(row_x_left=495, row_x_right=1000, cell_width_px=50) == 10
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd "c:/Users/ronelgazar/Documents/University/year2/Semester1/Research/SurgeryRobot"
python -m pytest tests/test_needle_tracker.py -v
```

Expected: `ImportError: cannot import name 'pixel_to_mm'`

- [ ] **Step 3: Implement the four functions**

Append to `needle_tracker.py` after the constants:

```python
# ─── Grid Math ────────────────────────────────────────────────────────────────

def compute_pixels_per_mm(p1: tuple, p2: tuple,
                           cell_size_mm: float = CELL_SIZE_MM) -> float:
    """Pixels per mm from two adjacent grid line clicks (one cell apart).
    Uses horizontal distance; falls back to vertical if horizontal is zero.
    """
    dist_px = abs(p2[0] - p1[0])
    if dist_px == 0:
        dist_px = abs(p2[1] - p1[1])
    return dist_px / cell_size_mm


def pixel_to_mm(x_pixel: float, row_x_right: float,
                pixels_per_mm: float) -> float:
    """Distance traveled from right edge of top row in mm (positive = moved left)."""
    return (row_x_right - x_pixel) / pixels_per_mm


def cell_number_from_x(x_pixel: float, row_x_right: float,
                        cell_width_px: float) -> int:
    """Cell number (1 = rightmost) for needle tip X pixel.
    Returns 0 if needle is to the right of the row (not yet entered).
    """
    offset_px = row_x_right - x_pixel
    if offset_px < 0:
        return 0
    return int(offset_px / cell_width_px) + 1


def compute_cell_count(row_x_left: float, row_x_right: float,
                        cell_width_px: float) -> int:
    """Number of cells in the top row."""
    return max(1, round((row_x_right - row_x_left) / cell_width_px))
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
python -m pytest tests/test_needle_tracker.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add needle_tracker.py tests/test_needle_tracker.py
git commit -m "feat: add grid math pure functions with tests"
```

---

## Task 3: KalmanTracker Class (TDD)

**Files:**
- Modify: `needle_tracker.py` — add `KalmanTracker` class
- Modify: `tests/test_needle_tracker.py` — add Kalman tests

- [ ] **Step 1: Write failing tests**

Append to `tests/test_needle_tracker.py`:

```python
from needle_tracker import KalmanTracker


def test_kalman_initializes_on_first_update():
    kt = KalmanTracker(dt=1.0)
    pos = kt.update((100.0, 200.0))
    assert abs(pos[0] - 100.0) < 1.0
    assert abs(pos[1] - 200.0) < 1.0


def test_kalman_predict_returns_tuple():
    kt = KalmanTracker(dt=1.0)
    kt.update((100.0, 200.0))
    pred = kt.predict()
    assert isinstance(pred, tuple)
    assert len(pred) == 2


def test_kalman_predict_before_init_returns_none():
    kt = KalmanTracker(dt=1.0)
    assert kt.predict() is None


def test_kalman_tracks_constant_motion():
    kt = KalmanTracker(dt=1.0)
    # Feed measurements moving left at 10px/frame
    for i in range(10):
        kt.update((500.0 - i * 10, 100.0))
    pos = kt.update((400.0, 100.0))
    # Should be close to 400
    assert abs(pos[0] - 400.0) < 5.0


def test_kalman_handles_noisy_measurement():
    import random
    random.seed(42)
    kt = KalmanTracker(dt=1.0)
    true_x = 500.0
    for _ in range(20):
        noisy = true_x + random.gauss(0, 3.0)
        kt.update((noisy, 100.0))
    pos = kt.update((true_x, 100.0))
    # Should be within 5px of true position after settling
    assert abs(pos[0] - true_x) < 5.0
```

- [ ] **Step 2: Run tests — expect failure**

```bash
python -m pytest tests/test_needle_tracker.py -v -k kalman
```

Expected: `ImportError: cannot import name 'KalmanTracker'`

- [ ] **Step 3: Implement KalmanTracker**

Append to `needle_tracker.py`:

```python
# ─── Kalman Tracker ───────────────────────────────────────────────────────────

class KalmanTracker:
    """2D constant-velocity Kalman filter for needle tip smoothing.

    State vector: [x, y, vx, vy]
    """

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        # State transition matrix
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=float)
        # Measurement matrix (we observe x, y only)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * 0.5   # process noise (position + velocity)
        self.R = np.eye(2) * 8.0   # measurement noise (pixel uncertainty)
        self.P = np.eye(4) * 200.0 # initial covariance (high uncertainty)
        self._x = None             # state [x, y, vx, vy]

    @property
    def initialized(self) -> bool:
        return self._x is not None

    def predict(self) -> tuple | None:
        """Advance state one step. Returns predicted (x, y) or None if not initialized."""
        if not self.initialized:
            return None
        self._x = self.F @ self._x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (float(self._x[0]), float(self._x[1]))

    def update(self, measurement: tuple) -> tuple:
        """Kalman correct step. Initializes on first call. Returns corrected (x, y)."""
        z = np.array(measurement, dtype=float)
        if not self.initialized:
            self._x = np.array([z[0], z[1], 0.0, 0.0])
            return (float(self._x[0]), float(self._x[1]))
        self.predict()
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innovation = z - self.H @ self._x
        self._x = self._x + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return (float(self._x[0]), float(self._x[1]))
```

- [ ] **Step 4: Run all tests — expect pass**

```bash
python -m pytest tests/test_needle_tracker.py -v
```

Expected: `14 passed`

- [ ] **Step 5: Commit**

```bash
git add needle_tracker.py tests/test_needle_tracker.py
git commit -m "feat: add KalmanTracker with tests"
```

---

## Task 4: Calibration UI

**Files:**
- Modify: `needle_tracker.py` — add `run_calibration()` function

No automated test (requires interactive OpenCV window). Tested manually in Task 8.

- [ ] **Step 1: Implement run_calibration()**

Append to `needle_tracker.py`:

```python
# ─── Calibration UI ───────────────────────────────────────────────────────────

def run_calibration(frame: np.ndarray) -> dict:
    """Interactive 4-click calibration.

    Click order:
      1. Left vertical grid line (calibration point A)
      2. Adjacent right vertical grid line (= 0.1mm away, calibration point B)
      3. Leftmost extent of the top row
      4. Rightmost extent of the top row

    Returns calibration dict with keys:
      pixels_per_mm, cell_width_px, row_x_left, row_x_right,
      row_y, n_cells
    """
    INSTRUCTIONS = [
        "1/4: Click LEFT adjacent grid line (calibration A)",
        "2/4: Click RIGHT adjacent grid line, exactly 0.1mm away (calibration B)",
        "3/4: Click LEFTMOST point of the top row",
        "4/4: Click RIGHTMOST point of the top row",
    ]
    WIN = "Calibration — follow prompts, ESC to quit"
    clicks: list = []
    display = frame.copy()

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(display, str(len(clicks)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    while len(clicks) < 4:
        overlay = display.copy()
        step = len(clicks)
        cv2.putText(overlay, INSTRUCTIONS[step], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        cv2.putText(overlay, INSTRUCTIONS[step], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
        cv2.imshow(WIN, overlay)
        key = cv2.waitKey(30)
        if key == 27:
            cv2.destroyWindow(WIN)
            raise RuntimeError("Calibration cancelled by user.")

    cv2.destroyWindow(WIN)

    p_a, p_b, row_left_pt, row_right_pt = clicks

    cell_width_px = abs(p_b[0] - p_a[0])
    if cell_width_px == 0:
        cell_width_px = abs(p_b[1] - p_a[1])  # fallback to vertical
    pixels_per_mm = cell_width_px / CELL_SIZE_MM

    row_x_left = min(row_left_pt[0], row_right_pt[0])
    row_x_right = max(row_left_pt[0], row_right_pt[0])
    row_y = (p_a[1] + p_b[1]) // 2
    n_cells = compute_cell_count(row_x_left, row_x_right, cell_width_px)

    print(f"  pixels/mm   = {pixels_per_mm:.1f}")
    print(f"  cell width  = {cell_width_px} px = {CELL_SIZE_MM} mm")
    print(f"  top row     = x[{row_x_left}..{row_x_right}], y≈{row_y}")
    print(f"  cells in row = {n_cells}")

    return {
        "pixels_per_mm": pixels_per_mm,
        "cell_width_px": float(cell_width_px),
        "row_x_left": row_x_left,
        "row_x_right": row_x_right,
        "row_y": row_y,
        "n_cells": n_cells,
    }
```

- [ ] **Step 2: Commit**

```bash
git add needle_tracker.py
git commit -m "feat: add interactive calibration UI"
```

---

## Task 5: ORBStabilizer + NeedleDetector

**Files:**
- Modify: `needle_tracker.py` — add two classes

- [ ] **Step 1: Implement ORBStabilizer**

Append to `needle_tracker.py`:

```python
# ─── ORB Stabilizer ───────────────────────────────────────────────────────────

class ORBStabilizer:
    """Stabilizes frames by warping them back to a reference frame via ORB matching.

    Handles mild camera shake/drift. Falls back to the raw frame if too few
    keypoints are found.
    """

    def __init__(self, reference_frame: np.ndarray, max_features: int = 500):
        self._orb = cv2.ORB_create(max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        self._ref_kp, self._ref_des = self._orb.detectAndCompute(ref_gray, None)
        self._h, self._w = reference_frame.shape[:2]

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """Return frame warped to align with the reference. Returns original on failure."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 4 or self._ref_des is None:
            return frame
        matches = self._matcher.match(self._ref_des, des)
        matches = sorted(matches, key=lambda m: m.distance)[:50]
        if len(matches) < 4:
            return frame
        src = np.float32([self._ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
        if M is None:
            return frame
        return cv2.warpPerspective(frame, M, (self._w, self._h))
```

- [ ] **Step 2: Implement NeedleDetector**

Append to `needle_tracker.py`:

```python
# ─── Needle Detector ─────────────────────────────────────────────────────────

class NeedleDetector:
    """Detects the needle tip using MOG2 background subtraction + blob analysis.

    Returns the centroid of the largest foreground blob, which tracks the
    moving needle. Uses a slow learning rate so the static grid is treated
    as background after a few frames.
    """

    def __init__(self, min_area: int = 30, max_area: int = 8000):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        self._min_area = min_area
        self._max_area = max_area
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray,
               learning_rate: float = 0.001) -> tuple | None:
        """Return (x, y) centroid of needle tip, or None if not detected."""
        fg = self._bg.apply(frame, learningRate=learning_rate)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours
                 if self._min_area < cv2.contourArea(c) < self._max_area]
        if not valid:
            return None
        largest = max(valid, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        return (M["m10"] / M["m00"], M["m01"] / M["m00"])
```

- [ ] **Step 3: Commit**

```bash
git add needle_tracker.py
git commit -m "feat: add ORBStabilizer and NeedleDetector classes"
```

---

## Task 6: Frame Annotator

**Files:**
- Modify: `needle_tracker.py` — add `annotate_frame()` function

- [ ] **Step 1: Implement annotate_frame()**

Append to `needle_tracker.py`:

```python
# ─── Frame Annotator ─────────────────────────────────────────────────────────

def annotate_frame(
    frame: np.ndarray,
    tip: tuple | None,
    trajectory: deque,
    calib: dict,
    cell_number: int,
    frame_idx: int,
    timestamp: float,
    x_mm: float | None,
) -> np.ndarray:
    """Draw tracking overlays onto a copy of frame and return it."""
    out = frame.copy()
    rx_l = calib["row_x_left"]
    rx_r = calib["row_x_right"]
    row_y = calib["row_y"]
    cell_w = calib["cell_width_px"]
    n_cells = calib["n_cells"]
    half_h = max(4, int(cell_w // 2))

    # Draw top row outline
    cv2.rectangle(out,
                  (rx_l, row_y - half_h), (rx_r, row_y + half_h),
                  (180, 180, 180), 1)

    # Draw cell dividers
    for i in range(1, n_cells):
        cx = int(rx_r - i * cell_w)
        cv2.line(out, (cx, row_y - half_h), (cx, row_y + half_h),
                 (140, 140, 140), 1)

    # Highlight current cell in green
    if 1 <= cell_number <= n_cells:
        c_right = int(rx_r - (cell_number - 1) * cell_w)
        c_left  = int(c_right - cell_w)
        cv2.rectangle(out,
                      (c_left, row_y - half_h), (c_right, row_y + half_h),
                      (0, 180, 0), -1)
        cv2.rectangle(out,
                      (c_left, row_y - half_h), (c_right, row_y + half_h),
                      (0, 255, 0), 1)

    # Draw trajectory trail
    pts = [p for p in trajectory if p is not None]
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        color = (int(255 * alpha), int(140 * alpha), 0)
        cv2.line(out, pts[i - 1], pts[i], color, 1)

    # Draw crosshair on needle tip
    if tip:
        tx, ty = int(tip[0]), int(tip[1])
        cv2.drawMarker(out, (tx, ty), (0, 0, 255),
                       cv2.MARKER_CROSS, 22, 2)

    # Text overlay (white + black shadow for readability)
    x_str = f"{x_mm:.3f} mm" if x_mm is not None else "---"
    cell_str = str(cell_number) if cell_number > 0 else "---"
    lines = [
        f"Frame {frame_idx}",
        f"t = {timestamp:.2f} s",
        f"X  = {x_str}",
        f"Cell {cell_str} / {n_cells}",
    ]
    for i, line in enumerate(lines):
        y_pos = 28 + i * 24
        cv2.putText(out, line, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(out, line, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1)

    return out
```

- [ ] **Step 2: Commit**

```bash
git add needle_tracker.py
git commit -m "feat: add frame annotator with trajectory and cell highlight"
```

---

## Task 7: Main Tracking Loop

**Files:**
- Modify: `needle_tracker.py` — add `run_tracking()` function

- [ ] **Step 1: Implement run_tracking()**

Append to `needle_tracker.py`:

```python
# ─── Tracking Loop ───────────────────────────────────────────────────────────

FIELDNAMES = [
    "frame_idx", "timestamp_sec",
    "x_pixel", "y_pixel",
    "x_mm", "delta_x_mm",
    "cell_number",
]


def run_tracking(video_path: Path, calib: dict, output_dir: Path) -> Path:
    """Run per-frame tracking pipeline. Returns path to CSV output file.

    Pipeline per frame:
      raw frame → ORB stabilize → MOG2 detect → Kalman correct
      → annotate → write video + CSV + live display
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame for stabilizer reference
    ret, ref_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    stabilizer = ORBStabilizer(ref_frame)
    detector   = NeedleDetector()
    tracker    = KalmanTracker(dt=1.0 / fps)

    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(
        str(output_dir / "needle_tracked.mp4"), fourcc, fps, (w, h)
    )

    csv_path = output_dir / "needle_data.csv"
    trajectory: deque = deque(maxlen=TRAJECTORY_LEN)
    prev_x_mm: float | None = None
    frame_idx = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            # --- stabilize ---
            stable = stabilizer.stabilize(frame)

            # --- detect ---
            raw_tip = detector.detect(stable)

            # --- Kalman ---
            if raw_tip is not None:
                tip = tracker.update(raw_tip)
                cell = cell_number_from_x(
                    tip[0], calib["row_x_right"], calib["cell_width_px"]
                )
                x_mm = pixel_to_mm(
                    tip[0], calib["row_x_right"], calib["pixels_per_mm"]
                )
            else:
                tip = tracker.predict()   # extrapolate
                cell = -1                 # no detection this frame
                x_mm = None

            delta_x = None
            if x_mm is not None and prev_x_mm is not None:
                delta_x = x_mm - prev_x_mm
            if x_mm is not None:
                prev_x_mm = x_mm

            # --- log ---
            writer.writerow({
                "frame_idx":    frame_idx,
                "timestamp_sec": round(timestamp, 4),
                "x_pixel":      round(tip[0], 1) if tip else "",
                "y_pixel":      round(tip[1], 1) if tip else "",
                "x_mm":         round(x_mm, 4)   if x_mm   is not None else "",
                "delta_x_mm":   round(delta_x, 4) if delta_x is not None else "",
                "cell_number":  cell,
            })

            trajectory.append(
                (int(tip[0]), int(tip[1])) if tip else None
            )

            # --- annotate + output ---
            annotated = annotate_frame(
                frame, tip, trajectory, calib,
                cell, frame_idx, timestamp, x_mm
            )
            video_out.write(annotated)
            cv2.imshow("Needle Tracker — ESC to stop", annotated)
            if cv2.waitKey(1) == 27:
                print("Stopped early by user.")
                break

            if frame_idx % 60 == 0:
                print(f"  frame {frame_idx}/{total}  t={timestamp:.1f}s  "
                      f"x={x_mm:.3f}mm  cell={cell}" if x_mm else
                      f"  frame {frame_idx}/{total}  t={timestamp:.1f}s  (no detection)")
            frame_idx += 1

    cap.release()
    video_out.release()
    cv2.destroyAllWindows()
    print(f"Video saved → {output_dir / 'needle_tracked.mp4'}")
    print(f"CSV saved  → {csv_path}")
    return csv_path
```

- [ ] **Step 2: Commit**

```bash
git add needle_tracker.py
git commit -m "feat: add main tracking loop with live display and CSV logging"
```

---

## Task 8: Graph Generator

**Files:**
- Modify: `needle_tracker.py` — add `generate_graphs()` function

- [ ] **Step 1: Implement generate_graphs()**

Append to `needle_tracker.py`:

```python
# ─── Graph Generator ─────────────────────────────────────────────────────────

def generate_graphs(csv_path: Path, output_dir: Path) -> None:
    """Generate position and velocity graphs as PNG (matplotlib) and HTML (plotly).

    Saves to output_dir:
      needle_position.png / needle_position.html
      needle_velocity.png / needle_velocity.html
    Opens both HTML files in the default browser.
    """
    df = pd.read_csv(csv_path)
    # Keep only frames where detection was successful
    df = df[df["x_mm"].notna() & (df["cell_number"] >= 0)].copy()
    df["x_mm"] = pd.to_numeric(df["x_mm"])
    df["delta_x_mm"] = pd.to_numeric(df["delta_x_mm"])
    df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"])

    t  = df["timestamp_sec"].values
    x  = df["x_mm"].values
    dx = df["delta_x_mm"].rolling(5, center=True, min_periods=1).mean().values

    # Cell crossing events
    df["prev_cell"] = df["cell_number"].shift(1)
    crossings = df[df["cell_number"] != df["prev_cell"]].copy()
    crossings = crossings[crossings["cell_number"] > 0]

    # ── Position PNG ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t, x, color="steelblue", linewidth=1, label="X position (mm)")
    for _, row in crossings.iterrows():
        ax.axvline(row["timestamp_sec"], color="gray",
                   linestyle="--", alpha=0.45, linewidth=0.8)
        ax.text(row["timestamp_sec"], ax.get_ylim()[1] * 0.95,
                f"C{int(row['cell_number'])}", ha="center",
                fontsize=7, color="gray")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("X from right edge (mm)")
    ax.set_title("Needle Tip X Position over Time")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pos_png = output_dir / "needle_position.png"
    fig.savefig(pos_png, dpi=150)
    plt.close(fig)
    print(f"Saved → {pos_png}")

    # ── Position HTML ─────────────────────────────────────────────────────────
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=t, y=x, mode="lines", name="X position (mm)"))
    for _, row in crossings.iterrows():
        fig_p.add_vline(x=row["timestamp_sec"], line_dash="dash",
                        line_color="gray",
                        annotation_text=f"C{int(row['cell_number'])}",
                        annotation_position="top")
    fig_p.update_layout(
        title="Needle Tip X Position over Time",
        xaxis_title="Time (s)",
        yaxis_title="X from right edge (mm)",
    )
    pos_html = output_dir / "needle_position.html"
    pio.write_html(fig_p, str(pos_html), auto_open=True)
    print(f"Saved → {pos_html}")

    # ── Velocity PNG ──────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(13, 5))
    ax2.plot(t, dx, color="darkorange", linewidth=1, label="ΔX (mm/frame, smoothed)")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("ΔX per frame (mm)")
    ax2.set_title("Needle Velocity (ΔX) over Time")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    vel_png = output_dir / "needle_velocity.png"
    fig2.savefig(vel_png, dpi=150)
    plt.close(fig2)
    print(f"Saved → {vel_png}")

    # ── Velocity HTML ─────────────────────────────────────────────────────────
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=t, y=dx, mode="lines",
                               name="ΔX mm/frame (smoothed)"))
    fig_v.add_hline(y=0, line_color="black", line_width=0.5)
    fig_v.update_layout(
        title="Needle Velocity (ΔX) over Time",
        xaxis_title="Time (s)",
        yaxis_title="ΔX per frame (mm)",
    )
    vel_html = output_dir / "needle_velocity.html"
    pio.write_html(fig_v, str(vel_html), auto_open=True)
    print(f"Saved → {vel_html}")
```

- [ ] **Step 2: Commit**

```bash
git add needle_tracker.py
git commit -m "feat: add graph generator (matplotlib PNG + plotly HTML)"
```

---

## Task 9: main() Entry Point + End-to-End Verification

**Files:**
- Modify: `needle_tracker.py` — add `main()` function at end of file

- [ ] **Step 1: Implement main()**

Append to `needle_tracker.py`:

```python
# ─── Entry Point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Needle tracker — microscope video analysis"
    )
    parser.add_argument(
        "video", nargs="?", default=VIDEO_PATH,
        help="Path to input video (default: hardcoded WhatsApp video)"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found:\n  {video_path}")
        sys.exit(1)

    output_dir = video_path.parent
    print(f"Input  : {video_path}")
    print(f"Output : {output_dir}")

    # ── Phase 1: Calibration ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("Error: cannot read first frame of video.")
        sys.exit(1)

    print("\n=== CALIBRATION ===")
    print("An OpenCV window will open. Click 4 points as instructed.")
    calib = run_calibration(first_frame)

    # ── Phase 2: Tracking ────────────────────────────────────────────────────
    print("\n=== TRACKING (press ESC in the window to stop early) ===")
    csv_path = run_tracking(video_path, calib, output_dir)

    # ── Phase 3: Graphs ──────────────────────────────────────────────────────
    print("\n=== GENERATING GRAPHS ===")
    generate_graphs(csv_path, output_dir)

    print("\nDone! Output files written to:", output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full unit test suite**

```bash
cd "c:/Users/ronelgazar/Documents/University/year2/Semester1/Research/SurgeryRobot"
python -m pytest tests/test_needle_tracker.py -v
```

Expected: `14 passed`

- [ ] **Step 3: Dry-run import check (no video required)**

```bash
python -c "import needle_tracker; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 4: End-to-end run with the actual video**

```bash
python needle_tracker.py
```

Expected sequence:
1. Calibration window opens on first frame — click 4 points as prompted
2. Terminal prints calibration values (px/mm, n_cells)
3. Tracking window opens showing live annotated video
4. Terminal prints per-60-frame progress lines
5. After video ends (or ESC): `Video saved → ...`, `CSV saved → ...`
6. Graphs generated: `Saved → needle_position.png` × 2, `Saved → needle_velocity.png` × 2
7. Browser opens with two interactive HTML graphs

If tracking looks wrong (needle not detected, or wrong blob):
- Increase/decrease `NeedleDetector(min_area=..., max_area=...)` to match needle size
- Adjust `varThreshold` in `createBackgroundSubtractorMOG2` (lower = more sensitive)

- [ ] **Step 5: Final commit**

```bash
git add needle_tracker.py
git commit -m "feat: add main() entry point — needle tracker complete"
```

---

## Tuning Reference

If detection is poor after running:

| Symptom | Fix |
|---------|-----|
| No blobs detected | Lower `min_area` or `varThreshold` |
| Grid lines detected as needle | Raise `min_area`, lower `max_area` |
| Needle position jumps | Raise `R` in `KalmanTracker` (trust measurements less) |
| Kalman lags behind needle | Lower `R` or raise `Q` |
| Camera shake not corrected | Check ORB: raise `max_features` to 1000 |
| Wrong blob tracked | Raise `min_area` to exclude small noise |
