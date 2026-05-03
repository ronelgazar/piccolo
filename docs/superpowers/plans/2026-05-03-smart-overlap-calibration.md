# Smart overlap calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 5-phase physical calibration wizard with a single-screen "Smart overlap calibration" panel that detects matching patterns between L and R eyes, draws coloured numbered markers connected by threads, and shows live diagnostic readouts. The "Apply detected scale" one-click button is preserved.

**Architecture:** Bottom-up. (1) Extract a shared `StereoFeatureMatcher` and `theil_sen` from `StereoAligner` into a new `src/stereo_matching.py` module — `StereoAligner` is refactored to call it (zero behaviour change). (2) Build the pure-Python `SmartOverlapAnalyzer` in `src/smart_overlap.py` with two modes (chessboard, live) using the shared matcher. (3) Wrap it in a `QThread` worker. (4) Replace the wizard `QGroupBox` in `calibration_tab.py`. (5) Delete the dead wizard files.

**Tech Stack:** Python 3.11+, NumPy, OpenCV (SIFT, Canny, FLANN, F-matrix RANSAC, drawing primitives), PyQt6, pytest. Existing helpers reused: `physical_grid_calibration.detect_grid` (chessboard mode) and `physical_grid_calibration.estimate_square_px` (zoom ratio in chessboard mode).

**Spec:** [docs/superpowers/specs/2026-05-03-smart-overlap-calibration-design.md](../specs/2026-05-03-smart-overlap-calibration-design.md)

---

## File structure

**Create:**
- `src/stereo_matching.py` — `MatchResult` dataclass, `StereoFeatureMatcher` class, `theil_sen` function
- `src/smart_overlap.py` — `OverlapPair`, `OverlapMetrics`, `SmartOverlapAnalyzer`, `render_overlay`
- `src/ui/smart_overlap_worker.py` — `SmartOverlapResult` payload, `SmartOverlapWorker(QThread)`
- `tests/test_stereo_matching.py` — covers `theil_sen` and `StereoFeatureMatcher`
- `tests/test_smart_overlap.py` — covers the analyzer and renderer

**Modify:**
- `src/stereo_align.py` — replace private match/regression helpers with calls into `stereo_matching` (no behaviour change)
- `src/config.py` — add `SmartOverlapCfg` field on `PiccoloCfg`
- `src/config_state.py` — persist new fields under `calibration_state`
- `config.yaml` — add `smart_overlap:` block; add `smart_overlap_mode` and `smart_overlap_pair_count` under `calibration_state`
- `src/ui/calibration_tab.py` — replace wizard `QGroupBox` with smart-overlap `QGroupBox`

**Delete:**
- `src/physical_cal.py`
- `src/ui/physical_cal_worker.py`
- `tests/test_physical_cal.py`

---

## Task 1: Extract `theil_sen` into `src/stereo_matching.py`

**Files:**
- Create: `src/stereo_matching.py`
- Create: `tests/test_stereo_matching.py`

The Theil-Sen robust regression already exists as a private static method on `StereoAligner` ([stereo_align.py:506-557](../../src/stereo_align.py#L506-L557)). We pull it out as a module-level function so `SmartOverlapAnalyzer` can also use it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stereo_matching.py
import numpy as np

from src.stereo_matching import theil_sen


def test_theil_sen_recovers_slope_and_intercept_on_clean_line():
    rng = np.random.default_rng(0)
    x = rng.uniform(-100, 100, size=80)
    true_slope = 0.05
    true_intercept = 1.2
    y = true_slope * x + true_intercept
    slope, intercept, rms = theil_sen(x, y)
    assert abs(slope - true_slope) < 1e-6
    assert abs(intercept - true_intercept) < 1e-6
    assert rms < 1e-6


def test_theil_sen_robust_to_30_pct_outliers():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.uniform(-100, 100, size=n)
    y = 0.02 * x + 0.5
    # Corrupt 30 % of points with very large residuals
    mask = rng.random(n) < 0.30
    y[mask] += rng.uniform(-50, 50, size=mask.sum())
    slope, intercept, _ = theil_sen(x, y)
    assert abs(slope - 0.02) < 0.01
    assert abs(intercept - 0.5) < 1.0


def test_theil_sen_zero_slope_when_no_x_spread():
    x = np.full(20, 5.0)
    y = np.linspace(0, 10, 20)
    slope, intercept, _ = theil_sen(x, y)
    # All x identical → no useful slope information → fall back to slope 0
    assert slope == 0.0
    assert abs(intercept - float(np.median(y))) < 1e-6
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_stereo_matching.py -v`
Expected: ImportError — `src.stereo_matching` does not exist yet.

- [ ] **Step 3: Implement `theil_sen` in the new module**

Create `src/stereo_matching.py`:

```python
"""Shared stereo feature matching + Theil-Sen regression.

Extracted from StereoAligner so SmartOverlapAnalyzer can reuse the same
matching pipeline. No behaviour change versus the original private methods
in stereo_align.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def theil_sen(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Theil-Sen robust linear fit: y = slope*x + intercept.

    Returns (slope, intercept, rms_residual). For n <= 200 uses all unique
    pairs; above that, samples 50_000 random pairs to keep runtime bounded.
    Tolerates ~29 % gross outliers.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    min_sep = 30.0  # minimum x-separation for a useful slope pair

    if n <= 200:
        ii, jj = np.triu_indices(n, k=1)
        dx = x[jj] - x[ii]
        valid = np.abs(dx) > min_sep
        if valid.sum() < 3:
            slope = 0.0
        else:
            slopes = (y[jj[valid]] - y[ii[valid]]) / dx[valid]
            slope = float(np.median(slopes))
    else:
        rng = np.random.default_rng(42)
        n_pairs = min(50_000, n * (n - 1) // 2)
        ai = rng.integers(0, n, n_pairs)
        bi = rng.integers(0, n, n_pairs)
        keep = ai != bi
        ai, bi = ai[keep], bi[keep]
        dx = x[bi] - x[ai]
        valid = np.abs(dx) > min_sep
        if valid.sum() < 3:
            slope = 0.0
        else:
            slopes = (y[bi[valid]] - y[ai[valid]]) / dx[valid]
            slope = float(np.median(slopes))

    intercepts = y - slope * x
    intercept = float(np.median(intercepts))
    residuals = y - (intercept + slope * x)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return slope, intercept, rms
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_stereo_matching.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stereo_matching.py tests/test_stereo_matching.py
git commit -m "refactor: extract theil_sen into src/stereo_matching.py"
```

---

## Task 2: Add `StereoFeatureMatcher` to `src/stereo_matching.py`

**Files:**
- Modify: `src/stereo_matching.py`
- Modify: `tests/test_stereo_matching.py`

The matcher pipeline lives inside `StereoAligner` private methods. We extract it as a standalone class with a `match(gray_l, gray_r) -> MatchResult` API.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stereo_matching.py`:

```python
import cv2

from src.stereo_matching import StereoFeatureMatcher, MatchResult


def _synthetic_feature_pair(shift_y: int = 0):
    """Build a 320x240 grey pair with strong corner features."""
    img_l = np.full((240, 320), 128, dtype=np.uint8)
    rng = np.random.default_rng(7)
    # Sprinkle high-contrast 6x6 squares to give SIFT something to match.
    for _ in range(60):
        x = int(rng.integers(20, 300))
        y = int(rng.integers(20, 220))
        img_l[y:y + 6, x:x + 6] = 255
    img_r = np.roll(img_l, shift_y, axis=0)
    return img_l, img_r


def test_feature_matcher_returns_inliers_for_feature_rich_pair():
    img_l, img_r = _synthetic_feature_pair(shift_y=3)
    m = StereoFeatureMatcher(max_features=500, match_ratio=0.75, ransac_thresh=2.0,
                             frame_w=320, frame_h=240)
    result = m.match(img_l, img_r)
    assert isinstance(result, MatchResult)
    assert result.pts_l.shape == result.pts_r.shape
    assert result.pts_l.shape[1] == 2
    assert result.pts_l.shape[0] >= 8


def test_feature_matcher_returns_empty_for_blank_pair():
    blank = np.full((240, 320), 128, dtype=np.uint8)
    m = StereoFeatureMatcher(max_features=500, match_ratio=0.75, ransac_thresh=2.0,
                             frame_w=320, frame_h=240)
    result = m.match(blank, blank.copy())
    assert result.pts_l.shape[0] == 0
    assert result.pts_r.shape[0] == 0
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_stereo_matching.py::test_feature_matcher_returns_inliers_for_feature_rich_pair -v`
Expected: ImportError on `StereoFeatureMatcher` / `MatchResult`.

- [ ] **Step 3: Implement `StereoFeatureMatcher`**

Append to `src/stereo_matching.py`:

```python
import cv2


@dataclass(frozen=True)
class MatchResult:
    """Inlier pixel coordinates for a stereo pair after full filtering.

    `pts_l[i]` and `pts_r[i]` are the same scene point seen from each eye.
    Both arrays have shape (N, 2) where N may be 0.
    """
    pts_l: np.ndarray
    pts_r: np.ndarray


class StereoFeatureMatcher:
    """SIFT + cross-check FLANN + grid distribution + F-matrix RANSAC.

    Identical pipeline to the original private methods in StereoAligner.
    Can be used for both auto-alignment (`StereoAligner`) and manual
    diagnostic visualisation (`SmartOverlapAnalyzer`).
    """

    _GRID = 6  # spatial-distribution grid (matches per cell are capped)

    def __init__(
        self,
        max_features: int,
        match_ratio: float,
        ransac_thresh: float,
        frame_w: int,
        frame_h: int,
    ):
        self.match_ratio = float(match_ratio)
        self.ransac_thresh = float(ransac_thresh)
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._sift = cv2.SIFT_create(nfeatures=int(max_features))
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=80)
        self._matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, gray_l: np.ndarray, gray_r: np.ndarray) -> MatchResult:
        gray_l = self._clahe.apply(gray_l)
        gray_r = self._clahe.apply(gray_r)

        kp_l, des_l = self._sift.detectAndCompute(gray_l, None)
        kp_r, des_r = self._sift.detectAndCompute(gray_r, None)
        if des_l is None or des_r is None or len(kp_l) < 10 or len(kp_r) < 10:
            return MatchResult(pts_l=np.empty((0, 2)), pts_r=np.empty((0, 2)))

        pairs = self._cross_check(des_l, des_r)
        if len(pairs) < 8:
            pairs = self._one_way(des_l, des_r)
            if len(pairs) < 8:
                return MatchResult(pts_l=np.empty((0, 2)), pts_r=np.empty((0, 2)))

        pts_l = np.float32([kp_l[a].pt for a, _ in pairs])
        pts_r = np.float32([kp_r[b].pt for _, b in pairs])

        pts_l, pts_r = self._enforce_distribution(pts_l, pts_r)
        if len(pts_l) < 8:
            return MatchResult(pts_l=np.empty((0, 2)), pts_r=np.empty((0, 2)))

        try:
            F, mask = cv2.findFundamentalMat(
                pts_l, pts_r,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
                confidence=0.999,
            )
        except cv2.error:
            F, mask = None, None
        if F is not None and mask is not None:
            inliers = mask.ravel() == 1
            pts_l = pts_l[inliers]
            pts_r = pts_r[inliers]

        return MatchResult(pts_l=pts_l, pts_r=pts_r)

    # --- internals ---------------------------------------------------------

    def _cross_check(self, des_l, des_r):
        try:
            raw_lr = self._matcher.knnMatch(des_l, des_r, k=2)
            raw_rl = self._matcher.knnMatch(des_r, des_l, k=2)
        except cv2.error:
            return []
        good_lr = {m.queryIdx: m.trainIdx
                   for pair in raw_lr if len(pair) == 2
                   for m, n in [pair] if m.distance < self.match_ratio * n.distance}
        good_rl = {m.queryIdx: m.trainIdx
                   for pair in raw_rl if len(pair) == 2
                   for m, n in [pair] if m.distance < self.match_ratio * n.distance}
        return [(l, r) for l, r in good_lr.items() if good_rl.get(r) == l]

    def _one_way(self, des_l, des_r):
        try:
            raw = self._matcher.knnMatch(des_l, des_r, k=2)
        except cv2.error:
            return []
        return [(m.queryIdx, m.trainIdx)
                for pair in raw if len(pair) == 2
                for m, n in [pair] if m.distance < self.match_ratio * n.distance]

    def _enforce_distribution(self, pts_l: np.ndarray, pts_r: np.ndarray):
        g = self._GRID
        cell_w = self.frame_w / g
        cell_h = self.frame_h / g
        buckets: dict[tuple, list[int]] = {}
        for i in range(len(pts_l)):
            gx = min(int(pts_l[i, 0] / cell_w), g - 1)
            gy = min(int(pts_l[i, 1] / cell_h), g - 1)
            buckets.setdefault((gx, gy), []).append(i)
        max_per = max(6, len(pts_l) // (g * g) + 1)
        keep = [i for indices in buckets.values() for i in indices[:max_per]]
        idx = np.array(keep)
        return pts_l[idx], pts_r[idx]
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `pytest tests/test_stereo_matching.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stereo_matching.py tests/test_stereo_matching.py
git commit -m "refactor: add StereoFeatureMatcher to src/stereo_matching.py"
```

---

## Task 3: Migrate `StereoAligner` to use the shared matcher

**Files:**
- Modify: `src/stereo_align.py`

`StereoAligner` keeps the same public API and behaviour. Internally we replace the private match/regression helpers with calls into `stereo_matching`. Existing tests in `tests/test_stereo_processor.py` validate no behaviour change.

- [ ] **Step 1: Capture the current pass-state of the existing test**

Run: `pytest tests/test_stereo_processor.py -v`
Expected: all green. Note the count.

- [ ] **Step 2: Edit `src/stereo_align.py`**

Replace the imports section (top of file) so it imports from the new module:

```python
from .config import AlignmentCfg
from .stereo_matching import StereoFeatureMatcher, theil_sen
```

In `StereoAligner.__init__`, replace the SIFT / FLANN / CLAHE creation with a single `StereoFeatureMatcher`:

```python
self._matcher = StereoFeatureMatcher(
    max_features=cfg.max_features,
    match_ratio=cfg.match_ratio,
    ransac_thresh=cfg.ransac_thresh,
    frame_w=frame_w,
    frame_h=frame_h,
)
```

Delete from `StereoAligner` (entire methods): `_cross_check_match`, `_one_way_match`, `_enforce_distribution`, `_theil_sen`. Also delete the `self._clahe`, `self._sift`, `self._matcher` (FLANN) attributes that are now owned by `StereoFeatureMatcher`.

In `_epipolar_align`, replace the body that did detection + matching + filtering with a single call:

```python
def _epipolar_align(self, gray_l, gray_r, scale):
    result = self._matcher.match(gray_l, gray_r)
    if len(result.pts_l) < self.cfg.min_matches:
        return None
    inv_s = 1.0 / scale
    pts_l = result.pts_l * inv_s
    pts_r = result.pts_r * inv_s

    dx = pts_r[:, 0] - pts_l[:, 0]
    dy = pts_r[:, 1] - pts_l[:, 1]
    max_horiz = self.frame_w * 0.4
    spatial_ok = (
        (dx > -max_horiz) & (dx < max_horiz)
        & (np.abs(dy) < self.cfg.max_correction_px * 1.5)
    )
    pts_l = pts_l[spatial_ok]
    pts_r = pts_r[spatial_ok]
    if len(pts_l) < self.cfg.min_matches:
        return None
    return self._epipolar_regression(pts_l, pts_r, len(pts_l))
```

In `_epipolar_regression`, replace the call to `self._theil_sen(...)` with the module-level helper:

```python
theta, offset, rms = theil_sen(x_centered, vert_diff)
```

Note: the matcher already owns CLAHE, so the existing CLAHE call in `update()` becomes redundant. Replace these two lines in `update()`:

```python
gray_l = self._clahe.apply(gray_l)
gray_r = self._clahe.apply(gray_r)
```

with bare grayscale only (CLAHE happens inside the matcher now):

```python
# (CLAHE moved into StereoFeatureMatcher)
```

(Leave the `cv2.cvtColor` lines that produce gray_l / gray_r unchanged.)

- [ ] **Step 3: Run all tests to confirm no regression**

Run: `pytest tests/ -v`
Expected: same count of green tests as Step 1 (matching tests still pass, new module's tests pass, no new failures).

- [ ] **Step 4: Commit**

```bash
git add src/stereo_align.py
git commit -m "refactor: migrate StereoAligner to shared StereoFeatureMatcher"
```

---

## Task 4: Define `OverlapPair` and `OverlapMetrics` dataclasses

**Files:**
- Create: `src/smart_overlap.py`
- Create: `tests/test_smart_overlap.py`

Pure dataclasses + the threshold helper that decides `align_ok` and `zoom_ok`. No matching logic yet.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smart_overlap.py
import numpy as np

from src.smart_overlap import (
    OverlapPair,
    OverlapMetrics,
    compute_align_ok,
    compute_zoom_ok,
)


def test_overlap_pair_holds_indices_color_and_xy():
    p = OverlapPair(index=2, color=(255, 100, 50), left_xy=(10.0, 20.0), right_xy=(11.0, 21.0))
    assert p.index == 2
    assert p.color == (255, 100, 50)
    assert p.left_xy == (10.0, 20.0)
    assert p.right_xy == (11.0, 21.0)


def test_overlap_metrics_defaults_to_empty():
    m = OverlapMetrics(mode="chessboard", pairs=[], vert_dy_px=0.0, rotation_deg=0.0,
                       zoom_ratio=None, n_inliers=0, n_requested=8,
                       align_ok=False, zoom_ok=False)
    assert m.pairs == []
    assert m.zoom_ratio is None
    assert m.align_ok is False


def test_compute_align_ok_within_thresholds():
    assert compute_align_ok(vert_dy_px=2.0, rotation_deg=0.2, n_inliers=5,
                            max_vert_dy_px=5.0, max_rotation_deg=0.5,
                            min_pairs_for_metrics=4) is True


def test_compute_align_ok_fails_on_low_inliers_even_when_metrics_perfect():
    assert compute_align_ok(vert_dy_px=0.0, rotation_deg=0.0, n_inliers=2,
                            max_vert_dy_px=5.0, max_rotation_deg=0.5,
                            min_pairs_for_metrics=4) is False


def test_compute_zoom_ok_within_tol():
    assert compute_zoom_ok(zoom_ratio=1.01, n_inliers=4,
                           max_zoom_ratio_err=0.02,
                           min_pairs_for_metrics=4) is True


def test_compute_zoom_ok_none_ratio_returns_false():
    assert compute_zoom_ok(zoom_ratio=None, n_inliers=4,
                           max_zoom_ratio_err=0.02,
                           min_pairs_for_metrics=4) is False
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_smart_overlap.py -v`
Expected: ImportError on `src.smart_overlap`.

- [ ] **Step 3: Create `src/smart_overlap.py`**

```python
"""Smart overlap calibration — diagnostic stereo-pair pattern matching.

Pure-Python core (no Qt). Two modes:
  * chessboard — uses physical_grid_calibration.detect_grid for perfect correspondence
  * live       — uses StereoFeatureMatcher for SIFT-based correspondence

The analyzer returns OverlapMetrics; the renderer draws coloured numbered
markers and connecting threads onto a side-by-side BGR frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class OverlapPair:
    index: int
    color: Tuple[int, int, int]            # BGR
    left_xy: Tuple[float, float]
    right_xy: Tuple[float, float]


@dataclass(frozen=True)
class OverlapMetrics:
    mode: str                               # "chessboard" | "live"
    pairs: list                             # list[OverlapPair]
    vert_dy_px: float
    rotation_deg: float
    zoom_ratio: Optional[float]             # None when not enough data
    n_inliers: int
    n_requested: int
    align_ok: bool
    zoom_ok: bool


def compute_align_ok(
    vert_dy_px: float,
    rotation_deg: float,
    n_inliers: int,
    max_vert_dy_px: float,
    max_rotation_deg: float,
    min_pairs_for_metrics: int,
) -> bool:
    return (
        n_inliers >= min_pairs_for_metrics
        and abs(vert_dy_px) <= max_vert_dy_px
        and abs(rotation_deg) <= max_rotation_deg
    )


def compute_zoom_ok(
    zoom_ratio: Optional[float],
    n_inliers: int,
    max_zoom_ratio_err: float,
    min_pairs_for_metrics: int,
) -> bool:
    if zoom_ratio is None:
        return False
    return (
        n_inliers >= min_pairs_for_metrics
        and abs(zoom_ratio - 1.0) <= max_zoom_ratio_err
    )
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_smart_overlap.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/smart_overlap.py tests/test_smart_overlap.py
git commit -m "feat: add OverlapPair, OverlapMetrics, and threshold helpers"
```

---

## Task 5: Implement chessboard-mode pair finder

**Files:**
- Modify: `src/smart_overlap.py`
- Modify: `tests/test_smart_overlap.py`

Use `physical_grid_calibration.detect_grid` to get corners in both eyes, then sample K well-distributed corners. Index correspondence is implicit: `corners_l[i]` and `corners_r[i]` are the same physical chessboard intersection.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smart_overlap.py`:

```python
import cv2
from src.physical_grid_calibration import generate_chessboard_page
from src.smart_overlap import find_chessboard_pairs


def _render_grid_eye(tmp_path):
    """Render the standard 9x6 grid PNG and return a BGR image."""
    out = tmp_path / "grid.png"
    generate_chessboard_page(out, inner_cols=9, inner_rows=6, square_mm=20.0, dpi=150)
    img = cv2.imread(str(out))
    assert img is not None
    return img


def test_find_chessboard_pairs_returns_K_pairs(tmp_path):
    img = _render_grid_eye(tmp_path)
    pairs, zoom_ratio = find_chessboard_pairs(img, img.copy(), pair_count=8)
    assert len(pairs) == 8
    assert all(len(p.left_xy) == 2 and len(p.right_xy) == 2 for p in pairs)
    # Identical inputs → ratio ≈ 1
    assert zoom_ratio is not None
    assert abs(zoom_ratio - 1.0) < 0.05


def test_find_chessboard_pairs_no_grid_returns_empty(tmp_path):
    blank = np.full((480, 640, 3), 200, dtype=np.uint8)
    pairs, zoom_ratio = find_chessboard_pairs(blank, blank.copy(), pair_count=8)
    assert pairs == []
    assert zoom_ratio is None


def test_find_chessboard_pairs_detects_injected_scale(tmp_path):
    img = _render_grid_eye(tmp_path)
    # Resize right eye to 90 % so right squares are smaller → ratio < 1.
    h, w = img.shape[:2]
    img_r = cv2.resize(img, (int(w * 0.9), int(h * 0.9)))
    img_r = cv2.copyMakeBorder(img_r, 0, h - img_r.shape[0],
                                0, w - img_r.shape[1],
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    _, ratio = find_chessboard_pairs(img, img_r, pair_count=8)
    assert ratio is not None
    assert ratio < 0.95
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_smart_overlap.py -v`
Expected: ImportError on `find_chessboard_pairs`.

- [ ] **Step 3: Implement `find_chessboard_pairs`**

Append to `src/smart_overlap.py`:

```python
import math

import numpy as np

from .physical_grid_calibration import GridDetection
# Imported lazily inside the function to keep top-level imports light:
#   from .physical_cal import _detect_grid_scaled, estimate_square_px


def _detect_grid(img, inner_cols=9, inner_rows=6, max_dim=420):
    """Wrapper around physical_grid_calibration.detect_grid that downscales for speed."""
    from .physical_grid_calibration import detect_grid
    h, w = img.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return detect_grid(img, inner_cols=inner_cols, inner_rows=inner_rows, exhaustive=False)
    scale = max_dim / largest
    small = _resize_for_detect(img, scale)
    detected = detect_grid(small, inner_cols=inner_cols, inner_rows=inner_rows, exhaustive=False)
    if detected is None:
        return None
    corners = detected.corners / scale
    center = (detected.center[0] / scale, detected.center[1] / scale)
    return GridDetection(corners=corners, center=center)


def _resize_for_detect(img, scale):
    import cv2
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def find_chessboard_pairs(
    eye_l: np.ndarray,
    eye_r: np.ndarray,
    pair_count: int,
    inner_cols: int = 9,
    inner_rows: int = 6,
):
    """Returns (pairs, zoom_ratio) for the chessboard mode.

    pairs is a list of OverlapPair (without color/index assigned — those
    come from the analyzer's stability tracking).
    zoom_ratio is right_square_px / left_square_px, or None if either eye
    failed grid detection.
    """
    from .physical_grid_calibration import estimate_square_px

    det_l = _detect_grid(eye_l, inner_cols=inner_cols, inner_rows=inner_rows)
    det_r = _detect_grid(eye_r, inner_cols=inner_cols, inner_rows=inner_rows)
    if det_l is None or det_r is None:
        return [], None

    n_corners = det_l.corners.shape[0]
    grid = max(1, int(math.ceil(math.sqrt(pair_count))))
    rows = inner_rows
    cols = inner_cols
    chosen: list[int] = []
    for gr in range(grid):
        for gc in range(grid):
            if len(chosen) >= pair_count:
                break
            r0 = int(gr * rows / grid)
            r1 = max(r0 + 1, int((gr + 1) * rows / grid))
            c0 = int(gc * cols / grid)
            c1 = max(c0 + 1, int((gc + 1) * cols / grid))
            r_centre = (r0 + r1) // 2
            c_centre = (c0 + c1) // 2
            idx = r_centre * cols + c_centre
            if 0 <= idx < n_corners and idx not in chosen:
                chosen.append(idx)
    chosen = chosen[:pair_count]

    pairs = [
        OverlapPair(
            index=-1,                               # filled in by analyzer
            color=(255, 255, 255),                  # filled in by analyzer
            left_xy=(float(det_l.corners[i, 0]), float(det_l.corners[i, 1])),
            right_xy=(float(det_r.corners[i, 0]), float(det_r.corners[i, 1])),
        )
        for i in chosen
    ]

    sq_l = estimate_square_px(det_l, inner_cols=inner_cols, inner_rows=inner_rows)
    sq_r = estimate_square_px(det_r, inner_cols=inner_cols, inner_rows=inner_rows)
    if sq_l and sq_r:
        zoom_ratio = float(sq_r) / float(sq_l)
    else:
        zoom_ratio = None
    return pairs, zoom_ratio
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `pytest tests/test_smart_overlap.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/smart_overlap.py tests/test_smart_overlap.py
git commit -m "feat: chessboard-mode pair finder with zoom-ratio"
```

---

## Task 6: Implement live-mode pair finder

**Files:**
- Modify: `src/smart_overlap.py`
- Modify: `tests/test_smart_overlap.py`

Uses `StereoFeatureMatcher` to get inliers, then samples K well-distributed pairs and computes `zoom_ratio` from pairwise distance ratios.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smart_overlap.py`:

```python
from src.smart_overlap import find_live_pairs
from src.stereo_matching import StereoFeatureMatcher


def _bgr_feature_pair(shift_y=3):
    """Return BGR feature-rich pair (640x480)."""
    rng = np.random.default_rng(11)
    img_l = np.full((480, 640, 3), 80, dtype=np.uint8)
    for _ in range(120):
        x = int(rng.integers(20, 600))
        y = int(rng.integers(20, 460))
        img_l[y:y + 6, x:x + 6] = 255
    img_r = np.roll(img_l, shift_y, axis=0)
    return img_l, img_r


def test_find_live_pairs_returns_at_most_K_pairs():
    matcher = StereoFeatureMatcher(max_features=500, match_ratio=0.75,
                                   ransac_thresh=2.0, frame_w=640, frame_h=480)
    img_l, img_r = _bgr_feature_pair(shift_y=3)
    pairs, zoom_ratio = find_live_pairs(img_l, img_r, matcher, pair_count=6)
    assert 1 <= len(pairs) <= 6
    assert zoom_ratio is not None
    assert 0.85 < zoom_ratio < 1.15  # identical-scale roll → ratio near 1


def test_find_live_pairs_blank_returns_empty():
    matcher = StereoFeatureMatcher(max_features=500, match_ratio=0.75,
                                   ransac_thresh=2.0, frame_w=640, frame_h=480)
    blank = np.full((480, 640, 3), 100, dtype=np.uint8)
    pairs, zoom_ratio = find_live_pairs(blank, blank.copy(), matcher, pair_count=8)
    assert pairs == []
    assert zoom_ratio is None
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_smart_overlap.py::test_find_live_pairs_returns_at_most_K_pairs -v`
Expected: ImportError on `find_live_pairs`.

- [ ] **Step 3: Implement `find_live_pairs`**

Append to `src/smart_overlap.py`:

```python
def find_live_pairs(
    eye_l: np.ndarray,
    eye_r: np.ndarray,
    matcher,
    pair_count: int,
    min_inliers: int = 4,
):
    """Returns (pairs, zoom_ratio) for the live mode.

    `matcher` is a StereoFeatureMatcher (passed in to avoid re-creating SIFT).
    `zoom_ratio` is computed from the median pairwise-distance ratio between
    inlier points, or None if there aren't at least 2 pairs.
    """
    import cv2
    gray_l = cv2.cvtColor(eye_l, cv2.COLOR_BGR2GRAY) if eye_l.ndim == 3 else eye_l
    gray_r = cv2.cvtColor(eye_r, cv2.COLOR_BGR2GRAY) if eye_r.ndim == 3 else eye_r

    result = matcher.match(gray_l, gray_r)
    n = len(result.pts_l)
    if n < min_inliers:
        return [], None

    h, w = eye_l.shape[:2]
    chosen = _sample_well_distributed(result.pts_l, pair_count, frame_w=w, frame_h=h)
    pairs = [
        OverlapPair(
            index=-1,
            color=(255, 255, 255),
            left_xy=(float(result.pts_l[i, 0]), float(result.pts_l[i, 1])),
            right_xy=(float(result.pts_r[i, 0]), float(result.pts_r[i, 1])),
        )
        for i in chosen
    ]
    zoom_ratio = _zoom_from_pair_distances(result.pts_l, result.pts_r)
    return pairs, zoom_ratio


def _sample_well_distributed(pts: np.ndarray, k: int, frame_w: int, frame_h: int) -> list[int]:
    if len(pts) == 0:
        return []
    grid = max(1, int(math.ceil(math.sqrt(k))))
    cell_w = frame_w / grid
    cell_h = frame_h / grid
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(pts):
        gx = min(int(x / cell_w), grid - 1)
        gy = min(int(y / cell_h), grid - 1)
        buckets.setdefault((gx, gy), []).append(i)
    chosen: list[int] = []
    # Prefer one per cell first
    for indices in buckets.values():
        chosen.append(indices[0])
        if len(chosen) >= k:
            break
    # Top up from leftover indices if we still need more
    if len(chosen) < k:
        leftover = [i for indices in buckets.values() for i in indices[1:]]
        chosen.extend(leftover[: k - len(chosen)])
    return chosen[:k]


def _zoom_from_pair_distances(pts_l: np.ndarray, pts_r: np.ndarray) -> Optional[float]:
    n = len(pts_l)
    if n < 2:
        return None
    ratios: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dl = float(np.hypot(pts_l[j, 0] - pts_l[i, 0], pts_l[j, 1] - pts_l[i, 1]))
            dr = float(np.hypot(pts_r[j, 0] - pts_r[i, 0], pts_r[j, 1] - pts_r[i, 1]))
            if dl > 5.0:
                ratios.append(dr / dl)
    if not ratios:
        return None
    return float(np.median(ratios))
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `pytest tests/test_smart_overlap.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/smart_overlap.py tests/test_smart_overlap.py
git commit -m "feat: live-mode pair finder with pair-distance zoom ratio"
```

---

## Task 7: Implement `SmartOverlapAnalyzer.analyze` orchestrator

**Files:**
- Modify: `src/smart_overlap.py`
- Modify: `tests/test_smart_overlap.py`

Combines the two finders, computes vert/rot metrics via `theil_sen`, fills colour and index from the palette, and produces the OK booleans. No stability tracking yet (Task 8 adds it).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smart_overlap.py`:

```python
from src.smart_overlap import SmartOverlapAnalyzer


def test_analyzer_chessboard_identical_eyes_metrics_near_zero(tmp_path):
    img = _render_grid_eye(tmp_path)
    a = SmartOverlapAnalyzer(
        max_vert_dy_px=5.0, max_rotation_deg=0.5, max_zoom_ratio_err=0.02,
        min_pairs_for_metrics=4, pair_stability_tol_px=30,
        matcher=None,
    )
    m = a.analyze(img, img.copy(), mode="chessboard", pair_count=8)
    assert m.mode == "chessboard"
    assert m.n_inliers == 8
    assert abs(m.vert_dy_px) < 1.0
    assert abs(m.rotation_deg) < 0.2
    assert m.align_ok is True
    assert m.zoom_ok is True
    # Colours assigned, indices 0..7
    assert sorted(p.index for p in m.pairs) == list(range(8))
    assert all(p.color != (255, 255, 255) for p in m.pairs)


def test_analyzer_chessboard_injected_vertical_offset(tmp_path):
    img = _render_grid_eye(tmp_path)
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 0], [0, 1, 8]])  # shift right eye down 8 px
    img_r = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    a = SmartOverlapAnalyzer(
        max_vert_dy_px=5.0, max_rotation_deg=0.5, max_zoom_ratio_err=0.02,
        min_pairs_for_metrics=4, pair_stability_tol_px=30, matcher=None,
    )
    m = a.analyze(img, img_r, mode="chessboard", pair_count=8)
    assert 6.0 < m.vert_dy_px < 10.0
    assert m.align_ok is False  # 8 px > 5 px threshold


def test_analyzer_returns_empty_metrics_when_no_grid(tmp_path):
    blank = np.full((480, 640, 3), 200, dtype=np.uint8)
    a = SmartOverlapAnalyzer(
        max_vert_dy_px=5.0, max_rotation_deg=0.5, max_zoom_ratio_err=0.02,
        min_pairs_for_metrics=4, pair_stability_tol_px=30, matcher=None,
    )
    m = a.analyze(blank, blank.copy(), mode="chessboard", pair_count=8)
    assert m.pairs == []
    assert m.n_inliers == 0
    assert m.align_ok is False
    assert m.zoom_ok is False


def test_analyzer_live_mode_runs_with_matcher():
    img_l, img_r = _bgr_feature_pair(shift_y=2)
    matcher = StereoFeatureMatcher(max_features=500, match_ratio=0.75,
                                   ransac_thresh=2.0, frame_w=640, frame_h=480)
    a = SmartOverlapAnalyzer(
        max_vert_dy_px=5.0, max_rotation_deg=0.5, max_zoom_ratio_err=0.02,
        min_pairs_for_metrics=4, pair_stability_tol_px=30, matcher=matcher,
    )
    m = a.analyze(img_l, img_r, mode="live", pair_count=6)
    assert m.mode == "live"
    assert m.n_inliers >= 1
    assert all(p.color != (255, 255, 255) for p in m.pairs)
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `pytest tests/test_smart_overlap.py -v`
Expected: ImportError on `SmartOverlapAnalyzer`.

- [ ] **Step 3: Implement `SmartOverlapAnalyzer`**

Append to `src/smart_overlap.py`:

```python
import colorsys

from .stereo_matching import theil_sen


_PALETTE_SIZE = 20  # max user-tunable pair count is 20


def _build_palette(n: int) -> list[Tuple[int, int, int]]:
    """Generate n visually distinct BGR colours via HSV spacing."""
    out = []
    for i in range(n):
        h = (i / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        out.append((int(b * 255), int(g * 255), int(r * 255)))
    return out


class SmartOverlapAnalyzer:
    """Diagnostic stereo overlap analysis. Holds previous-frame pairs for stability."""

    def __init__(
        self,
        max_vert_dy_px: float,
        max_rotation_deg: float,
        max_zoom_ratio_err: float,
        min_pairs_for_metrics: int,
        pair_stability_tol_px: float,
        matcher,                                    # StereoFeatureMatcher | None
    ):
        self.max_vert_dy_px = float(max_vert_dy_px)
        self.max_rotation_deg = float(max_rotation_deg)
        self.max_zoom_ratio_err = float(max_zoom_ratio_err)
        self.min_pairs_for_metrics = int(min_pairs_for_metrics)
        self.pair_stability_tol_px = float(pair_stability_tol_px)
        self.matcher = matcher
        self._palette = _build_palette(_PALETTE_SIZE)
        self._previous: list[OverlapPair] = []

    def reset(self) -> None:
        self._previous = []

    def analyze(
        self,
        eye_l: np.ndarray,
        eye_r: np.ndarray,
        mode: str,
        pair_count: int,
    ) -> OverlapMetrics:
        if mode == "chessboard":
            raw, zoom_ratio = find_chessboard_pairs(eye_l, eye_r, pair_count)
        elif mode == "live":
            if self.matcher is None:
                raise RuntimeError("live mode requires a matcher")
            raw, zoom_ratio = find_live_pairs(eye_l, eye_r, self.matcher, pair_count,
                                              min_inliers=self.min_pairs_for_metrics)
        else:
            raise ValueError(f"unknown mode: {mode}")

        if not raw:
            self._previous = []
            return OverlapMetrics(
                mode=mode, pairs=[], vert_dy_px=0.0, rotation_deg=0.0,
                zoom_ratio=zoom_ratio, n_inliers=0, n_requested=pair_count,
                align_ok=False, zoom_ok=False,
            )

        pairs = self._assign_colors_and_indices(raw)
        self._previous = pairs

        # Metrics
        h_w_cx = self._frame_cx(eye_l)
        x_centered = np.array([p.left_xy[0] - h_w_cx for p in pairs])
        dy = np.array([p.right_xy[1] - p.left_xy[1] for p in pairs])
        slope, dy_off, _ = theil_sen(x_centered, dy)
        vert_dy = float(dy_off)
        rotation_deg = float(math.degrees(slope))

        align_ok = compute_align_ok(
            vert_dy, rotation_deg, len(pairs),
            self.max_vert_dy_px, self.max_rotation_deg, self.min_pairs_for_metrics,
        )
        zoom_ok = compute_zoom_ok(
            zoom_ratio, len(pairs),
            self.max_zoom_ratio_err, self.min_pairs_for_metrics,
        )
        return OverlapMetrics(
            mode=mode, pairs=pairs, vert_dy_px=vert_dy, rotation_deg=rotation_deg,
            zoom_ratio=zoom_ratio, n_inliers=len(pairs), n_requested=pair_count,
            align_ok=align_ok, zoom_ok=zoom_ok,
        )

    # --- internals ---------------------------------------------------------

    @staticmethod
    def _frame_cx(eye_l: np.ndarray) -> float:
        return eye_l.shape[1] / 2.0

    def _assign_colors_and_indices(self, raw: list[OverlapPair]) -> list[OverlapPair]:
        """Stage 1: simple sequential assignment from the palette.

        Stability tracking (Task 8) replaces this with previous-frame inheritance.
        """
        out: list[OverlapPair] = []
        for i, p in enumerate(raw):
            colour = self._palette[i % len(self._palette)]
            out.append(OverlapPair(
                index=i,
                color=colour,
                left_xy=p.left_xy,
                right_xy=p.right_xy,
            ))
        return out
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `pytest tests/test_smart_overlap.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add src/smart_overlap.py tests/test_smart_overlap.py
git commit -m "feat: SmartOverlapAnalyzer orchestrator with metrics + OK"
```

---

## Task 8: Add pair stability tracking to the analyzer

**Files:**
- Modify: `src/smart_overlap.py`
- Modify: `tests/test_smart_overlap.py`

When the analyzer runs frame-to-frame, the same physical landmark should keep the same colour and index. We match each previous pair to the closest current pair (by `left_xy`) within `pair_stability_tol_px`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smart_overlap.py`:

```python
def test_pair_stability_preserves_colors_across_frames(tmp_path):
    img = _render_grid_eye(tmp_path)
    a = SmartOverlapAnalyzer(
        max_vert_dy_px=5.0, max_rotation_deg=0.5, max_zoom_ratio_err=0.02,
        min_pairs_for_metrics=4, pair_stability_tol_px=30, matcher=None,
    )
    first = a.analyze(img, img.copy(), mode="chessboard", pair_count=8)
    second = a.analyze(img, img.copy(), mode="chessboard", pair_count=8)
    # Same scene → every pair in the second frame should map to a previous
    # pair within the tolerance, inheriting its colour and index.
    second_by_xy = {p.left_xy: p for p in second.pairs}
    for first_pair in first.pairs:
        match = second_by_xy.get(first_pair.left_xy)
        assert match is not None
        assert match.color == first_pair.color
        assert match.index == first_pair.index


def test_reset_clears_previous_pairs(tmp_path):
    img = _render_grid_eye(tmp_path)
    a = SmartOverlapAnalyzer(
        max_vert_dy_px=5.0, max_rotation_deg=0.5, max_zoom_ratio_err=0.02,
        min_pairs_for_metrics=4, pair_stability_tol_px=30, matcher=None,
    )
    a.analyze(img, img.copy(), mode="chessboard", pair_count=8)
    a.reset()
    assert a._previous == []
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `pytest tests/test_smart_overlap.py::test_pair_stability_preserves_colors_across_frames -v`
Expected: AssertionError (sequential reassignment may produce different indices/colours when grid sampling is deterministic, but colours are at least guaranteed by index — confirm test fails first by inspecting output).

- [ ] **Step 3: Replace `_assign_colors_and_indices` with stability-aware version**

In `src/smart_overlap.py`, replace the `_assign_colors_and_indices` method on `SmartOverlapAnalyzer`:

```python
def _assign_colors_and_indices(self, raw: list[OverlapPair]) -> list[OverlapPair]:
    """Inherit colour/index from the closest previous pair within tolerance.

    Unmatched new pairs receive fresh palette slots not currently in use.
    """
    used_indices: set[int] = set()
    out: list[OverlapPair] = []
    tol = self.pair_stability_tol_px

    # First pass: try to match each new pair to a previous pair
    matched: list[Optional[OverlapPair]] = [None] * len(raw)
    available_prev = list(self._previous)
    for new_i, new_p in enumerate(raw):
        best_j = -1
        best_dist = tol
        for j, prev_p in enumerate(available_prev):
            if prev_p is None:
                continue
            d = math.hypot(new_p.left_xy[0] - prev_p.left_xy[0],
                           new_p.left_xy[1] - prev_p.left_xy[1])
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j >= 0:
            prev_p = available_prev[best_j]
            available_prev[best_j] = None  # consume
            matched[new_i] = OverlapPair(
                index=prev_p.index, color=prev_p.color,
                left_xy=new_p.left_xy, right_xy=new_p.right_xy,
            )
            used_indices.add(prev_p.index)

    # Second pass: assign fresh palette slots to unmatched pairs
    free_indices = [i for i in range(len(self._palette)) if i not in used_indices]
    free_iter = iter(free_indices)
    for new_i, new_p in enumerate(raw):
        if matched[new_i] is not None:
            out.append(matched[new_i])
            continue
        try:
            fresh = next(free_iter)
        except StopIteration:
            fresh = new_i % len(self._palette)
        out.append(OverlapPair(
            index=fresh, color=self._palette[fresh],
            left_xy=new_p.left_xy, right_xy=new_p.right_xy,
        ))
    return out
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `pytest tests/test_smart_overlap.py -v`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add src/smart_overlap.py tests/test_smart_overlap.py
git commit -m "feat: pair stability — inherit colour+index from previous frame"
```

---

## Task 9: Implement `render_overlay`

**Files:**
- Modify: `src/smart_overlap.py`
- Modify: `tests/test_smart_overlap.py`

Draws the coloured numbered markers (one per eye half) and connecting threads on the SBS frame. Used by the worker. The eye-gap divider line is also drawn.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smart_overlap.py`:

```python
from src.smart_overlap import render_overlay


def test_render_overlay_returns_same_shape_image_with_markers_drawn(tmp_path):
    img = _render_grid_eye(tmp_path)
    h, w = img.shape[:2]
    # Build an SBS frame from two copies of the same eye
    sbs = np.concatenate([img, img.copy()], axis=1)
    a = SmartOverlapAnalyzer(
        max_vert_dy_px=5.0, max_rotation_deg=0.5, max_zoom_ratio_err=0.02,
        min_pairs_for_metrics=4, pair_stability_tol_px=30, matcher=None,
    )
    metrics = a.analyze(img, img.copy(), mode="chessboard", pair_count=8)
    out = render_overlay(sbs, metrics)
    assert out.shape == sbs.shape
    # Some pixels must differ (markers + threads drawn)
    assert int(np.sum(out != sbs)) > 100


def test_render_overlay_empty_metrics_returns_input_unmodified():
    blank = np.full((200, 400, 3), 50, dtype=np.uint8)
    empty = OverlapMetrics(
        mode="chessboard", pairs=[], vert_dy_px=0.0, rotation_deg=0.0,
        zoom_ratio=None, n_inliers=0, n_requested=8,
        align_ok=False, zoom_ok=False,
    )
    out = render_overlay(blank, empty)
    assert out.shape == blank.shape
    # No pairs → only the divider line is drawn
    assert int(np.sum(out != blank)) < 5000
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `pytest tests/test_smart_overlap.py::test_render_overlay_returns_same_shape_image_with_markers_drawn -v`
Expected: ImportError on `render_overlay`.

- [ ] **Step 3: Implement `render_overlay`**

Append to `src/smart_overlap.py`:

```python
def render_overlay(sbs: np.ndarray, metrics: OverlapMetrics) -> np.ndarray:
    """Draw coloured numbered markers + connecting threads onto the SBS frame.

    `sbs` is a (H, 2W, 3) BGR image with left half + right half concatenated
    horizontally. Markers are drawn at each pair's left_xy (in left half) and
    right_xy (offset by W into the right half). Threads connect them.
    """
    import cv2
    out = sbs.copy()
    if out.ndim != 3 or out.shape[2] != 3:
        return out
    h, total_w, _ = out.shape
    eye_w = total_w // 2

    # Eye-gap divider
    cv2.line(out, (eye_w, 0), (eye_w, h), (60, 60, 60), 1, cv2.LINE_AA)

    for p in metrics.pairs:
        lx = int(round(p.left_xy[0]))
        ly = int(round(p.left_xy[1]))
        rx = int(round(p.right_xy[0])) + eye_w
        ry = int(round(p.right_xy[1]))
        # Marker circle and number per eye
        cv2.circle(out, (lx, ly), 9, p.color, 2, cv2.LINE_AA)
        cv2.circle(out, (rx, ry), 9, p.color, 2, cv2.LINE_AA)
        cv2.putText(out, str(p.index + 1), (lx - 4, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, p.color, 1, cv2.LINE_AA)
        cv2.putText(out, str(p.index + 1), (rx - 4, ry + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, p.color, 1, cv2.LINE_AA)
        # Connecting thread
        cv2.line(out, (lx, ly), (rx, ry), p.color, 1, cv2.LINE_AA)
    return out
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `pytest tests/test_smart_overlap.py -v`
Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add src/smart_overlap.py tests/test_smart_overlap.py
git commit -m "feat: render coloured markers and threads on SBS frame"
```

---

## Task 10: Add `SmartOverlapCfg` to `src/config.py` and `config.yaml`

**Files:**
- Modify: `src/config.py`
- Modify: `config.yaml`
- Modify: `tests/test_config_state.py` (add a focused load test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config_state.py`:

```python
from src.config import load_config, SmartOverlapCfg


def test_smart_overlap_defaults_when_missing(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("display:\n  width: 1920\n", encoding="utf-8")
    cfg = load_config(str(p))
    assert isinstance(cfg.smart_overlap, SmartOverlapCfg)
    assert cfg.smart_overlap.default_mode == "chessboard"
    assert cfg.smart_overlap.pair_count == 8
    assert cfg.smart_overlap.max_vert_dy_px == 5.0
    assert cfg.smart_overlap.max_rotation_deg == 0.5
    assert cfg.smart_overlap.max_zoom_ratio_err == 0.02


def test_smart_overlap_overrides_from_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "smart_overlap:\n"
        "  default_mode: live\n"
        "  pair_count: 12\n"
        "  max_vert_dy_px: 3.5\n",
        encoding="utf-8",
    )
    cfg = load_config(str(p))
    assert cfg.smart_overlap.default_mode == "live"
    assert cfg.smart_overlap.pair_count == 12
    assert cfg.smart_overlap.max_vert_dy_px == 3.5
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_config_state.py::test_smart_overlap_defaults_when_missing -v`
Expected: ImportError on `SmartOverlapCfg`.

- [ ] **Step 3: Add `SmartOverlapCfg` and wire it onto `PiccoloCfg`**

Edit `src/config.py`. Add the dataclass above `PiccoloCfg`:

```python
@dataclass
class SmartOverlapCfg:
    default_mode: str = "chessboard"        # "chessboard" | "live"
    pair_count: int = 8                     # default; user-adjustable in spinbox (4–20)
    min_pairs_for_metrics: int = 4
    max_vert_dy_px: float = 5.0
    max_rotation_deg: float = 0.5
    max_zoom_ratio_err: float = 0.02
    pair_stability_tol_px: float = 30.0
    worker_interval_sec: float = 0.2
```

Add the field on `PiccoloCfg` (next to existing fields):

```python
@dataclass
class PiccoloCfg:
    display: DisplayCfg = field(default_factory=DisplayCfg)
    cameras: CamerasCfg = field(default_factory=CamerasCfg)
    stereo: StereoCfg = field(default_factory=StereoCfg)
    calibration: CalibrationCfg = field(default_factory=CalibrationCfg)
    calibration_state: CalibrationStateCfg = field(default_factory=CalibrationStateCfg)
    controls: ControlsCfg = field(default_factory=ControlsCfg)
    stream: StreamCfg = field(default_factory=StreamCfg)
    smart_overlap: SmartOverlapCfg = field(default_factory=SmartOverlapCfg)
```

- [ ] **Step 4: Add the YAML block to `config.yaml`**

Append to `config.yaml`:

```yaml
smart_overlap:
  default_mode: chessboard
  pair_count: 8
  min_pairs_for_metrics: 4
  max_vert_dy_px: 5.0
  max_rotation_deg: 0.5
  max_zoom_ratio_err: 0.02
  pair_stability_tol_px: 30.0
  worker_interval_sec: 0.2
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `pytest tests/test_config_state.py -v`
Expected: all green (including the two new ones).

- [ ] **Step 6: Commit**

```bash
git add src/config.py config.yaml tests/test_config_state.py
git commit -m "feat: SmartOverlapCfg dataclass + config.yaml defaults"
```

---

## Task 11: Persist `smart_overlap_mode` and `smart_overlap_pair_count`

**Files:**
- Modify: `src/config.py` (extend `CalibrationStateCfg`)
- Modify: `src/config_state.py` (no changes needed if `asdict` covers it — verify)
- Modify: `tests/test_config_state.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config_state.py`:

```python
from src.config_state import save_calibration_state


def test_smart_overlap_mode_and_pair_count_persist(tmp_path):
    p = tmp_path / "config.yaml"
    cfg = load_config(str(p))
    cfg.calibration_state.smart_overlap_mode = "live"
    cfg.calibration_state.smart_overlap_pair_count = 12
    save_calibration_state(cfg, path=str(p))

    cfg2 = load_config(str(p))
    assert cfg2.calibration_state.smart_overlap_mode == "live"
    assert cfg2.calibration_state.smart_overlap_pair_count == 12
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_config_state.py::test_smart_overlap_mode_and_pair_count_persist -v`
Expected: AttributeError on `smart_overlap_mode`.

- [ ] **Step 3: Add the two fields to `CalibrationStateCfg`**

Edit `src/config.py`, replace the existing `CalibrationStateCfg`:

```python
@dataclass
class CalibrationStateCfg:
    """Persistent calibration state — saved to config.yaml on change."""
    nudge_left_x: int = 0
    nudge_right_x: int = 0
    nudge_left_y: int = 0
    nudge_right_y: int = 0
    convergence_offset: int = 0
    joint_zoom_center: int = 50
    joint_zoom_center_y: int = 50
    scale_left_pct: int = 100
    scale_right_pct: int = 100
    smart_overlap_mode: str = "chessboard"
    smart_overlap_pair_count: int = 8
```

`src/config_state.py` already calls `asdict(cfg.calibration_state)`, so the new fields are persisted automatically. No edit needed there.

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_config_state.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config_state.py
git commit -m "feat: persist smart_overlap_mode and pair_count in calibration_state"
```

---

## Task 12: Create `SmartOverlapWorker` (`QThread`)

**Files:**
- Create: `src/ui/smart_overlap_worker.py`
- Create: `tests/test_smart_overlap_worker.py`

Same shape as today's `PhysicalCalWorker`: `submit(sbs, mode, pair_count)` queues at most one frame; the worker analyses it and emits `result_ready` with a payload. Headless smoke test verifies the worker logic without a Qt event loop by calling `_process()` directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smart_overlap_worker.py
import numpy as np
import pytest

from src.config import SmartOverlapCfg
from src.stereo_matching import StereoFeatureMatcher
from src.smart_overlap import SmartOverlapAnalyzer


def test_worker_process_returns_payload_with_image_and_metrics():
    pytest.importorskip("PyQt6")
    from src.ui.smart_overlap_worker import SmartOverlapWorker, SmartOverlapResult

    cfg = SmartOverlapCfg()
    matcher = StereoFeatureMatcher(max_features=300, match_ratio=0.75,
                                   ransac_thresh=2.0, frame_w=320, frame_h=240)
    analyzer = SmartOverlapAnalyzer(
        max_vert_dy_px=cfg.max_vert_dy_px, max_rotation_deg=cfg.max_rotation_deg,
        max_zoom_ratio_err=cfg.max_zoom_ratio_err,
        min_pairs_for_metrics=cfg.min_pairs_for_metrics,
        pair_stability_tol_px=cfg.pair_stability_tol_px,
        matcher=matcher,
    )
    w = SmartOverlapWorker(analyzer)

    sbs = np.full((240, 640, 3), 50, dtype=np.uint8)
    result = w._process(sbs, mode="live", pair_count=4)
    assert isinstance(result, SmartOverlapResult)
    assert result.image is not None
    assert result.metrics.mode == "live"
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_smart_overlap_worker.py -v`
Expected: ImportError on `SmartOverlapWorker`.

- [ ] **Step 3: Implement the worker**

Create `src/ui/smart_overlap_worker.py`:

```python
"""Background analysis for the Smart overlap calibration panel."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..smart_overlap import OverlapMetrics, SmartOverlapAnalyzer, render_overlay
from .qt_helpers import ndarray_to_qimage


@dataclass
class SmartOverlapResult:
    image: object              # QImage
    metrics: OverlapMetrics


class SmartOverlapWorker(QThread):
    """Processes at most one latest SBS frame at a time and emits a rendered image + metrics."""

    result_ready = pyqtSignal(object)   # SmartOverlapResult

    def __init__(self, analyzer: SmartOverlapAnalyzer, parent=None):
        super().__init__(parent)
        self._analyzer = analyzer
        self._condition = threading.Condition()
        self._pending: Optional[tuple[np.ndarray, str, int]] = None
        self._stopping = False

    def submit(self, sbs: np.ndarray, mode: str, pair_count: int) -> None:
        with self._condition:
            self._pending = (sbs.copy(), mode, int(pair_count))
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()
        self.wait(2000)

    def reset_state(self) -> None:
        """Called when the surgeon presses Stop — clears stability tracking."""
        self._analyzer.reset()

    def run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                sbs, mode, pair_count = self._pending
                self._pending = None
            try:
                self.result_ready.emit(self._process(sbs, mode, pair_count))
            except Exception:
                continue

    def _process(self, sbs: np.ndarray, mode: str, pair_count: int) -> SmartOverlapResult:
        eye_w = sbs.shape[1] // 2
        eye_l = sbs[:, :eye_w]
        eye_r = sbs[:, eye_w:]
        metrics = self._analyzer.analyze(eye_l, eye_r, mode=mode, pair_count=pair_count)
        rendered = render_overlay(sbs, metrics)
        return SmartOverlapResult(
            image=ndarray_to_qimage(rendered),
            metrics=metrics,
        )
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_smart_overlap_worker.py -v`
Expected: 1 passed (or `skipped` if PyQt6 isn't installed in the test environment — that's acceptable).

- [ ] **Step 5: Commit**

```bash
git add src/ui/smart_overlap_worker.py tests/test_smart_overlap_worker.py
git commit -m "feat: SmartOverlapWorker QThread"
```

---

## Task 13: Replace wizard `QGroupBox` with smart-overlap panel in `calibration_tab.py`

**Files:**
- Modify: `src/ui/calibration_tab.py`

This task only edits the UI tab. It's a single logical change but touches many lines, so do all edits first, then verify imports + smoke-load.

- [ ] **Step 1: Update the imports at the top of `calibration_tab.py`**

Replace lines like:

```python
from ..physical_cal import GridPairMetrics, PhysicalCalSession
from .physical_cal_worker import PhysicalCalResult, PhysicalCalWorker
```

With:

```python
from ..smart_overlap import OverlapMetrics, SmartOverlapAnalyzer
from ..stereo_matching import StereoFeatureMatcher
from .smart_overlap_worker import SmartOverlapResult, SmartOverlapWorker
```

- [ ] **Step 2: Replace the `__init__` wiring**

In `CalibrationTab.__init__`, replace:

```python
self.session = PhysicalCalSession()
self.physical_worker = PhysicalCalWorker(self)
self.physical_worker.result_ready.connect(self._on_physical_result)
self.physical_worker.start()
self._latest_status: dict = {}
self._physical_active = False
...
self._latest_grid_metrics: GridPairMetrics | None = None
```

With:

```python
cfg_so = worker.cfg.smart_overlap if worker is not None else None
state = worker.cfg.calibration_state if worker is not None else None

eye_w = worker.cfg.cameras.left.width if worker is not None else 1920
eye_h = worker.cfg.cameras.left.height if worker is not None else 1080

matcher = StereoFeatureMatcher(
    max_features=500, match_ratio=0.75, ransac_thresh=2.0,
    frame_w=eye_w, frame_h=eye_h,
)
analyzer = SmartOverlapAnalyzer(
    max_vert_dy_px=cfg_so.max_vert_dy_px if cfg_so else 5.0,
    max_rotation_deg=cfg_so.max_rotation_deg if cfg_so else 0.5,
    max_zoom_ratio_err=cfg_so.max_zoom_ratio_err if cfg_so else 0.02,
    min_pairs_for_metrics=cfg_so.min_pairs_for_metrics if cfg_so else 4,
    pair_stability_tol_px=cfg_so.pair_stability_tol_px if cfg_so else 30.0,
    matcher=matcher,
)
self.smart_overlap_worker = SmartOverlapWorker(analyzer, self)
self.smart_overlap_worker.result_ready.connect(self._on_smart_overlap_result)
self.smart_overlap_worker.start()
self._smart_active = False
self._smart_mode = (state.smart_overlap_mode if state else "chessboard")
self._smart_pair_count = (state.smart_overlap_pair_count if state else 8)
self._latest_metrics: OverlapMetrics | None = None
```

Also remove the `self.session`, `self.physical_worker`, `self._physical_active`, `self._latest_status`, `self._latest_grid_metrics` references throughout the rest of the file.

- [ ] **Step 3: Replace `_make_wizard_group` with `_make_smart_overlap_group`**

Delete the entire `_make_wizard_group` method ([calibration_tab.py:412-444](../../src/ui/calibration_tab.py#L412-L444)) and add this new method in its place:

```python
def _make_smart_overlap_group(self) -> QGroupBox:
    box = QGroupBox("Smart overlap calibration", self)
    lay = QVBoxLayout(box)
    self.smart_preview = VideoWidget(box)
    lay.addWidget(self.smart_preview, stretch=1)

    # Readouts row (vert, rot, zoom, pairs)
    readouts = QHBoxLayout()
    self.lbl_vert = QLabel("Vert offset: --")
    self.lbl_rot = QLabel("Rotation: --")
    self.lbl_zoom = QLabel("Zoom ratio: --")
    self.lbl_pairs = QLabel("Match pairs: 0 / 0")
    for w in (self.lbl_vert, self.lbl_rot, self.lbl_zoom, self.lbl_pairs):
        w.setStyleSheet("font-family: monospace;")
        readouts.addWidget(w)
    readouts.addStretch(1)
    lay.addLayout(readouts)

    # Controls row
    controls = QHBoxLayout()
    controls.addWidget(QLabel("Mode:"))
    self.mode_combo = QComboBox(box)
    self.mode_combo.addItems(["Chessboard", "Live scene"])
    self.mode_combo.setCurrentIndex(0 if self._smart_mode == "chessboard" else 1)
    self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
    controls.addWidget(self.mode_combo)

    controls.addSpacing(12)
    controls.addWidget(QLabel("Pairs:"))
    self.pair_spin = QSpinBox(box)
    self.pair_spin.setRange(4, 20)
    self.pair_spin.setValue(self._smart_pair_count)
    self.pair_spin.valueChanged.connect(self._on_pair_count_changed)
    controls.addWidget(self.pair_spin)

    controls.addSpacing(12)
    self.lbl_align_badge = QLabel("ALIGN —")
    self.lbl_zoom_badge = QLabel("ZOOM —")
    for w in (self.lbl_align_badge, self.lbl_zoom_badge):
        w.setStyleSheet("font-family: monospace; padding: 2px 6px; border-radius: 3px;")
        controls.addWidget(w)

    controls.addStretch(1)
    self.btn_smart_start = QPushButton("Start")
    self.btn_smart_start.clicked.connect(self._toggle_smart_mode)
    controls.addWidget(self.btn_smart_start)

    self.btn_apply_scale = QPushButton("Apply detected scale")
    self.btn_apply_scale.setEnabled(False)
    self.btn_apply_scale.clicked.connect(self._apply_detected_scale)
    controls.addWidget(self.btn_apply_scale)

    lay.addLayout(controls)
    return box
```

- [ ] **Step 4: Replace the wizard handlers and slots**

Delete these methods entirely from `CalibrationTab`:
- `_toggle_physical_mode`, `_start_physical_mode`, `_stop_physical_mode`
- `_on_status_for_wizard`, `_on_physical_result`
- `_update_wizard_readout`, `_wizard_next`, `_wizard_prev`
- The wizard branch and the related sbs handling inside `_on_frame_for_wizard` (rewrite the whole method per below)

Replace `_on_frame_for_wizard` with `_on_frame_for_smart_overlap` (still wired to `worker.sbs_frame_ready`):

```python
@pyqtSlot(object)
def _on_frame_for_smart_overlap(self, sbs) -> None:
    if not self.isVisible():
        return
    now = time.perf_counter()
    interval = 0.1 if self._overlay_active else (0.2 if self._smart_active else 1.0)
    if now - self._last_wizard_render_t < interval:
        return
    self._last_wizard_render_t = now
    if self._overlay_active:
        self._overlay_last_sbs = sbs.copy()
        self._render_overlay_preview(self._overlay_last_sbs)
        return
    if not self._smart_active:
        return
    if sbs is None or sbs.ndim != 3 or sbs.shape[2] != 3:
        return
    self.smart_overlap_worker.submit(sbs, self._smart_mode, self._smart_pair_count)
```

In `__init__` change the connection from `_on_frame_for_wizard` to `_on_frame_for_smart_overlap` (and drop the `worker.status_tick.connect(...)` line — no longer needed).

Add the result handler:

```python
@pyqtSlot(object)
def _on_smart_overlap_result(self, result: SmartOverlapResult) -> None:
    self._latest_metrics = result.metrics
    self.smart_preview.set_frame(result.image)
    self._update_smart_readouts(result.metrics)

def _update_smart_readouts(self, m: OverlapMetrics) -> None:
    if m.n_inliers < (self.worker.cfg.smart_overlap.min_pairs_for_metrics if self.worker else 4):
        self.lbl_vert.setText("Vert offset: --")
        self.lbl_rot.setText("Rotation: --")
        self.lbl_zoom.setText("Zoom ratio: --")
    else:
        self.lbl_vert.setText(f"Vert offset: {m.vert_dy_px:+.1f} px")
        self.lbl_rot.setText(f"Rotation: {m.rotation_deg:+.2f}°")
        zr = "--" if m.zoom_ratio is None else f"{m.zoom_ratio:.3f}"
        self.lbl_zoom.setText(f"Zoom ratio: {zr}")
    self.lbl_pairs.setText(f"Match pairs: {m.n_inliers} / {m.n_requested}")

    self._set_badge(self.lbl_align_badge, "ALIGN", m.align_ok, neutral=m.n_inliers == 0)
    self._set_badge(self.lbl_zoom_badge, "ZOOM", m.zoom_ok, neutral=m.zoom_ratio is None)
    self.btn_apply_scale.setEnabled(
        self._smart_active and m.zoom_ratio is not None
        and m.n_inliers >= (self.worker.cfg.smart_overlap.min_pairs_for_metrics if self.worker else 4)
    )

@staticmethod
def _set_badge(label: QLabel, prefix: str, ok: bool, neutral: bool) -> None:
    if neutral:
        label.setText(f"{prefix} —")
        label.setStyleSheet("font-family: monospace; padding: 2px 6px; border-radius: 3px;"
                            "background: #2a2a2a; color: #aaa;")
        return
    if ok:
        label.setText(f"{prefix} OK")
        label.setStyleSheet("font-family: monospace; padding: 2px 6px; border-radius: 3px;"
                            "background: #2d6a3d; color: #fff;")
    else:
        label.setText(f"{prefix} ADJUST")
        label.setStyleSheet("font-family: monospace; padding: 2px 6px; border-radius: 3px;"
                            "background: #8a4a1a; color: #fff;")
```

Add the start/stop and control handlers:

```python
def _toggle_smart_mode(self) -> None:
    if self.worker is None:
        return
    if self._smart_active:
        self._stop_smart_mode()
    else:
        self._start_smart_mode()

def _start_smart_mode(self) -> None:
    if self._overlay_active:
        self._cancel_overlay_mode()
    self._smart_active = True
    self.btn_smart_start.setText("Stop")
    self._update_worker_raw_rate()

def _stop_smart_mode(self) -> None:
    if not self._smart_active:
        return
    self._smart_active = False
    self.btn_smart_start.setText("Start")
    self.smart_overlap_worker.reset_state()
    self.btn_apply_scale.setEnabled(False)
    self._update_worker_raw_rate()

def _on_mode_changed(self, idx: int) -> None:
    self._smart_mode = "chessboard" if idx == 0 else "live"
    self.smart_overlap_worker.reset_state()
    if self.worker is not None:
        self.worker.cfg.calibration_state.smart_overlap_mode = self._smart_mode
        save_calibration_state(self.worker.cfg)

def _on_pair_count_changed(self, value: int) -> None:
    self._smart_pair_count = int(value)
    self.smart_overlap_worker.reset_state()
    if self.worker is not None:
        self.worker.cfg.calibration_state.smart_overlap_pair_count = int(value)
        save_calibration_state(self.worker.cfg)
```

Replace the existing `_apply_detected_scale` with one that reads from `OverlapMetrics`:

```python
def _apply_detected_scale(self) -> None:
    metrics = self._latest_metrics
    if self.worker is None or metrics is None or metrics.zoom_ratio is None:
        return
    right_scale = int(round(100.0 / metrics.zoom_ratio))
    right_scale = max(80, min(120, right_scale))
    self.sld_lscale.setValue(100)
    self.sld_rscale.setValue(right_scale)
    self._on_lscale(100)
    self._on_rscale(right_scale)
```

Update `_update_worker_raw_rate` to use the new `_smart_active` flag:

```python
def _update_worker_raw_rate(self) -> None:
    if self.worker is None:
        return
    if self._overlay_active:
        self.worker.raw_frame_interval = 0.1
    elif self._smart_active:
        self.worker.raw_frame_interval = 0.2
    else:
        self.worker.raw_frame_interval = 1.0
```

In `stop_background_work`, replace `self.physical_worker.stop()` with `self.smart_overlap_worker.stop()` (and remove `self._stop_physical_mode()`).

Finally, in the `__init__`, replace the line:

```python
root.addWidget(self._make_wizard_group(), stretch=1)
```

with:

```python
root.addWidget(self._make_smart_overlap_group(), stretch=1)
```

- [ ] **Step 5: Run the existing UI smoke test**

Run: `pytest tests/test_ui_smoke.py -v`
Expected: pass (the smoke test imports the tab module — any import-time error is caught here).

If a Qt event loop is required and not available, the test may skip; the important thing is no `ImportError` or `AttributeError`.

- [ ] **Step 6: Commit**

```bash
git add src/ui/calibration_tab.py
git commit -m "feat: replace wizard QGroupBox with Smart overlap calibration panel"
```

---

## Task 14: Delete the obsolete physical-cal files

**Files:**
- Delete: `src/physical_cal.py`
- Delete: `src/ui/physical_cal_worker.py`
- Delete: `tests/test_physical_cal.py`

- [ ] **Step 1: Confirm no remaining references**

Run (use the Grep tool, not raw grep):
- pattern `from .*physical_cal|PhysicalCalSession|PatternRenderer|PhysicalCalWorker|PhysicalCalResult|GridPairMetrics|GridEyeMetrics`
- output_mode `files_with_matches`

Expected: only docs/specs/plans (acceptable) and the three files about to be deleted. If any other source file still imports these names, fix it before deleting.

- [ ] **Step 2: Delete the three files**

```bash
git rm src/physical_cal.py src/ui/physical_cal_worker.py tests/test_physical_cal.py
```

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all green. The physical-cal tests are gone; smart-overlap tests, stereo-matching tests, calibration tests, config-state tests, stereo-processor tests all pass.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove dead physical-cal wizard files"
```

---

## Task 15: Manual UI smoke verification (no automated test)

This task has no commit; it's a checkpoint before the work is considered done.

- [ ] **Step 1: Launch the app**

Run: `python piccolo.py` (or the platform-specific equivalent already used by the project).

- [ ] **Step 2: Verify the Calibration tab**

- The Calibration tab opens without error
- The "Smart overlap calibration" group is visible (replacing the old 5-phase wizard)
- The per-eye nudge sliders, reset row, and "Overlay manual calibration" group are still present and functional

- [ ] **Step 3: Verify the panel works on real cameras**

- Press `Start` — the preview begins updating, coloured numbered markers appear (chessboard mode by default)
- Toggle to `Live scene` — markers reappear from SIFT inliers
- Adjust `Pairs` spinbox — pair count in readout matches new value
- Adjust the cameras until threads sit horizontal — `ALIGN OK` lights green
- If `ZOOM ADJUST` is showing, click `Apply detected scale` — scale slider for the right eye updates and the reading clears
- Press `Stop` — stability state resets, the preview freezes

- [ ] **Step 4: Verify persistence**

- Quit the app, change `Pairs` to 12 and toggle to Live mode
- Quit and relaunch — `Pairs` should still read 12 and the mode should still be Live (read from `calibration_state` in `config.yaml`)

---

## Self-review checklist (run before handing off)

1. **Spec coverage** — every section in the design doc maps to a task above:
   - §2.1 Panel layout → Task 13
   - §2.2 Surgeon workflow → Task 13 + Task 15 (manual)
   - §2.3 Empty / failure states → Tasks 7, 9 (analyzer + renderer handle empty); Task 13 (UI badges)
   - §3.1 New files → Tasks 1, 2, 4–9, 12
   - §3.2 Modified files → Tasks 3, 10, 11, 13
   - §3.3 Deleted files → Task 14
   - §3.4 Untouched → not a task; verified by `git status`
   - §4.1 Chessboard mode → Task 5
   - §4.2 Live mode → Task 6
   - §4.3 Metrics → Task 7
   - §4.4 Threshold checks → Task 4 (helpers) + Task 7 (analyzer wires them)
   - §4.5 Pair stability → Task 8 (incl. `reset()`)
   - §4.6 Threading & timing → Task 12 (worker) + Task 13 (raw_frame_interval)
   - §5.1 config.yaml → Task 10
   - §5.2 Persistence → Task 11
   - §6.1 Unit tests → Tasks 1, 2, 4–11, 12 (each task includes its tests)
   - §6.2 Manual smoke → Task 15
   - §7 Migration → no code; Task 14 deletes; release-note line is documentation-only
2. **Placeholders** — none. Every step has full code or full commands.
3. **Type/method consistency** — `OverlapPair`, `OverlapMetrics`, `MatchResult` field names are consistent across Tasks 1–13. The `analyzer` argument shape (`SmartOverlapAnalyzer` with the named kwargs in Task 7) is what Task 12 and Task 13 instantiate.
