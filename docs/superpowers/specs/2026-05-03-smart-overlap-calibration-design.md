# Smart overlap calibration — design

**Date:** 2026-05-03
**Status:** Approved (awaiting implementation plan)
**Replaces:** [2026-04-13-physical-calibration-design.md](2026-04-13-physical-calibration-design.md) (the 5-phase wizard)

## 1. Goal & scope

Replace the existing 5-phase physical calibration wizard with a single **Smart overlap calibration** panel in the Calibration tab. The new panel is a **purely diagnostic** tool: it detects matching patterns between left and right camera frames, draws coloured numbered markers connected by threads, and shows numeric readouts so the surgeon can mechanically adjust mounts and lens controls until the threads sit horizontal and the badges show OK.

The one **non-diagnostic** action retained from the wizard is the **Apply detected scale** one-click button, because dialing lens zoom rings to single-percent precision by hand is impractical.

The existing **Overlay manual calibration** block (flash + arrow keys → software nudge) and the **per-eye nudge sliders** are unchanged. They serve a different need (software fine-tune).

## 2. User experience

### 2.1 Panel layout

A single `QGroupBox` titled "Smart overlap calibration" replaces today's wizard `QGroupBox`. Contents:

- **Preview area** — `VideoWidget` showing the live SBS frame with overlay:
  - Per matched pair: same-coloured numbered circle in each eye
  - Thread line drawn from each left marker to its right marker across the eye-gap
  - Threads are horizontal when vertical alignment is good; tilt indicates dy; spread of slopes indicates rotation
- **Readouts row** (4 cells, each green when in tolerance, orange when not, neutral grey when no data):
  - `Vert offset` (px)
  - `Rotation` (°)
  - `Zoom ratio`
  - `Match pairs` (`n_inliers / n_requested`)
- **Controls row**:
  - Mode toggle: `Chessboard` ↔ `Live scene`
  - `Pairs` spinbox (default 8, range 4–20)
  - `ALIGN OK` / `ALIGN ADJUST` badge (combined vert + rot)
  - `ZOOM OK` / `ZOOM ADJUST` badge
  - `Start` / `Stop` button
  - `Apply detected scale` button — enabled only when the zoom ratio reading is high-confidence (`n_inliers >= min_pairs_for_metrics`)

### 2.2 Surgeon workflow

1. Open Calibration tab. Smart-overlap panel is visible.
2. Pick mode (chessboard for first-time precise setup, live for quick re-checks).
3. Press **Start**. Coloured pairs appear in the preview, threads connect them, readouts begin updating.
4. Adjust camera mounts and lens controls (focus, zoom rings, mount tilt) until threads are horizontal and `ALIGN OK` lights up.
5. If `ZOOM ADJUST` is showing, either continue dialing the lens zoom rings or press **Apply detected scale** to absorb the residual into software scale.
6. Press **Stop** when satisfied.

### 2.3 Empty / failure states

If no chessboard is detected (chessboard mode) or insufficient inliers (live mode):

- Preview renders the SBS frame without overlay
- All four readouts show `—`
- Both badges grey, with text "no chessboard" or "looking for matches…"
- Start/Stop button continues to function; Apply-detected-scale stays disabled

## 3. Architecture & modules

### 3.1 New files

- **`src/stereo_matching.py`** — extracted shared stereo matcher
  - `StereoFeatureMatcher(max_features, match_ratio, ransac_thresh)` — owns SIFT, CLAHE, FLANN
  - `match(gray_l, gray_r) -> MatchResult` where `MatchResult` is a dataclass with two fields: `pts_l: ndarray[N, 2]` and `pts_r: ndarray[N, 2]` (post cross-check + grid distribution + F-matrix RANSAC; `N` may be 0)
  - `theil_sen(x, y) -> (slope, intercept, rms)`
- **`src/smart_overlap.py`** — pure-Python core (no Qt)
  - `OverlapPair` dataclass: `index`, `color`, `left_xy`, `right_xy`
  - `OverlapMetrics` dataclass: `mode`, `pairs`, `vert_dy_px`, `rotation_deg`, `zoom_ratio`, `n_inliers`, `n_requested`, `align_ok` (bool, computed by analyzer per §4.4), `zoom_ok` (bool, computed by analyzer per §4.4)
  - `SmartOverlapAnalyzer` class:
    - `analyze(eye_l, eye_r, mode, pair_count) -> OverlapMetrics` — computes pairs, metrics, and the OK booleans in one pass
    - Holds previous-frame pairs for colour/number stability (see §4.5)
    - `reset()` — clears stability state (called when worker stops)
    - Private finders: `_find_chessboard_pairs`, `_find_live_pairs`
  - `render_overlay(sbs, metrics) -> ndarray` — draws markers + threads on SBS frame
- **`src/ui/smart_overlap_worker.py`** — `QThread` background worker mirroring today's `PhysicalCalWorker` pattern
  - `submit(sbs, mode, pair_count)` — queues at most one frame
  - `result_ready` signal emits `(QImage, OverlapMetrics)`

### 3.2 Modified files

- **`src/ui/calibration_tab.py`** — drop wizard helpers, add smart-overlap helpers
  - Remove: `_make_wizard_group`, `_on_status_for_wizard`, the wizard branch of `_on_frame_for_wizard`, `_on_physical_result`, `_update_wizard_readout`, `_wizard_next`, `_wizard_prev`, `_toggle_physical_mode`, `_start_physical_mode`, `_stop_physical_mode`, all `physical_*` fields and the `physical_mode_changed` / `physical_frame_ready` signals (verify no other tab consumes them before deletion)
  - Add: `_make_smart_overlap_group`, `_on_smart_overlap_result`, `_on_mode_toggle`, `_on_pair_count_changed`, kept `_apply_detected_scale` (now reads `OverlapMetrics.zoom_ratio`)
  - Wire to `SmartOverlapWorker` instead of `PhysicalCalWorker`
- **`src/config.py`** — add `SmartOverlapCfg` dataclass parsed from the new `smart_overlap` YAML block
- **`config.yaml`** — add `smart_overlap` section (see §4)
- **`src/config_state.py`** — persist `smart_overlap_mode` and `smart_overlap_pair_count` under existing `calibration_state`
- **`src/stereo_align.py`** — internal refactor only: replace private matching methods with calls into `StereoFeatureMatcher`. No behaviour change. No public API change.

### 3.3 Deleted files

- `src/physical_cal.py` — full file. The `zoom_ratio = right.square_px / left.square_px` formula moves into `smart_overlap.py`'s chessboard finder.
- `src/ui/physical_cal_worker.py` — full file
- `tests/test_physical_cal.py` — full file (replaced by `tests/test_smart_overlap.py`)

### 3.4 Untouched

- `src/calibration.py` — nudge logic stays
- `src/physical_grid_calibration.py` — `detect_grid` is reused by chessboard mode
- The standalone repo-root `physical_calibration.py` — legacy script, separate from the Qt app

## 4. Algorithms

### 4.1 Chessboard mode

```
detect_grid(left)  → corners_l (54 corners for 9×6)
detect_grid(right) → corners_r
if either is None: return empty metrics
choose K corner indices spread across the grid: divide the grid's index space (rows × cols) into ⌈√K⌉ × ⌈√K⌉ sub-regions
                                                 and pick one index from each (centre-most), capped at K
for each chosen index i: pairs.append((corners_l[i], corners_r[i]))   # detect_grid emits both eyes in the same row-major order
zoom_ratio = estimate_square_px(right) / estimate_square_px(left)
```

### 4.2 Live mode

```
gray_l, gray_r = grayscale of L and R
result = StereoFeatureMatcher.match(gray_l, gray_r)        # result.pts_l, result.pts_r — shape [N, 2]
if N < min_pairs_for_metrics: return empty metrics
sample K well-distributed inlier indices (frame divided into ⌈√K⌉ × ⌈√K⌉ cells; pick the highest-confidence match in each cell)
pairs = [(result.pts_l[i], result.pts_r[i]) for i in chosen_indices]
zoom_ratio = median over (i, j) pairs of dist(pts_r[i], pts_r[j]) / dist(pts_l[i], pts_l[j])
```

### 4.3 Metrics (both modes)

```
dy_per_pair = pair.right.y − pair.left.y                    for each pair
x_per_pair  = pair.left.x − frame_w/2                       for each pair
slope, dy_off, rms = theil_sen(x_per_pair, dy_per_pair)
metrics.vert_dy_px    = dy_off
metrics.rotation_deg  = degrees(slope)
metrics.zoom_ratio    = (per-mode formula above)
metrics.n_inliers     = len(pairs)
```

### 4.4 Threshold checks

```
align_ok = |vert_dy_px| ≤ max_vert_dy_px AND
           |rotation_deg| ≤ max_rotation_deg AND
           n_inliers ≥ min_pairs_for_metrics
zoom_ok  = |zoom_ratio − 1.0| ≤ max_zoom_ratio_err AND
           n_inliers ≥ min_pairs_for_metrics
```

If `n_inliers < min_pairs_for_metrics`, both badges render neutral grey, not orange.

### 4.5 Pair stability across frames

- First successful `analyze()` after Start → assign colours and indices from a stable HSV-spaced palette
- Subsequent calls → for each previous pair, find the closest current pair by `left_xy` within `pair_stability_tol_px`. If matched, inherit its colour and index.
- Unmatched new pairs receive fresh palette slots
- Previous pairs that no longer match are dropped
- When the worker is stopped (`Stop` button or tab hidden), the analyzer's `reset()` is called. The next Start begins a fresh palette — no carry-over across sessions.

Chessboard mode is trivially stable (corners have a fixed index ordering); the stability logic matters mainly for live mode.

### 4.6 Threading & timing

- `SmartOverlapWorker(QThread)` mirrors today's `PhysicalCalWorker` pattern: `submit()` keeps only the latest frame
- `worker_interval_sec = 0.2` (5 Hz analysis rate; matches today's wizard rate)
- The existing `pipeline_worker` raw-frame interval already throttles to 0.2 s when the calibration tab is visible — that stays
- Tab-visibility check in `_on_frame_for_smart_overlap` (`if not self.isVisible(): return`)

## 5. Configuration

### 5.1 `config.yaml`

```yaml
smart_overlap:
  default_mode: chessboard         # "chessboard" | "live"
  pair_count: 8                    # default; user-adjustable in spinbox (range 4–20)
  min_pairs_for_metrics: 4
  max_vert_dy_px: 5.0              # was VERTICAL_DELTA_OK_PX
  max_rotation_deg: 0.5            # was ROTATION_DELTA_OK_DEG
  max_zoom_ratio_err: 0.02         # was ZOOM_RATIO_TOL
  pair_stability_tol_px: 30
  worker_interval_sec: 0.2
```

### 5.2 Persistence (`src/config_state.py`)

Add under existing `calibration_state` block:

```yaml
calibration_state:
  ...
  smart_overlap_mode: chessboard
  smart_overlap_pair_count: 8
```

Save/load follows the existing `save_calibration_state()` pattern.

## 6. Testing

### 6.1 Unit tests

- **`tests/test_smart_overlap.py`** (new):
  - Chessboard mode, synthetic 9×6 grid in both eyes → `vert_dy_px ≈ 0`, `rotation_deg ≈ 0`, `zoom_ratio ≈ 1.0`, exactly K pairs
  - Chessboard with injected vertical offset → `vert_dy_px ≈ injected`
  - Chessboard with injected rotation → `rotation_deg ≈ injected`
  - Chessboard with injected scale → `zoom_ratio ≈ injected`
  - Live mode with synthetic feature-rich pair → non-empty pairs, metrics shape correct
  - Empty / no-detect → both badges false, `n_inliers == 0`
  - Pair stability → same scene over two consecutive `analyze()` calls preserves colour/index assignments
- **`tests/test_stereo_matching.py`** (new):
  - `theil_sen` on noisy linear data with outliers → recovers slope within tolerance
  - `StereoFeatureMatcher.match` on a synthetic feature-rich pair → returns inliers
  - `StereoFeatureMatcher.match` on a feature-less pair → returns empty
- **`tests/test_stereo_processor.py`** (existing) — must still pass after `StereoAligner`'s internal refactor (no behaviour change expected)

### 6.2 Manual UI smoke test

Open the Calibration tab on the desktop app, confirm the panel renders, threads draw at ~5 Hz on real cameras, badges flip correctly across visible mis-/aligned configurations, mode toggle works, Apply-detected-scale writes to the right scale slider.

## 7. Migration notes

- The 5-phase wizard is removed in a single change. No deprecated shims.
- All persisted calibration values (nudge sliders, scale percentages, `convergence_offset`) are unaffected and continue to apply.
- One-line release note: *"Replaced 5-phase physical calibration wizard with single-screen Smart overlap calibration. Same goals (zoom, vertical alignment, rotation), simpler workflow."*

## 8. Out of scope (explicit non-goals)

- Auto-applying alignment fixes (vertical/rotation) to a software warp — `StereoAligner` already does that automatically; the new panel is for the **physical** correction step that comes before
- Replacing or modifying the Overlay manual calibration block (the flash + arrow-key software nudge tool)
- Replacing or modifying the per-eye nudge sliders
- Persisting "is calibrated?" status — the panel is diagnostic; the surgeon decides when to stop
- Touching `physical_calibration.py` at the repo root (legacy script outside the Qt app)
