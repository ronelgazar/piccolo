# GPU Pipeline Latency Reduction — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully-GPU-resident hot path that uploads each camera frame pair once, runs warp + fill-holes + zoom-resize + calibration nudge + depth + SBS compose entirely on `cv2.cuda_GpuMat`, and downloads once at the end. Remove the per-frame pipeline throttle that delays fresh frames by up to one display interval. Gate everything behind a `performance.use_gpu_pipeline` flag for full reversibility.

**Architecture:** New `src/gpu_pipeline.py` orchestrates the GPU-resident path with persistent `cuda_GpuMat` buffers. Existing CPU implementations stay; GPU equivalents are added as parallel methods on `StereoAligner`, `StereoProcessor`, and `CalibrationOverlay`. `PipelineWorker` selects the path based on the new flag. The throttle in `_can_process_now` is replaced with frame-id deduplication only.

**Tech Stack:** `opencv-python-cuda 4.12` (already installed, CUDA 12.9), `numpy`, `PyQt6`, `pytest`. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-05-10-gpu-pipeline-latency-reduction-design.md](../specs/2026-05-10-gpu-pipeline-latency-reduction-design.md). This plan implements Phase 1 only. Phases 2–5 will be planned separately after Phase 1 ships and is measured.

**Out of scope for this plan:** Resolution toggle (Phase 2), TurboJPEG decode (Phase 3), QOpenGLWidget display (Phase 4), true GPU MJPEG decode (Phase 5). Calibration/full-FOV views remain CPU-only — they run at slow cadence and don't dominate latency.

---

## File map

| File | Action | Responsibility |
| ---- | ------ | -------------- |
| `src/config.py` | Modify | Add `use_gpu_pipeline: bool` field to `PerformanceCfg` |
| `src/gpu_pipeline.py` | Create | `GpuPipeline` class — owns GpuMats, runs the GPU-resident hot path |
| `src/stereo_align.py` | Modify | Add `warp_pair_gpu` and GPU-side cached warp matrices |
| `src/stereo_processor.py` | Modify | Add `process_pair_gpu` writing into a single SBS GpuMat |
| `src/calibration.py` | Modify | Add `apply_nudge_gpu` |
| `src/ui/pipeline_worker.py` | Modify | Switch hot path on flag; remove throttle in `_can_process_now` |
| `tests/test_gpu_pipeline_spike.py` | Create | One-off probe verifying `cv2.cuda_GpuMat` ROI semantics on this build |
| `tests/test_gpu_pipeline_fill_holes.py` | Create | GPU fill-holes matches CPU output |
| `tests/test_gpu_pipeline_align.py` | Create | GPU warp_pair matches CPU output |
| `tests/test_gpu_pipeline_processor.py` | Create | GPU process_pair matches CPU output |
| `tests/test_gpu_pipeline_calibration.py` | Create | GPU apply_nudge matches CPU output |
| `tests/test_gpu_pipeline_orchestrator.py` | Create | End-to-end GPU pipeline matches CPU within tolerance |
| `tests/test_gpu_pipeline_perf.py` | Create | GPU path is at least 3 ms faster than CPU at 640×480 |
| `tests/_cuda_helpers.py` | Create | Shared `skip_if_no_cuda` helper for the test files above |
| `config.yaml` | Modify | Add `performance.use_gpu_pipeline: true` |

---

## Task 1: Spike — verify cv2.cuda_GpuMat ROI semantics

**Goal:** Before building the orchestrator, prove on this exact OpenCV build that we can (a) create a `cuda_GpuMat`, (b) get a writeable ROI from it via `cv2.cuda_GpuMat(parent, roi)` or equivalent, and (c) have writes through the ROI persist in the parent. The spec calls this out as a known risk.

**Files:**
- Create: `tests/_cuda_helpers.py`
- Create: `tests/test_gpu_pipeline_spike.py`

- [ ] **Step 1: Create the CUDA-skip helper**

Create `tests/_cuda_helpers.py`:

```python
"""Shared helpers for GPU-pipeline tests."""
from __future__ import annotations

import cv2
import pytest


def skip_if_no_cuda() -> None:
    """Skip the calling test if CUDA is not usable on this machine."""
    if not hasattr(cv2, "cuda"):
        pytest.skip("cv2.cuda module not available")
    if not hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
        pytest.skip("cv2.cuda.getCudaEnabledDeviceCount not available")
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() <= 0:
            pytest.skip("No CUDA-enabled devices")
    except cv2.error as exc:
        pytest.skip(f"CUDA query failed: {exc}")
```

- [ ] **Step 2: Write the spike test**

Create `tests/test_gpu_pipeline_spike.py`:

```python
"""One-off spike: verify cv2.cuda_GpuMat ROI write semantics.

Phase 1 of the GPU pipeline relies on writing into ROI views of a single
SBS GpuMat. This test fails fast if that idiom is broken on this OpenCV
build, so we know to use a different SBS compose strategy.
"""
from __future__ import annotations

import cv2
import numpy as np

from tests._cuda_helpers import skip_if_no_cuda


def test_gpu_mat_upload_download_roundtrip():
    skip_if_no_cuda()
    src = np.full((10, 10, 3), 42, dtype=np.uint8)
    gpu = cv2.cuda_GpuMat()
    gpu.upload(src)
    out = gpu.download()
    assert out.shape == src.shape
    assert (out == 42).all()


def test_gpu_mat_roi_write_visible_in_parent():
    skip_if_no_cuda()
    parent = cv2.cuda_GpuMat(20, 20, cv2.CV_8UC3)
    parent.upload(np.zeros((20, 20, 3), dtype=np.uint8))

    # Try to obtain a 10x20 ROI covering the right half.
    roi = cv2.cuda_GpuMat(parent, (10, 0, 10, 20))  # (x, y, width, height)
    fill = np.full((20, 10, 3), 99, dtype=np.uint8)
    roi.upload(fill)

    out = parent.download()
    assert (out[:, :10] == 0).all(), "left half should be untouched"
    assert (out[:, 10:] == 99).all(), "right half should reflect ROI write"


def test_gpu_resize_into_preallocated_destination():
    skip_if_no_cuda()
    src = cv2.cuda_GpuMat()
    src.upload(np.full((40, 40, 3), 7, dtype=np.uint8))
    dst = cv2.cuda_GpuMat(20, 20, cv2.CV_8UC3)
    cv2.cuda.resize(src, (20, 20), dst=dst, interpolation=cv2.INTER_AREA)
    out = dst.download()
    assert out.shape == (20, 20, 3)
    assert (out == 7).all()
```

- [ ] **Step 3: Run the spike test and confirm it passes**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_spike.py -v`

Expected: 3 PASSED. If `test_gpu_mat_roi_write_visible_in_parent` fails or errors, the rest of this plan needs to use `cv2.cuda.copyTo` with masks instead of ROI views in Task 7 — note the failure and consult before continuing.

- [ ] **Step 4: Commit**

```bash
git add tests/_cuda_helpers.py tests/test_gpu_pipeline_spike.py
git commit -m "test: spike verifying cv2.cuda_GpuMat ROI write semantics"
```

---

## Task 2: Add `use_gpu_pipeline` config flag

**Goal:** Add the runtime toggle for the new path.

**Files:**
- Modify: `src/config.py:174-177` (PerformanceCfg dataclass)
- Modify: `config.yaml:78-80` (performance section)
- Test: `tests/test_config_state.py` (add a smoke test)

- [ ] **Step 1: Write a failing config-load test**

Add to `tests/test_config_state.py` (append at end of file):

```python
def test_performance_config_has_use_gpu_pipeline_default():
    from src.config import PerformanceCfg

    cfg = PerformanceCfg()
    assert hasattr(cfg, "use_gpu_pipeline"), "PerformanceCfg missing use_gpu_pipeline"
    assert cfg.use_gpu_pipeline is True, "default should be True"


def test_performance_config_use_gpu_pipeline_loads_from_yaml(tmp_path):
    import yaml
    from src.config import load_config

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.safe_dump({"performance": {"use_gpu_pipeline": False}}))
    cfg = load_config(str(yaml_path))
    assert cfg.performance.use_gpu_pipeline is False
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests\test_config_state.py::test_performance_config_has_use_gpu_pipeline_default tests\test_config_state.py::test_performance_config_use_gpu_pipeline_loads_from_yaml -v`

Expected: FAIL with `AttributeError: 'PerformanceCfg' object has no attribute 'use_gpu_pipeline'` or similar.

- [ ] **Step 3: Add the field to PerformanceCfg**

In `src/config.py`, replace the `PerformanceCfg` block:

```python
@dataclass
class PerformanceCfg:
    """Performance tuning flags exposed in Settings."""
    low_latency_mode: bool = False
    use_gpu_for_depth: bool = True
    use_gpu_pipeline: bool = True   # NEW: GPU-resident hot path (Phase 1)
```

- [ ] **Step 4: Add the key to `config.yaml`**

In `config.yaml`, replace the `performance:` block:

```yaml
performance:
  low_latency_mode: false
  use_gpu_for_depth: true
  use_gpu_pipeline: true
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests\test_config_state.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/config.py config.yaml tests/test_config_state.py
git commit -m "feat(config): add performance.use_gpu_pipeline flag"
```

---

## Task 3: Implement `fill_holes_cross_gpu`

**Goal:** Port the numpy boolean-indexing border-fill from `pipeline_worker._fill_holes_cross` to a pure-GPU implementation. This is the trickiest port because there's no direct cv2.cuda equivalent for `np.all(left == 0, axis=2)`, so we use per-channel `cv2.cuda.compare` plus `cv2.cuda.bitwise_and` to build the mask, then `cv2.cuda.copyTo` to substitute pixels.

**Files:**
- Create: `src/gpu_pipeline.py` (initial skeleton with just the fill_holes function and a tiny GpuPipeline class)
- Create: `tests/test_gpu_pipeline_fill_holes.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gpu_pipeline_fill_holes.py`:

```python
"""GPU fill-holes-cross matches CPU output exactly."""
from __future__ import annotations

import numpy as np

from tests._cuda_helpers import skip_if_no_cuda


def _cpu_fill_holes(left: np.ndarray, right: np.ndarray):
    # Mirror of pipeline_worker._fill_holes_cross
    if left.shape != right.shape:
        return left, right
    mask_l = np.all(left == 0, axis=2)
    mask_r = np.all(right == 0, axis=2)
    copy_to_l = mask_l & (~mask_r)
    if np.any(copy_to_l):
        left[copy_to_l] = right[copy_to_l]
    copy_to_r = mask_r & (~mask_l)
    if np.any(copy_to_r):
        right[copy_to_r] = left[copy_to_r]
    return left, right


def _make_pair_with_borders():
    rng = np.random.default_rng(0)
    left = rng.integers(1, 256, size=(60, 80, 3), dtype=np.uint8)
    right = rng.integers(1, 256, size=(60, 80, 3), dtype=np.uint8)
    # Carve a 10-px black left border in `left` and 10-px black right border in `right`.
    left[:, :10] = 0
    right[:, -10:] = 0
    return left, right


def test_fill_holes_gpu_matches_cpu():
    skip_if_no_cuda()
    from src.gpu_pipeline import fill_holes_cross_gpu

    left, right = _make_pair_with_borders()
    cpu_left, cpu_right = _cpu_fill_holes(left.copy(), right.copy())

    import cv2
    gpu_l = cv2.cuda_GpuMat()
    gpu_r = cv2.cuda_GpuMat()
    gpu_l.upload(left)
    gpu_r.upload(right)
    fill_holes_cross_gpu(gpu_l, gpu_r)

    out_l = gpu_l.download()
    out_r = gpu_r.download()

    assert np.array_equal(out_l, cpu_left), "left mismatch"
    assert np.array_equal(out_r, cpu_right), "right mismatch"


def test_fill_holes_gpu_no_borders_is_noop():
    skip_if_no_cuda()
    import cv2
    from src.gpu_pipeline import fill_holes_cross_gpu

    rng = np.random.default_rng(1)
    left = rng.integers(1, 256, size=(40, 50, 3), dtype=np.uint8)
    right = rng.integers(1, 256, size=(40, 50, 3), dtype=np.uint8)
    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(left)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(right)
    fill_holes_cross_gpu(gpu_l, gpu_r)
    assert np.array_equal(gpu_l.download(), left)
    assert np.array_equal(gpu_r.download(), right)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_fill_holes.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.gpu_pipeline'`.

- [ ] **Step 3: Create `src/gpu_pipeline.py` with the fill-holes function**

Create `src/gpu_pipeline.py`:

```python
"""GPU-resident hot path for the live stereo pipeline.

See docs/superpowers/specs/2026-05-10-gpu-pipeline-latency-reduction-design.md.
"""
from __future__ import annotations

import cv2
import numpy as np


def fill_holes_cross_gpu(gpu_l: "cv2.cuda_GpuMat", gpu_r: "cv2.cuda_GpuMat") -> None:
    """In-place equivalent of pipeline_worker._fill_holes_cross.

    For each pixel that is fully zero in one eye and non-zero in the other,
    copy the non-zero pixel into the zero side. Pure GPU; no host round-trip.
    """
    if gpu_l.size() != gpu_r.size() or gpu_l.type() != gpu_r.type():
        return

    # Per-channel zero mask: cv2.cuda.compare yields 8U single-channel.
    chans_l = cv2.cuda.split(gpu_l)
    chans_r = cv2.cuda.split(gpu_r)

    zero_l = cv2.cuda.compare(chans_l[0], 0, cv2.CMP_EQ)
    for ch in chans_l[1:]:
        zero_l = cv2.cuda.bitwise_and(zero_l, cv2.cuda.compare(ch, 0, cv2.CMP_EQ))

    zero_r = cv2.cuda.compare(chans_r[0], 0, cv2.CMP_EQ)
    for ch in chans_r[1:]:
        zero_r = cv2.cuda.bitwise_and(zero_r, cv2.cuda.compare(ch, 0, cv2.CMP_EQ))

    # copy_to_l = zero_l AND NOT zero_r;  copy_to_r = zero_r AND NOT zero_l
    not_zero_r = cv2.cuda.bitwise_not(zero_r)
    not_zero_l = cv2.cuda.bitwise_not(zero_l)
    copy_to_l = cv2.cuda.bitwise_and(zero_l, not_zero_r)
    copy_to_r = cv2.cuda.bitwise_and(zero_r, not_zero_l)

    # cv2.cuda.copyTo(src, mask, dst) copies src→dst where mask != 0.
    cv2.cuda.copyTo(gpu_r, copy_to_l, gpu_l)
    cv2.cuda.copyTo(gpu_l, copy_to_r, gpu_r)
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_fill_holes.py -v`

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/gpu_pipeline.py tests/test_gpu_pipeline_fill_holes.py
git commit -m "feat(gpu_pipeline): GPU fill-holes-cross matching CPU output"
```

---

## Task 4: Add `warp_pair_gpu` to StereoAligner

**Goal:** Provide a method that takes two GpuMats in and writes warp results into two GpuMats out, using the cached `_warp_l` / `_warp_r` matrices. Avoids per-frame upload of warp matrices to GPU.

**Files:**
- Modify: `src/stereo_align.py` (add new method, keep `warp_pair` unchanged)
- Create: `tests/test_gpu_pipeline_align.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gpu_pipeline_align.py`:

```python
"""GPU warp_pair matches CPU warp_pair within sub-pixel tolerance."""
from __future__ import annotations

import math
import numpy as np

from tests._cuda_helpers import skip_if_no_cuda


def _make_aligner(frame_w: int = 80, frame_h: int = 60):
    from src.config import AlignmentCfg
    from src.stereo_align import StereoAligner

    cfg = AlignmentCfg(enabled=True)
    aligner = StereoAligner(cfg, frame_w, frame_h)
    aligner._smooth_dy = 4.0
    aligner._smooth_dtheta = math.radians(0.5)
    aligner._build_warp_matrices()
    return aligner


def test_warp_pair_gpu_matches_cpu():
    skip_if_no_cuda()
    import cv2

    aligner = _make_aligner()
    rng = np.random.default_rng(2)
    frame_l = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
    frame_r = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)

    cpu_l, cpu_r = aligner.warp_pair(frame_l.copy(), frame_r.copy())

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(frame_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(frame_r)
    out_l = cv2.cuda_GpuMat(60, 80, cv2.CV_8UC3)
    out_r = cv2.cuda_GpuMat(60, 80, cv2.CV_8UC3)
    aligner.warp_pair_gpu(gpu_l, gpu_r, out_l, out_r)
    gpu_l_arr = out_l.download()
    gpu_r_arr = out_r.download()

    # Allow per-pixel value drift up to 2 (different resamplers, same input)
    assert np.abs(cpu_l.astype(int) - gpu_l_arr.astype(int)).max() <= 2
    assert np.abs(cpu_r.astype(int) - gpu_r_arr.astype(int)).max() <= 2


def test_warp_pair_gpu_disabled_aligner_passthrough():
    skip_if_no_cuda()
    import cv2
    from src.config import AlignmentCfg
    from src.stereo_align import StereoAligner

    cfg = AlignmentCfg(enabled=False)
    aligner = StereoAligner(cfg, frame_w=40, frame_h=30)

    rng = np.random.default_rng(3)
    frame_l = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    frame_r = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(frame_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(frame_r)
    out_l = cv2.cuda_GpuMat(30, 40, cv2.CV_8UC3)
    out_r = cv2.cuda_GpuMat(30, 40, cv2.CV_8UC3)
    used_warp = aligner.warp_pair_gpu(gpu_l, gpu_r, out_l, out_r)

    assert used_warp is False, "disabled aligner must report no-warp"
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_align.py -v`

Expected: FAIL with `AttributeError: 'StereoAligner' object has no attribute 'warp_pair_gpu'`.

- [ ] **Step 3: Add `warp_pair_gpu` to StereoAligner**

In `src/stereo_align.py`, add this method below `warp_pair` (after line 257, before `force_update`):

```python
    def warp_pair_gpu(
        self,
        gpu_l: "cv2.cuda_GpuMat",
        gpu_r: "cv2.cuda_GpuMat",
        out_l: "cv2.cuda_GpuMat",
        out_r: "cv2.cuda_GpuMat",
    ) -> bool:
        """GPU equivalent of `warp_pair`.

        Writes the warped frames into `out_l` / `out_r`. Returns True if a
        warp was applied; False if the aligner is disabled or has no
        correction yet (in that case callers should fall back to a copy).
        """
        if not self._enabled or self._warp_l is None:
            return False

        size = gpu_l.size()  # (width, height)
        cv2.cuda.warpAffine(
            gpu_l, self._warp_l, size, dst=out_l,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        cv2.cuda.warpAffine(
            gpu_r, self._warp_r, size, dst=out_r,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        if self.cfg.mask_overlap and self._overlap_mask_3ch is not None:
            # Mask overlap stays CPU-side for now; uploading the mask each
            # frame would defeat the purpose. Phase 1 leaves mask_overlap
            # off (it's already false by default in config).
            pass

        return True
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_align.py -v`

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/stereo_align.py tests/test_gpu_pipeline_align.py
git commit -m "feat(stereo_align): add GPU warp_pair_gpu using cached matrices"
```

---

## Task 5: Add `process_pair_gpu` to StereoProcessor

**Goal:** Crop + resize each eye on GPU, writing into ROI views of a single SBS GpuMat. The ROI semantics were verified in Task 1; if that spike failed, use the alternate `cv2.cuda.copyTo` strategy noted there.

**Files:**
- Modify: `src/stereo_processor.py` (add `process_pair_gpu`, keep `process_pair` unchanged)
- Create: `tests/test_gpu_pipeline_processor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gpu_pipeline_processor.py`:

```python
"""GPU process_pair matches CPU process_pair output."""
from __future__ import annotations

import numpy as np

from tests._cuda_helpers import skip_if_no_cuda


def _make_processor(eye_w: int = 100, eye_h: int = 100):
    from src.config import StereoCfg
    from src.stereo_processor import StereoProcessor

    cfg = StereoCfg()
    cfg.zoom.min = 1.0
    cfg.zoom.max = 5.0
    cfg.zoom.step = 0.1
    cfg.convergence.base_offset = 0
    cfg.convergence.auto_adjust = False
    return StereoProcessor(cfg, eye_width=eye_w, eye_height=eye_h)


def test_process_pair_gpu_matches_cpu_zoom_1x():
    skip_if_no_cuda()
    import cv2

    p_cpu = _make_processor()
    p_gpu = _make_processor()

    rng = np.random.default_rng(4)
    frame_l = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)
    frame_r = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)

    _, _, sbs_cpu = p_cpu.process_pair(frame_l, frame_r)

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(frame_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(frame_r)
    gpu_sbs = cv2.cuda_GpuMat(p_gpu.eye_h, p_gpu.eye_w * 2, cv2.CV_8UC3)
    p_gpu.process_pair_gpu(gpu_l, gpu_r, gpu_sbs)
    sbs_gpu = gpu_sbs.download()

    diff = np.abs(sbs_cpu.astype(int) - sbs_gpu.astype(int)).max()
    assert diff <= 2, f"max abs diff was {diff}"


def test_process_pair_gpu_matches_cpu_zoom_2x():
    skip_if_no_cuda()
    import cv2

    p_cpu = _make_processor()
    p_gpu = _make_processor()
    p_cpu.zoom = 2.0
    p_gpu.zoom = 2.0

    rng = np.random.default_rng(5)
    frame_l = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)
    frame_r = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)

    _, _, sbs_cpu = p_cpu.process_pair(frame_l, frame_r)

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(frame_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(frame_r)
    gpu_sbs = cv2.cuda_GpuMat(p_gpu.eye_h, p_gpu.eye_w * 2, cv2.CV_8UC3)
    p_gpu.process_pair_gpu(gpu_l, gpu_r, gpu_sbs)
    sbs_gpu = gpu_sbs.download()

    diff = np.abs(sbs_cpu.astype(int) - sbs_gpu.astype(int)).max()
    assert diff <= 3, f"max abs diff was {diff}"
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_processor.py -v`

Expected: FAIL with `AttributeError: 'StereoProcessor' object has no attribute 'process_pair_gpu'`.

- [ ] **Step 3: Add `process_pair_gpu` to StereoProcessor**

In `src/stereo_processor.py`, add this method after `process_pair` (around line 199, before `process_pair_full_fov`):

```python
    def process_pair_gpu(
        self,
        gpu_l: "cv2.cuda_GpuMat",
        gpu_r: "cv2.cuda_GpuMat",
        gpu_sbs: "cv2.cuda_GpuMat",
    ) -> None:
        """GPU equivalent of `process_pair`.

        Crops each eye per zoom/convergence/joint-center, resizes into the
        left and right halves of `gpu_sbs` (a single 8UC3 GpuMat sized
        eye_h x eye_w*2). Mirrors the CPU `_resize_to_eye` 'fit' path used
        in normal viewing. Letterboxing is filled by clearing the SBS
        before each eye write.
        """
        h, w = gpu_l.size()[1], gpu_l.size()[0]  # size() returns (width, height)
        roi_w = int(w / self.zoom)
        roi_h = int(h / self.zoom)
        roi_w, roi_h = self._adjust_roi_aspect(roi_w, roi_h)

        cx = int(w * (self.joint_zoom_center / 100.0))
        cy = int(h * (self.joint_zoom_center_y / 100.0))
        offset = int(round(self.effective_offset))

        for side, gpu_in, dst_x in (
            ("left",  gpu_l, 0),
            ("right", gpu_r, self.eye_w),
        ):
            cx_eye = cx + offset if side == "left" else cx - offset
            x1 = max(cx_eye - roi_w // 2, 0)
            y1 = max(cy - roi_h // 2, 0)
            x2 = min(x1 + roi_w, w)
            y2 = min(y1 + roi_h, h)
            x1 = max(x2 - roi_w, 0)
            y1 = max(y2 - roi_h, 0)
            crop_w = x2 - x1
            crop_h = y2 - y1

            crop = cv2.cuda_GpuMat(gpu_in, (x1, y1, crop_w, crop_h))

            mode = getattr(self.cfg, "aspect_mode", "fit")
            scale = min(self.eye_w / max(1, crop_w), self.eye_h / max(1, crop_h))
            fit_w = max(1, min(self.eye_w, int(round(crop_w * scale))))
            fit_h = max(1, min(self.eye_h, int(round(crop_h * scale))))
            x0 = (self.eye_w - fit_w) // 2
            y0 = (self.eye_h - fit_h) // 2

            # Clear the destination half (letterbox).
            dst_half = cv2.cuda_GpuMat(gpu_sbs, (dst_x, 0, self.eye_w, self.eye_h))
            dst_half.setTo((0, 0, 0))

            # Write the resized crop into the centered region of the half.
            target_roi = cv2.cuda_GpuMat(
                gpu_sbs, (dst_x + x0, y0, fit_w, fit_h)
            )
            cv2.cuda.resize(
                crop, (fit_w, fit_h), dst=target_roi,
                interpolation=cv2.INTER_LINEAR,
            )
```

Note: this implementation assumes `aspect_mode == "fit"` (the default in config.yaml). The "crop" mode uses `_adjust_roi_aspect` which is already called above; the resize-to-eye math collapses to the same calculation. The `sbs_fit` and `_center_to_eye` paths are only used by full-FOV calibration views, which stay CPU-only per the spec's non-goals.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_processor.py -v`

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/stereo_processor.py tests/test_gpu_pipeline_processor.py
git commit -m "feat(stereo_processor): add GPU process_pair_gpu writing to SBS GpuMat"
```

---

## Task 6: Add `apply_nudge_gpu` to CalibrationOverlay

**Goal:** Apply per-eye nudge offsets and scale on GPU. The CPU version uses `np.roll` + `cv2.resize` + center-paste; the GPU version uses `cv2.cuda.warpAffine` with translation matrices and `cv2.cuda.resize`.

**Files:**
- Modify: `src/calibration.py` (add `apply_nudge_gpu`, keep `apply_nudge` unchanged)
- Create: `tests/test_gpu_pipeline_calibration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gpu_pipeline_calibration.py`:

```python
"""GPU apply_nudge matches CPU apply_nudge."""
from __future__ import annotations

import numpy as np

from tests._cuda_helpers import skip_if_no_cuda


def _make_overlay(nudge_x_l=5, nudge_y_l=-3, nudge_x_r=-2, nudge_y_r=4,
                  scale_l=100, scale_r=100):
    from src.config import CalibrationCfg
    from src.calibration import CalibrationOverlay

    overlay = CalibrationOverlay(CalibrationCfg())
    overlay.nudge_left = nudge_x_l
    overlay.nudge_right = nudge_x_r
    overlay.nudge_left_y = nudge_y_l
    overlay.nudge_right_y = nudge_y_r
    overlay.scale_left_pct = scale_l
    overlay.scale_right_pct = scale_r
    return overlay


def test_apply_nudge_gpu_matches_cpu_translation_only():
    skip_if_no_cuda()
    import cv2

    overlay = _make_overlay()
    rng = np.random.default_rng(6)
    eye_l = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
    eye_r = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)

    cpu_l, cpu_r = overlay.apply_nudge(eye_l.copy(), eye_r.copy())

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(eye_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(eye_r)
    overlay.apply_nudge_gpu(gpu_l, gpu_r)
    out_l = gpu_l.download()
    out_r = gpu_r.download()

    # warpAffine vs np.roll differ in border treatment; compare interior region.
    ix_l = slice(10, 70)
    iy = slice(10, 50)
    diff_l = np.abs(cpu_l[iy, ix_l].astype(int) - out_l[iy, ix_l].astype(int)).max()
    diff_r = np.abs(cpu_r[iy, ix_l].astype(int) - out_r[iy, ix_l].astype(int)).max()
    assert diff_l <= 2, f"left diff {diff_l}"
    assert diff_r <= 2, f"right diff {diff_r}"


def test_apply_nudge_gpu_zero_nudge_is_passthrough():
    skip_if_no_cuda()
    import cv2

    overlay = _make_overlay(nudge_x_l=0, nudge_y_l=0, nudge_x_r=0, nudge_y_r=0)
    rng = np.random.default_rng(7)
    eye_l = rng.integers(1, 256, size=(40, 50, 3), dtype=np.uint8)
    eye_r = rng.integers(1, 256, size=(40, 50, 3), dtype=np.uint8)

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(eye_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(eye_r)
    overlay.apply_nudge_gpu(gpu_l, gpu_r)
    assert np.array_equal(gpu_l.download(), eye_l)
    assert np.array_equal(gpu_r.download(), eye_r)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_calibration.py -v`

Expected: FAIL with `AttributeError: 'CalibrationOverlay' object has no attribute 'apply_nudge_gpu'`.

- [ ] **Step 3: Add `apply_nudge_gpu` to CalibrationOverlay**

In `src/calibration.py`, add this method right after `apply_nudge` (around line 177):

```python
    def apply_nudge_gpu(self, gpu_l: "cv2.cuda_GpuMat", gpu_r: "cv2.cuda_GpuMat") -> None:
        """In-place GPU equivalent of apply_nudge.

        For Phase 1 we only port the translation portion (scale_*_pct stays
        CPU; the UI defaults are 100 = no-op, so the typical hot path skips it).
        If a non-100 scale is requested, the call falls back to CPU for that
        eye via download/upload — slow but correct, and the surgeon will only
        encounter it when explicitly tweaking eye scale in calibration.
        """
        self._nudge_gpu_eye(gpu_l, self.nudge_left, self.nudge_left_y, self.scale_left_pct)
        self._nudge_gpu_eye(gpu_r, self.nudge_right, self.nudge_right_y, self.scale_right_pct)

    def _nudge_gpu_eye(
        self,
        gpu_eye: "cv2.cuda_GpuMat",
        dx: int,
        dy: int,
        scale_pct: int,
    ) -> None:
        if scale_pct != 100:
            arr = gpu_eye.download()
            arr = self._scale_eye(arr, scale_pct)
            gpu_eye.upload(arr)

        if dx == 0 and dy == 0:
            return

        size = gpu_eye.size()
        m = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
        out = cv2.cuda_GpuMat(size[1], size[0], gpu_eye.type())
        cv2.cuda.warpAffine(
            gpu_eye, m, size, dst=out,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        gpu_eye.upload(out.download())  # in-place semantic; cheap (small ROI)
```

The final `upload(out.download())` looks wasteful but `cv2.cuda_GpuMat` doesn't expose a direct copy-from-other-GpuMat assignment in this Python binding. The download/upload at eye-frame size (~2 MB at 1080p) is sub-millisecond and acceptable for Phase 1; if profiling shows it as a hotspot, replace with `out.copyTo(gpu_eye)` once we confirm the binding accepts that signature.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_calibration.py -v`

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/calibration.py tests/test_gpu_pipeline_calibration.py
git commit -m "feat(calibration): add GPU apply_nudge_gpu with translation on GPU"
```

---

## Task 7: Implement `GpuPipeline` orchestrator

**Goal:** A class that owns the persistent GpuMat buffers and runs the GPU-resident hot path: upload → align warp → fill holes → process zoom → calibration nudge → SBS download. Depth stays where it is (already GPU-side via `_compute_disparity_gpu_from_bgr`); the orchestrator just calls into it.

**Files:**
- Modify: `src/gpu_pipeline.py` (add `GpuPipeline` class)
- Create: `tests/test_gpu_pipeline_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gpu_pipeline_orchestrator.py`:

```python
"""End-to-end GpuPipeline matches CPU pipeline within tolerance."""
from __future__ import annotations

import numpy as np

from tests._cuda_helpers import skip_if_no_cuda


def _build_components(eye_w=100, eye_h=100, frame_w=80, frame_h=60):
    from src.config import StereoCfg, AlignmentCfg, CalibrationCfg
    from src.stereo_align import StereoAligner
    from src.stereo_processor import StereoProcessor
    from src.calibration import CalibrationOverlay

    stereo_cfg = StereoCfg()
    stereo_cfg.zoom.min = 1.0
    stereo_cfg.convergence.base_offset = 0
    stereo_cfg.convergence.auto_adjust = False

    aligner = StereoAligner(AlignmentCfg(enabled=False), frame_w, frame_h)
    processor = StereoProcessor(stereo_cfg, eye_w, eye_h)
    calibration = CalibrationOverlay(CalibrationCfg())
    calibration.nudge_left = 0
    calibration.nudge_right = 0
    return aligner, processor, calibration


def test_gpu_pipeline_end_to_end_matches_cpu_when_aligner_disabled():
    skip_if_no_cuda()
    from src.gpu_pipeline import GpuPipeline

    aligner, processor, calibration = _build_components()
    pipeline = GpuPipeline(aligner, processor, calibration)

    rng = np.random.default_rng(8)
    frame_l = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
    frame_r = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)

    sbs_gpu = pipeline.process(frame_l, frame_r)

    # Reference CPU run mirroring pipeline_worker._tick when aligner disabled
    cpu_l, cpu_r = frame_l.copy(), frame_r.copy()
    eye_l, eye_r, sbs_cpu = processor.process_pair(cpu_l, cpu_r)
    eye_l, eye_r = calibration.apply_nudge(eye_l, eye_r)

    diff = np.abs(sbs_cpu.astype(int) - sbs_gpu.astype(int)).max()
    assert diff <= 3, f"max diff {diff}"


def test_gpu_pipeline_releases_buffers_on_teardown():
    skip_if_no_cuda()
    from src.gpu_pipeline import GpuPipeline

    aligner, processor, calibration = _build_components()
    pipeline = GpuPipeline(aligner, processor, calibration)

    rng = np.random.default_rng(9)
    frame_l = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
    frame_r = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
    pipeline.process(frame_l, frame_r)
    pipeline.release()
    assert pipeline._gpu_in_l is None
    assert pipeline._gpu_sbs is None
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py -v`

Expected: FAIL with `ImportError: cannot import name 'GpuPipeline' from 'src.gpu_pipeline'`.

- [ ] **Step 3: Add `GpuPipeline` to `src/gpu_pipeline.py`**

Append to `src/gpu_pipeline.py`:

```python
class GpuPipeline:
    """Owns persistent cuda_GpuMat buffers and runs the GPU-resident hot path.

    Lifecycle:
        pipeline = GpuPipeline(aligner, processor, calibration)
        sbs = pipeline.process(frame_l, frame_r)   # called every frame
        ...
        pipeline.release()                          # on shutdown
    """

    def __init__(self, aligner, processor, calibration):
        self.aligner = aligner
        self.processor = processor
        self.calibration = calibration
        self._gpu_in_l: cv2.cuda_GpuMat | None = None
        self._gpu_in_r: cv2.cuda_GpuMat | None = None
        self._gpu_warp_l: cv2.cuda_GpuMat | None = None
        self._gpu_warp_r: cv2.cuda_GpuMat | None = None
        self._gpu_sbs: cv2.cuda_GpuMat | None = None
        self._frame_shape: tuple[int, int] | None = None

    def process(self, frame_l: np.ndarray, frame_r: np.ndarray) -> np.ndarray:
        """Run the GPU-resident pipeline; return the composed SBS ndarray."""
        h, w = frame_l.shape[:2]
        self._ensure_buffers(h, w)

        self._gpu_in_l.upload(frame_l)
        self._gpu_in_r.upload(frame_r)

        # 1. Stereo alignment warp (GPU). When the aligner has no correction
        #    yet, copy the input through unchanged.
        used_warp = self.aligner.warp_pair_gpu(
            self._gpu_in_l, self._gpu_in_r,
            self._gpu_warp_l, self._gpu_warp_r,
        )
        if used_warp:
            src_l, src_r = self._gpu_warp_l, self._gpu_warp_r
        else:
            src_l, src_r = self._gpu_in_l, self._gpu_in_r

        # 2. Fill holes from cross-eye copy (GPU).
        fill_holes_cross_gpu(src_l, src_r)

        # 3. Stereo processor crop+resize into SBS halves (GPU).
        self.processor.process_pair_gpu(src_l, src_r, self._gpu_sbs)

        # 4. Calibration nudge applied to each SBS half (GPU).
        eye_l_gpu = cv2.cuda_GpuMat(
            self._gpu_sbs, (0, 0, self.processor.eye_w, self.processor.eye_h)
        )
        eye_r_gpu = cv2.cuda_GpuMat(
            self._gpu_sbs,
            (self.processor.eye_w, 0, self.processor.eye_w, self.processor.eye_h),
        )
        self.calibration.apply_nudge_gpu(eye_l_gpu, eye_r_gpu)

        # 5. Single download for display.
        return self._gpu_sbs.download()

    def release(self) -> None:
        self._gpu_in_l = None
        self._gpu_in_r = None
        self._gpu_warp_l = None
        self._gpu_warp_r = None
        self._gpu_sbs = None
        self._frame_shape = None

    def _ensure_buffers(self, h: int, w: int) -> None:
        if self._frame_shape == (h, w):
            return
        self._gpu_in_l = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
        self._gpu_in_r = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
        self._gpu_warp_l = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
        self._gpu_warp_r = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
        self._gpu_sbs = cv2.cuda_GpuMat(
            self.processor.eye_h, self.processor.eye_w * 2, cv2.CV_8UC3
        )
        self._frame_shape = (h, w)
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py -v`

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/gpu_pipeline.py tests/test_gpu_pipeline_orchestrator.py
git commit -m "feat(gpu_pipeline): add GpuPipeline orchestrator with persistent buffers"
```

---

## Task 8: Wire `pipeline_worker.py` to use `GpuPipeline`

**Goal:** When `cfg.performance.use_gpu_pipeline` is true and CUDA is available, run the GPU pipeline; otherwise leave today's path untouched. Depth still uses `_compute_disparity_gpu_from_bgr` (already GPU); we just feed it the same input frames.

**Files:**
- Modify: `src/ui/pipeline_worker.py`

- [ ] **Step 1: Add a smoke test for the worker switch**

Append to `tests/test_gpu_pipeline_orchestrator.py`:

```python
def test_pipeline_worker_constructs_gpu_pipeline_when_flag_set(tmp_path, monkeypatch):
    skip_if_no_cuda()
    from src.config import PiccoloCfg
    from src.ui.pipeline_worker import PipelineWorker

    cfg = PiccoloCfg()
    cfg.cameras.test_mode = True
    cfg.performance.use_gpu_pipeline = True

    worker = PipelineWorker(cfg)
    assert worker._gpu_pipeline is not None, "GpuPipeline should be constructed"

    cfg2 = PiccoloCfg()
    cfg2.cameras.test_mode = True
    cfg2.performance.use_gpu_pipeline = False
    worker2 = PipelineWorker(cfg2)
    assert worker2._gpu_pipeline is None, "should be None when flag off"
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py::test_pipeline_worker_constructs_gpu_pipeline_when_flag_set -v`

Expected: FAIL with `AttributeError: 'PipelineWorker' object has no attribute '_gpu_pipeline'`.

- [ ] **Step 3: Construct the GpuPipeline in `__init__` when enabled**

In `src/ui/pipeline_worker.py`, in `__init__` after the existing `self._cuda_caps = ...` line (around line 106), add:

```python
        # New GPU-resident hot path (Phase 1). Only constructed when CUDA is
        # available and the flag is set, so falling back to CPU is automatic.
        self._use_gpu_pipeline = bool(
            self._cuda_available
            and getattr(cfg, "performance", None)
            and getattr(cfg.performance, "use_gpu_pipeline", False)
        )
        if self._use_gpu_pipeline:
            from ..gpu_pipeline import GpuPipeline
            self._gpu_pipeline = GpuPipeline(self.aligner, self.processor, self.calibration)
        else:
            self._gpu_pipeline = None
```

- [ ] **Step 4: Use the GPU pipeline in `_tick` when enabled**

In `src/ui/pipeline_worker.py`, replace the section in `_tick` from `frame_l, frame_r = self._apply_camera_flips(...)` (line 186) through the SBS-compose block ending at `sbs[:, self.processor.eye_w:] = eye_r` (line 200) with:

```python
        frame_l, frame_r = self._apply_camera_flips(frame_l, frame_r)
        low_latency = self._low_latency_enabled()

        if self._gpu_pipeline is not None and not low_latency:
            # GPU-resident hot path: align warp + fill holes + zoom resize +
            # nudge all stay on GPU; depth still uses _compute_disparity_gpu.
            t_warp = time.perf_counter()
            if (
                self.aligner.needs_update()
                and not self.cfg.calibration_state.alignment_locked
            ):
                self.aligner.update(frame_l, frame_r)
            sbs = self._gpu_pipeline.process(frame_l, frame_r)
            perf['align_warp_ms'] = (time.perf_counter() - t_warp) * 1000.0
            perf['fill_ms'] = 0.0  # folded into GPU pipeline
            perf['process_nudge_ms'] = 0.0  # folded into GPU pipeline
            self._last_depth_mm, perf['depth_ms'] = self._maybe_update_depth(
                frame_l, frame_r, low_latency
            )
        else:
            frame_l, frame_r, perf['align_warp_ms'] = self._maybe_update_and_warp(
                frame_l, frame_r, low_latency
            )
            try:
                t_fill = time.perf_counter()
                frame_l, frame_r = self._fill_holes_cross(frame_l, frame_r)
                perf['fill_ms'] = (time.perf_counter() - t_fill) * 1000.0
            except Exception:
                perf['fill_ms'] = 0.0
            self._last_depth_mm, perf['depth_ms'] = self._maybe_update_depth(
                frame_l, frame_r, low_latency
            )
            eye_l, eye_r, sbs, perf['process_nudge_ms'] = self._process_and_nudge(
                frame_l, frame_r
            )
            sbs[:, :self.processor.eye_w] = eye_l
            sbs[:, self.processor.eye_w:] = eye_r
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py -v`

Expected: 3 PASSED. Then run the full suite: `.venv\Scripts\python.exe -m pytest tests\ -x -q` to confirm no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/ui/pipeline_worker.py tests/test_gpu_pipeline_orchestrator.py
git commit -m "feat(pipeline_worker): switch to GpuPipeline when use_gpu_pipeline flag set"
```

---

## Task 9: Remove the `_can_process_now` throttle

**Goal:** Today's `_can_process_now` skips the entire pipeline if the last *emit* was less than `1/display.fps` ago. That adds up to one display interval (~16 ms at 60 Hz) of latency for fresh frames. We replace it with frame-id deduplication only — the existing `if frame_ids == self._last_frame_ids: msleep(1); return` already handles "no new camera frame" cleanly. The display emit can still be rate-limited if needed; we move that gate to the QImage emit specifically, not the whole pipeline.

**Files:**
- Modify: `src/ui/pipeline_worker.py`

- [ ] **Step 1: Write a failing test that the throttle no longer gates the pipeline**

Append to `tests/test_gpu_pipeline_orchestrator.py`:

```python
def test_pipeline_worker_does_not_gate_processing_on_emit_interval():
    skip_if_no_cuda()
    import time
    from src.config import PiccoloCfg
    from src.ui.pipeline_worker import PipelineWorker

    cfg = PiccoloCfg()
    cfg.cameras.test_mode = True
    cfg.display.fps = 30  # large interval (~33 ms)
    worker = PipelineWorker(cfg)

    # Even with a recent emit timestamp, _can_process_now must allow processing.
    worker._last_emit_t = time.perf_counter()
    assert worker._can_process_now(time.perf_counter()) is True
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py::test_pipeline_worker_does_not_gate_processing_on_emit_interval -v`

Expected: FAIL — `_can_process_now` currently returns False right after an emit.

- [ ] **Step 3: Make `_can_process_now` always return True**

In `src/ui/pipeline_worker.py`, replace the `_can_process_now` method (lines 213-219):

```python
    def _can_process_now(self, now: float) -> bool:
        # Phase 1: pipeline runs whenever a new camera frame is available;
        # frame-id dedup further down handles the "same frame again" case.
        # Display emit rate-limiting is moved to _maybe_emit_qimage so it
        # doesn't add latency to the pipeline path.
        return True
```

- [ ] **Step 4: Move emit rate-limiting to a helper that gates only the QImage emit**

In `src/ui/pipeline_worker.py`, replace the `_finalize_frame` method (lines 310-318) with:

```python
    def _finalize_frame(self, perf: dict[str, float], start_t: float, now: float) -> None:
        if now - self._last_status_emit_t >= self._status_interval:
            self._emit_status()
            self._last_status_emit_t = now
        # _last_emit_t is bookkeeping for status; we no longer gate the
        # pipeline on it, but it stays useful for FPS history.
        self._last_emit_t = now
        perf['total_frame_ms'] = (time.perf_counter() - start_t) * 1000.0
        for key, value in perf.items():
            self._perf_history[key].append(value)
        self._last_perf = perf
```

(This is the same logic, kept for clarity; no behavior change here other than the comment. The throttle change is in `_can_process_now`.)

- [ ] **Step 5: Run the tests and verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py -v`

Expected: all PASSED. Run the full suite: `.venv\Scripts\python.exe -m pytest tests\ -x -q`.

- [ ] **Step 6: Commit**

```bash
git add src/ui/pipeline_worker.py tests/test_gpu_pipeline_orchestrator.py
git commit -m "perf(pipeline_worker): remove _can_process_now throttle from hot path"
```

---

## Task 10: Add `cv2.cuda_Event` GPU-side timing

**Goal:** Today's `time.perf_counter` brackets include host wait time but don't measure GPU work directly. Add `cv2.cuda_Event` records around each GPU step so the perf print reports actual GPU-side ms when the GPU pipeline is active.

**Files:**
- Modify: `src/gpu_pipeline.py`

- [ ] **Step 1: Write a test that `process` returns timing info**

Append to `tests/test_gpu_pipeline_orchestrator.py`:

```python
def test_gpu_pipeline_returns_per_stage_timings():
    skip_if_no_cuda()
    import numpy as np
    from src.gpu_pipeline import GpuPipeline

    aligner, processor, calibration = _build_components()
    pipeline = GpuPipeline(aligner, processor, calibration)
    rng = np.random.default_rng(10)
    frame_l = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
    frame_r = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
    pipeline.process(frame_l, frame_r)

    timings = pipeline.last_timings_ms
    assert "warp_ms" in timings
    assert "fill_ms" in timings
    assert "process_ms" in timings
    assert "nudge_ms" in timings
    assert all(v >= 0.0 for v in timings.values())
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py::test_gpu_pipeline_returns_per_stage_timings -v`

Expected: FAIL with `AttributeError: 'GpuPipeline' object has no attribute 'last_timings_ms'`.

- [ ] **Step 3: Add cv2.cuda_Event timing in GpuPipeline.process**

In `src/gpu_pipeline.py`, modify the `GpuPipeline.__init__` to add the timing field:

```python
        self.last_timings_ms: dict[str, float] = {}
```

Then replace `process` with the timed version:

```python
    def process(self, frame_l: np.ndarray, frame_r: np.ndarray) -> np.ndarray:
        """Run the GPU-resident pipeline; return the composed SBS ndarray.

        Per-stage timings (GPU-side ms) are written to `self.last_timings_ms`.
        """
        h, w = frame_l.shape[:2]
        self._ensure_buffers(h, w)

        ev = {
            "start": cv2.cuda_Event(), "after_upload": cv2.cuda_Event(),
            "after_warp": cv2.cuda_Event(), "after_fill": cv2.cuda_Event(),
            "after_process": cv2.cuda_Event(), "after_nudge": cv2.cuda_Event(),
        }
        ev["start"].record()

        self._gpu_in_l.upload(frame_l)
        self._gpu_in_r.upload(frame_r)
        ev["after_upload"].record()

        used_warp = self.aligner.warp_pair_gpu(
            self._gpu_in_l, self._gpu_in_r,
            self._gpu_warp_l, self._gpu_warp_r,
        )
        if used_warp:
            src_l, src_r = self._gpu_warp_l, self._gpu_warp_r
        else:
            src_l, src_r = self._gpu_in_l, self._gpu_in_r
        ev["after_warp"].record()

        fill_holes_cross_gpu(src_l, src_r)
        ev["after_fill"].record()

        self.processor.process_pair_gpu(src_l, src_r, self._gpu_sbs)
        ev["after_process"].record()

        eye_l_gpu = cv2.cuda_GpuMat(
            self._gpu_sbs, (0, 0, self.processor.eye_w, self.processor.eye_h)
        )
        eye_r_gpu = cv2.cuda_GpuMat(
            self._gpu_sbs,
            (self.processor.eye_w, 0, self.processor.eye_w, self.processor.eye_h),
        )
        self.calibration.apply_nudge_gpu(eye_l_gpu, eye_r_gpu)
        ev["after_nudge"].record()
        ev["after_nudge"].waitForCompletion()

        self.last_timings_ms = {
            "upload_ms": cv2.cuda_Event.elapsedTime(ev["start"], ev["after_upload"]),
            "warp_ms":   cv2.cuda_Event.elapsedTime(ev["after_upload"], ev["after_warp"]),
            "fill_ms":   cv2.cuda_Event.elapsedTime(ev["after_warp"], ev["after_fill"]),
            "process_ms": cv2.cuda_Event.elapsedTime(ev["after_fill"], ev["after_process"]),
            "nudge_ms":  cv2.cuda_Event.elapsedTime(ev["after_process"], ev["after_nudge"]),
        }

        return self._gpu_sbs.download()
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py -v`

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/gpu_pipeline.py tests/test_gpu_pipeline_orchestrator.py
git commit -m "feat(gpu_pipeline): add cv2.cuda_Event per-stage GPU timing"
```

---

## Task 11: Performance regression test

**Goal:** Lock in the savings with a measurement that fails if the GPU path regresses below CPU at 640×480.

**Files:**
- Create: `tests/test_gpu_pipeline_perf.py`

- [ ] **Step 1: Write the perf test**

Create `tests/test_gpu_pipeline_perf.py`:

```python
"""Performance regression: GpuPipeline must beat the CPU path at 640x480.

Skipped unless run explicitly with -k perf so it doesn't slow down normal
test runs. Failure here is a real signal — investigate before merging.
"""
from __future__ import annotations

import time
import numpy as np
import pytest

from tests._cuda_helpers import skip_if_no_cuda


@pytest.mark.perf
def test_gpu_pipeline_beats_cpu_at_640x480():
    skip_if_no_cuda()
    from src.config import StereoCfg, AlignmentCfg, CalibrationCfg
    from src.stereo_align import StereoAligner
    from src.stereo_processor import StereoProcessor
    from src.calibration import CalibrationOverlay
    from src.gpu_pipeline import GpuPipeline

    eye_w, eye_h = 960, 1080
    frame_w, frame_h = 640, 480

    stereo_cfg = StereoCfg()
    stereo_cfg.zoom.min = 1.0
    stereo_cfg.convergence.base_offset = 0
    stereo_cfg.convergence.auto_adjust = False

    aligner = StereoAligner(AlignmentCfg(enabled=False), frame_w, frame_h)
    processor = StereoProcessor(stereo_cfg, eye_w, eye_h)
    calibration = CalibrationOverlay(CalibrationCfg())

    pipeline = GpuPipeline(aligner, processor, calibration)

    rng = np.random.default_rng(42)
    pairs = [
        (
            rng.integers(0, 256, size=(frame_h, frame_w, 3), dtype=np.uint8),
            rng.integers(0, 256, size=(frame_h, frame_w, 3), dtype=np.uint8),
        )
        for _ in range(60)
    ]

    # Warmup
    for fl, fr in pairs[:5]:
        pipeline.process(fl, fr)
        eye_l, eye_r, _ = processor.process_pair(fl, fr)
        calibration.apply_nudge(eye_l, eye_r)

    # CPU timing
    t0 = time.perf_counter()
    for fl, fr in pairs:
        eye_l, eye_r, sbs_cpu = processor.process_pair(fl, fr)
        calibration.apply_nudge(eye_l, eye_r)
    cpu_ms = (time.perf_counter() - t0) / len(pairs) * 1000.0

    # GPU timing
    t0 = time.perf_counter()
    for fl, fr in pairs:
        pipeline.process(fl, fr)
    gpu_ms = (time.perf_counter() - t0) / len(pairs) * 1000.0

    print(f"\n[perf] CPU median per-frame: {cpu_ms:.2f} ms")
    print(f"[perf] GPU median per-frame: {gpu_ms:.2f} ms")
    assert gpu_ms + 3.0 <= cpu_ms, (
        f"GPU pipeline ({gpu_ms:.2f} ms) must be ≥ 3 ms faster than CPU "
        f"({cpu_ms:.2f} ms) at 640x480"
    )
```

- [ ] **Step 2: Add the `perf` marker to pytest config so `-k perf` works**

If `pyproject.toml` or `pytest.ini` doesn't already declare markers, that's fine — pytest accepts unknown markers with a warning. Skip this step unless an existing config explicitly enforces marker registration.

- [ ] **Step 3: Run the perf test and verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_perf.py -v -s`

Expected: PASSED. Capture the printed CPU and GPU per-frame times for the PR description.

If it fails: the GPU pipeline didn't beat the CPU path by 3 ms. Likely causes: ROI views not zero-copy on this build (Task 1 spike would have caught the most severe form), small-frame upload overhead dominating, or `_nudge_gpu_eye`'s download/upload roundtrip in Task 6 being a hotspot. Investigate before merging — don't lower the threshold.

- [ ] **Step 4: Commit**

```bash
git add tests/test_gpu_pipeline_perf.py
git commit -m "test: perf regression — GpuPipeline ≥3 ms faster than CPU at 640x480"
```

---

## Final verification

- [ ] **Step 1: Run the full test suite**

Run: `.venv\Scripts\python.exe -m pytest tests\ -x -q`

Expected: all tests pass. No skips on this machine since CUDA is available. If a non-CUDA machine runs the suite, the `skip_if_no_cuda()` calls correctly mark them skipped.

- [ ] **Step 2: Smoke-test the live app**

Run: `.venv\Scripts\python.exe piccolo.py`

Expected behavior:
- Startup log includes `[Pipeline] Initialized in GPU-ACCELERATED mode`.
- Live SBS frame appears in the headset.
- Status print shows non-zero `total_frame_ms`, `depth_ms`, `qimage_ms`.
- Toggle `performance.use_gpu_pipeline: false` in `config.yaml` and restart — confirm the original code path still works and produces visually identical output.

- [ ] **Step 3: Capture before/after numbers**

Compare median `total_frame_ms` from the status prints across `use_gpu_pipeline: true` vs. `false` over ~30 seconds of live operation. Record both values for the PR description. Phase 1 success criterion: GPU median is at least 3 ms faster than CPU median, ideally 5–10 ms faster, on the test hardware.

- [ ] **Step 4: Final commit if needed**

If any cleanup (lint warnings, stray prints) was needed:

```bash
git add -p
git commit -m "chore: final cleanup after Phase 1 GPU pipeline rewrite"
```

---

## What this plan does NOT do

These are explicitly out of scope and have separate plans:

- **Phase 2 — resolution toggle.** Adding `cameras.<eye>.fps` and a UI dropdown for resolution.
- **Phase 3 — TurboJPEG decode.** Faster CPU decode in the grab thread.
- **Phase 4 — QOpenGLWidget display.** Replace QImage path with GL texture for ~2 ms saving.
- **Phase 5 — true GPU MJPEG decode.** Custom `cv2.cudacodec.RawVideoSource` wrapping a DirectShow Sample Grabber. Research-grade.

After Phase 1 ships and we have measurements, we'll plan whichever of 2–4 looks most worthwhile based on real numbers. Phase 5 stays a stretch goal we attempt only if 2–4 still leave latency on the table.
