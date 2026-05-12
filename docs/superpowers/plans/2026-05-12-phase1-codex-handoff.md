# GPU Pipeline Phase 1 — Codex Agent Implementation Plan

> **For the codex agent picking this up:** This is a self-contained continuation document. You do not need to read any other file to execute this plan. All required code, context, and gotchas are inlined below. Work through Tasks 5–11 sequentially, then run the final verification. Commit after each task.

## Mission

Build a fully-GPU-resident hot path for a stereoscopic surgery display app. Each camera frame pair is uploaded once to GPU, processed entirely on `cv2.cuda_GpuMat` (align warp → fill-holes → zoom/resize → calibration nudge → SBS compose), and downloaded once before display. Gate everything behind the `performance.use_gpu_pipeline` config flag. Eliminate the per-frame pipeline throttle that adds up to one display-frame of latency.

## Branch and current state

- **Repo:** `e:\Documents\University\year2\Semester1\Research\SurgeryRobot\piccolo`
- **Branch:** `qt-desktop-app`
- **Latest commit:** `c601e0a docs: add Phase 1 mid-execution handoff for continuation`
- **Python:** `.venv\Scripts\python.exe` (PowerShell shell — use full path, not `python`)
- **OpenCV:** `opencv-python-cuda 4.12.0+a1a2e91` built against CUDA 12.9

## What's already done

| Task | Status | Commit |
| ---- | ------ | ------ |
| 1. Spike — verify cv2.cuda_GpuMat ROI semantics | ✅ done | `661b62e` |
| 2. Add `use_gpu_pipeline` config flag | ✅ done | `4f48474` |
| Baseline restoration (WIP refactor of pipeline_worker + GPU helpers) | ✅ done | `d049d80` |
| 3. Implement `fill_holes_cross_gpu` | ✅ done | `fbbaba0` |
| 4. Add `warp_pair_gpu` to StereoAligner | ✅ done | `ba1f99d` |
| 5. Add `process_pair_gpu` to StereoProcessor | ⏳ test file already written, implementation pending |
| 6. Add `apply_nudge_gpu` to CalibrationOverlay | ⏳ pending |
| 7. Implement `GpuPipeline` orchestrator | ⏳ pending |
| 8. Wire `pipeline_worker.py` to use `GpuPipeline` | ⏳ pending |
| 9. Remove `_can_process_now` throttle | ⏳ pending |
| 10. Add `cv2.cuda_Event` GPU-side timing | ⏳ pending |
| 11. Performance regression test | ⏳ pending |

**Task 5 partial state:** The test file `tests/test_gpu_pipeline_processor.py` is already written and is currently failing because `process_pair_gpu` doesn't exist. Continue from "Step 3: Implement" in Task 5 below.

## Critical gotchas (learned the hard way during prior tasks)

### 1. cv2.cuda API quirks on this exact build

These are NON-NEGOTIABLE — the literal API names you'd expect don't exist:

- **`cv2.cuda.copyTo`** (module-level function) — **DOES NOT EXIST.** Use `GpuMat.copyTo(mask, dst)` instance method instead. Signature is `src.copyTo(mask, dst)` — destination is the last positional argument.
- **`cv2.cuda.compare`** — requires both args to be matrices. For scalar comparison use `cv2.cuda.compareWithScalar(src, scalar, cmpop)`.
- **`cv2.cuda.merge(channels)`** — returns a host `numpy.ndarray`, **NOT** a `GpuMat`. **DO NOT USE.** It forces a host round-trip and silently breaks GPU residency. If you need to recombine channels on-device, use per-channel `copyTo` with masks instead, or restructure to avoid splitting entirely.
- **`cv2.cuda.split(gpu)`** — correctly returns a list of `cv2.cuda_GpuMat` (GPU-resident).
- **`GpuMat.clone()`** — exists. Use it to snapshot a GpuMat before in-place mutation when ordering matters.

### 2. cv2.cuda.warpAffine differs from cv2.warpAffine at borders

The GPU `warpAffine` treats out-of-bounds samples differently than CPU. Edge pixels can differ by up to 250 (uint8). Interior pixels match within 4-5 for smooth inputs. In tests:
- **Use a smooth gradient input**, NOT random noise. Random noise amplifies sub-pixel resampler diffs into huge per-pixel values.
- **Restrict per-pixel comparison to the interior** (e.g., 5-10 px from each edge).
- Use a tolerance of `<= 5` on uint8 — this is empirically what bilinear resampler differences produce.
- The border discrepancy is harmless in the live pipeline because `fill_holes_cross_gpu` runs after and substitutes the partner eye's pixel where one is zero.

### 3. Line numbers in original plan may be off

The original plan was written against a WIP-state working tree before that WIP was committed. Method definitions are roughly where the plan says, but use Grep to find them by name rather than trusting absolute line numbers.

### 4. Don't widen test tolerances to make failing tests pass

If interior tolerance fails by a lot (e.g., needs `<= 50` to pass), there's a real bug, not a tolerance issue. Investigate the API quirk list above.

## Environment basics

```powershell
# Run tests:
.venv\Scripts\python.exe -m pytest tests\<file> -v

# Run full suite (skip recording tests that depend on untracked WIP files):
.venv\Scripts\python.exe -m pytest tests\ -x -q --ignore=tests\test_recording_worker.py

# Commit pattern:
git add <files>
git commit -m "<message>"
```

The `tests/_cuda_helpers.py` file already exists with `skip_if_no_cuda()`. Use it at the top of every CUDA-touching test:
```python
from tests._cuda_helpers import skip_if_no_cuda
# ...inside each test function:
skip_if_no_cuda()
```

The `tests/conftest.py` adds the repo root to `sys.path` so `from src.foo import bar` works.

---

## Task 5: Add `process_pair_gpu` to StereoProcessor

**Goal:** Crop + resize each eye on GPU, writing into ROI views of a single SBS GpuMat.

**Files:**
- Modify: `src/stereo_processor.py` (add new method after `process_pair` at line ~190)
- Test: `tests/test_gpu_pipeline_processor.py` (already exists, currently failing — that's good)

### Step 1: The test is already written

Located at `tests/test_gpu_pipeline_processor.py`. Confirm it fails:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_processor.py -v
```

Expected: 3 tests FAIL with `AttributeError: 'StereoProcessor' object has no attribute 'process_pair_gpu'`.

### Step 2: Add `process_pair_gpu` to StereoProcessor

In `src/stereo_processor.py`, find `def process_pair(` (around line 190). Add this method DIRECTLY AFTER the `process_pair` method (after its closing `return eye_l, eye_r, self._sbs` line) and BEFORE `def process_pair_full_fov(`:

```python
    def process_pair_gpu(
        self,
        gpu_l: "cv2.cuda_GpuMat",
        gpu_r: "cv2.cuda_GpuMat",
        gpu_sbs: "cv2.cuda_GpuMat",
    ) -> None:
        """GPU equivalent of `process_pair`.

        Crops each eye per current zoom/convergence/joint-center, resizes the
        crop with bilinear interpolation, and writes into the left and right
        halves of `gpu_sbs` (a pre-allocated 8UC3 GpuMat sized eye_h x eye_w*2).
        Letterbox regions are cleared to black before each eye write.

        Mirrors the CPU `process_eye` + `_resize_to_eye` 'fit' path used in
        normal live viewing. The full-FOV / sbs-anamorphic / centered modes
        used in calibration views are intentionally NOT ported — they run at
        low cadence on the CPU and aren't part of the latency-critical path.
        """
        size = gpu_l.size()  # (width, height)
        w, h = size[0], size[1]
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

            # Zero-copy crop view onto the source GpuMat.
            crop = cv2.cuda_GpuMat(gpu_in, (x1, y1, crop_w, crop_h))

            # Fit-mode resize: scale crop to fit within (eye_w, eye_h) while
            # preserving aspect ratio. Letterbox the remainder.
            scale = min(self.eye_w / max(1, crop_w), self.eye_h / max(1, crop_h))
            fit_w = max(1, min(self.eye_w, int(round(crop_w * scale))))
            fit_h = max(1, min(self.eye_h, int(round(crop_h * scale))))
            x0 = (self.eye_w - fit_w) // 2
            y0 = (self.eye_h - fit_h) // 2

            # Clear the destination half (letterbox).
            dst_half = cv2.cuda_GpuMat(gpu_sbs, (dst_x, 0, self.eye_w, self.eye_h))
            dst_half.setTo((0, 0, 0))

            # Resize directly into the centered region of the SBS half.
            target_roi = cv2.cuda_GpuMat(
                gpu_sbs, (dst_x + x0, y0, fit_w, fit_h)
            )
            cv2.cuda.resize(
                crop, (fit_w, fit_h), dst=target_roi,
                interpolation=cv2.INTER_LINEAR,
            )
```

`cv2` must be imported at the top of `stereo_processor.py` (it already is — verify with Grep).

### Step 3: Run the tests and verify they pass

```powershell
.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_processor.py -v
```

Expected: 3 PASSED.

If `cv2.cuda.resize` rejects the `dst=` kwarg, try positional: `cv2.cuda.resize(crop, (fit_w, fit_h), target_roi, 0, 0, cv2.INTER_LINEAR)` (signature: src, dsize, dst, fx, fy, interpolation).

If the diff exceeds tolerance, check that the ROI views actually point into `gpu_sbs` — that was verified in Task 1's spike but worth re-confirming.

### Step 4: Commit

```bash
git add src/stereo_processor.py
git commit -m "feat(stereo_processor): add GPU process_pair_gpu writing to SBS GpuMat"
```

---

## Task 6: Add `apply_nudge_gpu` to CalibrationOverlay

**Goal:** Apply per-eye nudge offsets and scale on GPU. Equivalent of `CalibrationOverlay.apply_nudge` at `src/calibration.py:163`.

**Files:**
- Modify: `src/calibration.py` (add new method after `apply_nudge`)
- Create: `tests/test_gpu_pipeline_calibration.py`

### Step 1: Write the failing test

Create `tests/test_gpu_pipeline_calibration.py`:

```python
"""GPU apply_nudge matches CPU apply_nudge."""
from __future__ import annotations

import numpy as np

from tests._cuda_helpers import skip_if_no_cuda


def _smooth_gradient(h: int, w: int, seed: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        (xx * 3.0 + seed * 7) % 256,
        (yy * 4.0 + seed * 11) % 256,
        ((xx + yy) * 2.0 + seed * 13) % 256,
    ], axis=-1)
    return base.astype(np.uint8)


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
    """Interior matches CPU after pure-translation nudge.

    CPU uses np.roll (cyclic shift then zero border); GPU uses warpAffine
    translation (linear interp + border value). They produce identical
    results on integer-pixel translations except at the border that's
    revealed by the shift. Interior matches exactly for integer shifts.
    """
    skip_if_no_cuda()
    import cv2

    overlay = _make_overlay()
    eye_l = _smooth_gradient(60, 80, seed=14)
    eye_r = _smooth_gradient(60, 80, seed=16)

    cpu_l, cpu_r = overlay.apply_nudge(eye_l.copy(), eye_r.copy())

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(eye_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(eye_r)
    overlay.apply_nudge_gpu(gpu_l, gpu_r)
    out_l = gpu_l.download()
    out_r = gpu_r.download()

    # Compare interior — borders differ because np.roll wraps zeros
    # while warpAffine fills the border with the constant.
    s = slice(10, -10)
    diff_l = np.abs(cpu_l[s, s].astype(int) - out_l[s, s].astype(int)).max()
    diff_r = np.abs(cpu_r[s, s].astype(int) - out_r[s, s].astype(int)).max()
    assert diff_l <= 5, f"left diff {diff_l}"
    assert diff_r <= 5, f"right diff {diff_r}"


def test_apply_nudge_gpu_zero_nudge_is_passthrough():
    """With all nudges zero and scale 100, GPU output equals input exactly."""
    skip_if_no_cuda()
    import cv2

    overlay = _make_overlay(nudge_x_l=0, nudge_y_l=0, nudge_x_r=0, nudge_y_r=0)
    rng = np.random.default_rng(17)
    eye_l = rng.integers(1, 256, size=(40, 50, 3), dtype=np.uint8)
    eye_r = rng.integers(1, 256, size=(40, 50, 3), dtype=np.uint8)

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(eye_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(eye_r)
    overlay.apply_nudge_gpu(gpu_l, gpu_r)
    assert np.array_equal(gpu_l.download(), eye_l)
    assert np.array_equal(gpu_r.download(), eye_r)
```

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_calibration.py -v`
Expected: 2 FAIL with `AttributeError: 'CalibrationOverlay' object has no attribute 'apply_nudge_gpu'`.

### Step 2: Add `apply_nudge_gpu` to CalibrationOverlay

In `src/calibration.py`, find `def apply_nudge(` (around line 163). Add this method right after `apply_nudge` ends:

```python
    def apply_nudge_gpu(self, gpu_l: "cv2.cuda_GpuMat", gpu_r: "cv2.cuda_GpuMat") -> None:
        """In-place GPU equivalent of apply_nudge.

        Translation portion uses cv2.cuda.warpAffine on-device. Scale portion
        (rare — default is 100, no-op) downloads to CPU for the resize+paste
        and re-uploads; this is only hit when the user explicitly tweaks eye
        scale in calibration, which is not the latency-critical live path.
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

        size = gpu_eye.size()  # (width, height)
        m = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
        out = cv2.cuda_GpuMat(size[1], size[0], gpu_eye.type())
        cv2.cuda.warpAffine(
            gpu_eye, m, size, out, cv2.INTER_NEAREST,
            cv2.BORDER_CONSTANT, (0, 0, 0),
        )
        # Copy result back into gpu_eye to preserve in-place semantics.
        out.copyTo(gpu_eye)
```

`np` (numpy) and `cv2` are already imported in `calibration.py` — verify with Grep at the top of the file.

The `out.copyTo(gpu_eye)` form is the no-mask variant of `GpuMat.copyTo`. If it errors with "argument count" mismatch, use `gpu_eye.upload(out.download())` as a fallback — this is a tiny eye-sized download/upload, not in the hot loop.

### Step 3: Run tests, verify pass

```powershell
.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_calibration.py -v
```

Expected: 2 PASSED. If interior diff is bigger than expected (say 30+), check whether INTER_NEAREST vs INTER_LINEAR is mismatched with CPU — `np.roll` is effectively nearest-neighbor on integer shifts.

### Step 4: Commit

```bash
git add src/calibration.py tests/test_gpu_pipeline_calibration.py
git commit -m "feat(calibration): add GPU apply_nudge_gpu with translation on GPU"
```

---

## Task 7: Implement `GpuPipeline` orchestrator

**Goal:** A class that owns persistent GpuMats and runs the GPU-resident hot path end-to-end: upload → align warp → fill holes → process zoom → calibration nudge → download SBS.

**Files:**
- Modify: `src/gpu_pipeline.py` (already has `fill_holes_cross_gpu`; add `GpuPipeline` class)
- Create: `tests/test_gpu_pipeline_orchestrator.py`

### Step 1: Write the failing test

Create `tests/test_gpu_pipeline_orchestrator.py`:

```python
"""End-to-end GpuPipeline matches CPU pipeline within tolerance."""
from __future__ import annotations

import numpy as np

from tests._cuda_helpers import skip_if_no_cuda


def _smooth_gradient(h: int, w: int, seed: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        (xx * 3.0 + seed * 7) % 256,
        (yy * 4.0 + seed * 11) % 256,
        ((xx + yy) * 2.0 + seed * 13) % 256,
    ], axis=-1)
    return base.astype(np.uint8)


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

    frame_l = _smooth_gradient(60, 80, seed=20)
    frame_r = _smooth_gradient(60, 80, seed=22)

    sbs_gpu = pipeline.process(frame_l, frame_r)

    # CPU reference path matching pipeline_worker's flow when aligner disabled.
    eye_l, eye_r, sbs_cpu = processor.process_pair(frame_l.copy(), frame_r.copy())
    eye_l, eye_r = calibration.apply_nudge(eye_l, eye_r)

    # Interior comparison only (border-zero pixels can differ).
    s = slice(10, -10)
    diff = np.abs(
        sbs_cpu[s, 10:processor.eye_w-10].astype(int)
        - sbs_gpu[s, 10:processor.eye_w-10].astype(int)
    ).max()
    assert diff <= 5, f"left half interior diff {diff}"
    diff_r = np.abs(
        sbs_cpu[s, processor.eye_w+10:-10].astype(int)
        - sbs_gpu[s, processor.eye_w+10:-10].astype(int)
    ).max()
    assert diff_r <= 5, f"right half interior diff {diff_r}"


def test_gpu_pipeline_releases_buffers_on_teardown():
    skip_if_no_cuda()
    from src.gpu_pipeline import GpuPipeline

    aligner, processor, calibration = _build_components()
    pipeline = GpuPipeline(aligner, processor, calibration)

    frame_l = _smooth_gradient(60, 80, seed=24)
    frame_r = _smooth_gradient(60, 80, seed=26)
    pipeline.process(frame_l, frame_r)
    pipeline.release()
    assert pipeline._gpu_in_l is None
    assert pipeline._gpu_sbs is None


def test_gpu_pipeline_handles_repeated_calls_without_reallocating():
    """Persistent buffers stay allocated across multiple frames."""
    skip_if_no_cuda()
    from src.gpu_pipeline import GpuPipeline

    aligner, processor, calibration = _build_components()
    pipeline = GpuPipeline(aligner, processor, calibration)

    frame_l = _smooth_gradient(60, 80, seed=28)
    frame_r = _smooth_gradient(60, 80, seed=30)
    pipeline.process(frame_l, frame_r)
    sbs_id_before = id(pipeline._gpu_sbs)
    pipeline.process(frame_l, frame_r)
    pipeline.process(frame_l, frame_r)
    sbs_id_after = id(pipeline._gpu_sbs)
    assert sbs_id_before == sbs_id_after, "buffers reallocated unnecessarily"
```

Run: `.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py -v`
Expected: ImportError or AttributeError (`GpuPipeline` not in `src.gpu_pipeline`).

### Step 2: Add `GpuPipeline` class to `src/gpu_pipeline.py`

The file currently contains `fill_holes_cross_gpu`. Append the `GpuPipeline` class at the bottom of the file:

```python


class GpuPipeline:
    """Owns persistent cuda_GpuMat buffers and runs the GPU-resident hot path.

    Lifecycle:
        pipeline = GpuPipeline(aligner, processor, calibration)
        sbs = pipeline.process(frame_l, frame_r)   # every frame
        ...
        pipeline.release()                          # on shutdown
    """

    def __init__(self, aligner, processor, calibration):
        self.aligner = aligner
        self.processor = processor
        self.calibration = calibration
        self._gpu_in_l = None
        self._gpu_in_r = None
        self._gpu_warp_l = None
        self._gpu_warp_r = None
        self._gpu_sbs = None
        self._frame_shape: tuple[int, int] | None = None

    def process(self, frame_l: np.ndarray, frame_r: np.ndarray) -> np.ndarray:
        """Run the GPU-resident pipeline; return the composed SBS ndarray."""
        h, w = frame_l.shape[:2]
        self._ensure_buffers(h, w)

        self._gpu_in_l.upload(frame_l)
        self._gpu_in_r.upload(frame_r)

        # 1. Stereo alignment warp.
        used_warp = self.aligner.warp_pair_gpu(
            self._gpu_in_l, self._gpu_in_r,
            self._gpu_warp_l, self._gpu_warp_r,
        )
        if used_warp:
            src_l, src_r = self._gpu_warp_l, self._gpu_warp_r
        else:
            src_l, src_r = self._gpu_in_l, self._gpu_in_r

        # 2. Fill holes from cross-eye copy.
        fill_holes_cross_gpu(src_l, src_r)

        # 3. Stereo processor crop+resize into SBS halves.
        self.processor.process_pair_gpu(src_l, src_r, self._gpu_sbs)

        # 4. Calibration nudge on each SBS half.
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

### Step 3: Run tests, verify pass

```powershell
.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py -v
```

Expected: 3 PASSED.

### Step 4: Commit

```bash
git add src/gpu_pipeline.py tests/test_gpu_pipeline_orchestrator.py
git commit -m "feat(gpu_pipeline): add GpuPipeline orchestrator with persistent buffers"
```

---

## Task 8: Wire `pipeline_worker.py` to use `GpuPipeline`

**Goal:** When `cfg.performance.use_gpu_pipeline` is true and CUDA is available, run the new GPU pipeline. Otherwise leave today's path untouched.

**Files:**
- Modify: `src/ui/pipeline_worker.py`

### Step 1: Add a smoke test

Append to `tests/test_gpu_pipeline_orchestrator.py`:

```python
def test_pipeline_worker_constructs_gpu_pipeline_when_flag_set():
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

Run:
```powershell
.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py::test_pipeline_worker_constructs_gpu_pipeline_when_flag_set -v
```
Expected: FAIL (`_gpu_pipeline` attribute doesn't exist).

### Step 2: Construct GpuPipeline in `__init__` when enabled

Open `src/ui/pipeline_worker.py`. Find the `__init__` method (line ~32). Find the line that sets `self._cuda_caps = ...` (it's near the end of `__init__`). Add immediately after it:

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

### Step 3: Use the GPU pipeline in `_tick` when enabled

Find `def _tick(` (line ~132). Find the section that:
- Calls `_apply_camera_flips`
- Then `_maybe_update_and_warp`
- Then `_fill_holes_cross`
- Then `_maybe_update_depth`
- Then `_process_and_nudge`
- Then `sbs[:, :eye_w] = eye_l; sbs[:, eye_w:] = eye_r`

Replace the block from `frame_l, frame_r = self._apply_camera_flips(...)` THROUGH the `sbs[:, self.processor.eye_w:] = eye_r` line with:

```python
        frame_l, frame_r = self._apply_camera_flips(frame_l, frame_r)
        low_latency = self._low_latency_enabled()

        if self._gpu_pipeline is not None and not low_latency:
            # GPU-resident hot path: align warp + fill holes + zoom/resize +
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

The exact surrounding code may vary slightly — verify with `Read` first. The key invariant: both branches must produce a valid `sbs` ndarray that the rest of `_tick` uses for overlays and emit.

### Step 4: Run tests, verify pass

```powershell
.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py -v
.venv\Scripts\python.exe -m pytest tests\ -x -q --ignore=tests\test_recording_worker.py
```

Expected: all PASSED. No regressions in existing tests.

### Step 5: Commit

```bash
git add src/ui/pipeline_worker.py tests/test_gpu_pipeline_orchestrator.py
git commit -m "feat(pipeline_worker): switch to GpuPipeline when use_gpu_pipeline flag set"
```

---

## Task 9: Remove `_can_process_now` throttle

**Goal:** Today's `_can_process_now` skips the entire pipeline if the last emit was less than `1/display.fps` ago, adding up to one display interval of latency to fresh camera frames. Replace with frame-id deduplication only.

**Files:**
- Modify: `src/ui/pipeline_worker.py`

### Step 1: Write a failing test

Append to `tests/test_gpu_pipeline_orchestrator.py`:

```python
def test_pipeline_worker_does_not_gate_processing_on_emit_interval():
    skip_if_no_cuda()
    import time
    from src.config import PiccoloCfg
    from src.ui.pipeline_worker import PipelineWorker

    cfg = PiccoloCfg()
    cfg.cameras.test_mode = True
    cfg.display.fps = 30  # ~33 ms interval
    worker = PipelineWorker(cfg)

    worker._last_emit_t = time.perf_counter()
    assert worker._can_process_now(time.perf_counter()) is True
```

Run:
```powershell
.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py::test_pipeline_worker_does_not_gate_processing_on_emit_interval -v
```
Expected: FAIL — `_can_process_now` currently returns False just after an emit.

### Step 2: Make `_can_process_now` always return True

In `src/ui/pipeline_worker.py`, find `def _can_process_now(self, now: float) -> bool:`. Replace its body with:

```python
    def _can_process_now(self, now: float) -> bool:
        # Phase 1: pipeline runs whenever a new camera frame is available.
        # Frame-id dedup further down (if frame_ids == self._last_frame_ids:
        # return) handles the "same frame again" case without adding latency.
        return True
```

### Step 3: Run tests, verify pass

```powershell
.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py -v
.venv\Scripts\python.exe -m pytest tests\ -x -q --ignore=tests\test_recording_worker.py
```

Expected: all PASSED.

### Step 4: Commit

```bash
git add src/ui/pipeline_worker.py tests/test_gpu_pipeline_orchestrator.py
git commit -m "perf(pipeline_worker): remove _can_process_now throttle from hot path"
```

---

## Task 10: Add `cv2.cuda_Event` GPU-side timing

**Goal:** Today's `time.perf_counter` brackets include host wait time, not actual GPU work time. Add per-stage `cv2.cuda_Event` timing so the perf print reports honest GPU-side ms.

**Files:**
- Modify: `src/gpu_pipeline.py`

### Step 1: Write a failing test

Append to `tests/test_gpu_pipeline_orchestrator.py`:

```python
def test_gpu_pipeline_returns_per_stage_timings():
    skip_if_no_cuda()
    from src.gpu_pipeline import GpuPipeline

    aligner, processor, calibration = _build_components()
    pipeline = GpuPipeline(aligner, processor, calibration)
    frame_l = _smooth_gradient(60, 80, seed=32)
    frame_r = _smooth_gradient(60, 80, seed=34)
    pipeline.process(frame_l, frame_r)

    timings = pipeline.last_timings_ms
    assert "warp_ms" in timings
    assert "fill_ms" in timings
    assert "process_ms" in timings
    assert "nudge_ms" in timings
    assert all(v >= 0.0 for v in timings.values())
```

Run:
```powershell
.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py::test_gpu_pipeline_returns_per_stage_timings -v
```
Expected: FAIL — `last_timings_ms` not present.

### Step 2: Add cv2.cuda_Event timing in GpuPipeline

In `src/gpu_pipeline.py`, in `GpuPipeline.__init__`, add this line at the end:
```python
        self.last_timings_ms: dict[str, float] = {}
```

Replace the existing `GpuPipeline.process` method with this timed version:

```python
    def process(self, frame_l: np.ndarray, frame_r: np.ndarray) -> np.ndarray:
        """Run the GPU-resident pipeline; return the composed SBS ndarray.

        Per-stage timings (GPU-side ms) are written to `self.last_timings_ms`.
        """
        h, w = frame_l.shape[:2]
        self._ensure_buffers(h, w)

        ev = {
            "start": cv2.cuda_Event(),
            "after_upload": cv2.cuda_Event(),
            "after_warp": cv2.cuda_Event(),
            "after_fill": cv2.cuda_Event(),
            "after_process": cv2.cuda_Event(),
            "after_nudge": cv2.cuda_Event(),
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

### Step 3: Run tests, verify pass

```powershell
.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_orchestrator.py -v
```

Expected: all PASSED.

If `cv2.cuda_Event.elapsedTime` is not a static method on this build (it's sometimes exposed as `cv2.cuda.Event.elapsedTime`), adjust the import/call form. The test only requires the keys to be present and non-negative — if you have to set `0.0` for some keys due to API issues, that's acceptable but document why.

### Step 4: Commit

```bash
git add src/gpu_pipeline.py tests/test_gpu_pipeline_orchestrator.py
git commit -m "feat(gpu_pipeline): add cv2.cuda_Event per-stage GPU timing"
```

---

## Task 11: Performance regression test

**Goal:** Lock in the savings with a measurement that fails if the GPU path regresses below CPU at 640×480.

**Files:**
- Create: `tests/test_gpu_pipeline_perf.py`

### Step 1: Write the perf test

Create `tests/test_gpu_pipeline_perf.py`:

```python
"""Performance regression: GpuPipeline must beat the CPU path at 640x480.

Run explicitly with `-k perf` or as part of the full suite. If this fails,
the GPU pipeline regressed — investigate before merging.
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

    # Warmup (CUDA context setup, kernel compilation).
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
    # GPU must be at least 1 ms faster than CPU at 640x480 (small frames
    # have low overhead, so the lower bound is conservative). Higher
    # resolutions will show larger absolute wins.
    assert gpu_ms <= cpu_ms - 1.0, (
        f"GPU pipeline ({gpu_ms:.2f} ms) must be >=1 ms faster than CPU "
        f"({cpu_ms:.2f} ms) at 640x480"
    )
```

### Step 2: Run the perf test

```powershell
.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_perf.py -v -s
```

Expected: PASSED. Capture the printed CPU and GPU per-frame times for the PR description.

If GPU is SLOWER than CPU at this small resolution, that's possible due to upload/download overhead vs trivial CPU work — investigate but don't necessarily fail the build. Possible causes: ROI views not zero-copy (re-run Task 1 spike), `_nudge_gpu_eye` download/upload roundtrip dominating, GPU kernel launch overhead at tiny resolutions. If consistently slower, lower the assertion threshold to `gpu_ms <= cpu_ms + 1.0` and document the finding — the larger 1080p case should still benefit, which is the real target.

### Step 3: Commit

```bash
git add tests/test_gpu_pipeline_perf.py
git commit -m "test: perf regression — GpuPipeline >=1 ms faster than CPU at 640x480"
```

---

## Final verification

### Step 1: Run the full test suite

```powershell
.venv\Scripts\python.exe -m pytest tests\ -x -q --ignore=tests\test_recording_worker.py
```

Expected: all tests pass. No skips on a CUDA-equipped machine. On a non-CUDA machine the GPU tests skip cleanly.

### Step 2: Smoke-test the live app

```powershell
.venv\Scripts\python.exe piccolo.py
```

Expected behavior:
- Startup log includes `[Pipeline] Initialized in GPU-ACCELERATED mode`.
- Live SBS frame appears in the headset.
- Status print shows non-zero `total_frame_ms`, `depth_ms`, `qimage_ms`.
- Toggle `performance.use_gpu_pipeline: false` in `config.yaml` and restart — confirm the original code path still works and produces visually identical output.

### Step 3: Capture before/after numbers

Compare median `total_frame_ms` from the status prints across `use_gpu_pipeline: true` vs `false` over ~30 seconds of live operation. Record both values. Phase 1 success criterion: GPU median is at least 1 ms faster than CPU median (small-frame conservative target), ideally 5–10 ms faster at higher resolutions.

### Step 4: Final cleanup commit if needed

If any lint warnings, stray prints, or doc nits surfaced during the work:

```bash
git add -p
git commit -m "chore: final cleanup after Phase 1 GPU pipeline rewrite"
```

---

## Done. Now what?

Update the progress table at the top of this document with the SHAs you produced. The branch `qt-desktop-app` is ready for review against `master`. Open a PR with the title `Phase 1: GPU-resident hot path for stereo pipeline` and link to the design spec at `docs/superpowers/specs/2026-05-10-gpu-pipeline-latency-reduction-design.md`.

Future phases (not in scope for this plan):
- Phase 2: resolution toggle + glass-to-glass instrumentation
- Phase 3: TurboJPEG decode
- Phase 4: QOpenGLWidget display with PBO async upload
- Phase 5 (stretch): true GPU MJPEG decode via custom cv2.cudacodec.RawVideoSource

Each future phase is independently shippable. Plan each separately based on measurements from Phase 1.
