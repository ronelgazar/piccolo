# GPU Pipeline Latency Reduction — Design

**Date:** 2026-05-10
**Status:** Draft → user review
**Goal:** Reduce glass-to-glass latency on the live stereo path by keeping every frame on the GPU from upload to display, removing redundant PCIe round-trips, and eliminating an unnecessary pipeline-throttle that adds up to one display-frame interval of latency. Cameras: ELP-USB500W05G-MFV (5MP USB 2.0 UVC, MJPG/YUY2 only — no raw Bayer over USB). Display: Goovis HMD via SBS HDMI.

## Motivation

Today's pipeline does the following per frame, in [`src/ui/pipeline_worker.py`](../../../src/ui/pipeline_worker.py):

1. CPU decode of MJPEG in camera grab thread (libjpeg).
2. Upload to GPU, GPU warp, **download** to ndarray.
3. CPU `_fill_holes_cross` (numpy boolean indexing).
4. CPU `process_pair` (zoom crop + resize) and calibration `apply_nudge`.
5. Upload to GPU again for depth, **download** disparity.
6. CPU `ndarray_to_qimage` for Qt display.
7. The pipeline is throttled to `display.fps` via `_can_process_now`, which can delay a fresh camera frame by up to ~16 ms before processing begins (pipeline_worker.py:213-219).

The pipeline is hybrid CPU/GPU with 4 ping-pong PCIe transfers per frame and a software-controllable throttle that adds latency for no glass-to-glass benefit. The existing GPU acceleration helpers (`_warp_affine_fast`, `_cvtcolor_and_resize_fast`) each do their own upload + download, which is what they're built for, but in the context of the whole pipeline that means the data crosses PCIe many times.

This design replaces the hot path with a fully-GPU-resident pipeline: one upload after camera read, one download before display. The pipeline throttle is removed (or scoped to emit gating only).

## Non-goals

- Reduce sensor exposure or USB transport latency (out of our control).
- Reduce Goovis HMD internal latency (out of our control).
- Change calibration views (`process_pair_full_fov*`), smart-overlap detection, or alignment update cadence — these run at slow cadence and CPU is fine.
- Bayer-raw access from these UVC cameras (the sensor ISP is on-camera; raw is not exposed over USB).

## Scope summary

| Phase       | Adds                                                     | Saves (est.)          | Risk   |
| ----------- | -------------------------------------------------------- | --------------------- | ------ |
| 1           | GPU pipeline + throttle removal                          | 5–10 ms               | medium |
| 2           | Resolution toggle + glass-to-glass instrumentation       | measurement only      | low    |
| 3           | TurboJPEG CPU decode                                     | 1–3 ms + CPU headroom | low    |
| 4           | QOpenGLWidget + PBO display                              | 1–3 ms                | medium |
| 5 (stretch) | True GPU MJPEG decode via `cv2.cudacodec.RawVideoSource` | 1–2 ms                | high   |

Total realistic pipeline savings at 640×480: **~7–13 ms**. Larger absolute savings expected at 1080p (more pixels per saved round-trip).

## Architecture

### Pipeline shape

```text
Camera grab thread → MJPEG/BGR frame
    ↓ single upload, pinned host memory
GpuMat[L], GpuMat[R]
    ↓ stereo align warp        (GPU; warp matrices cached on GPU)
    ↓ fill-holes cross         (GPU mask ops via cv2.cuda.compare + copyTo)
    ↓ stereo processor zoom    (GPU ROI + cv2.cuda.resize)
    ↓ calibration nudge        (GPU ROI shift)
    ↓ depth disparity          (GPU, decimated as today)
    ↓ compose SBS              (single GpuMat, two ROI views)
    ↓ smart-overlap overlay    (GPU when active)
    ↓ single download → texture upload (Phase 4: PBO)
QOpenGLWidget
```

### Key invariant

Between the upload (after camera read) and the final download (before display) there are zero `cv2.cuda_GpuMat.download()` calls in the hot path. Today there are 4. This is the principal source of savings.

### Reversibility

Every phase is gated by a config flag. Phase 1 is gated by `performance.use_gpu_pipeline`. When false, today's code path runs unchanged. The same pattern applies to Phases 3 and 4.

## Components

### New files

**`src/gpu_pipeline.py`** — Orchestrates the GpuMat-resident pipeline.

- Allocates persistent `cuda_GpuMat` buffers for: input L/R, warped L/R, post-fill L/R, processed L/R (views into a single SBS GpuMat).
- Caches warp matrices on GPU, reuploads only when alignment changes.
- Implements `fill_holes_cross_gpu` using `cv2.cuda.compare` (mask: pixel == 0 across all 3 channels) plus `cv2.cuda.copyTo` to substitute the partner eye's pixel.
- Public API:
  - `process(bgr_l, bgr_r) -> ndarray_sbs` (download path) — used in Phase 1.
  - `process_to_texture(bgr_l, bgr_r, gl_pbo) -> None` — Phase 4 entry point.
- Owns the GPU stream(s) for async work overlap.

**`src/ui/gl_display_widget.py`** — `QOpenGLWidget` subclass.

- Holds one GL texture sized to the SBS output (1920×1080).
- Supports two upload paths:
  - Sync: `update_from_ndarray(sbs_bgr)` (Phase 1 fallback).
  - Async: `update_from_pbo(pbo_id)` (Phase 4 default).
- Renders a textured quad. No filtering, exact-pixel mapping to the surface.
- Uses Qt's `QOpenGLContext` and double-buffered swap.

### Modified files

**`src/stereo_processor.py`**

Add `process_pair_gpu(gpu_l, gpu_r, gpu_sbs) -> (gpu_eye_l, gpu_eye_r, gpu_sbs)`. The crop is a `cv2.cuda_GpuMat` ROI (zero-copy view). The resize uses `cv2.cuda.resize` writing into pre-allocated GpuMat ROIs of `gpu_sbs`. Existing CPU `process_pair` stays unchanged.

**`src/stereo_align.py`**

Add `warp_pair_gpu(gpu_l, gpu_r, gpu_warped_l, gpu_warped_r)`. The warp matrices are cached on GPU as 2x3 float32 GpuMats and only re-uploaded when `_build_warp_matrices` runs (i.e., when alignment updates).

**`src/calibration.py`**

Add `apply_nudge_gpu(gpu_l, gpu_r)`. Nudge is implemented as a `cv2.cuda.warpAffine` with a translation matrix, reusing existing nudge logic.

**`src/camera.py`**

- Add `decode_backend: 'opencv' | 'turbojpeg'` parameter to `CameraCapture`.
- When `'turbojpeg'`, the grab thread uses `cv2.VideoCapture` with `CAP_PROP_FORMAT = -1` (raw mode) where supported, or installs a sample-grabber to capture the MJPEG bytes, then decodes via `PyTurboJPEG.decode`. If raw mode is not available on the chosen backend, log a warning and fall back to OpenCV decode.
- Add `fps: int` field per-camera in config (today hardcoded to 60 in `_open_opencv`).

**`src/ui/pipeline_worker.py`**

- Replace per-stage CPU/GPU helpers with a single `_run_gpu_pipeline(frame_l, frame_r) -> sbs` call when `performance.use_gpu_pipeline` is true.
- Remove the per-frame throttle in `_can_process_now`. The display emit can still be rate-limited (Qt receives at most one paint per vsync anyway), but the pipeline runs whenever a new camera frame is available.
- Add per-stage GPU timing via `cv2.cuda_Event` so we have honest GPU-side metrics (today's `time.perf_counter` brackets include host wait time, not actual GPU work time).

**`src/ui/live_tab.py`** (and wherever the SBS QLabel is used)

When `performance.use_gl_display` is true, swap the QLabel for `GLDisplayWidget`. Same parent layout, same size hints.

**`src/config.py`**

```yaml
performance:
  use_gpu_pipeline: true     # NEW (default true when CUDA available)
  use_turbojpeg: false       # NEW (default false; opt in)
  use_gl_display: false      # NEW (default false; opt in)
  low_latency_mode: false    # existing
  use_gpu_for_depth: true    # existing

cameras:
  left:
    index: 2
    width: 640                # bumped via UI dropdown
    height: 480
    fps: 60                  # NEW
    flip_180: true
    decode_backend: opencv   # NEW: opencv | turbojpeg
  right: { ... same ... }
```

### Stretch (Phase 5)

True GPU MJPEG decode via `cv2.cudacodec.RawVideoSource`. Implement a Python subclass that pulls MJPEG bytes from a custom DirectShow Sample Grabber and yields them as packets to the NVDEC video reader. This is a research effort; only attempt if measured savings from Phases 1–4 are insufficient. Documented here so we don't lose the option.

## Data flow & invariants

### GpuMat lifetime

The pipeline allocates these GpuMats once at startup and reuses them every frame:

- `_gpu_in_l`, `_gpu_in_r` — destination of camera upload (BGR, camera resolution).
- `_gpu_warp_l`, `_gpu_warp_r` — alignment warp output (BGR, camera resolution).
- `_gpu_filled_l`, `_gpu_filled_r` — post fill-holes (BGR, camera resolution).
- `_gpu_sbs` — final SBS frame (BGR, 1920×1080). `_gpu_eye_l = _gpu_sbs.colRange(0, 960)`, `_gpu_eye_r = _gpu_sbs.colRange(960, 1920)`.
- `_gpu_disp_l`, `_gpu_disp_r`, `_gpu_disparity` — disparity inputs and output, downscaled.

Reallocation only occurs when camera resolution changes (new resolution toggle in Phase 2).

### CUDA streams

Phase 1 uses the default stream (single-threaded GPU work) for simplicity. If profiling shows opportunities, Phase 4 can introduce a second stream for the async PBO upload, overlapping it with the next frame's pipeline.

### Pinned host memory

Camera frames are uploaded into a pinned (page-locked) host buffer via `cv2.cuda_GpuMat.upload(numpy_array)` where numpy_array's storage was registered with `cv2.cuda.HostMem`. The grab thread writes new frames into this pinned buffer. This makes uploads asynchronous-capable and ~2× faster than pageable host memory.

### Throttle removal

`_can_process_now` is removed. The pipeline runs every iteration. New rate control:

- The pipeline only does work when `frame_ids != self._last_frame_ids` (already in place).
- Display emit still rate-limited via `_emit_interval` so we don't queue paints faster than the display can swap.

This means CPU work increases slightly (we now process every camera frame, not every display interval), but glass-to-glass latency drops by up to one display interval (~16 ms at 60 Hz).

## Glass-to-glass measurement

Set `performance.latency_watermark: true` in `config.yaml` and restart the app. The pipeline draws a monotonic-clock timestamp in milliseconds into the top-left corner of every emitted SBS frame.

To estimate glass-to-glass latency, record the headset output with a high-speed camera or phone slow-motion mode while the same recording also captures a visible system clock. For a sampled frame:

1. Read the clock time visible to the recording device.
2. Read the timestamp watermark visible in the headset image.
3. Compute `latency_ms = camera_clock_at_shutter_ms - watermark_value_ms`.

Use the same phone, lighting, scene, and resolution for every comparison. The absolute value is approximate, but the delta between configurations is good enough to compare `use_gpu_pipeline`, resolution presets, TurboJPEG, and GL display changes.

## Testing

### Smoke

For Phase 1: with `use_gpu_pipeline: true`, the pipeline starts, both eyes display, depth values appear, and `[GPU]` perf prints show non-zero per-stage timings. Toggling the flag to `false` reverts to today's behavior; both paths must produce visually identical output (allowing for pixel-level GPU-vs-CPU resampler differences).

### Unit-level

- `test_gpu_pipeline_fill_holes_matches_cpu` — same input → same output as `_fill_holes_cross` within ε.
- `test_gpu_pipeline_warp_matches_cpu` — alignment warp output matches CPU within sub-pixel tolerance.
- `test_gpu_pipeline_processor_matches_cpu` — `process_pair_gpu` SBS bytes match `process_pair` SBS bytes for fixed input.

### Performance regression

Add `tests/perf/test_gpu_pipeline_perf.py` that opens test-pattern cameras and measures `total_frame_ms` median over 200 frames. Asserts the GPU path is at least 3 ms faster than CPU at 640×480.

### Manual verification (the user can't claim "fixed" without this)

Before merging Phase 1: run the live app, watch the perf line, confirm `[GPU]` median `total_frame_ms` is below the same-machine CPU baseline by at least 3 ms. Phase 4 verification: confirm visual output is identical to QLabel path (no color shift, correct scaling, correct flip).

## Open questions

1. **Pinned memory cost on small frames.** At 640×480 the upload is sub-millisecond; pinned memory might be unnecessary overhead. Will measure and drop it if it doesn't help.
2. **`fill_holes_cross_gpu` correctness.** Today's CPU version uses `np.all(left == 0, axis=2)` to detect borders. The GPU version needs `cv2.cuda.compare(eye, 0, CMP_EQ)` per channel and an AND across channels. Verify the mask matches CPU exactly on a test set with known border pixels.
3. **TurboJPEG raw-byte access on Windows.** OpenCV's `CAP_PROP_FORMAT = -1` raw mode is not universally implemented across DirectShow / Media Foundation. May need to use `pyturbojpeg` only after grabbing already-decoded BGR (which provides no benefit), unless we install a custom DirectShow filter to grab MJPEG bytes pre-decode. If raw access turns out to be infeasible, **drop Phase 3** rather than fight it.
4. **Resolution toggle UX.** Should the resolution dropdown be in the existing settings tab or a new "Performance" tab? Likely settings tab; user can confirm during Phase 2.

## Risks

- **`cv2.cuda` API surface gaps.** Some operations (e.g., per-channel boolean indexing) don't have direct cv2.cuda equivalents. Mitigations: use `cv2.cuda.compare` + `cv2.cuda.copyTo`; if a stage can't be cleanly ported, fall back to CPU for just that stage.
- **GpuMat ROI semantics.** Unlike numpy slicing, `cuda_GpuMat.colRange()` may or may not produce a writeable view depending on the OpenCV build. Verify on this OpenCV 4.12 build during Phase 1 spike; if not writeable, use `cv2.cuda.copyTo` with a region instead.
- **QOpenGLWidget threading.** Qt requires GL context interaction on the main thread. The pipeline runs on a worker QThread. Need a thread-safe handoff: pipeline writes to a shared GpuMat → main thread downloads via PBO. Phase 4 design must account for this; if it gets messy, ship Phase 1–3 without Phase 4.
- **Visual regression on the headset.** A color-space or scaling bug in the GL path could be subtle. Mandate side-by-side QLabel vs. GLDisplayWidget verification before promoting `use_gl_display: true` as default.

## Summary

The biggest single latency contributor we can fix is the pipeline throttle in `_can_process_now` (free, ~5–15 ms tail). The next biggest is the four PCIe round-trips per frame, addressed by the GPU pipeline rewrite. Everything else (TurboJPEG, GL display, true GPU JPEG decode) yields diminishing returns and is sequenced as optional follow-ups, each behind a flag.
