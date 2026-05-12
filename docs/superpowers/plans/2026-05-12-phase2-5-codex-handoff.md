# GPU Pipeline Phases 2–5 — Codex Agent Implementation Plan

> **For the codex agent picking this up:** This is a self-contained continuation document. Phase 1 is complete and committed. Work through Phases 2, 3, 4, and (optionally) 5 in order. Each phase is independently shippable behind a feature flag. After each phase, run the full test suite and capture a measurement before moving on — the whole point of phasing is that we keep what helps and skip what doesn't.

## Mission recap

Reduce **glass-to-glass latency** on the live stereo surgery display path. Phase 1 built a fully-GPU-resident hot path and removed the per-frame throttle. Phase 1 alone is roughly neutral at 640×480 and saves up to one display frame (~16 ms) of throttle latency.

The remaining phases each chip away at the rest of the latency budget:

- **Phase 2 — resolution toggle + measurement.** Unblock honest comparisons at 720p/1080p where the GPU pipeline actually pulls ahead. Add a glass-to-glass timing probe.
- **Phase 3 — TurboJPEG decode.** Replace OpenCV's libjpeg with TurboJPEG (~3× faster) in the camera grab thread. Frees CPU and reduces decode-induced jitter.
- **Phase 4 — QOpenGLWidget + PBO display.** Replace the QLabel/QImage paint path with a GL texture upload, optionally PBO-async. Saves ~1–3 ms at the end of the pipeline.
- **Phase 5 (stretch) — true GPU MJPEG decode** via `cv2.cudacodec.RawVideoSource`. Research-grade. Only attempt if Phases 2–4 leave latency on the table.

## Branch and current state at handoff

- **Repo:** `e:\Documents\University\year2\Semester1\Research\SurgeryRobot\piccolo`
- **Branch:** `qt-desktop-app`
- **Latest commit:** `90397f8 docs: update Phase 1 GPU pipeline handoff progress`
- **Python:** `.venv\Scripts\python.exe` (PowerShell shell — use full path, not `python`)
- **OpenCV:** `opencv-python-cuda 4.12.0+a1a2e91` against CUDA 12.9
- **Display:** Goovis HMD via SBS HDMI; cameras: ELP-USB500W05G-MFV (USB 2.0 UVC, MJPG)

## Carry-over gotchas from Phase 1

These bit us in Phase 1. Don't relearn them.

### cv2.cuda API quirks on this exact build

- `cv2.cuda.copyTo` (module function) — **does not exist.** Use `GpuMat.copyTo(mask, dst)` instance method.
- `cv2.cuda.compare` requires both args to be matrices; for scalar use `cv2.cuda.compareWithScalar(src, scalar, cmpop)`.
- `cv2.cuda.merge(channels)` — returns a **host ndarray, not GpuMat.** Do not use; it forces a host round-trip. Use `GpuMat.copyTo` patterns instead.
- `cv2.cuda.split(gpu)` correctly returns a list of `cv2.cuda_GpuMat`.
- `GpuMat.clone()` exists. Use it to snapshot before in-place mutation when ordering matters.

### Test design

- Use **smooth gradients**, not random noise. Random noise amplifies sub-pixel resampler diffs into spurious huge per-pixel differences.
- Restrict comparisons to **image interior** (e.g., 5–10 px from each edge). CPU and GPU `warpAffine` differ at borders by design.
- A tolerance of `<= 5` on uint8 is realistic for bilinear resampler diffs. Do **not** widen tolerance to mask a real bug.

### Environment

```powershell
.venv\Scripts\python.exe -m pytest tests\<file> -v
.venv\Scripts\python.exe -m pytest tests\ -x -q --ignore=tests\test_recording_worker.py
```

The `tests/_cuda_helpers.py` file provides `skip_if_no_cuda()`. Use it at the top of every CUDA-touching test.

---

# Phase 2: Resolution toggle + glass-to-glass instrumentation

**Why first:** Phase 1's GPU pipeline gains real ground at higher resolutions. Without a resolution toggle and a glass-to-glass measurement we can't honestly compare Phase 1 to baseline. This phase changes no algorithm — it just unblocks measurement.

**Estimated savings:** 0 ms by itself. Unblocks the real measurements that motivate Phases 3 and 4.

---

## Task 2.1: Add per-camera `fps` and a resolution preset list to config

**Files:**
- Modify: `src/config.py` (add `fps` field to `CameraDeviceCfg`, optional `presets`)
- Modify: `config.yaml`

### Step 1: Failing test

Append to `tests/test_config_state.py`:

```python
def test_camera_device_cfg_has_fps_default():
    from src.config import CameraDeviceCfg
    cfg = CameraDeviceCfg()
    assert hasattr(cfg, "fps")
    assert cfg.fps == 60


def test_camera_fps_loads_from_yaml(tmp_path):
    import yaml
    from src.config import load_config
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "cameras": {
            "left": {"index": 2, "width": 1280, "height": 720, "fps": 30},
        }
    }))
    cfg = load_config(str(path))
    assert cfg.cameras.left.fps == 30
    assert cfg.cameras.left.width == 1280
    assert cfg.cameras.left.height == 720
```

Run: `.venv\Scripts\python.exe -m pytest tests\test_config_state.py -v` — expect 2 FAIL with `AttributeError: fps`.

### Step 2: Add `fps` to `CameraDeviceCfg`

In `src/config.py`, update the `CameraDeviceCfg` dataclass:

```python
@dataclass
class CameraDeviceCfg:
    index: int = 0
    width: int = 640
    height: int = 480
    flip_180: bool = False
    fps: int = 60  # requested camera framerate; driver may negotiate lower
```

### Step 3: Add `fps` keys to `config.yaml`

Edit `config.yaml` to include `fps: 60` under each of `cameras.left` and `cameras.right`. Example:

```yaml
cameras:
  backend: opencv
  left:
    index: 2
    width: 640
    height: 480
    fps: 60
    flip_180: true
  right:
    index: 1
    width: 640
    height: 480
    fps: 60
    flip_180: true
  test_mode: false
```

### Step 4: Pass `fps` through to `CameraCapture`

In `src/camera.py`, `CameraCapture.__init__` takes width/height. Add `fps` and store it. Then in `_open_opencv`, replace the hardcoded `60` with `self.fps`. Specifically:

- Constructor signature becomes:
  ```python
  def __init__(self, index=0, width=640, height=480, fps=60, backend="opencv", name="camera"):
      ...
      self.fps = fps
  ```
- In `_open_opencv`, change `self._cap.set(cv2.CAP_PROP_FPS, 60)` (two occurrences — the initial set and the fallback after MJPG retry) to `self._cap.set(cv2.CAP_PROP_FPS, self.fps)`.

In `src/ui/pipeline_worker.py`, find where `CameraCapture(...)` is instantiated (look for `CameraCapture(c.left.index, c.left.width, c.left.height, ...)` — there are two calls, one per eye). Add `fps=c.left.fps` and `fps=c.right.fps` as keyword arguments.

### Step 5: Run tests, verify pass

```powershell
.venv\Scripts\python.exe -m pytest tests\test_config_state.py -v
.venv\Scripts\python.exe -m pytest tests\ -x -q --ignore=tests\test_recording_worker.py
```

Expected: all PASSED.

### Step 6: Commit

```bash
git add src/config.py src/camera.py src/ui/pipeline_worker.py config.yaml tests/test_config_state.py
git commit -m "feat(camera): add per-camera fps field; wire through CameraCapture"
```

---

## Task 2.2: Add a resolution-preset dropdown to the Settings tab

**Goal:** Let the user pick 480p / 720p / 1080p at runtime without editing YAML. Selecting a preset writes to the config and requires a worker restart to take effect (camera reopens at the new resolution).

**Files:**
- Modify: `src/ui/settings_tab.py`
- Modify: `src/ui/pipeline_worker.py` (add a public `restart()` method, or a signal the parent listens to)
- Modify: `src/ui/main_window.py` (subscribe to settings change → restart worker)

### Step 1: Failing smoke test

Append to `tests/test_ui_smoke.py`:

```python
def test_settings_tab_exposes_resolution_dropdown(qtbot):
    from src.config import PiccoloCfg
    from src.ui.pipeline_worker import PipelineWorker
    from src.ui.settings_tab import SettingsTab

    cfg = PiccoloCfg()
    cfg.cameras.test_mode = True
    worker = PipelineWorker(cfg)
    tab = SettingsTab(worker)
    qtbot.addWidget(tab)
    assert hasattr(tab, "resolution_combo"), "SettingsTab must expose resolution_combo"
    items = [tab.resolution_combo.itemText(i) for i in range(tab.resolution_combo.count())]
    assert "640x480 @ 60" in items
    assert "1280x720 @ 60" in items
    assert "1920x1080 @ 30" in items
```

Run: `.venv\Scripts\python.exe -m pytest tests\test_ui_smoke.py::test_settings_tab_exposes_resolution_dropdown -v` — expect FAIL.

### Step 2: Add a `RESOLUTION_PRESETS` constant and dropdown

In `src/ui/settings_tab.py`, near the top of the file:

```python
RESOLUTION_PRESETS: list[tuple[str, int, int, int]] = [
    ("640x480 @ 60",   640,  480,  60),
    ("1280x720 @ 60",  1280, 720,  60),
    ("1920x1080 @ 30", 1920, 1080, 30),
]
```

In `SettingsTab.__init__`, add a `QComboBox` populated from `RESOLUTION_PRESETS`. The exact place to put it depends on the existing tab layout — find the existing widgets (toggles for low-latency, GPU depth, etc.) and add the combo in the same `QFormLayout` or similar. Pattern:

```python
from PyQt6.QtWidgets import QComboBox, QPushButton

self.resolution_combo = QComboBox(self)
for label, *_ in RESOLUTION_PRESETS:
    self.resolution_combo.addItem(label)
# Select current setting based on cfg.cameras.left.{width,height}
current = (worker.cfg.cameras.left.width, worker.cfg.cameras.left.height)
for i, (_, w, h, _) in enumerate(RESOLUTION_PRESETS):
    if (w, h) == current:
        self.resolution_combo.setCurrentIndex(i)
        break
self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)

self.apply_resolution_btn = QPushButton("Apply (restart camera)")
self.apply_resolution_btn.clicked.connect(self._apply_resolution)
```

Where the existing layout adds rows, add:

```python
form_layout.addRow("Resolution preset:", self.resolution_combo)
form_layout.addRow(self.apply_resolution_btn)
```

And the handlers:

```python
def _on_resolution_changed(self, idx: int) -> None:
    self._pending_resolution = RESOLUTION_PRESETS[idx]

def _apply_resolution(self) -> None:
    if not hasattr(self, "_pending_resolution"):
        return
    label, w, h, fps = self._pending_resolution
    cfg = self.worker.cfg.cameras
    for eye in (cfg.left, cfg.right):
        eye.width = w
        eye.height = h
        eye.fps = fps
    # Persist to disk and request worker restart.
    from src.config_state_persistence import save_calibration_state  # if present
    self.worker.request_restart.emit()  # signal added below
```

### Step 3: Add a `request_restart` signal to PipelineWorker

In `src/ui/pipeline_worker.py`, near the top of the class (with the other `pyqtSignal` declarations):

```python
    request_restart = pyqtSignal()
```

### Step 4: Wire restart in `MainWindow`

In `src/ui/main_window.py`, after constructing the worker (look for `self.worker = PipelineWorker(cfg)`), connect:

```python
self.worker.request_restart.connect(self._restart_worker)
```

And add a method:

```python
def _restart_worker(self) -> None:
    self.worker.stop()
    self.worker.wait(3000)
    # Re-instantiate using current cfg state.
    self.worker = PipelineWorker(self.cfg)
    self._rewire_worker_signals()  # whatever existing pattern wires SBS frame to widgets
    self.worker.start()
```

If the existing code already has a wire-up function reuse it; otherwise inline the connection lines used in `__init__`.

### Step 5: Save the new resolution to config.yaml

The codebase already has `save_calibration_state` (or similar) for persistence. Extend or add a `save_camera_settings(cfg, path)` that writes the camera section back. Minimum:

```python
def save_camera_settings(cfg, path) -> None:
    """Persist current cameras.left/right width/height/fps to config.yaml."""
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cams = raw.setdefault("cameras", {})
    for side, dev in (("left", cfg.cameras.left), ("right", cfg.cameras.right)):
        s = cams.setdefault(side, {})
        s["width"] = dev.width
        s["height"] = dev.height
        s["fps"] = dev.fps
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
```

Call it from `_apply_resolution` after updating the dataclass values.

### Step 6: Run tests, smoke-test, commit

```powershell
.venv\Scripts\python.exe -m pytest tests\ -x -q --ignore=tests\test_recording_worker.py
```

Expected: all PASSED including the new smoke test.

```bash
git add src/ui/settings_tab.py src/ui/pipeline_worker.py src/ui/main_window.py src/config_state_persistence.py tests/test_ui_smoke.py
git commit -m "feat(ui): add resolution preset dropdown with worker restart"
```

---

## Task 2.3: Glass-to-glass measurement (timestamp watermark)

**Goal:** Estimate end-to-end latency without external hardware. Render a millisecond-precision timestamp into a small black-on-white square in a corner of every output SBS frame. The user can then point the Goovis at a high-speed camera (a phone slow-mo recording works) pointed at the system clock, and compute glass-to-glass = (camera-clock-time-of-shutter − watermark-timestamp). Not nanosecond-precise, but consistent and good enough to compare phases.

**Files:**
- Create: `src/ui/latency_watermark.py`
- Modify: `src/ui/pipeline_worker.py` (call watermark before emit, behind a flag)
- Modify: `src/config.py` (new `performance.latency_watermark: bool = False`)

### Step 1: Failing test

Create `tests/test_latency_watermark.py`:

```python
"""Latency watermark draws a readable timestamp on top-left."""
from __future__ import annotations

import numpy as np


def test_latency_watermark_draws_text_in_top_left():
    from src.ui.latency_watermark import draw_timestamp_watermark

    sbs = np.full((100, 200, 3), 200, dtype=np.uint8)
    out = draw_timestamp_watermark(sbs, timestamp_ms=12345.678)
    # Top-left region (where the watermark lives) must differ from the background.
    assert not np.array_equal(out[0:30, 0:120], sbs[0:30, 0:120])
    # Lower-right region (untouched) must be unchanged.
    assert np.array_equal(out[50:, 150:], sbs[50:, 150:])


def test_latency_watermark_returns_same_shape_and_dtype():
    from src.ui.latency_watermark import draw_timestamp_watermark
    sbs = np.zeros((60, 120, 3), dtype=np.uint8)
    out = draw_timestamp_watermark(sbs, timestamp_ms=0.0)
    assert out.shape == sbs.shape
    assert out.dtype == sbs.dtype
```

Run, expect ImportError.

### Step 2: Implement `draw_timestamp_watermark`

Create `src/ui/latency_watermark.py`:

```python
"""Glass-to-glass latency watermark.

Renders the current monotonic clock (ms) into a small high-contrast block
in the top-left of the SBS frame. Pair with an external high-speed camera
filming both the system clock and the headset to compute glass-to-glass:

    latency_ms = camera_clock_at_shutter_ms - watermark_value_ms
"""
from __future__ import annotations

import cv2
import numpy as np


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.7
_THICKNESS = 2
_BG_PAD = 4


def draw_timestamp_watermark(sbs: np.ndarray, timestamp_ms: float) -> np.ndarray:
    """In-place watermark. Returns the same array for chaining."""
    text = f"{timestamp_ms:.1f}"
    (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS)
    x, y = 6, 6 + th
    # White background block under the text for contrast against any scene.
    cv2.rectangle(
        sbs,
        (x - _BG_PAD, y - th - _BG_PAD),
        (x + tw + _BG_PAD, y + baseline + _BG_PAD),
        (255, 255, 255), thickness=-1,
    )
    cv2.putText(sbs, text, (x, y), _FONT, _FONT_SCALE, (0, 0, 0), _THICKNESS, cv2.LINE_AA)
    return sbs
```

### Step 3: Add config flag

In `src/config.py`, extend `PerformanceCfg`:

```python
@dataclass
class PerformanceCfg:
    low_latency_mode: bool = False
    use_gpu_for_depth: bool = True
    use_gpu_pipeline: bool = True
    latency_watermark: bool = False  # NEW: enable to measure glass-to-glass
```

Add a corresponding `latency_watermark: false` line in `config.yaml`'s `performance:` block.

### Step 4: Call from `pipeline_worker._tick`

In `src/ui/pipeline_worker.py`, just before the `self.sbs_qimage_ready.emit(ndarray_to_qimage(sbs))` line, add:

```python
if getattr(self.cfg.performance, "latency_watermark", False):
    from .latency_watermark import draw_timestamp_watermark
    import time as _time
    draw_timestamp_watermark(sbs, _time.monotonic() * 1000.0)
```

(The import is inline to avoid loading the helper when the flag is off.)

### Step 5: Run tests, commit

```powershell
.venv\Scripts\python.exe -m pytest tests\test_latency_watermark.py -v
.venv\Scripts\python.exe -m pytest tests\ -x -q --ignore=tests\test_recording_worker.py
```

```bash
git add src/ui/latency_watermark.py src/ui/pipeline_worker.py src/config.py config.yaml tests/test_latency_watermark.py
git commit -m "feat(ui): add glass-to-glass timestamp watermark behind flag"
```

### Step 6: Document the measurement procedure (user-facing)

Append a brief section to `docs/superpowers/specs/2026-05-10-gpu-pipeline-latency-reduction-design.md` describing how to use the watermark with a phone slow-mo recording to estimate glass-to-glass. (Optional but valuable.) Commit as `docs: add glass-to-glass measurement procedure`.

---

## Phase 2 sanity check

After Tasks 2.1–2.3:

```powershell
.venv\Scripts\python.exe piccolo.py
```

Try each resolution preset. Confirm the camera reopens and the pipeline produces frames. Enable `latency_watermark: true` in `config.yaml`, restart, and confirm the timestamp prints on every emitted SBS frame.

**Capture before/after numbers:** With `use_gpu_pipeline: true` vs `false`, at each of 480p / 720p / 1080p, record median `total_frame_ms` over ~30 seconds. The GPU path should show a measurable win at 720p and a clear win at 1080p. If it doesn't, investigate before starting Phase 3.

---

# Phase 3: TurboJPEG CPU decode

**Why:** OpenCV's `cap.read()` decodes MJPEG with libjpeg, which is the CPU bottleneck in the camera grab thread at 1080p. TurboJPEG (libjpeg-turbo) is roughly 3× faster, freeing CPU and reducing decode-induced FPS jitter.

**Estimated savings:** 1–3 ms per frame at 1080p in the grab thread + lower CPU usage. The decode is parallel to the pipeline, so the saving shows up as headroom (the pipeline runs without stealing CPU from decode), not directly in `total_frame_ms`.

**Risk:** Medium. The fast path requires raw MJPEG byte access, which is not straightforward on Windows DirectShow. We provide a **fallback** that uses OpenCV decode but installs PyTurboJPEG for future use, so this phase is shippable even if raw byte access fails.

---

## Task 3.1: Install PyTurboJPEG and add config flag

**Files:**
- Modify: `requirements.txt`
- Modify: `src/config.py` (add `decode_backend` to `CameraDeviceCfg`)
- Modify: `config.yaml`

### Step 1: Add dependency

In `requirements.txt`, append:

```
PyTurboJPEG>=1.7
```

Install:
```powershell
.venv\Scripts\python.exe -m pip install "PyTurboJPEG>=1.7"
```

PyTurboJPEG requires the native `libturbojpeg` shared library. On Windows it's usually bundled with the wheel. If `import turbojpeg; turbojpeg.TurboJPEG()` raises `ImportError: Could not find libturbojpeg`, download the libjpeg-turbo Windows installer from sourceforge.net/projects/libjpeg-turbo/ and set `TURBOJPEG=C:\libjpeg-turbo64\bin\turbojpeg.dll` in the environment. If after 30 minutes you can't get it working on this machine, **stop and report** — this phase is not load-bearing for Phase 4.

### Step 2: Failing test

Append to `tests/test_config_state.py`:

```python
def test_camera_decode_backend_default():
    from src.config import CameraDeviceCfg
    assert CameraDeviceCfg().decode_backend == "opencv"


def test_camera_decode_backend_loads_from_yaml(tmp_path):
    import yaml
    from src.config import load_config
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"cameras": {"left": {"decode_backend": "turbojpeg"}}}))
    cfg = load_config(str(path))
    assert cfg.cameras.left.decode_backend == "turbojpeg"
```

### Step 3: Add field

In `src/config.py`, `CameraDeviceCfg`:

```python
@dataclass
class CameraDeviceCfg:
    index: int = 0
    width: int = 640
    height: int = 480
    flip_180: bool = False
    fps: int = 60
    decode_backend: str = "opencv"  # NEW: "opencv" | "turbojpeg"
```

In `config.yaml`, add `decode_backend: opencv` under each of `cameras.left` and `cameras.right`.

### Step 4: Run, commit

```powershell
.venv\Scripts\python.exe -m pytest tests\test_config_state.py -v
```

```bash
git add requirements.txt src/config.py config.yaml tests/test_config_state.py
git commit -m "feat(config): add cameras.<eye>.decode_backend flag; install PyTurboJPEG"
```

---

## Task 3.2: Replace decode in the grab thread when `decode_backend == "turbojpeg"`

**Goal:** When the flag is set, use TurboJPEG instead of OpenCV's internal decode. The challenge: extracting the raw MJPEG bytes from a UVC stream on Windows.

**Approach (in order of attempt):**

1. **Attempt A — `cv2.VideoCapture` with `CAP_PROP_FORMAT = -1`.** Some OpenCV backends expose raw mode this way, returning the encoded bytes from `read()`. If this works, TurboJPEG decodes them. Implementation:
   ```python
   self._cap.set(cv2.CAP_PROP_FORMAT, -1)
   # then in the grab loop:
   ret, raw = cap.read()
   if isinstance(raw, np.ndarray) and raw.ndim == 1:
       frame = self._tj.decode(raw.tobytes(), pixel_format=turbojpeg.TJPF_BGR)
   ```
   On Windows DirectShow this **may not work** — `CAP_PROP_FORMAT` raw mode is backend-specific. Test with `print(self._cap.get(cv2.CAP_PROP_FORMAT))` after setting; if it returns `-1` and `cap.read()` yields a 1-D ndarray, you're in business.

2. **Attempt B — fall back to OpenCV decode but still construct the `TurboJPEG` instance** (for future use in Phase 5 or for documentation). This makes the flag a no-op in practice on Windows but keeps the config / wiring in place.

3. **If A works, prefer it. If only B is viable, document it clearly in the commit message** and lower expectations for this phase's savings.

**Files:**
- Modify: `src/camera.py`

### Step 1: Test infrastructure

Append to `tests/test_camera_decode.py` (create the file):

```python
"""TurboJPEG fast-path falls back to OpenCV gracefully when raw mode unavailable."""
from __future__ import annotations

import pytest


def test_turbojpeg_decode_backend_constructs_without_error():
    from src.camera import CameraCapture

    # Sanity-only: construct in test mode equivalent path and confirm the
    # decode_backend field is honored without crashing on import / construction.
    # (Actual camera open is not asserted — would require hardware.)
    cap = CameraCapture(
        index=0, width=640, height=480, fps=60,
        backend="opencv", name="test-cam",
    )
    cap.decode_backend = "turbojpeg"  # set directly to simulate config
    # Calling _try_init_turbojpeg should either succeed or fall back without
    # raising. The actual camera is not opened in this unit test.
    assert hasattr(cap, "_try_init_turbojpeg")
    cap._try_init_turbojpeg()
    # _tj is either a TurboJPEG instance or None (graceful fallback).
    assert cap._tj is None or hasattr(cap._tj, "decode")
```

### Step 2: Implement the TurboJPEG hook in `CameraCapture`

In `src/camera.py`:

Add to the top of the file:
```python
try:
    from turbojpeg import TurboJPEG, TJPF_BGR
    _TJ_AVAILABLE = True
except ImportError:
    _TJ_AVAILABLE = False
```

Add to `CameraCapture.__init__` (extend signature):
```python
def __init__(
    self,
    index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 60,
    backend: str = "opencv",
    name: str = "camera",
    decode_backend: str = "opencv",
):
    ...
    self.decode_backend = decode_backend
    self._tj = None
    self._raw_mode_ok = False
```

Add helper methods:
```python
    def _try_init_turbojpeg(self) -> None:
        """Try to construct a TurboJPEG instance. Sets self._tj or leaves None."""
        if self.decode_backend != "turbojpeg":
            return
        if not _TJ_AVAILABLE:
            print(f"[camera] {self.name}: PyTurboJPEG not installed; falling back to OpenCV decode")
            return
        try:
            self._tj = TurboJPEG()
        except Exception as exc:
            print(f"[camera] {self.name}: TurboJPEG init failed ({exc}); falling back")
            self._tj = None

    def _try_enable_raw_mode(self) -> None:
        """Try to put VideoCapture into raw-bytes mode for TurboJPEG decode."""
        if self._cap is None or self._tj is None:
            return
        try:
            self._cap.set(cv2.CAP_PROP_FORMAT, -1)
            fmt = self._cap.get(cv2.CAP_PROP_FORMAT)
            self._raw_mode_ok = (int(fmt) == -1)
            if self._raw_mode_ok:
                print(f"[camera] {self.name}: TurboJPEG fast-path enabled (raw MJPEG bytes)")
            else:
                print(f"[camera] {self.name}: raw mode unsupported; TurboJPEG installed but unused")
        except Exception as exc:
            print(f"[camera] {self.name}: raw-mode probe failed ({exc})")
            self._raw_mode_ok = False
```

Call `_try_init_turbojpeg()` early in `start()` (before the grab thread starts) and `_try_enable_raw_mode()` at the end of `_open_opencv()`.

Modify `_grab_loop`:
```python
def _grab_loop(self):
    if self.backend == "picamera2":
        self._grab_loop_picamera()
        return
    cap = self._cap
    while self._running:
        ret, payload = cap.read()
        if not ret or payload is None:
            continue
        if self._raw_mode_ok and self._tj is not None and payload.ndim == 1:
            try:
                frame = self._tj.decode(payload.tobytes(), pixel_format=TJPF_BGR)
            except Exception:
                frame = None
        else:
            frame = payload
        if frame is None:
            continue
        with self._lock:
            self._frame = frame
            self._frame_id += 1
```

Wire `decode_backend` through `pipeline_worker._open_cameras`:
```python
self.cam_l = CameraCapture(
    c.left.index, c.left.width, c.left.height,
    fps=c.left.fps, backend=c.backend, name="cam-L",
    decode_backend=c.left.decode_backend,
).start()
# ...same for cam_r...
```

### Step 3: Smoke-run

Set `cameras.left.decode_backend: turbojpeg` and `cameras.right.decode_backend: turbojpeg` in `config.yaml`. Launch the app, check the log output:

- If you see `TurboJPEG fast-path enabled (raw MJPEG bytes)` — Phase 3 is delivering its design intent.
- If you see `raw mode unsupported` — TurboJPEG won't be doing decode work, but the flag is in place. Document this in the commit and the design spec.

### Step 4: Commit

If A worked:
```bash
git add src/camera.py src/ui/pipeline_worker.py tests/test_camera_decode.py
git commit -m "feat(camera): TurboJPEG fast-path decode for MJPEG streams"
```

If only fallback worked:
```bash
git add src/camera.py src/ui/pipeline_worker.py tests/test_camera_decode.py
git commit -m "feat(camera): wire decode_backend flag (TurboJPEG raw-byte path unavailable on Windows DirectShow; fallback to OpenCV decode)"
```

---

## Phase 3 sanity check

Run the full app and watch CPU usage with the cameras at 1080p, `decode_backend: turbojpeg`:
- If raw-mode worked: CPU usage in the grab threads should drop noticeably (~30–50%).
- If not: this phase contributed only the config plumbing — Phase 4 is where the next concrete win lives.

Either way, run the full test suite and confirm no regressions:
```powershell
.venv\Scripts\python.exe -m pytest tests\ -x -q --ignore=tests\test_recording_worker.py
```

---

# Phase 4: QOpenGLWidget + PBO display

**Why:** Today's display path is `cuda_GpuMat.download() → ndarray → QImage → QLabel.setPixmap`. That's a download, a copy into a QImage, and a Qt paint conversion. A `QOpenGLWidget` rendering a single textured quad bypasses the QImage step and lets the GPU upload happen asynchronously via a Pixel Buffer Object (PBO).

**Estimated savings:** 1–3 ms at the end of each frame, plus reduced CPU work in Qt's paint pipeline.

**Risk:** Medium. Touching the display path can introduce subtle visual regressions (color space, flip, scaling). Side-by-side QLabel vs GL widget verification is mandatory.

---

## Task 4.1: Add `use_gl_display` config flag and a stub GL widget

**Files:**
- Modify: `src/config.py` (`performance.use_gl_display: bool = False`)
- Create: `src/ui/gl_display_widget.py`

### Step 1: Failing test

Append to `tests/test_config_state.py`:
```python
def test_performance_use_gl_display_default():
    from src.config import PerformanceCfg
    assert PerformanceCfg().use_gl_display is False
```

Append to `tests/test_ui_smoke.py`:
```python
def test_gl_display_widget_constructs(qtbot):
    from src.ui.gl_display_widget import GLDisplayWidget
    w = GLDisplayWidget()
    qtbot.addWidget(w)
    assert w is not None
```

### Step 2: Add the config field

In `src/config.py`:
```python
@dataclass
class PerformanceCfg:
    low_latency_mode: bool = False
    use_gpu_for_depth: bool = True
    use_gpu_pipeline: bool = True
    latency_watermark: bool = False
    use_gl_display: bool = False  # NEW
```

Add `use_gl_display: false` to `config.yaml`.

### Step 3: Create `GLDisplayWidget` stub

Create `src/ui/gl_display_widget.py`:

```python
"""QOpenGLWidget that renders an SBS BGR frame as a fullscreen textured quad.

For Phase 4 we keep the implementation deliberately minimal: synchronous
glTexSubImage2D upload from a host ndarray. PBO async upload is a follow-up.
The widget exposes `set_frame(ndarray)` so existing code can swap in for
QLabel.setPixmap() without other changes.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QOpenGLContext

try:
    from OpenGL import GL
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False


class GLDisplayWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self._texture: int = 0
        self._tex_w: int = 0
        self._tex_h: int = 0
        if not _GL_AVAILABLE:
            print("[GLDisplayWidget] PyOpenGL not installed; widget will be inert")

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """Set the next frame to render. Triggers a paint."""
        self._frame = frame_bgr
        self.update()  # schedule paintGL on the main thread

    def initializeGL(self):
        if not _GL_AVAILABLE:
            return
        self._texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)

    def resizeGL(self, w: int, h: int):
        if not _GL_AVAILABLE:
            return
        GL.glViewport(0, 0, w, h)

    def paintGL(self):
        if not _GL_AVAILABLE or self._frame is None:
            return
        h, w = self._frame.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        if (w, h) != (self._tex_w, self._tex_h):
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                w, h, 0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE,
                self._frame,
            )
            self._tex_w, self._tex_h = w, h
        else:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0,
                w, h, GL.GL_BGR, GL.GL_UNSIGNED_BYTE,
                self._frame,
            )
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        # Immediate-mode quad (fine for a single full-screen blit).
        GL.glMatrixMode(GL.GL_PROJECTION); GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_MODELVIEW);  GL.glLoadIdentity()
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBegin(GL.GL_QUADS)
        for (u, v, x, y) in (
            (0.0, 1.0, -1.0, -1.0),
            (1.0, 1.0,  1.0, -1.0),
            (1.0, 0.0,  1.0,  1.0),
            (0.0, 0.0, -1.0,  1.0),
        ):
            GL.glTexCoord2f(u, v); GL.glVertex2f(x, y)
        GL.glEnd()
        GL.glDisable(GL.GL_TEXTURE_2D)
```

Install PyOpenGL:
```powershell
.venv\Scripts\python.exe -m pip install PyOpenGL PyOpenGL_accelerate
```
Add `PyOpenGL>=3.1` and `PyOpenGL_accelerate>=3.1` to `requirements.txt`.

### Step 4: Run tests, commit

```powershell
.venv\Scripts\python.exe -m pytest tests\test_config_state.py tests\test_ui_smoke.py -v
```

```bash
git add requirements.txt src/config.py src/ui/gl_display_widget.py config.yaml tests/test_config_state.py tests/test_ui_smoke.py
git commit -m "feat(ui): add stub GLDisplayWidget with use_gl_display flag"
```

---

## Task 4.2: Wire `GLDisplayWidget` into `LiveTab` behind the flag

**Goal:** When `performance.use_gl_display: true`, replace the existing QLabel-based `VideoWidget` with `GLDisplayWidget` for SBS rendering. The pipeline currently emits `sbs_qimage_ready` (QImage); we'll need a second signal `sbs_ndarray_ready` (or convert internally) — the simplest path is to feed the GL widget the ndarray from `sbs_frame_ready` (which already exists per pipeline_worker.py).

**Files:**
- Modify: `src/ui/live_tab.py`

### Step 1: Smoke test

Append to `tests/test_ui_smoke.py`:
```python
def test_live_tab_uses_gl_display_when_flag_set(qtbot):
    from src.config import PiccoloCfg
    from src.ui.pipeline_worker import PipelineWorker
    from src.ui.live_tab import LiveTab
    from src.ui.gl_display_widget import GLDisplayWidget

    cfg = PiccoloCfg()
    cfg.cameras.test_mode = True
    cfg.performance.use_gl_display = True
    worker = PipelineWorker(cfg)
    tab = LiveTab(worker)
    qtbot.addWidget(tab)
    assert isinstance(tab.preview, GLDisplayWidget)
```

### Step 2: Switch widget based on flag

In `src/ui/live_tab.py`, at the point where `self.preview = VideoWidget(...)` is constructed, add a conditional:

```python
use_gl = bool(
    getattr(worker, "cfg", None)
    and getattr(worker.cfg, "performance", None)
    and getattr(worker.cfg.performance, "use_gl_display", False)
)
if use_gl:
    from .gl_display_widget import GLDisplayWidget
    self.preview = GLDisplayWidget(preview_wrap)
else:
    self.preview = VideoWidget(preview_wrap)
```

Then change the signal connection at the bottom of `__init__`:

```python
if worker is not None:
    if use_gl:
        # GL widget consumes raw BGR ndarrays directly.
        worker.sbs_frame_ready.connect(self.preview.set_frame)
    else:
        worker.sbs_qimage_ready.connect(self.preview.set_frame)
```

Note: `sbs_frame_ready` already exists per `pipeline_worker.py` — it emits the ndarray SBS (currently used for the calibration overlay path). If it's not always emitted on every frame, you'll need to either always emit it (cheap — it's a signal, not a conversion) or convert in the GL widget from QImage. Prefer "always emit it" for simplicity. If you add an unconditional emit, make sure the calibration overlay path still distinguishes its own use case (it currently uses `_last_raw_emit_t` to throttle).

If `sbs_frame_ready` is gated by `raw_frame_requested`, the cleanest fix is to introduce a new signal `sbs_ndarray_ready` that always fires unconditionally with the raw ndarray, and wire the GL widget to it. This keeps the existing calibration semantics intact.

### Step 3: Visual parity check

This is mandatory before promoting `use_gl_display` past "opt-in":
1. Start the app with `use_gl_display: false`, screenshot the live SBS view.
2. Set `use_gl_display: true`, restart, screenshot the same scene.
3. The two screenshots must be visually identical except for paint-engine differences (font anti-aliasing, edge pixel rounding). If colors look wrong (R/B swapped, washed out, etc.), the GL path's color format is wrong — usually `GL_BGR` vs `GL_RGB` in `glTexImage2D`.

### Step 4: Commit

```bash
git add src/ui/live_tab.py src/ui/pipeline_worker.py tests/test_ui_smoke.py
git commit -m "feat(ui): use GLDisplayWidget for live SBS when use_gl_display flag set"
```

---

## Task 4.3 (optional): PBO async upload

**Goal:** With PBO, the host→GPU texture upload runs asynchronously while the previous frame is still being drawn. This trims another ~1–2 ms off the display latency.

**Why optional:** PBO double-buffering adds real complexity. Skip if Task 4.2 already shows the savings you need.

### Approach

In `GLDisplayWidget`:
- Allocate 2 PBOs in `initializeGL`.
- On `set_frame`, copy the ndarray bytes into the PBO mapped pointer (next PBO in rotation).
- In `paintGL`, bind the OTHER PBO and call `glTexSubImage2D(..., None)` — this tells GL to copy from the bound PBO.

This requires `GL.glGenBuffers`, `GL_PIXEL_UNPACK_BUFFER`, and `glMapBufferRange`. Reference implementation pattern (Khronos wiki, or any "OpenGL PBO async texture upload" tutorial) — pseudocode:

```python
def initializeGL(self):
    super().initializeGL()
    self._pbos = list(GL.glGenBuffers(2))
    self._pbo_idx = 0
    # ...
```

```python
def set_frame(self, frame):
    self._next_frame = frame
    self.update()

def paintGL(self):
    # Upload via current PBO
    cur = self._pbos[self._pbo_idx]
    nxt = self._pbos[1 - self._pbo_idx]
    h, w = self._next_frame.shape[:2]
    nbytes = w * h * 3

    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, nxt)
    GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, nbytes, None, GL.GL_STREAM_DRAW)
    ptr = GL.glMapBufferRange(GL.GL_PIXEL_UNPACK_BUFFER, 0, nbytes,
                              GL.GL_MAP_WRITE_BIT | GL.GL_MAP_INVALIDATE_BUFFER_BIT)
    if ptr:
        import ctypes
        ctypes.memmove(int(ptr), self._next_frame.ctypes.data, nbytes)
        GL.glUnmapBuffer(GL.GL_PIXEL_UNPACK_BUFFER)

    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, cur)
    GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
    GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                       GL.GL_BGR, GL.GL_UNSIGNED_BYTE, None)
    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
    # ... draw the quad as before ...
    self._pbo_idx = 1 - self._pbo_idx
```

If implementing, gate behind a new sub-flag `performance.use_gl_pbo: bool = False` so it can be toggled independently. Commit:

```bash
git add src/ui/gl_display_widget.py src/config.py config.yaml
git commit -m "feat(ui): add PBO async upload path to GLDisplayWidget"
```

---

## Phase 4 sanity check

With `latency_watermark: true` (Phase 2), capture phone slow-mo recordings of the headset at:
- `use_gl_display: false` (baseline)
- `use_gl_display: true`

Compute glass-to-glass for each. The GL path should be 1–3 ms lower. If it's worse, suspect:
- `glClear` is happening but the quad isn't drawing (check `paintGL`).
- The texture is uploaded but `glClearColor` is overdrawing it (check call order).
- VSync is off in the OS but on in Qt's swap, doubling the apparent latency. Set `QSurfaceFormat::setSwapInterval(0)` in `MainWindow.__init__` if so.

---

# Phase 5 (stretch): true GPU MJPEG decode via `cv2.cudacodec.RawVideoSource`

**Status:** Research. Only attempt if Phases 2–4 are not enough.

**Why it's hard:** `cv2.cudacodec.VideoReader` is designed for files (mp4, mkv) — it does not accept a USB UVC stream directly. The workaround is `cv2.cudacodec.RawVideoSource`, an abstract base class you subclass in Python to yield MJPEG packets one at a time.

The Python binding for `RawVideoSource` may or may not be fully usable on this OpenCV build — verify first:
```powershell
.venv\Scripts\python.exe -c "import cv2; print(hasattr(cv2.cudacodec, 'RawVideoSource'))"
```

If `True`, dig further. If not, this phase is blocked at the API layer and you should stop.

### Sketch (no full code — this is research)

1. **Acquire MJPEG bytes from the UVC stream without OpenCV's decode.** On Windows the cleanest path is a custom DirectShow Sample Grabber filter inserted before the decode stage. This is a C++ component you'd wrap with `pybind11` or call via `ctypes` + the Windows Media Foundation API. Significant effort.

2. **Subclass `cv2.cudacodec.RawVideoSource`** in Python:
   ```python
   class UvcMjpegSource(cv2.cudacodec.RawVideoSource):
       def __init__(self, sample_grabber):
           super().__init__()
           self._grabber = sample_grabber
       def getNextPacket(self, raw_data, size):
           pkt = self._grabber.next_packet()
           ...
       def lastPacketContainsKeyFrame(self): return True  # every MJPEG frame is a key frame
       def format(self):
           fmt = cv2.cudacodec.FormatInfo()
           fmt.codec = cv2.cudacodec.JPEG
           fmt.chromaFormat = cv2.cudacodec.YUV420
           fmt.width = self._width
           fmt.height = self._height
           return fmt
   ```

3. **Feed it to `cv2.cudacodec.createVideoReader`:**
   ```python
   reader = cv2.cudacodec.createVideoReader(UvcMjpegSource(grabber))
   ret, gpu_frame = reader.nextFrame()  # gpu_frame is a cuda_GpuMat in NV12 format
   ```

4. **Convert NV12 → BGR on-device** with `cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_YUV2BGR_NV12)`.

5. **Hand the BGR GpuMat to `GpuPipeline`** instead of uploading from CPU.

**Risk assessment:** the DirectShow Sample Grabber piece is the hardest part. Without GStreamer (this OpenCV build has no GStreamer support), you're writing platform code for Windows specifically. Allocate at least a week of focused work. Document findings even if you don't ship — the design spec should capture what's possible and what isn't.

If you ship this:
```bash
git commit -m "feat(camera): GPU MJPEG decode via cudacodec.RawVideoSource"
```

If you investigate and abandon:
```bash
git commit -m "docs: document why GPU MJPEG decode is not viable on Windows DirectShow"
```
…and update the design spec's Phase 5 section with the findings.

---

# Final verification (after each phase)

```powershell
# 1. Full test suite — no regressions.
.venv\Scripts\python.exe -m pytest tests\ -x -q --ignore=tests\test_recording_worker.py

# 2. Smoke-test the live app.
.venv\Scripts\python.exe piccolo.py

# 3. Measure. With latency_watermark: true and a phone slow-mo of the headset:
#    a) note baseline glass-to-glass (use_gpu_pipeline: false, no GL display).
#    b) toggle the phase you just shipped and re-measure.
#    c) record both numbers in the PR description.
```

**Stop conditions:** If a phase's measured improvement is below 1 ms AND adds dependency / risk, document it and consider reverting the flag's default to false. The phases are designed to be independently shippable so we can keep what helps and skip what doesn't.

---

# Cross-cutting reminders

- **Commit after each task.** Don't batch.
- **Each phase is independently shippable behind a flag.** If you have to stop mid-phase, the previous phase's flag stays in place and the partial work is just dead code behind an off flag.
- **No silent host round-trips in the hot path.** This was a recurring failure mode in Phase 1. Grep your work for `download` / `upload` / `merge` in `src/gpu_pipeline.py` and `src/ui/gl_display_widget.py` — those should appear only in the documented upload-once / download-once positions or in clearly out-of-hot-path code.
- **Don't widen test tolerances to mask failures.** If a test starts failing because the GPU path differs from CPU, find the API quirk before relaxing the assertion.

---

# Where to publish results

After each phase ships, append to `docs/superpowers/specs/2026-05-10-gpu-pipeline-latency-reduction-design.md` under a new "Measured results" section:

```markdown
## Measured results

| Configuration | Resolution | Median total_frame_ms | Glass-to-glass (ms) |
| ------------- | ---------- | --------------------- | -------------------- |
| Pre-Phase 1 (baseline CPU)  | 640x480 | ... | ... |
| Phase 1 GPU pipeline        | 640x480 | ... | ... |
| Phase 1 GPU pipeline        | 1920x1080 | ... | ... |
| + Phase 3 TurboJPEG         | 1920x1080 | ... | ... |
| + Phase 4 GL display        | 1920x1080 | ... | ... |
```

This is what reviewers will look at to decide whether to merge. Empty rows are honest if a phase didn't help on the hardware available.
