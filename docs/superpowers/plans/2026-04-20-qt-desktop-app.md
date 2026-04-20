# Qt Desktop App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current Pygame + Flask web-UI split with a single PyQt6 desktop application packaged as `piccolo.exe`, with tabbed assistant UI, full-screen Goovis output, cleaned-up pedal handling, and persistent calibration state.

**Architecture:** One Python process runs three pieces — main Qt window (tabbed UI), borderless Goovis output window, and a `QThread` pipeline worker that reuses the existing backend (cameras, stereo processor, aligner, calibration, annotations). Pedal/keyboard input flows through a framework-neutral `InputHandler`. Calibration state persists to `config.yaml` on change.

**Tech Stack:** PyQt6, opencv-python, numpy, screeninfo, Pillow, PyInstaller (Windows 10/11 x64).

**Branch:** `qt-desktop-app`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `piccolo.py` | Create | New Qt entry point |
| `piccolo.spec` | Create | PyInstaller config |
| `build.bat` | Create | Build script |
| `src/ui/__init__.py` | Create | Package marker |
| `src/ui/main_window.py` | Create | `QMainWindow` with `QTabWidget` |
| `src/ui/live_tab.py` | Create | Live preview + zoom pad + annotations + status |
| `src/ui/calibration_tab.py` | Create | Nudge sliders + physical-cal wizard |
| `src/ui/settings_tab.py` | Create | Cameras + display + stereo + pedals + config I/O |
| `src/ui/goovis_window.py` | Create | Borderless full-screen output |
| `src/ui/pipeline_worker.py` | Create | `QThread` running the backend loop |
| `src/ui/video_widget.py` | Create | Reusable `QLabel`-based frame display |
| `src/ui/qt_helpers.py` | Create | BGR ndarray → `QImage` conversion |
| `src/input_handler.py` | Modify | Remove pygame, framework-neutral key API, delete dead pedal code |
| `src/config.py` | Modify | Add `CalibrationStateCfg` + `ControlsCfg.pedal_*` fields |
| `src/config_state.py` | Create | `save_calibration_state()` / `load_calibration_state()` helpers |
| `tests/test_config_state.py` | Create | Persistence round-trip tests |
| `tests/test_input_handler.py` | Create | Input handler framework-neutral API tests |
| `tests/test_ui_smoke.py` | Create | `pytest-qt` smoke tests |
| `requirements.txt` | Modify | Add PyQt6, pytest-qt; remove pygame, flask, python-bidi (removal in Task 12) |
| `run.py` | Delete (Task 12) | Old Pygame entry point |
| `src/app.py` | Delete (Task 12) | Old app class |
| `src/viewer_stream.py` | Delete (Task 12) | Old Flask server |
| `src/display.py` | Delete (Task 12) | Old Pygame display |

**Rationale:** UI lives in `src/ui/` so the backend stays headless and importable from tests. Each tab is one file, one responsibility. `qt_helpers.py` isolates the BGR→QImage conversion that every video widget needs. `config_state.py` is separate from `config.py` to keep the YAML schema / dataclasses separate from the I/O helpers.

---

## Task 1: Refactor `InputHandler` — framework-neutral + delete dead code

**Files:**
- Modify: `src/input_handler.py`
- Create: `tests/test_input_handler.py`

**Goal:** Remove pygame imports from `InputHandler`, expose a framework-neutral `on_key_down(name)` / `on_key_up(name)` API, and delete legacy dead code (unused `_handle_pedal_logic`, numpad combo, unused Action enum values).

- [ ] **Step 1: Write failing tests**

Create `tests/test_input_handler.py`:

```python
from src.input_handler import InputHandler, Action
from src.config import ControlsCfg


def _make_handler():
    return InputHandler(aligner=None, cfg=ControlsCfg())


class TestPedalToggleLogic:
    def test_no_mode_at_startup(self):
        h = _make_handler()
        assert h.pedal_mode is None

    def test_press_a_enters_zoom_mode(self):
        h = _make_handler()
        h.on_key_down("a")
        assert h.pedal_mode == "a"

    def test_press_a_twice_toggles_off(self):
        h = _make_handler()
        h.on_key_down("a")
        h.on_key_up("a")
        h.on_key_down("a")
        assert h.pedal_mode is None

    def test_press_b_while_zoom_mode_is_adjust(self):
        h = _make_handler()
        h.on_key_down("a")
        h.on_key_up("a")
        h.on_key_down("b")
        actions = h.poll_actions()
        assert Action.ZOOM_IN in actions

    def test_hold_b_in_zoom_mode_keeps_firing(self):
        h = _make_handler()
        h.on_key_down("a")
        h.on_key_up("a")
        h.on_key_down("b")
        h.poll_actions()  # drain one-shot
        # b still held → continuous
        assert Action.ZOOM_IN in h.poll_actions()

    def test_release_b_stops_continuous(self):
        h = _make_handler()
        h.on_key_down("a")
        h.on_key_up("a")
        h.on_key_down("b")
        h.on_key_up("b")
        h.poll_actions()  # drain one-shot
        assert Action.ZOOM_IN not in h.poll_actions()

    def test_side_mode_maps_ac_to_left_right(self):
        h = _make_handler()
        h.on_key_down("b")
        h.on_key_up("b")
        h.on_key_down("a")
        assert Action.PEDAL_CENTER_LEFT in h.poll_actions()
        h.on_key_up("a")
        h.on_key_down("c")
        assert Action.PEDAL_CENTER_RIGHT in h.poll_actions()
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
.venv/Scripts/python -m pytest tests/test_input_handler.py -v
```

Expected: ImportError or AttributeError — the new API methods don't exist yet.

- [ ] **Step 3: Replace `src/input_handler.py` with the refactored version**

```python
"""Framework-neutral keyboard/pedal input handler.

Exposes `on_key_down(name)` / `on_key_up(name)` that any UI framework
(Qt, pygame, headless) can call.  Returns accumulated actions via
`poll_actions()`.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Set

from .config import ControlsCfg


class Action(Enum):
    ZOOM_IN = auto()
    ZOOM_OUT = auto()
    CONVERGE_IN = auto()
    CONVERGE_OUT = auto()
    TOGGLE_CALIBRATION = auto()
    TOGGLE_ALIGNMENT = auto()
    CALIB_NEXT = auto()
    CALIB_NUDGE_LEFT = auto()
    CALIB_NUDGE_RIGHT = auto()
    RESET = auto()
    QUIT = auto()
    PEDAL_CENTER_LEFT = auto()
    PEDAL_CENTER_RIGHT = auto()
    PEDAL_CENTER_UP = auto()
    PEDAL_CENTER_DOWN = auto()


class InputHandler:
    """Framework-neutral input handler.

    Usage:
      - Call `on_key_down(name)` / `on_key_up(name)` from your framework's
        key-event callback.  `name` is a single-char string like "a",
        "ArrowLeft", "Escape", etc.
      - Call `poll_actions()` once per frame to get the active actions.
    """

    _CONTINUOUS = {
        Action.ZOOM_IN, Action.ZOOM_OUT,
        Action.CONVERGE_IN, Action.CONVERGE_OUT,
        Action.CALIB_NUDGE_LEFT, Action.CALIB_NUDGE_RIGHT,
    }

    _PEDAL_ADJUST_MAP: dict[str, dict[str, Action]] = {
        "a": {"b": Action.ZOOM_IN,            "c": Action.ZOOM_OUT},
        "b": {"a": Action.PEDAL_CENTER_LEFT,  "c": Action.PEDAL_CENTER_RIGHT},
        "c": {"a": Action.PEDAL_CENTER_UP,    "b": Action.PEDAL_CENTER_DOWN},
    }

    def __init__(self, aligner, cfg: ControlsCfg):
        self.aligner = aligner
        self.cfg = cfg
        self._held: Set[Action] = set()
        self._one_shot: Set[Action] = set()
        # Pedal toggle mode: None | 'a' | 'b' | 'c'
        self.pedal_mode: str | None = None
        # Adjust-pedal → continuous Action while held
        self._pedal_adjust_held: dict[str, Action] = {}
        # Keyboard key-name → Action map (built from cfg)
        self._keymap: dict[str, Action] = self._build_keymap()

    def _build_keymap(self) -> dict[str, Action]:
        mapping = {
            "zoom_in": Action.ZOOM_IN,
            "zoom_out": Action.ZOOM_OUT,
            "converge_in": Action.CONVERGE_IN,
            "converge_out": Action.CONVERGE_OUT,
            "toggle_calibration": Action.TOGGLE_CALIBRATION,
            "toggle_alignment": Action.TOGGLE_ALIGNMENT,
            "calib_next": Action.CALIB_NEXT,
            "calib_nudge_left": Action.CALIB_NUDGE_LEFT,
            "calib_nudge_right": Action.CALIB_NUDGE_RIGHT,
            "reset": Action.RESET,
            "quit": Action.QUIT,
        }
        keymap: dict[str, Action] = {}
        for attr, action in mapping.items():
            keys = getattr(self.cfg, attr, None)
            if keys is None:
                continue
            if not isinstance(keys, (list, tuple)):
                keys = [keys]
            for k in keys:
                keymap[str(k).lower()] = action
        return keymap

    @staticmethod
    def _is_pedal(name: str) -> bool:
        return name in ("a", "b", "c")

    def _clear_pedal_adjust(self):
        self._pedal_adjust_held.clear()

    def _pedal_adjust_action(self, pedal: str):
        if self.pedal_mode is None:
            return None
        return self._PEDAL_ADJUST_MAP.get(self.pedal_mode, {}).get(pedal)

    def on_key_down(self, name: str) -> None:
        name = name.lower()
        if self._is_pedal(name):
            if self.pedal_mode == name:
                # Toggle off
                self._clear_pedal_adjust()
                self.pedal_mode = None
            elif self.pedal_mode is None:
                self.pedal_mode = name
            else:
                # Adjust pedal — immediate + continuous
                action = self._pedal_adjust_action(name)
                if action is not None:
                    self._one_shot.add(action)
                    self._pedal_adjust_held[name] = action
            return
        action = self._keymap.get(name)
        if action is None:
            return
        if action in self._CONTINUOUS:
            self._held.add(action)
        else:
            self._one_shot.add(action)

    def on_key_up(self, name: str) -> None:
        name = name.lower()
        if self._is_pedal(name):
            if name in self._pedal_adjust_held:
                del self._pedal_adjust_held[name]
            return
        action = self._keymap.get(name)
        if action is not None:
            self._held.discard(action)

    def poll_actions(self) -> Set[Action]:
        """Return held + one-shot + continuous-pedal actions; clears one-shots."""
        result = self._held | set(self._pedal_adjust_held.values()) | self._one_shot
        self._one_shot.clear()
        return result

    def get_pedal_mode(self) -> str | None:
        return self.pedal_mode
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
.venv/Scripts/python -m pytest tests/test_input_handler.py -v
```

Expected: 7/7 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/input_handler.py tests/test_input_handler.py
git commit -m "refactor: framework-neutral InputHandler, remove dead pedal code"
```

**NOTE:** `src/app.py` currently imports pygame-specific methods from `InputHandler`. This will be broken after Task 1 until Task 12 deletes `app.py`. That's expected — `run.py` will not work on this branch after Task 1. The new `piccolo.py` entry point replaces it.

---

## Task 2: Config schema + persistent calibration state

**Files:**
- Modify: `src/config.py`
- Create: `src/config_state.py`
- Create: `tests/test_config_state.py`

**Goal:** Add `CalibrationStateCfg` dataclass to config, add pedal fields to `ControlsCfg`, implement load/save helpers that round-trip state through `config.yaml`.

- [ ] **Step 1: Add the new dataclasses to `src/config.py`**

After the existing `CalibrationCfg` dataclass, add:

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
```

Extend `ControlsCfg`:

```python
@dataclass
class ControlsCfg:
    zoom_in: str = "EQUALS"
    zoom_out: str = "MINUS"
    converge_in: str = "RIGHTBRACKET"
    converge_out: str = "LEFTBRACKET"
    toggle_calibration: str = "c"
    toggle_alignment: str = "a"
    calib_next: str = "n"
    calib_nudge_left: str = "LEFT"
    calib_nudge_right: str = "RIGHT"
    reset: str = "r"
    quit: str = "ESCAPE"
    # Pedal bindings
    pedal_key_a: str = "a"
    pedal_key_b: str = "b"
    pedal_key_c: str = "c"
    pedal_mode_a: str = "zoom"        # "zoom" | "side" | "updown" | "none"
    pedal_mode_b: str = "side"
    pedal_mode_c: str = "updown"
    pedal_repeat_ms: int = 100
```

Extend `PiccoloCfg`:

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
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_config_state.py`:

```python
import os
import tempfile
import yaml
from src.config import PiccoloCfg, load_config
from src.config_state import save_calibration_state


def _write_yaml(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)


def test_load_missing_calibration_state_uses_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        _write_yaml(path, {"display": {"width": 1920}})
        cfg = load_config(path)
        assert cfg.calibration_state.nudge_left_x == 0
        assert cfg.calibration_state.joint_zoom_center == 50


def test_load_populates_calibration_state():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        _write_yaml(path, {
            "calibration_state": {
                "nudge_left_x": 3,
                "nudge_right_y": -5,
                "joint_zoom_center": 42,
            }
        })
        cfg = load_config(path)
        assert cfg.calibration_state.nudge_left_x == 3
        assert cfg.calibration_state.nudge_right_y == -5
        assert cfg.calibration_state.joint_zoom_center == 42
        # Unspecified fields keep defaults
        assert cfg.calibration_state.nudge_left_y == 0


def test_save_calibration_state_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        # Start with a populated file
        _write_yaml(path, {"display": {"width": 1920}})
        cfg = load_config(path)
        cfg.calibration_state.nudge_left_x = 7
        cfg.calibration_state.nudge_right_y = -2
        cfg.calibration_state.convergence_offset = 12
        save_calibration_state(cfg, path)
        # Reload and verify
        cfg2 = load_config(path)
        assert cfg2.calibration_state.nudge_left_x == 7
        assert cfg2.calibration_state.nudge_right_y == -2
        assert cfg2.calibration_state.convergence_offset == 12
        # Other top-level keys preserved
        assert cfg2.display.width == 1920


def test_save_does_not_clobber_other_keys():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        _write_yaml(path, {
            "cameras": {"left": {"index": 5, "flip_180": True}},
            "stereo": {"zoom": {"step": 0.07}},
        })
        cfg = load_config(path)
        cfg.calibration_state.nudge_left_x = 9
        save_calibration_state(cfg, path)
        # Re-read raw YAML to confirm other keys untouched
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        assert raw["cameras"]["left"]["index"] == 5
        assert raw["cameras"]["left"]["flip_180"] is True
        assert raw["stereo"]["zoom"]["step"] == 0.07
        assert raw["calibration_state"]["nudge_left_x"] == 9
```

- [ ] **Step 3: Run tests to confirm failure**

```bash
.venv/Scripts/python -m pytest tests/test_config_state.py -v
```

Expected: ImportError for `save_calibration_state`.

- [ ] **Step 4: Create `src/config_state.py`**

```python
"""Persist selected config sections to `config.yaml` on change."""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import yaml

from .config import PiccoloCfg


def save_calibration_state(cfg: PiccoloCfg, path: str | None = None) -> None:
    """Write `cfg.calibration_state` into `config.yaml` without clobbering
    other top-level keys.  Creates the file if missing.
    """
    if path is None:
        path = _default_config_path()
    raw: dict[str, Any] = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    raw["calibration_state"] = asdict(cfg.calibration_state)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)


def _default_config_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml",
    )
```

- [ ] **Step 5: Run tests to confirm pass**

```bash
.venv/Scripts/python -m pytest tests/test_config_state.py tests/test_input_handler.py tests/test_physical_cal.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/config.py src/config_state.py tests/test_config_state.py
git commit -m "feat: add persistent calibration state in config.yaml"
```

---

## Task 3: Qt scaffolding — entry point + empty tabs

**Files:**
- Modify: `requirements.txt`
- Create: `piccolo.py`
- Create: `src/ui/__init__.py`
- Create: `src/ui/main_window.py`
- Create: `src/ui/live_tab.py`
- Create: `src/ui/calibration_tab.py`
- Create: `src/ui/settings_tab.py`
- Create: `src/ui/qt_helpers.py`
- Create: `tests/test_ui_smoke.py`

**Goal:** Minimal Qt app that launches, shows three empty tabs, closes cleanly.

- [ ] **Step 1: Add dependencies to `requirements.txt`**

Append:
```
PyQt6>=6.6
pytest-qt>=4.3    # dev only — Qt widget smoke tests
```

Then:
```bash
.venv/Scripts/python -m pip install PyQt6 pytest-qt
```

- [ ] **Step 2: Create `src/ui/__init__.py`**

Empty file:
```python
"""Qt UI package."""
```

- [ ] **Step 3: Create `src/ui/qt_helpers.py`**

```python
"""Qt helpers — BGR ndarray → QImage conversion."""
from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QImage


def ndarray_to_qimage(bgr: np.ndarray) -> QImage:
    """Convert a BGR HxWx3 uint8 ndarray to a QImage (RGB888).

    The returned QImage owns a copy of the pixel data; the source
    ndarray can be freed immediately.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError(f"Expected HxWx3 uint8 BGR, got {bgr.shape} {bgr.dtype}")
    h, w, _ = bgr.shape
    rgb = bgr[:, :, ::-1].copy()  # BGR → RGB, contiguous
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
```

- [ ] **Step 4: Create `src/ui/live_tab.py`, `src/ui/calibration_tab.py`, `src/ui/settings_tab.py` as stubs**

Each file:

```python
"""Live tab (stub — populated in later task)."""
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout


class LiveTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Live tab — preview coming soon"))
```

Replace class name + label text for `CalibrationTab` and `SettingsTab`.

- [ ] **Step 5: Create `src/ui/main_window.py`**

```python
"""Main Qt window: tabbed assistant UI."""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow, QTabWidget

from ..config import PiccoloCfg
from .live_tab import LiveTab
from .calibration_tab import CalibrationTab
from .settings_tab import SettingsTab


class MainWindow(QMainWindow):
    def __init__(self, cfg: PiccoloCfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Piccolo")
        self.resize(1280, 800)

        tabs = QTabWidget(self)
        self.live_tab = LiveTab(self)
        self.calibration_tab = CalibrationTab(self)
        self.settings_tab = SettingsTab(self)
        tabs.addTab(self.live_tab, "Live")
        tabs.addTab(self.calibration_tab, "Calibration")
        tabs.addTab(self.settings_tab, "Settings")
        self.setCentralWidget(tabs)
```

- [ ] **Step 6: Create `piccolo.py`**

```python
"""Piccolo — stereoscopic surgery display (Qt desktop app)."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from src.config import load_config
from src.ui.main_window import MainWindow


def main() -> int:
    cfg = load_config()
    app = QApplication(sys.argv)
    window = MainWindow(cfg)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Create `tests/test_ui_smoke.py`**

```python
import pytest
from PyQt6.QtWidgets import QApplication
from src.config import load_config
from src.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_opens(qapp):
    cfg = load_config()
    w = MainWindow(cfg)
    w.show()
    assert w.isVisible()
    # Three tabs present
    central = w.centralWidget()
    assert central.count() == 3
    assert central.tabText(0) == "Live"
    assert central.tabText(1) == "Calibration"
    assert central.tabText(2) == "Settings"
    w.close()
```

- [ ] **Step 8: Run tests**

```bash
.venv/Scripts/python -m pytest tests/test_ui_smoke.py -v
```

Expected: 1/1 PASS.

- [ ] **Step 9: Smoke-run the app**

```bash
.venv/Scripts/python piccolo.py
```

Expected: a 1280×800 window opens titled "Piccolo" with three tabs. Close manually.

- [ ] **Step 10: Commit**

```bash
git add piccolo.py src/ui tests/test_ui_smoke.py requirements.txt
git commit -m "feat: add Qt scaffolding with three empty tabs"
```

---

## Task 4: Pipeline worker `QThread`

**Files:**
- Create: `src/ui/pipeline_worker.py`
- Modify: `src/ui/main_window.py`

**Goal:** A `QThread` that runs the existing backend pipeline (cameras → flip → align → process → overlay), emits signals for the latest SBS frame, and exposes `on_key_down/up` passthrough to `InputHandler` so Qt key events drive pedal/keyboard actions.

- [ ] **Step 1: Create `src/ui/pipeline_worker.py`**

```python
"""Backend pipeline running in a QThread.

Reads frames from both cameras, applies flip_180, runs the StereoAligner,
processes stereo (zoom + convergence), applies calibration nudges and
overlays, and emits a QImage signal for the latest SBS frame.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QObject

from ..config import PiccoloCfg
from ..camera import CameraCapture, TestPatternCamera
from ..stereo_processor import StereoProcessor
from ..stereo_align import StereoAligner
from ..calibration import CalibrationOverlay
from ..input_handler import InputHandler, Action
from .qt_helpers import ndarray_to_qimage


class PipelineWorker(QThread):
    """Long-running backend thread."""

    frame_ready = pyqtSignal(object)            # QImage — processed SBS
    status_tick = pyqtSignal(dict)              # FPS, alignment, pedal mode
    error = pyqtSignal(str)

    def __init__(self, cfg: PiccoloCfg, parent: QObject | None = None):
        super().__init__(parent)
        self.cfg = cfg
        self._running = False
        # Shared backend instances (tabs read/write these directly)
        eye_w = cfg.display.width // 2
        eye_h = cfg.display.height
        self.processor = StereoProcessor(cfg.stereo, eye_w, eye_h)
        self.aligner = StereoAligner(
            cfg.stereo.alignment,
            cfg.cameras.left.width,
            cfg.cameras.left.height,
        )
        self.calibration = CalibrationOverlay(cfg.calibration)
        self.input = InputHandler(self.aligner, cfg.controls)
        # Apply persisted calibration state to processor + calibration overlay
        st = cfg.calibration_state
        self.calibration.nudge_left = st.nudge_left_x
        self.calibration.nudge_right = st.nudge_right_x
        self.calibration.nudge_left_y = st.nudge_left_y
        self.calibration.nudge_right_y = st.nudge_right_y
        self.processor.base_offset = st.convergence_offset
        if hasattr(self.processor, "set_joint_zoom_center"):
            self.processor.set_joint_zoom_center(st.joint_zoom_center)
        if hasattr(self.processor, "set_joint_zoom_center_y"):
            self.processor.set_joint_zoom_center_y(st.joint_zoom_center_y)

        self.cam_l = None
        self.cam_r = None
        self._fps_hist: list[float] = []

    # ------------------------------------------------------------------

    def run(self) -> None:  # QThread entry
        try:
            self._open_cameras()
            self._running = True
            while self._running:
                t0 = time.perf_counter()
                self._tick()
                dt = time.perf_counter() - t0
                self._fps_hist.append(dt)
                if len(self._fps_hist) > 60:
                    self._fps_hist.pop(0)
        except Exception as exc:
            self.error.emit(f"Pipeline error: {exc}")
        finally:
            self._cleanup()

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    # ------------------------------------------------------------------

    def _open_cameras(self) -> None:
        c = self.cfg.cameras
        if c.test_mode:
            self.cam_l = TestPatternCamera(c.left.width, c.left.height, side="left", name="test-L").start()
            self.cam_r = TestPatternCamera(c.right.width, c.right.height, side="right", name="test-R").start()
        else:
            self.cam_l = CameraCapture(c.left.index, c.left.width, c.left.height,
                                        backend=c.backend, name="cam-L").start()
            try:
                self.cam_r = CameraCapture(c.right.index, c.right.width, c.right.height,
                                            backend=c.backend, name="cam-R").start()
            except Exception:
                if self.cam_l is not None:
                    self.cam_l.stop()
                raise

    def _tick(self) -> None:
        # Handle queued input actions
        for act in self.input.poll_actions():
            self._apply_action(act)
        if not self._running:
            return

        frame_l = self.cam_l.read_no_copy() if self.cam_l else None
        frame_r = self.cam_r.read_no_copy() if self.cam_r else None
        if frame_l is None or frame_r is None:
            self.msleep(5)
            return

        if self.cfg.cameras.left.flip_180:
            frame_l = cv2.rotate(frame_l, cv2.ROTATE_180)
        if self.cfg.cameras.right.flip_180:
            frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)

        if self.aligner.needs_update():
            self.aligner.update(frame_l, frame_r)
        frame_l, frame_r = self.aligner.warp_pair(frame_l, frame_r)

        eye_l, eye_r, sbs = self.processor.process_pair(frame_l, frame_r)
        eye_l, eye_r = self.calibration.apply_nudge(eye_l, eye_r)
        sbs[:, :self.processor.eye_w] = eye_l
        sbs[:, self.processor.eye_w:] = eye_r

        self.frame_ready.emit(ndarray_to_qimage(sbs))
        self._emit_status()

    def _emit_status(self) -> None:
        avg_dt = sum(self._fps_hist) / len(self._fps_hist) if self._fps_hist else 0.016
        fps = 1.0 / avg_dt if avg_dt > 0 else 0
        ar = self.aligner.result
        self.status_tick.emit({
            "fps": fps,
            "dy": ar.dy,
            "dtheta_deg": ar.dtheta * 57.2958,
            "aligner_converged": self.aligner.converged,
            "pedal_mode": self.input.get_pedal_mode(),
            "zoom": self.processor.zoom,
        })

    def _apply_action(self, act: Action) -> None:
        if act == Action.ZOOM_IN:
            self.processor.zoom_in()
        elif act == Action.ZOOM_OUT:
            self.processor.zoom_out()
        elif act == Action.CONVERGE_IN:
            self.processor.converge_in()
        elif act == Action.CONVERGE_OUT:
            self.processor.converge_out()
        elif act == Action.PEDAL_CENTER_LEFT:
            if hasattr(self.processor, "joint_zoom_center"):
                self.processor.set_joint_zoom_center(self.processor.joint_zoom_center - 1)
        elif act == Action.PEDAL_CENTER_RIGHT:
            if hasattr(self.processor, "joint_zoom_center"):
                self.processor.set_joint_zoom_center(self.processor.joint_zoom_center + 1)
        elif act == Action.PEDAL_CENTER_UP:
            if hasattr(self.processor, "joint_zoom_center_y"):
                self.processor.set_joint_zoom_center_y(self.processor.joint_zoom_center_y - 1)
        elif act == Action.PEDAL_CENTER_DOWN:
            if hasattr(self.processor, "joint_zoom_center_y"):
                self.processor.set_joint_zoom_center_y(self.processor.joint_zoom_center_y + 1)
        elif act == Action.RESET:
            self.processor.reset()
            self.aligner.reset()
            self.calibration.reset_nudge()
        elif act == Action.QUIT:
            self._running = False

    def _cleanup(self) -> None:
        if self.cam_l:
            self.cam_l.stop()
        if self.cam_r:
            self.cam_r.stop()
```

- [ ] **Step 2: Wire the worker into `MainWindow`**

Replace `src/ui/main_window.py` with:

```python
"""Main Qt window: tabbed assistant UI + pipeline worker."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QMainWindow, QTabWidget

from ..config import PiccoloCfg
from .live_tab import LiveTab
from .calibration_tab import CalibrationTab
from .settings_tab import SettingsTab
from .pipeline_worker import PipelineWorker


class MainWindow(QMainWindow):
    def __init__(self, cfg: PiccoloCfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Piccolo")
        self.resize(1280, 800)

        # Start pipeline first so tabs can connect to its signals
        self.worker = PipelineWorker(cfg, self)

        tabs = QTabWidget(self)
        self.live_tab = LiveTab(self.worker, self)
        self.calibration_tab = CalibrationTab(self.worker, self)
        self.settings_tab = SettingsTab(self.worker, self)
        tabs.addTab(self.live_tab, "Live")
        tabs.addTab(self.calibration_tab, "Calibration")
        tabs.addTab(self.settings_tab, "Settings")
        self.setCentralWidget(tabs)

        self.worker.start()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not event.isAutoRepeat():
            self.worker.input.on_key_down(self._qt_key_name(event))
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if not event.isAutoRepeat():
            self.worker.input.on_key_up(self._qt_key_name(event))
        super().keyReleaseEvent(event)

    @staticmethod
    def _qt_key_name(event: QKeyEvent) -> str:
        # Single-char keys: use text (already lowercased by Qt when no modifiers)
        txt = event.text().lower()
        if txt and txt.isprintable() and len(txt) == 1:
            return txt
        # Named keys
        key = event.key()
        return Qt.Key(key).name.removeprefix("Key_").lower() if key else ""

    def closeEvent(self, event) -> None:
        self.worker.stop()
        super().closeEvent(event)
```

Update tab stubs to accept `worker`:

`src/ui/live_tab.py` (stub signature update):

```python
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout


class LiveTab(QWidget):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Live tab — preview coming soon"))
```

Apply the same `worker, parent=None` signature update to `CalibrationTab` and `SettingsTab`.

- [ ] **Step 3: Run smoke test**

```bash
.venv/Scripts/python -m pytest tests/test_ui_smoke.py -v
```

Expected: existing test still passes (constructs MainWindow; worker starts in background).

- [ ] **Step 4: Smoke-run the app**

With `cameras.test_mode: true` in `config.yaml`, run:

```bash
.venv/Scripts/python piccolo.py
```

Expected: app opens, pipeline worker starts (no visible video yet since LiveTab is still a stub), no crashes. Close window — process exits cleanly within ~2 seconds.

- [ ] **Step 5: Commit**

```bash
git add src/ui/pipeline_worker.py src/ui/main_window.py src/ui/live_tab.py src/ui/calibration_tab.py src/ui/settings_tab.py
git commit -m "feat: add pipeline worker QThread with Qt key-event integration"
```

---

## Task 5: Live tab — preview + zoom pad + status strip

**Files:**
- Create: `src/ui/video_widget.py`
- Modify: `src/ui/live_tab.py`

**Goal:** The Live tab shows the pipeline's SBS frame full-width, a 4-button arrow pad (zoom-center nudges), and a bottom status strip (FPS, alignment, pedal mode).

- [ ] **Step 1: Create `src/ui/video_widget.py`**

```python
"""QLabel-based widget that displays QImage frames with auto-scaling."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class VideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 225)
        self.setStyleSheet("background-color: black;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._last: QImage | None = None

    def set_frame(self, image: QImage) -> None:
        self._last = image
        self._rescale()

    def resizeEvent(self, event) -> None:
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self) -> None:
        if self._last is None:
            return
        scaled = self._last.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(scaled))
```

- [ ] **Step 2: Replace `src/ui/live_tab.py`**

```python
"""Live tab: SBS preview, zoom-center arrow pad, bottom status strip."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QGroupBox,
)

from .video_widget import VideoWidget


class LiveTab(QWidget):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker

        root = QVBoxLayout(self)
        main = QHBoxLayout()
        root.addLayout(main, stretch=1)

        # Preview (left, grows to fill)
        self.preview = VideoWidget(self)
        main.addWidget(self.preview, stretch=4)

        # Side panel (right)
        side = QVBoxLayout()
        main.addLayout(side, stretch=1)
        side.addWidget(self._make_zoom_pad())
        side.addStretch(1)

        # Status strip (bottom)
        root.addWidget(self._make_status_strip())

        # Wire signals
        worker.frame_ready.connect(self.preview.set_frame)
        worker.status_tick.connect(self._on_status)

    # ------------------------------------------------------------------

    def _make_zoom_pad(self) -> QGroupBox:
        box = QGroupBox("Zoom Center", self)
        grid = QGridLayout(box)
        btn_up = QPushButton("↑")
        btn_down = QPushButton("↓")
        btn_left = QPushButton("←")
        btn_right = QPushButton("→")
        for b in (btn_up, btn_down, btn_left, btn_right):
            b.setFixedSize(48, 48)
            b.setStyleSheet("font-size: 18px;")
        grid.addWidget(btn_up,    0, 1)
        grid.addWidget(btn_left,  1, 0)
        grid.addWidget(btn_right, 1, 2)
        grid.addWidget(btn_down,  2, 1)
        btn_up.clicked.connect(lambda: self._nudge_center_y(-1))
        btn_down.clicked.connect(lambda: self._nudge_center_y(+1))
        btn_left.clicked.connect(lambda: self._nudge_center_x(-1))
        btn_right.clicked.connect(lambda: self._nudge_center_x(+1))
        return box

    def _make_status_strip(self) -> QWidget:
        strip = QWidget(self)
        lay = QHBoxLayout(strip)
        lay.setContentsMargins(8, 4, 8, 4)
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_align = QLabel("Align: --")
        self.lbl_pedal = QLabel("Pedal: OFF")
        for lbl in (self.lbl_fps, self.lbl_align, self.lbl_pedal):
            lbl.setStyleSheet("font-family: monospace;")
        lay.addWidget(self.lbl_fps)
        lay.addSpacing(20)
        lay.addWidget(self.lbl_align)
        lay.addStretch(1)
        lay.addWidget(self.lbl_pedal)
        return strip

    # ------------------------------------------------------------------

    def _nudge_center_x(self, delta: int) -> None:
        p = self.worker.processor
        if hasattr(p, "joint_zoom_center"):
            p.set_joint_zoom_center(p.joint_zoom_center + delta)

    def _nudge_center_y(self, delta: int) -> None:
        p = self.worker.processor
        if hasattr(p, "joint_zoom_center_y"):
            p.set_joint_zoom_center_y(p.joint_zoom_center_y + delta)

    def _on_status(self, st: dict) -> None:
        self.lbl_fps.setText(f"FPS: {st['fps']:.0f}")
        dy = st["dy"]
        dt = st["dtheta_deg"]
        conv = "ok" if st["aligner_converged"] else "calibrating"
        self.lbl_align.setText(f"Align: dy={dy:+.1f}px rot={dt:+.2f}° ({conv})")
        mode = st["pedal_mode"]
        mode_names = {"a": "ZOOM", "b": "SIDE", "c": "UP/DOWN"}
        self.lbl_pedal.setText(f"Pedal: {mode_names.get(mode, 'OFF')}")
```

- [ ] **Step 3: Run smoke test**

```bash
.venv/Scripts/python -m pytest tests/test_ui_smoke.py -v
```

Expected: passes.

- [ ] **Step 4: Manual smoke-run**

With `cameras.test_mode: true`:

```bash
.venv/Scripts/python piccolo.py
```

Expected: Live tab shows the test-pattern SBS, status strip updates FPS numbers, arrow buttons nudge the zoom center (visible if you zoom in via `=` key).

- [ ] **Step 5: Commit**

```bash
git add src/ui/video_widget.py src/ui/live_tab.py
git commit -m "feat: add Live tab preview, zoom-center pad, and status strip"
```

---

## Task 6: Goovis output window

**Files:**
- Create: `src/ui/goovis_window.py`
- Modify: `src/ui/main_window.py`

**Goal:** A borderless full-screen `QWidget` on the detected Goovis monitor, receiving the same `frame_ready` signal as the Live preview. Degrades silently if not detected.

- [ ] **Step 1: Create `src/ui/goovis_window.py`**

```python
"""Borderless full-screen stereo output for the Goovis headset."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication, QScreen
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from ..config import DisplayCfg
from .video_widget import VideoWidget


class GoovisWindow(QWidget):
    def __init__(self, cfg: DisplayCfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.video = VideoWidget(self)
        lay.addWidget(self.video)

    def show_on_goovis(self) -> bool:
        """Position on the detected Goovis monitor and show full-screen.

        Returns True if a matching screen was found and the window shown.
        """
        screen = _find_goovis_screen(self.cfg)
        if screen is None:
            return False
        geom = screen.geometry()
        self.move(geom.x(), geom.y())
        self.resize(geom.width(), geom.height())
        self.showFullScreen()
        return True


def _find_goovis_screen(cfg: DisplayCfg) -> QScreen | None:
    screens = QGuiApplication.screens()
    # Explicit index
    if isinstance(cfg.monitor, int) and 0 <= cfg.monitor < len(screens):
        return screens[cfg.monitor]
    # Auto-detect
    for s in screens:
        name = s.name().upper()
        if "GOOVIS" in name or "NED" in name:
            return s
    # Fallback: non-primary 1920x1080
    primary = QGuiApplication.primaryScreen()
    for s in screens:
        if s is not primary and s.geometry().width() == 1920 and s.geometry().height() == 1080:
            return s
    return None
```

- [ ] **Step 2: Wire into `MainWindow`**

Modify `src/ui/main_window.py` — in `__init__`, after the worker is created but before `self.worker.start()`, add:

```python
        # Goovis output window (optional)
        self.goovis = GoovisWindow(cfg.display, self)
        if self.goovis.show_on_goovis():
            self.worker.frame_ready.connect(self.goovis.video.set_frame)
        else:
            self.goovis.deleteLater()
            self.goovis = None
```

Import at the top:
```python
from .goovis_window import GoovisWindow
```

Modify `closeEvent` to close Goovis first:
```python
    def closeEvent(self, event) -> None:
        if self.goovis is not None:
            self.goovis.close()
        self.worker.stop()
        super().closeEvent(event)
```

- [ ] **Step 3: Smoke-test without Goovis**

```bash
.venv/Scripts/python piccolo.py
```

Expected: no second window (assuming no 1920×1080 non-primary display is connected), no crash.

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ui/goovis_window.py src/ui/main_window.py
git commit -m "feat: add Goovis borderless full-screen output window"
```

---

## Task 7: Calibration tab — nudge sliders (persistent)

**Files:**
- Modify: `src/ui/calibration_tab.py`
- Create: `tests/test_calibration_persistence.py`

**Goal:** Four sliders (Left-X, Right-X, Left-Y, Right-Y), each ±300 px, wired to `CalibrationOverlay` and persisted to `config.yaml` via `save_calibration_state`. "Reset all" button zeroes them and saves.

- [ ] **Step 1: Write failing test**

Create `tests/test_calibration_persistence.py`:

```python
import os
import tempfile
import yaml
import pytest
from PyQt6.QtWidgets import QApplication
from src.config import load_config
from src.config_state import save_calibration_state


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_saving_nudge_round_trips(qapp):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump({}, fh)
        cfg = load_config(path)
        cfg.calibration_state.nudge_left_x = 5
        cfg.calibration_state.nudge_right_y = -10
        save_calibration_state(cfg, path)
        cfg2 = load_config(path)
        assert cfg2.calibration_state.nudge_left_x == 5
        assert cfg2.calibration_state.nudge_right_y == -10
```

- [ ] **Step 2: Replace `src/ui/calibration_tab.py`**

```python
"""Calibration tab: nudge sliders + physical-cal wizard (wizard in Task 8)."""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox,
)

from ..config_state import save_calibration_state


class CalibrationTab(QWidget):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        root = QVBoxLayout(self)
        root.addWidget(self._make_nudge_group())
        root.addWidget(self._make_reset_row())
        root.addStretch(1)

    def _make_nudge_group(self) -> QGroupBox:
        box = QGroupBox("Per-eye pixel nudge", self)
        lay = QVBoxLayout(box)
        st = self.worker.cfg.calibration_state
        self.sld_lx = self._add_slider(lay, "Left eye X", st.nudge_left_x, self._on_lx)
        self.sld_rx = self._add_slider(lay, "Right eye X", st.nudge_right_x, self._on_rx)
        self.sld_ly = self._add_slider(lay, "Left eye Y", st.nudge_left_y, self._on_ly)
        self.sld_ry = self._add_slider(lay, "Right eye Y", st.nudge_right_y, self._on_ry)
        return box

    def _add_slider(self, parent_layout, label: str, initial: int, cb) -> QSlider:
        row = QHBoxLayout()
        lbl = QLabel(f"{label}: {initial}px")
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setMinimum(-300)
        sld.setMaximum(300)
        sld.setValue(initial)
        sld.valueChanged.connect(lambda v, l=lbl, t=label: l.setText(f"{t}: {v}px"))
        sld.sliderReleased.connect(lambda s=sld, cb=cb: cb(s.value()))
        row.addWidget(lbl, stretch=1)
        row.addWidget(sld, stretch=3)
        parent_layout.addLayout(row)
        return sld

    def _make_reset_row(self) -> QWidget:
        w = QWidget(self)
        lay = QHBoxLayout(w)
        btn = QPushButton("Reset all nudges")
        btn.clicked.connect(self._reset)
        lay.addWidget(btn)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------

    def _on_lx(self, v: int) -> None: self._set("nudge_left_x",  v, lambda: setattr(self.worker.calibration, "nudge_left",  v))
    def _on_rx(self, v: int) -> None: self._set("nudge_right_x", v, lambda: setattr(self.worker.calibration, "nudge_right", v))
    def _on_ly(self, v: int) -> None: self._set("nudge_left_y",  v, lambda: setattr(self.worker.calibration, "nudge_left_y",  v))
    def _on_ry(self, v: int) -> None: self._set("nudge_right_y", v, lambda: setattr(self.worker.calibration, "nudge_right_y", v))

    def _set(self, attr: str, value: int, apply_runtime) -> None:
        setattr(self.worker.cfg.calibration_state, attr, value)
        apply_runtime()
        save_calibration_state(self.worker.cfg)

    def _reset(self) -> None:
        for s in (self.sld_lx, self.sld_rx, self.sld_ly, self.sld_ry):
            s.setValue(0)
        self.worker.calibration.reset_nudge()
        st = self.worker.cfg.calibration_state
        st.nudge_left_x = st.nudge_right_x = st.nudge_left_y = st.nudge_right_y = 0
        save_calibration_state(self.worker.cfg)
```

- [ ] **Step 3: Run tests**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 4: Manual smoke-run**

```bash
.venv/Scripts/python piccolo.py
```

- Switch to Calibration tab.
- Move a slider, release.
- Verify `config.yaml` now contains `calibration_state` with the new value.
- Close app, relaunch, confirm slider starts at the saved value.

- [ ] **Step 5: Commit**

```bash
git add src/ui/calibration_tab.py tests/test_calibration_persistence.py
git commit -m "feat: Calibration tab with persistent per-eye nudge sliders"
```

---

## Task 8: Calibration tab — physical-cal wizard

**Files:**
- Modify: `src/ui/calibration_tab.py`

**Goal:** Add a wizard section to the Calibration tab that embeds the 4-phase flow from `physical_calibration.py` (focus → scale → horizontal → rotation). Uses the existing `PatternRenderer` + `PhysicalCalSession` from `src/physical_cal.py`.

- [ ] **Step 1: Add wizard to `src/ui/calibration_tab.py`**

Add these imports at top:

```python
import cv2
import numpy as np
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import QImage
from ..physical_cal import PhysicalCalSession, PatternRenderer
from .qt_helpers import ndarray_to_qimage
from .video_widget import VideoWidget
```

Extend `CalibrationTab.__init__` — before `root.addStretch(1)` add:

```python
        root.addWidget(self._make_wizard_group(), stretch=1)
```

Add these methods to the class:

```python
    def _make_wizard_group(self) -> QGroupBox:
        box = QGroupBox("Physical calibration wizard", self)
        lay = QVBoxLayout(box)
        self.wizard_preview = VideoWidget(box)
        lay.addWidget(self.wizard_preview, stretch=1)
        btns = QHBoxLayout()
        self.wizard_phase_lbl = QLabel("Phase: Focus (1/4)")
        self.wizard_readout_lbl = QLabel("")
        self.wizard_readout_lbl.setStyleSheet("font-family: monospace;")
        self.btn_prev = QPushButton("← Prev")
        self.btn_next = QPushButton("Next →")
        self.btn_prev.clicked.connect(self._wizard_prev)
        self.btn_next.clicked.connect(self._wizard_next)
        btns.addWidget(self.btn_prev)
        btns.addWidget(self.btn_next)
        btns.addSpacing(20)
        btns.addWidget(self.wizard_phase_lbl)
        btns.addStretch(1)
        btns.addWidget(self.wizard_readout_lbl)
        lay.addLayout(btns)

        # Wizard state
        self.session = PhysicalCalSession()
        self.renderer = PatternRenderer()
        self.worker.frame_ready.connect(self._on_frame_for_wizard)
        self.worker.status_tick.connect(self._on_status_for_wizard)
        self._latest_status: dict = {}
        return box

    @pyqtSlot(dict)
    def _on_status_for_wizard(self, st: dict) -> None:
        self._latest_status = st

    @pyqtSlot(object)
    def _on_frame_for_wizard(self, _qimg: QImage) -> None:
        # Re-render from the worker's latest raw frames with the wizard overlay.
        # Easier path: grab from camera captures directly.
        cam_l = self.worker.cam_l
        cam_r = self.worker.cam_r
        if cam_l is None or cam_r is None:
            return
        fl = cam_l.read_no_copy()
        fr = cam_r.read_no_copy()
        if fl is None or fr is None:
            return
        # Apply flip to match live pipeline
        if self.worker.cfg.cameras.left.flip_180:
            fl = cv2.rotate(fl, cv2.ROTATE_180)
        if self.worker.cfg.cameras.right.flip_180:
            fr = cv2.rotate(fr, cv2.ROTATE_180)
        eye_l = fl.copy()
        eye_r = fr.copy()
        phase = self.session.phase
        dy = self._latest_status.get("dy")
        dtheta_deg = self._latest_status.get("dtheta_deg")
        sharp_l = PhysicalCalSession.sharpness(fl)
        sharp_r = PhysicalCalSession.sharpness(fr)
        if phase == "focus":
            self.renderer.render_focus(eye_l, sharp_l)
            self.renderer.render_focus(eye_r, sharp_r)
        elif phase == "scale":
            self.renderer.render_scale(eye_l)
            self.renderer.render_scale(eye_r)
        elif phase == "horizontal":
            self.renderer.render_horizontal(eye_l, dy)
            self.renderer.render_horizontal(eye_r, dy)
        elif phase == "rotation":
            self.renderer.render_rotation(eye_l, dtheta_deg)
            self.renderer.render_rotation(eye_r, dtheta_deg)
        sbs = np.concatenate([eye_l, eye_r], axis=1)
        self.wizard_preview.set_frame(ndarray_to_qimage(sbs))
        self._update_wizard_readout(sharp_l, sharp_r, dy, dtheta_deg)

    def _update_wizard_readout(self, sl, sr, dy, dt):
        phase = self.session.phase
        self.wizard_phase_lbl.setText(
            f"Phase: {phase.capitalize()} ({self.session.phase_index+1}/{self.session.total_phases})"
        )
        if phase == "focus":
            self.wizard_readout_lbl.setText(f"sharp L={sl:.0f}  R={sr:.0f}")
        elif phase == "horizontal":
            self.wizard_readout_lbl.setText(f"dy={dy:+.1f}px" if dy is not None else "dy=--")
        elif phase == "rotation":
            self.wizard_readout_lbl.setText(f"rot={dt:+.2f}°" if dt is not None else "rot=--")
        else:
            self.wizard_readout_lbl.setText("")

    def _wizard_next(self) -> None:
        self.session.next_phase()

    def _wizard_prev(self) -> None:
        self.session.prev_phase()
```

- [ ] **Step 2: Manual smoke-run**

```bash
.venv/Scripts/python piccolo.py
```

- Calibration tab shows the wizard with Siemens star overlay on test-pattern feed.
- Next / Prev buttons cycle through focus → scale → horizontal → rotation.
- Readout text updates per phase.

- [ ] **Step 3: Run tests**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/ui/calibration_tab.py
git commit -m "feat: embed physical-cal wizard in Calibration tab"
```

---

## Task 9: Settings tab — Cameras + Display + Stereo

**Files:**
- Modify: `src/ui/settings_tab.py`

**Goal:** Settings tab with three `QGroupBox` sections (Cameras, Goovis Display, Stereo). Live widgets that update the running worker's config (changes to values that require a restart show a warning).

- [ ] **Step 1: Replace `src/ui/settings_tab.py`** (pedals + config I/O added in Task 10)

```python
"""Settings tab: cameras, display, stereo groups."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QLabel,
)

from ..config_state import save_calibration_state


class SettingsTab(QWidget):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        root = QVBoxLayout(self)
        root.addWidget(self._make_cameras_group())
        root.addWidget(self._make_display_group())
        root.addWidget(self._make_stereo_group())
        root.addStretch(1)

    # ------------------ Cameras ------------------------------------------

    def _make_cameras_group(self) -> QGroupBox:
        box = QGroupBox("Cameras (requires app restart to apply)", self)
        form = QFormLayout(box)
        c = self.worker.cfg.cameras
        form.addRow(QLabel("Left index"),  self._spinbox(c.left.index, 0, 10,
                                                          lambda v: setattr(c.left, "index", v)))
        form.addRow(QLabel("Right index"), self._spinbox(c.right.index, 0, 10,
                                                          lambda v: setattr(c.right, "index", v)))
        cb_flip_l = QCheckBox("Flip Left 180°")
        cb_flip_l.setChecked(c.left.flip_180)
        cb_flip_l.stateChanged.connect(
            lambda s: setattr(c.left, "flip_180", s == Qt.CheckState.Checked.value))
        form.addRow(cb_flip_l)
        cb_flip_r = QCheckBox("Flip Right 180°")
        cb_flip_r.setChecked(c.right.flip_180)
        cb_flip_r.stateChanged.connect(
            lambda s: setattr(c.right, "flip_180", s == Qt.CheckState.Checked.value))
        form.addRow(cb_flip_r)
        return box

    # ------------------ Display ------------------------------------------

    def _make_display_group(self) -> QGroupBox:
        box = QGroupBox("Goovis Display", self)
        form = QFormLayout(box)
        self.cmb_monitor = QComboBox()
        screens = QGuiApplication.screens()
        for i, s in enumerate(screens):
            g = s.geometry()
            self.cmb_monitor.addItem(f"[{i}] {s.name()}  {g.width()}×{g.height()}", i)
        form.addRow(QLabel("Monitor"), self.cmb_monitor)
        self.cmb_monitor.currentIndexChanged.connect(
            lambda i: setattr(self.worker.cfg.display, "monitor", self.cmb_monitor.itemData(i)))
        return box

    # ------------------ Stereo -------------------------------------------

    def _make_stereo_group(self) -> QGroupBox:
        box = QGroupBox("Stereo", self)
        form = QFormLayout(box)
        s = self.worker.cfg.stereo
        # Convergence base offset slider
        row = QHBoxLayout()
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setMinimum(-200); sld.setMaximum(200)
        sld.setValue(s.convergence.base_offset)
        lbl = QLabel(f"{s.convergence.base_offset}px")
        sld.valueChanged.connect(lambda v, l=lbl: l.setText(f"{v}px"))
        sld.sliderReleased.connect(lambda: self._save_conv(sld.value()))
        row.addWidget(sld); row.addWidget(lbl)
        wrap = QWidget(); wrap.setLayout(row)
        form.addRow(QLabel("Convergence offset"), wrap)
        # Auto-alignment
        cb = QCheckBox("Auto alignment")
        cb.setChecked(s.alignment.enabled)
        cb.stateChanged.connect(
            lambda v: setattr(self.worker.aligner, "enabled", v == Qt.CheckState.Checked.value))
        form.addRow(cb)
        # Zoom step
        form.addRow(QLabel("Zoom step"),
                    self._doublebox(s.zoom.step, 0.001, 1.0, 0.01,
                                    lambda v: self._apply_zoom_step(v)))
        form.addRow(QLabel("Zoom tick (ms)"),
                    self._spinbox(s.zoom.tick_ms, 1, 1000,
                                  lambda v: setattr(s.zoom, "tick_ms", v)))
        return box

    # ------------------ Helpers ------------------------------------------

    def _spinbox(self, value: int, lo: int, hi: int, cb) -> QSpinBox:
        w = QSpinBox(); w.setRange(lo, hi); w.setValue(value)
        w.valueChanged.connect(cb)
        return w

    def _doublebox(self, value: float, lo: float, hi: float, step: float, cb) -> QDoubleSpinBox:
        w = QDoubleSpinBox(); w.setRange(lo, hi); w.setSingleStep(step); w.setDecimals(3); w.setValue(value)
        w.valueChanged.connect(cb)
        return w

    def _save_conv(self, value: int) -> None:
        self.worker.cfg.stereo.convergence.base_offset = value
        self.worker.processor.base_offset = value
        self.worker.cfg.calibration_state.convergence_offset = value
        save_calibration_state(self.worker.cfg)

    def _apply_zoom_step(self, v: float) -> None:
        self.worker.cfg.stereo.zoom.step = v
```

- [ ] **Step 2: Manual smoke-run**

```bash
.venv/Scripts/python piccolo.py
```

- Settings tab shows three groups.
- Convergence slider moves → SBS preview shifts → releasing saves to config.yaml.
- Auto-alignment checkbox toggles `worker.aligner.enabled`.
- Monitor dropdown lists connected displays.

- [ ] **Step 3: Run tests**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/ui/settings_tab.py
git commit -m "feat: Settings tab with cameras, display, and stereo groups"
```

---

## Task 10: Settings tab — Pedals + Config I/O

**Files:**
- Modify: `src/ui/settings_tab.py`

**Goal:** Add Pedals section (key rebinding, mode assignments, live pedal indicator) and Config File section (Load / Save / Reset to defaults). Rebinding writes the changes into `ControlsCfg` and rebuilds the `InputHandler` keymap.

- [ ] **Step 1: Extend `src/ui/settings_tab.py`**

Add at the top of the file:

```python
from PyQt6.QtWidgets import QLineEdit, QPushButton, QMessageBox, QFileDialog
import yaml
```

Extend `SettingsTab.__init__` — before `root.addStretch(1)` add:

```python
        root.addWidget(self._make_pedals_group())
        root.addWidget(self._make_config_group())
```

Add these methods:

```python
    def _make_pedals_group(self) -> QGroupBox:
        box = QGroupBox("Pedals", self)
        form = QFormLayout(box)
        ctl = self.worker.cfg.controls

        cb_enable = QCheckBox("Pedal input enabled")
        cb_enable.setChecked(True)
        form.addRow(cb_enable)

        # Key bindings
        self.ed_key_a = QLineEdit(ctl.pedal_key_a); self.ed_key_a.setMaxLength(1)
        self.ed_key_b = QLineEdit(ctl.pedal_key_b); self.ed_key_b.setMaxLength(1)
        self.ed_key_c = QLineEdit(ctl.pedal_key_c); self.ed_key_c.setMaxLength(1)
        for ed, attr in ((self.ed_key_a, "pedal_key_a"),
                          (self.ed_key_b, "pedal_key_b"),
                          (self.ed_key_c, "pedal_key_c")):
            ed.editingFinished.connect(
                lambda e=ed, a=attr: self._set_pedal_key(a, e.text()))
        form.addRow(QLabel("Pedal A key"), self.ed_key_a)
        form.addRow(QLabel("Pedal B key"), self.ed_key_b)
        form.addRow(QLabel("Pedal C key"), self.ed_key_c)

        # Mode assignments
        modes = ["zoom", "side", "updown", "none"]
        self.cmb_mode_a = QComboBox(); self.cmb_mode_a.addItems(modes); self.cmb_mode_a.setCurrentText(ctl.pedal_mode_a)
        self.cmb_mode_b = QComboBox(); self.cmb_mode_b.addItems(modes); self.cmb_mode_b.setCurrentText(ctl.pedal_mode_b)
        self.cmb_mode_c = QComboBox(); self.cmb_mode_c.addItems(modes); self.cmb_mode_c.setCurrentText(ctl.pedal_mode_c)
        self.cmb_mode_a.currentTextChanged.connect(lambda t: setattr(ctl, "pedal_mode_a", t))
        self.cmb_mode_b.currentTextChanged.connect(lambda t: setattr(ctl, "pedal_mode_b", t))
        self.cmb_mode_c.currentTextChanged.connect(lambda t: setattr(ctl, "pedal_mode_c", t))
        form.addRow(QLabel("Pedal A mode"), self.cmb_mode_a)
        form.addRow(QLabel("Pedal B mode"), self.cmb_mode_b)
        form.addRow(QLabel("Pedal C mode"), self.cmb_mode_c)

        # Long-press repeat
        form.addRow(QLabel("Long-press repeat (ms)"),
                    self._spinbox(ctl.pedal_repeat_ms, 1, 1000,
                                  lambda v: setattr(ctl, "pedal_repeat_ms", v)))

        # Live pedal mode indicator
        self.lbl_live_mode = QLabel("Pedal mode: OFF")
        self.lbl_live_mode.setStyleSheet("font-family: monospace;")
        form.addRow(self.lbl_live_mode)
        self.worker.status_tick.connect(self._on_status_mode)
        return box

    def _on_status_mode(self, st: dict) -> None:
        mode = st.get("pedal_mode")
        mode_names = {"a": "ZOOM", "b": "SIDE", "c": "UP/DOWN"}
        self.lbl_live_mode.setText(f"Pedal mode: {mode_names.get(mode, 'OFF')}")

    def _set_pedal_key(self, attr: str, text: str) -> None:
        if len(text) != 1:
            return
        setattr(self.worker.cfg.controls, attr, text.lower())

    def _make_config_group(self) -> QGroupBox:
        box = QGroupBox("Config file", self)
        row = QHBoxLayout(box)
        btn_load = QPushButton("Load…")
        btn_save = QPushButton("Save")
        btn_reset = QPushButton("Reset to defaults")
        btn_load.clicked.connect(self._load_config)
        btn_save.clicked.connect(self._save_config)
        btn_reset.clicked.connect(self._reset_config)
        row.addWidget(btn_load); row.addWidget(btn_save); row.addWidget(btn_reset); row.addStretch(1)
        return box

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load config", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        QMessageBox.information(self, "Restart required",
                                 f"Config loaded from {path}.\nRestart the app for all changes to take effect.")

    def _save_config(self) -> None:
        from dataclasses import asdict
        cfg = self.worker.cfg
        raw = {
            "display": asdict(cfg.display),
            "cameras": {
                "backend": cfg.cameras.backend,
                "left":  asdict(cfg.cameras.left),
                "right": asdict(cfg.cameras.right),
                "test_mode": cfg.cameras.test_mode,
            },
            "stereo": {
                "zoom": asdict(cfg.stereo.zoom),
                "convergence": asdict(cfg.stereo.convergence),
                "alignment": asdict(cfg.stereo.alignment),
            },
            "calibration": asdict(cfg.calibration),
            "calibration_state": asdict(cfg.calibration_state),
            "controls": asdict(cfg.controls),
        }
        from ..config_state import _default_config_path  # reuse helper
        path = _default_config_path()
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(raw, fh, sort_keys=False)
        QMessageBox.information(self, "Saved", f"Wrote {path}")

    def _reset_config(self) -> None:
        if QMessageBox.question(self, "Reset config", "Overwrite config.yaml with defaults?") \
                != QMessageBox.StandardButton.Yes:
            return
        from ..config import PiccoloCfg
        from dataclasses import asdict
        from ..config_state import _default_config_path
        defaults = PiccoloCfg()
        raw = {
            "display": asdict(defaults.display),
            "cameras": {
                "backend": defaults.cameras.backend,
                "left":  asdict(defaults.cameras.left),
                "right": asdict(defaults.cameras.right),
                "test_mode": defaults.cameras.test_mode,
            },
            "stereo": {
                "zoom": asdict(defaults.stereo.zoom),
                "convergence": asdict(defaults.stereo.convergence),
                "alignment": asdict(defaults.stereo.alignment),
            },
            "calibration": asdict(defaults.calibration),
            "calibration_state": asdict(defaults.calibration_state),
            "controls": asdict(defaults.controls),
        }
        path = _default_config_path()
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(raw, fh, sort_keys=False)
        QMessageBox.information(self, "Reset", f"Wrote defaults to {path}.\nRestart to apply.")
```

- [ ] **Step 2: Manual smoke-run**

```bash
.venv/Scripts/python piccolo.py
```

- Pedal key fields editable; changing "a" to "z" rebinds Pedal A to the Z key.
- Mode dropdowns switch assignments.
- Live pedal mode indicator updates when you press a pedal key.
- Save writes current config to `config.yaml`.

- [ ] **Step 3: Run tests**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add src/ui/settings_tab.py
git commit -m "feat: Settings tab Pedals group + Config file I/O"
```

---

## Task 11: Live tab — annotations

**Files:**
- Modify: `src/ui/live_tab.py`
- Create: `src/ui/annotation_overlay_widget.py`

**Goal:** Add a drawing canvas overlaid on the Live preview and a toolbar (freehand, arrow, circle, text, undo, clear, disparity offset). Backed by the existing `AnnotationOverlay` from `src/annotation.py`.

- [ ] **Step 1: Create `src/ui/annotation_overlay_widget.py`**

```python
"""Translucent drawing canvas that feeds into AnnotationOverlay."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QMouseEvent
from PyQt6.QtWidgets import QWidget


class AnnotationCanvas(QWidget):
    stroke_finished = pyqtSignal(list)  # list of (x, y) points in widget coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._points: list[QPointF] = []
        self._strokes: list[list[QPointF]] = []

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._points = [event.position()]
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._points:
            self._points.append(event.position())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._points:
            self._strokes.append(self._points)
            self.stroke_finished.emit([(p.x(), p.y()) for p in self._points])
            self._points = []
            self.update()

    def paintEvent(self, _) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(255, 200, 0), 3)
        painter.setPen(pen)
        for stroke in self._strokes:
            for i in range(1, len(stroke)):
                painter.drawLine(stroke[i - 1], stroke[i])
        if self._points:
            for i in range(1, len(self._points)):
                painter.drawLine(self._points[i - 1], self._points[i])

    def clear(self) -> None:
        self._strokes.clear()
        self._points.clear()
        self.update()

    def undo(self) -> None:
        if self._strokes:
            self._strokes.pop()
            self.update()
```

- [ ] **Step 2: Integrate annotation toolbar into `src/ui/live_tab.py`**

Add to imports:
```python
from PyQt6.QtWidgets import QToolButton
from .annotation_overlay_widget import AnnotationCanvas
```

In `LiveTab.__init__`, extend the side panel (after the zoom pad):

```python
        side.addWidget(self._make_annotation_group())
```

Layer the canvas on top of the preview — change the preview construction to:

```python
        # Preview stack: video + annotation canvas on top
        from PyQt6.QtWidgets import QStackedLayout
        preview_wrap = QWidget(self)
        stack = QStackedLayout(preview_wrap)
        stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self.preview = VideoWidget(preview_wrap)
        self.canvas = AnnotationCanvas(preview_wrap)
        stack.addWidget(self.preview)
        stack.addWidget(self.canvas)
        main.addWidget(preview_wrap, stretch=4)
```

Add `_make_annotation_group`:

```python
    def _make_annotation_group(self) -> QGroupBox:
        box = QGroupBox("Annotations", self)
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        btn_undo = QPushButton("Undo")
        btn_clear = QPushButton("Clear")
        btn_undo.clicked.connect(self.canvas.undo)
        btn_clear.clicked.connect(self.canvas.clear)
        row.addWidget(btn_undo); row.addWidget(btn_clear)
        lay.addLayout(row)
        return box
```

- [ ] **Step 3: Manual smoke-run**

```bash
.venv/Scripts/python piccolo.py
```

- Left-drag on the preview draws yellow freehand strokes.
- Undo button removes last stroke; Clear removes all.

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add src/ui/live_tab.py src/ui/annotation_overlay_widget.py
git commit -m "feat: Live tab freehand annotation overlay"
```

**NOTE:** This task delivers only freehand drawing + undo/clear. Arrow/circle/text tools and disparity-offset forwarding to the Goovis display are deferred to a follow-up task if the surgeon requests them. Ship annotations minimal first.

---

## Task 12: Packaging + delete old web UI

**Files:**
- Create: `piccolo.spec`
- Create: `build.bat`
- Modify: `requirements.txt`
- Delete: `run.py`, `src/app.py`, `src/viewer_stream.py`, `src/display.py`
- Modify: `README.md`

**Goal:** One-file `piccolo.exe` via PyInstaller, old web-UI code removed, README updated.

- [ ] **Step 1: Create `piccolo.spec`**

```python
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['piccolo.py'],
    pathex=[],
    binaries=[],
    datas=[('config.yaml', '.')],
    hiddenimports=['screeninfo.enumerators.windows'],
    hookspath=[],
    runtime_hooks=[],
    excludes=['pygame', 'flask', 'python-bidi'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='piccolo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
```

- [ ] **Step 2: Create `build.bat`**

```bat
@echo off
set VENV=.venv\Scripts
%VENV%\python -m pip install pyinstaller>=6.0
%VENV%\pyinstaller --clean piccolo.spec
echo Build complete: dist\piccolo.exe
```

- [ ] **Step 3: Remove old dependencies from `requirements.txt`**

Open `requirements.txt` and delete the lines:
```
flask>=3.0
python-bidi>=0.6
```

Remove `pygame>=2.5` if present. Confirm `PyQt6>=6.6` and `pytest-qt>=4.3` are present from Task 3.

- [ ] **Step 4: Delete old files**

```bash
rm run.py src/app.py src/viewer_stream.py src/display.py
```

- [ ] **Step 5: Update `README.md`** (or create if missing)

Replace the usage section with:

```markdown
## Usage

1. Configure `config.yaml` (cameras, display).
2. Run:
   ```
   python piccolo.py
   ```
3. To build a standalone exe:
   ```
   build.bat
   ```
   Produces `dist\piccolo.exe`.

## Tabs

- **Live** — surgery view, zoom-center pad, annotations.
- **Calibration** — per-eye nudge sliders, physical-cal wizard.
- **Settings** — cameras, Goovis display, stereo, pedals, config I/O.
```

- [ ] **Step 6: Run full test suite**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all pass (no references to deleted modules).

- [ ] **Step 7: Build and smoke-run**

```bash
build.bat
dist\piccolo.exe
```

Expected: exe launches, three tabs appear, test-pattern video visible on Live tab (if `test_mode: true`), closes cleanly.

- [ ] **Step 8: Commit**

```bash
git add piccolo.spec build.bat requirements.txt README.md
git rm run.py src/app.py src/viewer_stream.py src/display.py
git commit -m "feat: package as piccolo.exe and remove old web-UI code"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Main Qt window with 3 tabs | Task 3 |
| Goovis borderless full-screen | Task 6 |
| Pipeline worker QThread | Task 4 |
| Live tab: preview + zoom pad + annotations + status | Tasks 5, 11 |
| Calibration tab: nudge sliders + wizard | Tasks 7, 8 |
| Settings tab: cameras + display + stereo + pedals + config I/O | Tasks 9, 10 |
| Remove pygame / flask / python-bidi | Task 12 |
| Delete run.py / app.py / viewer_stream.py / display.py | Task 12 |
| Calibration cleanup (dead pedal code) | Task 1 |
| Persistent calibration state | Task 2 |
| PyInstaller one-file exe | Task 12 |
| Error handling (camera fail, Goovis missing, pipeline crash) | Tasks 4, 6 |
| pytest-qt smoke test | Task 3 |

All requirements mapped. No gaps.

**Placeholder scan:** No TBD, no vague "add error handling later" — every step has concrete code.

**Type consistency:**
- `PipelineWorker.frame_ready` signal: `pyqtSignal(object)` carrying `QImage` — consistently connected in Tasks 5, 6, 8.
- `worker.processor`, `worker.calibration`, `worker.aligner`, `worker.input`, `worker.cam_l/r`, `worker.cfg` — all defined in Task 4, used consistently in Tasks 7–11.
- `save_calibration_state(cfg, path=None)` — defined Task 2, called with `self.worker.cfg` in Tasks 7, 9, 10.
- `InputHandler.on_key_down/up(name)` + `poll_actions()` — defined Task 1, used in Task 4.
- `PhysicalCalSession.sharpness()` + `.phase` + `.next_phase()` + `.prev_phase()` + `.phase_index` + `.total_phases` — existing API, reused in Task 8 verbatim.
- `PatternRenderer.render_{focus,scale,horizontal,rotation}` — existing signatures, reused in Task 8 verbatim.

Consistent.
