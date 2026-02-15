"""Configuration loader for Piccolo.

Reads config.yaml and exposes a flat namespace with sane defaults so the
rest of the codebase never has to worry about missing keys.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Data-classes that mirror config.yaml
# ---------------------------------------------------------------------------

@dataclass
class CameraDeviceCfg:
    index: int = 0
    width: int = 1920
    height: int = 1080


@dataclass
class CamerasCfg:
    backend: str = "opencv"
    left: CameraDeviceCfg = field(default_factory=lambda: CameraDeviceCfg(index=0))
    right: CameraDeviceCfg = field(default_factory=lambda: CameraDeviceCfg(index=1))
    test_mode: bool = False


@dataclass
class ZoomCfg:
    min: float = 1.0
    max: float = 5.0
    step: float = 0.02
    tick_ms: int = 30


@dataclass
class ConvergenceCfg:
    base_offset: int = 0
    step: int = 1
    auto_adjust: bool = True


@dataclass
class AlignmentCfg:
    enabled: bool = True
    interval_sec: float = 2.0
    min_matches: int = 12          # SIFT matches are higher quality → lower threshold
    max_features: int = 1500       # SIFT benefits from more features
    match_ratio: float = 0.75      # Lowe's ratio test threshold
    ransac_thresh: float = 2.0     # tighter for Fundamental Matrix
    max_correction_px: float = 80.0
    max_correction_deg: float = 2.0
    smoothing: float = 0.25        # slightly faster convergence
    detection_scale: float = 0.5


@dataclass
class StereoCfg:
    zoom: ZoomCfg = field(default_factory=ZoomCfg)
    convergence: ConvergenceCfg = field(default_factory=ConvergenceCfg)
    alignment: AlignmentCfg = field(default_factory=AlignmentCfg)


@dataclass
class CalibrationCfg:
    crosshair_color: Tuple[int, int, int] = (0, 255, 0)
    crosshair_thickness: int = 2
    crosshair_size: int = 40
    blink_interval_sec: float = 1.0
    blink_cycles: int = 3


@dataclass
class DisplayCfg:
    width: int = 1920
    height: int = 1080
    fullscreen: bool = True
    fps: int = 60
    monitor: str | int = "auto"   # "auto" detects Goovis, or integer index


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


@dataclass
class StreamCfg:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    jpeg_quality: int = 80


@dataclass
class PiccoloCfg:
    display: DisplayCfg = field(default_factory=DisplayCfg)
    cameras: CamerasCfg = field(default_factory=CamerasCfg)
    stereo: StereoCfg = field(default_factory=StereoCfg)
    calibration: CalibrationCfg = field(default_factory=CalibrationCfg)
    controls: ControlsCfg = field(default_factory=ControlsCfg)
    stream: StreamCfg = field(default_factory=StreamCfg)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _merge(dataclass_obj, raw_dict: dict | None):
    """Recursively overwrite dataclass fields from a plain dict."""
    if raw_dict is None:
        return dataclass_obj
    for key, value in raw_dict.items():
        if not hasattr(dataclass_obj, key):
            continue
        current = getattr(dataclass_obj, key)
        if hasattr(current, "__dataclass_fields__"):
            _merge(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(dataclass_obj, key, tuple(value))
        else:
            setattr(dataclass_obj, key, value)
    return dataclass_obj


def load_config(path: str | None = None) -> PiccoloCfg:
    """Load configuration from *path* (defaults to ``config.yaml`` next to
    the project root).  Missing keys silently fall back to defaults."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    cfg = PiccoloCfg()
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        _merge(cfg, raw)
    return cfg
