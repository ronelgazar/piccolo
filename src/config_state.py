"""Persist selected config sections to `config.yaml` on change."""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import yaml

from .config import PiccoloCfg, default_config_path


def save_calibration_state(cfg: PiccoloCfg, path: str | None = None) -> None:
    """Write `cfg.calibration_state` into `config.yaml` without clobbering
    other top-level keys.  Creates the file if missing.
    """
    if path is None:
        path = _default_config_path()
    raw: dict[str, Any] = _cfg_to_dict(cfg)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    raw["calibration_state"] = asdict(cfg.calibration_state)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)


def save_camera_settings(cfg: PiccoloCfg, path: str | None = None) -> None:
    """Persist current camera width, height, and FPS without clobbering YAML."""
    if path is None:
        path = _default_config_path()
    raw: dict[str, Any] = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    cams = raw.setdefault("cameras", {})
    for side, dev in (("left", cfg.cameras.left), ("right", cfg.cameras.right)):
        section = cams.setdefault(side, {})
        section["width"] = dev.width
        section["height"] = dev.height
        section["fps"] = dev.fps

    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)


def _default_config_path() -> str:
    return default_config_path()


def _cfg_to_dict(cfg: PiccoloCfg) -> dict[str, Any]:
    return {
        "display": asdict(cfg.display),
        "cameras": {
            "backend": cfg.cameras.backend,
            "left": asdict(cfg.cameras.left),
            "right": asdict(cfg.cameras.right),
            "test_mode": cfg.cameras.test_mode,
        },
        "stereo": {
            "zoom": asdict(cfg.stereo.zoom),
            "convergence": asdict(cfg.stereo.convergence),
            "alignment": asdict(cfg.stereo.alignment),
            "aspect_mode": cfg.stereo.aspect_mode,
        },
        "calibration": asdict(cfg.calibration),
        "calibration_state": asdict(cfg.calibration_state),
        "controls": asdict(cfg.controls),
        "stream": asdict(cfg.stream),
    }
