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
