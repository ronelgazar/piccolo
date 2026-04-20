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
