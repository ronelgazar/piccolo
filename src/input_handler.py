"""Keyboard (and future GPIO / pedal) input handler.

Translates Pygame key events into high-level *actions* that the main
application loop can consume.  Supports **hold-to-repeat** for zoom
and convergence keys – matching the OBS Lua prototype behaviour.

For the final pedal integration, subclass :class:`InputHandler` and
override :meth:`poll` to read from GPIO / serial.
"""

from __future__ import annotations

import pygame
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


def _key_const(name: str) -> int:
    """Resolve a human-friendly key name (from config) to a Pygame key
    constant.  E.g. ``'EQUALS'`` → ``pygame.K_EQUALS``."""
    attr = f"K_{name.lower()}"
    val = getattr(pygame, attr, None)
    if val is not None:
        return val
    # Try single character
    if len(name) == 1:
        return ord(name.lower())
    raise ValueError(f"Unknown key name: {name!r}")


class InputHandler:
    """Polls Pygame events and exposes the set of currently active actions."""

    def __init__(self, cfg: ControlsCfg):
        self.cfg = cfg
        self._keymap: dict[int, Action] = {}
        self._build_keymap()

        # Keys currently held down → continuous actions (zoom, convergence)
        self._held: Set[Action] = set()

    def _build_keymap(self):
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
        for attr, action in mapping.items():
            key_name = getattr(self.cfg, attr, None)
            if key_name:
                try:
                    self._keymap[_key_const(key_name)] = action
                except ValueError:
                    pass  # skip unmappable keys

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Actions that are held-to-repeat (continuous)
    _CONTINUOUS = {
        Action.ZOOM_IN, Action.ZOOM_OUT,
        Action.CONVERGE_IN, Action.CONVERGE_OUT,
        Action.CALIB_NUDGE_LEFT, Action.CALIB_NUDGE_RIGHT,
    }

    def poll(self) -> Set[Action]:
        """Process all pending Pygame events and return the set of actions
        that should be executed this frame.

        Continuous actions (zoom / convergence) are returned for every frame
        the key is held.  One-shot actions (calibrate, reset, quit) are
        returned only once on key-down.
        """
        one_shot: Set[Action] = set()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                one_shot.add(Action.QUIT)
            elif event.type == pygame.KEYDOWN:
                action = self._keymap.get(event.key)
                if action is None:
                    continue
                if action in self._CONTINUOUS:
                    self._held.add(action)
                else:
                    one_shot.add(action)
            elif event.type == pygame.KEYUP:
                action = self._keymap.get(event.key)
                if action is not None:
                    self._held.discard(action)

        return self._held | one_shot
