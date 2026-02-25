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
from .stereo_align import StereoAligner


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
    PEDAL_MODE_TOGGLE = auto()
    PEDAL_MODE_CENTER_X = auto()
    PEDAL_MODE_CENTER_Y = auto()
    PEDAL_ZOOM_IN = auto()
    PEDAL_ZOOM_OUT = auto()
    PEDAL_CENTER_LEFT = auto()
    PEDAL_CENTER_RIGHT = auto()
    PEDAL_CENTER_UP = auto()
    PEDAL_CENTER_DOWN = auto()


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

    def __init__(self, aligner: StereoAligner, cfg: ControlsCfg):
        self.aligner = aligner
        self.cfg = cfg
        self._keymap: dict[int, Action] = {}
        self._build_keymap()
        self._held: Set[Action] = set()
        # Pedal mode: 0=zoom, 1=center-x, 2=center-y
        self.pedal_mode = 0
        # Track held numpad keys for pedal combos
        self._npad_held = set()

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
            "pedal_mode_toggle": Action.PEDAL_MODE_TOGGLE,
            "pedal_mode_center_x": Action.PEDAL_MODE_CENTER_X,
            "pedal_mode_center_y": Action.PEDAL_MODE_CENTER_Y,
            "pedal_zoom_in": Action.PEDAL_ZOOM_IN,
            "pedal_zoom_out": Action.PEDAL_ZOOM_OUT,
            "pedal_center_left": Action.PEDAL_CENTER_LEFT,
            "pedal_center_right": Action.PEDAL_CENTER_RIGHT,
            "pedal_center_up": Action.PEDAL_CENTER_UP,
            "pedal_center_down": Action.PEDAL_CENTER_DOWN,
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

    def _handle_pedal_logic(self, event, one_shot):
        """Handle pedal-related actions based on the current pedal mode."""
        if event.key == pygame.K_F1:  # Left pedal
            self.pedal_mode = (self.pedal_mode + 1) % 3
            if self.pedal_mode == 0:
                one_shot.add(Action.PEDAL_MODE_TOGGLE)
            elif self.pedal_mode == 1:
                one_shot.add(Action.PEDAL_MODE_CENTER_X)
            elif self.pedal_mode == 2:
                one_shot.add(Action.PEDAL_MODE_CENTER_Y)
        elif event.key == pygame.K_F2:  # Middle pedal
            if self.pedal_mode == 0:
                one_shot.add(Action.PEDAL_ZOOM_IN)
            elif self.pedal_mode == 1:
                one_shot.add(Action.PEDAL_CENTER_LEFT)
            elif self.pedal_mode == 2:
                one_shot.add(Action.PEDAL_CENTER_UP)
        elif event.key == pygame.K_F3:  # Right pedal
            if self.pedal_mode == 0:
                one_shot.add(Action.PEDAL_ZOOM_OUT)
            elif self.pedal_mode == 1:
                one_shot.add(Action.PEDAL_CENTER_RIGHT)
            elif self.pedal_mode == 2:
                one_shot.add(Action.PEDAL_CENTER_DOWN)

    def _handle_key_event(self, event, one_shot):
        """Handle key press and release events."""
        if event.type == pygame.KEYDOWN:
            self._handle_pedal_logic(event, one_shot)
            if event.key in (pygame.K_KP4, pygame.K_KP5, pygame.K_KP6):
                self._npad_held.add(event.key)
            action = self._keymap.get(event.key)
            if action is not None:
                if action in self._CONTINUOUS:
                    self._held.add(action)
                else:
                    one_shot.add(action)
        elif event.type == pygame.KEYUP:
            if event.key in (pygame.K_KP4, pygame.K_KP5, pygame.K_KP6):
                self._npad_held.discard(event.key)
            action = self._keymap.get(event.key)
            if action is not None:
                self._held.discard(action)

    def _determine_combo_action(self, pressed):
        """Determine combo actions based on numpad keys."""
        kp4, kp5, kp6 = pygame.K_KP4, pygame.K_KP5, pygame.K_KP6
        combo_keys = [kp4, kp5, kp6]
        pressed_numpad = [k for k in combo_keys if pressed[k]]

        if len(pressed_numpad) < 2:
            return None

        first, second = self._get_first_and_second_keys(self._npad_held, pressed_numpad)
        return self._map_combo_to_action(first, second)

    def _get_first_and_second_keys(self, npad_held, pressed_numpad):
        """Get the first and second keys from the pressed numpad keys."""
        npad_held_order = list(npad_held)
        if not npad_held_order:
            return None, None

        first = npad_held_order[0]
        second = next((k for k in pressed_numpad if k != first), None)
        return first, second

    def _map_combo_to_action(self, first, second):
        """Map the first and second keys to a combo action (pedal logic)."""
        kp4, kp5, kp6 = pygame.K_KP4, pygame.K_KP5, pygame.K_KP6

        # Middle pedal held
        if first == kp5:
            if second == kp4:
                return Action.ZOOM_IN
            elif second == kp6:
                return Action.ZOOM_OUT
        # Right pedal held
        elif first == kp6:
            if second == kp5:
                return Action.CALIB_NUDGE_RIGHT
            elif second == kp4:
                return Action.CALIB_NUDGE_LEFT
        # Left pedal held
        elif first == kp4:
            if second == kp5:
                return Action.PEDAL_CENTER_UP
            elif second == kp6:
                return Action.PEDAL_CENTER_DOWN

        return None

    def poll(self) -> Set[Action]:
        """Process all pending Pygame events and return the set of actions."""
        one_shot: Set[Action] = set()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                one_shot.add(Action.QUIT)
            else:
                self._handle_key_event(event, one_shot)

        pressed = pygame.key.get_pressed()
        combo_action = self._determine_combo_action(pressed)

        # Remove all combo actions from self._held
        for act in [Action.ZOOM_OUT, Action.ZOOM_IN, Action.CALIB_NUDGE_LEFT, Action.CALIB_NUDGE_RIGHT, Action.PEDAL_CENTER_DOWN, Action.PEDAL_CENTER_UP]:
            self._held.discard(act)

        # Return combo actions if present, otherwise return held and one-shot actions
        if combo_action:
            return {combo_action} | one_shot
        return self._held | one_shot
