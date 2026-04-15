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
        # Pedal toggle mode: None | 'a' (zoom) | 'b' (side) | 'c' (up/down)
        self.pedal_mode: str | None = None
        # Maps adjust-pedal key → action, for continuous long-press
        self._pedal_adjust_held: dict[str, Action] = {}
        # Track held numpad keys for pedal combos (legacy)
        self._npad_held = set()
    def _pedal_key(self, event):
        # Map pedal keys: a=left, b=middle, c=right
        if not hasattr(event, "key"):
            return None
        if event.key == ord('a'):
            return 'a'
        elif event.key == ord('b'):
            return 'b'
        elif event.key == ord('c'):
            return 'c'
        return None

    # Pedal mode → (adjust-pedal-b action, adjust-pedal-c action or adjust-pedal-a action)
    # 'a' = zoom:      b → ZOOM_IN,           c → ZOOM_OUT
    # 'b' = side:      a → CENTER_LEFT,        c → CENTER_RIGHT
    # 'c' = up/down:   a → CENTER_UP,          b → CENTER_DOWN
    _PEDAL_ADJUST_MAP: dict[str, dict[str, Action]] = {
        'a': {'b': Action.ZOOM_IN,            'c': Action.ZOOM_OUT},
        'b': {'a': Action.PEDAL_CENTER_LEFT,  'c': Action.PEDAL_CENTER_RIGHT},
        'c': {'a': Action.PEDAL_CENTER_UP,    'b': Action.PEDAL_CENTER_DOWN},
    }

    def _pedal_adjust_action(self, pedal: str) -> Action | None:
        """Return the action for an adjust pedal given the current mode."""
        if self.pedal_mode is None:
            return None
        return self._PEDAL_ADJUST_MAP.get(self.pedal_mode, {}).get(pedal)

    def get_pedal_mode(self):
        return self.pedal_mode

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
            key_names = getattr(self.cfg, attr, None)
            if key_names:
                if not isinstance(key_names, (list, tuple)):
                    key_names = [key_names]
                for key_name in key_names:
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
        """Handle key press and release events, including pedal toggle logic."""
        pedal = self._pedal_key(event)
        if event.type == pygame.KEYDOWN:
            if pedal:
                if self.pedal_mode == pedal:
                    # Same pedal pressed again → toggle mode off, stop any adjust
                    self._clear_pedal_adjust()
                    self.pedal_mode = None
                    print(f"[pedal] mode OFF")
                elif self.pedal_mode is None:
                    # No mode active → toggle this pedal's mode on
                    self.pedal_mode = pedal
                    mode_names = {'a': 'ZOOM', 'b': 'SIDE', 'c': 'UP/DOWN'}
                    print(f"[pedal] mode → {mode_names.get(pedal, pedal)}")
                else:
                    # Mode active, different pedal → adjust (immediate + continuous)
                    action = self._pedal_adjust_action(pedal)
                    if action is not None:
                        one_shot.add(action)
                        self._pedal_adjust_held[pedal] = action

            # Legacy numpad support
            if event.key in (pygame.K_KP4, pygame.K_KP5, pygame.K_KP6):
                self._npad_held.add(event.key)
            action = self._keymap.get(event.key)
            if action is not None:
                if action in self._CONTINUOUS:
                    self._held.add(action)
                else:
                    one_shot.add(action)

        elif event.type == pygame.KEYUP:
            if pedal and pedal in self._pedal_adjust_held:
                # Adjust pedal released → stop continuous action
                del self._pedal_adjust_held[pedal]
            if event.key in (pygame.K_KP4, pygame.K_KP5, pygame.K_KP6):
                self._npad_held.discard(event.key)
            action = self._keymap.get(event.key)
            if action is not None:
                self._held.discard(action)

    def _clear_pedal_adjust(self):
        """Stop all continuous pedal adjust actions."""
        self._pedal_adjust_held.clear()

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

        # Continuous pedal adjust actions (long-press held pedals)
        pedal_continuous = set(self._pedal_adjust_held.values())

        if combo_action:
            return {combo_action} | one_shot | pedal_continuous
        return self._held | one_shot | pedal_continuous
