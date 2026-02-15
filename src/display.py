"""Pygame-based stereo display targeting a specific monitor (Goovis glasses).

Detects connected monitors and opens a borderless full-screen window on the
Goovis display (or whichever ``display.monitor`` is set to in config.yaml).
The surgeon's goggles see *only* the stereo SBS feed; the computer screen
stays free for the web viewer and other tools.

**Low-latency path:**
- A single ``pygame.Surface`` is pre-allocated once and reused every frame.
- ``pygame.surfarray.blit_array()`` writes directly into the surface
  (no allocation, no intermediate copies).
- BGR→RGB conversion uses a pre-allocated numpy buffer with
  ``cv2.cvtColor()`` (faster than numpy slice reversal for large frames).

The display **must** run on the main thread (platform requirement for
Pygame / SDL).
"""

from __future__ import annotations

import os
import pygame
import cv2
import numpy as np

from .config import DisplayCfg


def _list_monitors() -> list[dict]:
    """Return a list of dicts with keys: name, width, height, x, y, is_primary."""
    try:
        from screeninfo import get_monitors  # type: ignore
        return [
            {"name": m.name, "width": m.width, "height": m.height,
             "x": m.x, "y": m.y, "is_primary": m.is_primary}
            for m in get_monitors()
        ]
    except ImportError:
        # Fallback: only know about the Pygame default display
        pygame.display.init()
        sizes = pygame.display.get_desktop_sizes()
        result = []
        x_offset = 0
        for i, (w, h) in enumerate(sizes):
            result.append({"name": f"Display {i}", "width": w, "height": h,
                           "x": x_offset, "y": 0, "is_primary": i == 0})
            x_offset += w
        return result


def _find_goovis(monitors: list[dict]) -> dict | None:
    """Auto-detect the Goovis by looking for 'GOOVIS' or 'NED' in the name,
    or a non-primary 1920x1080 display."""
    for m in monitors:
        name_upper = m["name"].upper()
        if "GOOVIS" in name_upper or "NED" in name_upper:
            return m
    # Fallback: first non-primary display that is 1920×1080
    for m in monitors:
        if not m["is_primary"] and m["width"] == 1920 and m["height"] == 1080:
            return m
    return None


class StereoDisplay:
    """Manages the full-screen Pygame window on the Goovis glasses."""

    def __init__(self, cfg: DisplayCfg):
        self.cfg = cfg
        self.width = cfg.width
        self.height = cfg.height
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self._target_monitor: dict | None = None
        # Pre-allocated buffers (created in open())
        self._surface: pygame.Surface | None = None
        self._rgb_buf: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self):
        # Enumerate monitors *before* fully initialising Pygame
        monitors = _list_monitors()
        print(f"[display] Found {len(monitors)} monitor(s):")
        for i, m in enumerate(monitors):
            tag = " [PRIMARY]" if m["is_primary"] else ""
            print(f"  #{i}: {m['name']}  {m['width']}x{m['height']}  "
                  f"at ({m['x']},{m['y']}){tag}")

        # Pick which monitor to use
        target = None
        monitor_idx = getattr(self.cfg, "monitor", "auto")

        if monitor_idx == "auto":
            target = _find_goovis(monitors)
            if target:
                print(f"[display] Auto-detected Goovis: {target['name']}")
            else:
                print("[display] Goovis not detected – using primary display.")
        elif isinstance(monitor_idx, int) and 0 <= monitor_idx < len(monitors):
            target = monitors[monitor_idx]
            print(f"[display] Using monitor #{monitor_idx}: {target['name']}")

        self._target_monitor = target

        # Position the SDL window on the target monitor
        if target and self.cfg.fullscreen:
            # SDL env vars must be set BEFORE pygame.display.set_mode()
            os.environ["SDL_VIDEO_WINDOW_POS"] = f"{target['x']},{target['y']}"
            self.width = target["width"]
            self.height = target["height"]

        pygame.init()

        if self.cfg.fullscreen and target:
            # NOFRAME borderless window at the exact monitor position + size
            # This avoids Pygame's FULLSCREEN which always targets display 0.
            flags = pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF
            self.screen = pygame.display.set_mode(
                (self.width, self.height), flags
            )
        elif self.cfg.fullscreen:
            flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
            self.screen = pygame.display.set_mode(
                (self.width, self.height), flags
            )
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption("Piccolo – Stereo Display")
        self.clock = pygame.time.Clock()

        # Pre-allocate the rendering surface and RGB buffer
        self._surface = pygame.Surface((self.width, self.height))
        self._rgb_buf = np.empty((self.height, self.width, 3), dtype=np.uint8)

        actual_w, actual_h = self.screen.get_size()
        print(f"[display] Window opened: {actual_w}x{actual_h}  "
              f"fullscreen={self.cfg.fullscreen}")

    def close(self):
        # Clean up the env var
        os.environ.pop("SDL_VIDEO_WINDOW_POS", None)
        pygame.quit()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def show(self, frame_bgr: np.ndarray):
        """Display a BGR numpy frame (expected shape ``(height, width, 3)``).

        Hot-path: avoids the expensive transpose + contiguous copy by
        using ``pygame.image.frombuffer`` which reads row-major RGB
        bytes directly – no (W,H,3) transposition needed.
        """
        if self.screen is None:
            return
        # Resize if frame doesn't exactly match display
        fh, fw = frame_bgr.shape[:2]
        if fw != self.width or fh != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

        # BGR → RGB into pre-allocated buffer (avoids allocation)
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB, dst=self._rgb_buf)

        # frombuffer reads row-major (H,W,3) bytes directly –
        # no transpose, no contiguous copy, no new Surface alloc.
        surf = pygame.image.frombuffer(
            bytes(self._rgb_buf.data), (self.width, self.height), "RGB"
        )
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    def tick(self) -> float:
        """Limit frame-rate and return the measured delta-time in seconds."""
        dt = self.clock.tick(self.cfg.fps) if self.clock else 16
        return dt / 1000.0
