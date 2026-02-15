"""Camera capture abstraction.

Provides a threaded camera reader that always exposes the *latest* frame
with minimal latency.  Three back-ends are supported:

* ``opencv``    – works everywhere (USB cameras, capture cards, etc.)
* ``dshow``     – Windows DirectShow; lower latency for USB cameras like ELP
* ``picamera2`` – native Raspberry Pi CSI cameras via libcamera
"""

from __future__ import annotations

import platform
import threading
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np


class CameraCapture:
    """Threaded camera capture that continuously grabs frames in the
    background and exposes the most recent one via :meth:`read`."""

    def __init__(
        self,
        index: int = 0,
        width: int = 1920,
        height: int = 1080,
        backend: str = "opencv",
        name: str = "camera",
    ):
        self.index = index
        self.width = width
        self.height = height
        self.backend = backend
        self.name = name

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "CameraCapture":
        """Open the camera and start the background grab thread."""
        if self.backend == "picamera2":
            self._open_picamera2()
        else:
            self._open_opencv()
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True, name=self.name)
        self._thread.start()
        # Warm up: wait for the first real frame (up to 2 s) so the
        # main loop doesn't start with stale / empty data.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    break
            time.sleep(0.02)
        return self

    def read(self) -> Optional[np.ndarray]:
        """Return the latest frame (BGR, np.ndarray) or *None*."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def read_no_copy(self) -> Optional[np.ndarray]:
        """Return latest frame **without** copying – faster, but the caller
        must NOT mutate the array."""
        with self._lock:
            return self._frame

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            if self.backend == "picamera2":
                self._cap.stop()
            else:
                self._cap.release()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_opencv(self):
        # On Windows, use DirectShow for USB cameras (lower latency)
        use_dshow = self.backend == "dshow" or (
            self.backend == "opencv" and platform.system() == "Windows"
        )
        if use_dshow:
            self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.index}.  "
                f"Run  python -m src.camera --list  to discover devices."
            )

        # Set resolution BEFORE fourcc – on DirectShow, setting
        # width/height after MJPG resets the fourcc back to YUY2.
        # FPS must also be set BEFORE fourcc for the same reason.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimal buffering

        # Try to negotiate MJPEG – ELP cameras decode MJPG in hardware
        # giving ~5× higher FPS at 1080p vs YUY2 (15fps vs 3fps).
        # This MUST be the last property set – any set after this
        # on DirectShow will reset the fourcc back to YUY2.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))

        # Verify what the driver actually gave us
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == 0 or actual_h == 0:
            # MJPG mode not supported – reopen without fourcc override
            self._cap.release()
            if use_dshow:
                self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            else:
                self._cap = cv2.VideoCapture(self.index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(
            chr(c) if 32 <= c < 127 else "?"
            for c in ((fourcc_int >> 8 * i) & 0xFF for i in range(4))
        )
        print(
            f"[camera] {self.name}: opened index={self.index}  "
            f"{actual_w}x{actual_h} @ {actual_fps:.0f}fps  fourcc={fourcc_str}"
        )

    def _open_picamera2(self):
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError:
            raise ImportError("picamera2 is required for RPi CSI cameras.  "
                              "Install with: pip install picamera2")
        self._cap = Picamera2(self.index)
        config = self._cap.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"},
            buffer_count=2,
        )
        self._cap.configure(config)
        self._cap.start()

    def _grab_loop(self):
        """Continuously grab the latest frame from the camera.

        Uses ``cap.read()`` because DirectShow's ``grab()`` +
        ``retrieve()`` can be unreliable from background threads.
        With ``CAP_PROP_BUFFERSIZE`` set to 1 the driver keeps at most
        one frame queued so the retrieved frame is close to real-time.
        """
        if self.backend == "picamera2":
            self._grab_loop_picamera()
            return

        cap = self._cap
        while self._running:
            ret, frame = cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
            # No sleep – read() already blocks until a frame arrives,
            # so this loop naturally runs at camera FPS.

    def _grab_loop_picamera(self):
        """Grab loop variant for picamera2."""
        while self._running:
            frame = self._cap.capture_array("main")  # type: ignore
            if frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.001)


# ---------------------------------------------------------------------------
# Camera discovery utility
# ---------------------------------------------------------------------------

def list_cameras(max_index: int = 10) -> List[Tuple[int, str]]:
    """Probe camera indices 0 … *max_index* and return a list of
    ``(index, description)`` tuples for every device that opens
    successfully.

    Useful for finding which indices the two ELP cameras are on::

        python -c "from src.camera import list_cameras; list_cameras()"
    """
    found: List[Tuple[int, str]] = []
    for idx in range(max_index):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
            desc = f"{w}x{h} fourcc={fourcc_str}"
            found.append((idx, desc))
            print(f"  [camera] index {idx}: {desc}")
            cap.release()
        else:
            cap.release()
    if not found:
        print("  [camera] No cameras found.")
    return found


# ---------------------------------------------------------------------------
# Test-pattern generator (no cameras needed)
# ---------------------------------------------------------------------------

class TestPatternCamera(CameraCapture):
    """Generates a synthetic stereo-friendly test pattern so the display and
    zoom logic can be tested without physical cameras."""

    def __init__(self, width: int = 1920, height: int = 1080, side: str = "left", name: str = "test"):
        super().__init__(index=-1, width=width, height=height, backend="test", name=name)
        self.side = side
        self._base_frame = self._generate_pattern()

    def start(self) -> "TestPatternCamera":
        self._running = True
        self._frame = self._base_frame.copy()
        # No background thread needed – the frame is static.
        return self

    def read(self) -> Optional[np.ndarray]:
        return self._base_frame.copy()

    def read_no_copy(self) -> Optional[np.ndarray]:
        return self._base_frame

    def stop(self):
        self._running = False

    def _generate_pattern(self) -> np.ndarray:
        """Create a stereo test pattern with real depth cues.

        Objects at different disparity offsets will appear at different
        depths when viewed through the Goovis auto-3D:
        - Zero disparity  → appears at screen depth
        - Positive shift (L shifts right, R shifts left) → appears in front
        - Negative shift  → appears behind the screen
        """
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cx, cy = self.width // 2, self.height // 2

        # --- Background checkerboard (zero disparity → screen depth) ---
        block = 60
        for y in range(0, self.height, block):
            for x in range(0, self.width, block):
                if ((x // block) + (y // block)) % 2 == 0:
                    img[y:y + block, x:x + block] = (40, 40, 40)
                else:
                    img[y:y + block, x:x + block] = (80, 80, 80)

        # Disparity sign: left eye shifts right (+), right eye shifts left (-)
        sign = 1 if self.side == "left" else -1

        # --- Ring at screen depth (0 px disparity) ---
        cv2.circle(img, (cx, cy), 250, (80, 80, 80), 2)
        cv2.putText(img, "screen", (cx - 45, cy + 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

        # --- Object BEHIND the screen (negative disparity, -20 px) ---
        d_back = -20
        ox_back = cx + sign * d_back
        cv2.circle(img, (ox_back, cy - 150), 60, (255, 180, 0), 3)
        cv2.putText(img, "FAR", (ox_back - 22, cy - 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

        # --- Object AT screen depth (0 px disparity) ---
        cv2.circle(img, (cx, cy), 80, (0, 255, 120), 3)
        cv2.putText(img, "MID", (cx - 25, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)

        # --- Object IN FRONT of the screen (positive disparity, +25 px) ---
        d_front = 25
        ox_front = cx + sign * d_front
        cv2.circle(img, (ox_front, cy + 160), 50, (0, 100, 255), 4)
        cv2.putText(img, "NEAR", (ox_front - 30, cy + 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        # --- Crosshair at centre (zero disparity) ---
        cv2.line(img, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 1)
        cv2.line(img, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 1)

        # --- Eye label ---
        color = (100, 200, 255) if self.side == "left" else (255, 200, 100)
        label = "L" if self.side == "left" else "R"
        cv2.putText(img, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        return img


# ---------------------------------------------------------------------------
# CLI: run  python -m src.camera  to discover cameras & preview
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Camera discovery & preview")
    parser.add_argument("--list", action="store_true", help="List available cameras and exit")
    parser.add_argument("--preview", type=int, nargs="*", default=None,
                        help="Preview one or more camera indices (e.g. --preview 0 1)")
    args = parser.parse_args()

    if args.list or args.preview is None:
        print("Scanning for cameras…")
        cams = list_cameras()
        if args.preview is None and cams:
            print(f"\nTo preview, run:  python -m src.camera --preview {cams[0][0]}")

    if args.preview is not None:
        indices = args.preview if args.preview else [0]
        caps = []
        opened_indices = []
        for idx in indices:
            try:
                c = CameraCapture(index=idx, width=1920, height=1080, name=f"cam-{idx}")
                c.start()
                caps.append(c)
                opened_indices.append(idx)
            except RuntimeError as e:
                print(f"  Skipping index {idx}: {e}")
        if not caps:
            print("No cameras could be opened.")
        else:
            print(f"Previewing camera(s) {opened_indices}.  Press Q to quit.")
            while True:
                for i, c in enumerate(caps):
                    frame = c.read()
                    if frame is not None:
                        # Resize for preview
                        small = cv2.resize(frame, (640, 360))
                        cv2.imshow(f"Camera {opened_indices[i]}", small)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            for c in caps:
                c.stop()
            cv2.destroyAllWindows()
