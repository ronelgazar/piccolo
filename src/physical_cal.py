"""Physical camera calibration — pattern rendering and session logic."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .physical_grid_calibration import GridDetection, detect_grid, estimate_square_px

PHASES: tuple[str, ...] = ("brightness", "focus", "scale", "horizontal", "rotation")

BRIGHTNESS_DELTA_OK = 10.0
SATURATION_OK_PCT = 2.0
ZOOM_RATIO_TOL = 0.02
VERTICAL_DELTA_OK_PX = 5.0
ROTATION_DELTA_OK_DEG = 0.5
FOCUS_PEAK_RATIO_OK = 0.90


@dataclass(frozen=True)
class GridEyeMetrics:
    detected: bool
    partial: bool = False
    brightness: float | None = None
    saturation_pct: float | None = None
    sharpness: float | None = None
    square_px: float | None = None
    center: tuple[float, float] | None = None
    row_angle_deg: float | None = None


@dataclass(frozen=True)
class GridPairMetrics:
    left: GridEyeMetrics
    right: GridEyeMetrics

    @property
    def zoom_ratio(self) -> float | None:
        if not self.left.square_px or not self.right.square_px:
            return None
        return self.right.square_px / self.left.square_px

    @property
    def zoom_error_pct(self) -> float | None:
        ratio = self.zoom_ratio
        if ratio is None:
            return None
        return (ratio - 1.0) * 100.0

    @property
    def brightness_delta(self) -> float | None:
        if self.left.brightness is None or self.right.brightness is None:
            return None
        return self.right.brightness - self.left.brightness

    @property
    def vertical_delta_px(self) -> float | None:
        if self.left.center is None or self.right.center is None:
            return None
        return self.right.center[1] - self.left.center[1]

    @property
    def rotation_delta_deg(self) -> float | None:
        if self.left.row_angle_deg is None or self.right.row_angle_deg is None:
            return None
        return self.right.row_angle_deg - self.left.row_angle_deg

    def brightness_ok(self) -> bool:
        delta = self.brightness_delta
        if delta is None:
            return False
        max_sat = max(self.left.saturation_pct or 100.0, self.right.saturation_pct or 100.0)
        return abs(delta) <= BRIGHTNESS_DELTA_OK and max_sat <= SATURATION_OK_PCT

    def zoom_ok(self) -> bool:
        ratio = self.zoom_ratio
        return ratio is not None and abs(ratio - 1.0) <= ZOOM_RATIO_TOL

    def zoom_status(self) -> str:
        ratio = self.zoom_ratio
        if ratio is None:
            return "NO GRID SCALE"
        if self.zoom_ok():
            return "MECHANICAL ZOOM MATCHED"
        if ratio > 1.0:
            return "RIGHT ZOOM LARGER"
        return "LEFT ZOOM LARGER"

    def vertical_ok(self) -> bool:
        delta = self.vertical_delta_px
        return delta is not None and abs(delta) <= VERTICAL_DELTA_OK_PX

    def rotation_ok(self) -> bool:
        delta = self.rotation_delta_deg
        return delta is not None and abs(delta) <= ROTATION_DELTA_OK_DEG


class PhysicalCalSession:
    """Phase state machine + per-eye sharpness metric."""

    def __init__(self):
        self._idx = 0
        self._best_focus_l = 0.0
        self._best_focus_r = 0.0

    @property
    def phase(self) -> str:
        return PHASES[self._idx]

    @property
    def phase_index(self) -> int:
        return self._idx

    @property
    def total_phases(self) -> int:
        return len(PHASES)

    def next_phase(self) -> bool:
        """Advance to next phase. Returns True when all phases are done."""
        if self._idx < len(PHASES) - 1:
            self._idx += 1
            return False
        return True

    def prev_phase(self) -> None:
        if self._idx > 0:
            self._idx -= 1

    def update_focus_peak(self, sharp_l: float, sharp_r: float) -> tuple[bool, float, float]:
        self._best_focus_l = max(self._best_focus_l, sharp_l)
        self._best_focus_r = max(self._best_focus_r, sharp_r)
        ok_l = self._best_focus_l > 0 and sharp_l >= self._best_focus_l * FOCUS_PEAK_RATIO_OK
        ok_r = self._best_focus_r > 0 and sharp_r >= self._best_focus_r * FOCUS_PEAK_RATIO_OK
        return ok_l and ok_r, self._best_focus_l, self._best_focus_r

    @staticmethod
    def sharpness(img: np.ndarray) -> float:
        """Laplacian variance of central 200×200 px ROI. Higher = sharper."""
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        roi = img[max(0, cy - 100):cy + 100, max(0, cx - 100):cx + 100]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else roi
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def grid_metrics(
        img: np.ndarray,
        inner_cols: int = 9,
        inner_rows: int = 6,
        max_detect_dim: int = 420,
    ) -> GridEyeMetrics:
        detection = _detect_grid_scaled(img, inner_cols, inner_rows, max_detect_dim)
        if detection is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            square_px = _estimate_partial_grid_square_px(gray)
            brightness = float(gray.mean())
            saturation_pct = float(((gray <= 5) | (gray >= 250)).mean() * 100.0)
            sharpness = PhysicalCalSession.sharpness(img)
            return GridEyeMetrics(
                detected=False,
                partial=square_px is not None,
                brightness=brightness,
                saturation_pct=saturation_pct,
                sharpness=sharpness,
                square_px=square_px,
            )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        x, y, w, h = _grid_bounds(detection, gray.shape)
        roi = gray[y:y + h, x:x + w]
        brightness = float(roi.mean())
        saturation_pct = float(((roi <= 5) | (roi >= 250)).mean() * 100.0)
        sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        square_px = estimate_square_px(detection, inner_cols=inner_cols, inner_rows=inner_rows)
        row_angle_deg = _grid_row_angle_deg(detection, inner_cols=inner_cols, inner_rows=inner_rows)
        return GridEyeMetrics(
            detected=True,
            partial=False,
            brightness=brightness,
            saturation_pct=saturation_pct,
            sharpness=sharpness,
            square_px=square_px,
            center=detection.center,
            row_angle_deg=row_angle_deg,
        )

    @staticmethod
    def grid_pair_metrics(
        left: np.ndarray,
        right: np.ndarray,
        inner_cols: int = 9,
        inner_rows: int = 6,
    ) -> GridPairMetrics:
        return GridPairMetrics(
            left=PhysicalCalSession.grid_metrics(left, inner_cols, inner_rows),
            right=PhysicalCalSession.grid_metrics(right, inner_cols, inner_rows),
        )


class PatternRenderer:
    """Draws phase-specific test patterns onto BGR images in-place."""

    def render_brightness(self, img: np.ndarray, metrics: GridEyeMetrics, ok: bool | None = None) -> None:
        h, w = img.shape[:2]
        color = _status_color(ok) if ok is not None else ((0, 255, 255) if metrics.detected else (0, 0, 255))
        cv2.putText(img, "BRIGHTNESS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        if metrics.detected:
            text = f"mean={metrics.brightness:.0f} sat={metrics.saturation_pct:.1f}%"
        else:
            text = "grid not detected"
        cv2.putText(img, text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
        if ok is not None:
            cv2.putText(img, _status_text(ok), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        cv2.putText(img, "target: similar L/R, no clipped whites/blacks", (10, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def render_focus(self, img: np.ndarray, sharpness: float) -> None:
        """Siemens star (18-line / 36-sector radial wheel) + sharpness score."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        radius = min(h, w) // 4
        for i in range(36):
            angle = i * math.pi / 36
            x1 = int(cx + radius * math.cos(angle))
            y1 = int(cy + radius * math.sin(angle))
            x2 = int(cx - radius * math.cos(angle))
            y2 = int(cy - radius * math.sin(angle))
            color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
            cv2.line(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, "FOCUS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"sharp: {sharpness:.0f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

    def render_grid_focus(self, img: np.ndarray, metrics: GridEyeMetrics, ok: bool | None = None) -> None:
        self.render_focus(img, metrics.sharpness if metrics.sharpness is not None else 0.0)
        status = "GRID OK" if metrics.detected else "partial grid" if metrics.partial else "grid not detected"
        cv2.putText(img, status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if metrics.detected or metrics.partial else (0, 0, 255), 1, cv2.LINE_AA)
        if ok is not None:
            cv2.putText(img, _status_text(ok), (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, _status_color(ok), 2, cv2.LINE_AA)

    def render_scale(self, img: np.ndarray, metrics: GridEyeMetrics | None = None, ok: bool | None = None) -> None:
        """Concentric circles at 10 %, 25 %, 50 % of frame height."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        color = (200, 200, 0)
        for pct in [10, 25, 50]:
            r = int(h * pct / 100)
            cv2.circle(img, (cx, cy), r, color, 1, cv2.LINE_AA)
            cv2.putText(img, f"{pct}%", (cx + r + 4, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(img, "SCALE  (match circle sizes visually)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        if metrics is not None:
            txt = f"square={metrics.square_px:.1f}px" if metrics.square_px is not None else "grid not detected"
            cv2.putText(img, txt, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
        if ok is not None:
            cv2.putText(img, _status_text(ok), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, _status_color(ok), 2, cv2.LINE_AA)
        cv2.putText(img, "adjust mechanical zoom until L/R square sizes match", (10, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def render_horizontal(self, img: np.ndarray, dy: float | None, ok: bool | None = None) -> None:
        """Horizontal grid lines + centre crosshair + vertical-offset readout."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        color = (0, 200, 255)
        for pct in [25, 50, 75]:
            y = int(h * pct / 100)
            cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx - 40, cy), (cx + 40, cy), (0, 255, 0), 2)
        cv2.line(img, (cx, cy - 40), (cx, cy + 40), (0, 255, 0), 2)
        cv2.putText(img, "HORIZONTAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        dy_txt = f"Vertical offset: {dy:+.1f} px" if dy is not None else "Vertical offset: --"
        cv2.putText(img, dy_txt, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
        if ok is not None:
            cv2.putText(img, _status_text(ok), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, _status_color(ok), 2, cv2.LINE_AA)

    def render_rotation(self, img: np.ndarray, dtheta_deg: float | None, ok: bool | None = None) -> None:
        """Diagonal reference lines + spirit-level arc + rotation readout."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        line_color = (255, 100, 0)
        cv2.line(img, (0, 0), (w, h), line_color, 1, cv2.LINE_AA)
        cv2.line(img, (w, 0), (0, h), line_color, 1, cv2.LINE_AA)
        arc_r = min(w, h) // 6
        arc_cx, arc_cy = cx, cy + arc_r + 20
        cv2.ellipse(img, (arc_cx, arc_cy), (arc_r, arc_r), 0, 180, 360,
                    (180, 180, 180), 1, cv2.LINE_AA)
        if dtheta_deg is not None:
            clamped = max(-5.0, min(5.0, dtheta_deg))
            dot_angle_rad = math.radians(270 + clamped * 18)
            dot_x = int(arc_cx + arc_r * math.cos(dot_angle_rad))
            dot_y = int(arc_cy + arc_r * math.sin(dot_angle_rad))
            cv2.circle(img, (dot_x, dot_y), 8, (0, 255, 0), -1)
        cv2.putText(img, "ROTATION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2, cv2.LINE_AA)
        rot_txt = (f"Rotation: {dtheta_deg:+.2f} deg"
                   if dtheta_deg is not None else "Rotation: --")
        cv2.putText(img, rot_txt, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_color, 1, cv2.LINE_AA)
        if ok is not None:
            cv2.putText(img, _status_text(ok), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, _status_color(ok), 2, cv2.LINE_AA)

    def render_pair_hud(
        self,
        img: np.ndarray,
        phase: str,
        metrics: GridPairMetrics,
        side: str,
    ) -> None:
        h, _ = img.shape[:2]
        full_detected = metrics.left.detected and metrics.right.detected
        scale_detected = metrics.left.square_px is not None and metrics.right.square_px is not None
        color = (0, 220, 0) if full_detected else (0, 200, 255) if scale_detected else (0, 0, 255)
        cv2.putText(img, f"PHYSICAL CAL: {phase.upper()}  eye={side.upper()}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)

        if not scale_detected:
            cv2.putText(img, "grid not detected in both eyes", (10, 178),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
            cv2.putText(img, "show at least 3-4 visible square edges", (10, 206),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
            return

        mode = "full grid" if full_detected else "partial grid scale"
        cv2.putText(img, mode, (10, 178),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)

        left_size = metrics.left.square_px
        right_size = metrics.right.square_px
        ratio = metrics.zoom_ratio
        size_txt = (
            f"grid square L={left_size:.1f}px R={right_size:.1f}px ratio={ratio:.3f}"
            if left_size is not None and right_size is not None and ratio is not None
            else "grid square L=-- R=-- ratio=--"
        )
        cv2.putText(img, size_txt, (10, 206),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)

        zoom_hint = _zoom_hint(metrics)
        zoom_color = (0, 220, 0) if metrics.zoom_ok() else (0, 120, 255)
        zoom_status = _zoom_status_line(metrics)
        cv2.putText(img, zoom_status, (10, 238),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, zoom_color, 2, cv2.LINE_AA)
        cv2.putText(img, zoom_hint, (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, zoom_color, 1, cv2.LINE_AA)
        self._draw_zoom_balance_bar(img, metrics, (10, 292), 260, 16)

        if metrics.vertical_delta_px is not None:
            cv2.putText(img, f"vertical grid dy={metrics.vertical_delta_px:+.1f}px", (10, h - 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 220, 0) if metrics.vertical_ok() else (0, 120, 255), 1, cv2.LINE_AA)
        if metrics.rotation_delta_deg is not None:
            cv2.putText(img, f"grid rotation={metrics.rotation_delta_deg:+.2f} deg", (10, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 220, 0) if metrics.rotation_ok() else (0, 120, 255), 1, cv2.LINE_AA)

    def _draw_zoom_balance_bar(
        self,
        img: np.ndarray,
        metrics: GridPairMetrics,
        origin: tuple[int, int],
        width: int,
        height: int,
    ) -> None:
        err = metrics.zoom_error_pct
        if err is None:
            return
        x, y = origin
        cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 1)
        mid = x + width // 2
        cv2.line(img, (mid, y - 4), (mid, y + height + 4), (255, 255, 255), 1)
        clamped = max(-10.0, min(10.0, err))
        marker_x = int(round(mid + (clamped / 10.0) * (width / 2)))
        color = (0, 220, 0) if metrics.zoom_ok() else (0, 120, 255)
        cv2.circle(img, (marker_x, y + height // 2), 7, color, -1, cv2.LINE_AA)
        cv2.putText(img, "L larger", (x, y + height + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, "R larger", (x + width - 70, y + height + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)


def _grid_bounds(detection: GridDetection, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    pts = detection.corners
    x0 = max(0, int(np.floor(pts[:, 0].min())) - 12)
    y0 = max(0, int(np.floor(pts[:, 1].min())) - 12)
    x1 = min(w, int(np.ceil(pts[:, 0].max())) + 12)
    y1 = min(h, int(np.ceil(pts[:, 1].max())) + 12)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _detect_grid_scaled(
    img: np.ndarray,
    inner_cols: int,
    inner_rows: int,
    max_dim: int,
) -> GridDetection | None:
    h, w = img.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return detect_grid(img, inner_cols=inner_cols, inner_rows=inner_rows, exhaustive=False)

    scale = max_dim / largest
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    detected = detect_grid(small, inner_cols=inner_cols, inner_rows=inner_rows, exhaustive=False)
    if detected is None:
        return None
    corners = detected.corners / scale
    center = (detected.center[0] / scale, detected.center[1] / scale)
    return GridDetection(corners=corners, center=center)


def _grid_row_angle_deg(detection: GridDetection, inner_cols: int, inner_rows: int) -> float:
    pts = detection.corners.reshape(inner_rows, inner_cols, 2)
    row_vectors = pts[:, -1, :] - pts[:, 0, :]
    angles = np.degrees(np.arctan2(row_vectors[:, 1], row_vectors[:, 0]))
    return float(np.median(angles))


def _estimate_partial_grid_square_px(gray: np.ndarray) -> float | None:
    """Estimate visible square spacing when the full chessboard is cropped.

    This fallback is intentionally only used for focus/scale guidance.  It
    does not provide corner identity, so vertical offset and rotation still
    require the full chessboard detection.
    """
    if gray.size == 0:
        return None
    h, w = gray.shape[:2]
    scale = 1.0
    largest = max(h, w)
    if largest > 640:
        scale = 640.0 / largest
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    spacings: list[float] = []
    for dx, dy, axis_len, reduce_axis in (
        (1, 0, blur.shape[1], 0),
        (0, 1, blur.shape[0], 1),
    ):
        grad = np.abs(cv2.Sobel(blur, cv2.CV_32F, dx, dy, ksize=3))
        profile = grad.mean(axis=reduce_axis)
        spacing = _median_peak_spacing(profile, min_gap=max(6, axis_len // 40))
        if spacing is not None:
            spacings.append(spacing / scale)
    if not spacings:
        return None
    return float(np.median(spacings))


def _median_peak_spacing(profile: np.ndarray, min_gap: int) -> float | None:
    if profile.size < min_gap * 3:
        return None
    window = max(5, (profile.size // 80) | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    smooth = np.convolve(profile.astype(np.float32), kernel, mode="same")
    threshold = max(float(np.percentile(smooth, 88)), float(smooth.mean() + smooth.std()))
    active = smooth >= threshold
    centers: list[float] = []
    i = 0
    while i < active.size:
        if not active[i]:
            i += 1
            continue
        start = i
        while i < active.size and active[i]:
            i += 1
        end = i
        if end - start >= 2:
            weights = smooth[start:end]
            xs = np.arange(start, end, dtype=np.float32)
            centers.append(float((xs * weights).sum() / max(float(weights.sum()), 1e-6)))
    if len(centers) < 3:
        return None
    diffs = np.diff(np.array(centers))
    diffs = diffs[diffs >= min_gap]
    if diffs.size < 2:
        return None
    median = float(np.median(diffs))
    return median if median > 0 else None


def _status_text(ok: bool) -> str:
    return "OK" if ok else "ADJUST"


def _status_color(ok: bool) -> tuple[int, int, int]:
    return (0, 220, 0) if ok else (0, 120, 255)


def _zoom_hint(metrics: GridPairMetrics) -> str:
    ratio = metrics.zoom_ratio
    if ratio is None:
        return "zoom: grid size unavailable"
    if metrics.zoom_ok():
        return "zoom: OK - mechanical zoom matched"
    if ratio > 1.0:
        return "zoom: RIGHT image is larger - reduce right zoom or increase left"
    return "zoom: LEFT image is larger - reduce left zoom or increase right"


def _zoom_status_line(metrics: GridPairMetrics) -> str:
    err = metrics.zoom_error_pct
    if err is None:
        return "ZOOM: NO GRID SCALE"
    return f"ZOOM: {metrics.zoom_status()}  error={err:+.1f}%"
