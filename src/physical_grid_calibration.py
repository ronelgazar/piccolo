"""Printable chessboard detection for physical stereo-offset calibration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class GridDetection:
    corners: np.ndarray
    center: tuple[float, float]


@dataclass(frozen=True)
class OffsetResult:
    nudge_right_x: int
    nudge_right_y: int
    dx: float
    dy: float
    samples: int
    expected_baseline_dx: float | None = None


def generate_chessboard_page(
    output_path: str | Path,
    inner_cols: int = 9,
    inner_rows: int = 6,
    square_mm: float = 20.0,
    dpi: int = 300,
    page_mm: tuple[float, float] = (210.0, 297.0),
) -> Path:
    """Generate an A4 printable chessboard page.

    ``inner_cols`` and ``inner_rows`` are the inner-corner counts passed to
    OpenCV. The rendered board therefore has one extra square in each axis.
    """
    out = Path(output_path)
    width_px = round(page_mm[0] / 25.4 * dpi)
    height_px = round(page_mm[1] / 25.4 * dpi)
    square_px = round(square_mm / 25.4 * dpi)
    squares_x = inner_cols + 1
    squares_y = inner_rows + 1
    board_w = squares_x * square_px
    board_h = squares_y * square_px
    origin_x = (width_px - board_w) // 2
    origin_y = (height_px - board_h) // 2
    if origin_x < 0 or origin_y < 0:
        raise ValueError("Board is larger than the requested page")

    img = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(img)
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                x0 = origin_x + x * square_px
                y0 = origin_y + y * square_px
                draw.rectangle((x0, y0, x0 + square_px, y0 + square_px), fill="black")

    draw.rectangle(
        (origin_x, origin_y, origin_x + board_w, origin_y + board_h),
        outline="black",
        width=max(2, dpi // 100),
    )
    label = f"Piccolo stereo calibration chessboard: {inner_cols}x{inner_rows} inner corners, {square_mm:g} mm squares"
    draw.text((origin_x, min(height_px - 80, origin_y + board_h + 24)), label, fill="black")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".pdf":
        img.save(out, "PDF", resolution=dpi)
    else:
        img.save(out)
    return out


def detect_grid(
    image: np.ndarray,
    inner_cols: int = 9,
    inner_rows: int = 6,
    exhaustive: bool = True,
) -> GridDetection | None:
    """Detect a chessboard and return subpixel corner locations."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    pattern = (inner_cols, inner_rows)
    found = False
    corners = None
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if exhaustive:
            flags |= cv2.CALIB_CB_EXHAUSTIVE
        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern,
            flags=flags,
        )
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.01,
            )
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    if not found or corners is None:
        return None

    pts = corners.reshape(-1, 2).astype(np.float32)
    center_xy = pts.mean(axis=0)
    return GridDetection(corners=pts, center=(float(center_xy[0]), float(center_xy[1])))


def draw_detection_overlay(
    image: np.ndarray,
    detection: GridDetection | None,
    inner_cols: int = 9,
    inner_rows: int = 6,
) -> np.ndarray:
    """Return an image annotated with chessboard detection status."""
    out = image.copy()
    if detection is None:
        cv2.putText(
            out,
            "NO GRID",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        return out

    corners = detection.corners.reshape(-1, 1, 2)
    cv2.drawChessboardCorners(out, (inner_cols, inner_rows), corners, True)
    cx, cy = detection.center
    cv2.circle(out, (round(cx), round(cy)), 8, (0, 255, 0), -1)
    cv2.putText(
        out,
        "GRID OK",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )
    return out


def estimate_square_px(
    detection: GridDetection,
    inner_cols: int = 9,
    inner_rows: int = 6,
) -> float:
    """Estimate printed square size in pixels from detected inner corners."""
    pts = detection.corners.reshape(inner_rows, inner_cols, 2)
    lengths: list[np.ndarray] = []
    if inner_cols > 1:
        lengths.append(np.linalg.norm(np.diff(pts, axis=1), axis=2).ravel())
    if inner_rows > 1:
        lengths.append(np.linalg.norm(np.diff(pts, axis=0), axis=2).ravel())
    if not lengths:
        raise ValueError("Need at least two detected corners to estimate square size")
    return float(np.median(np.concatenate(lengths)))


def expected_horizontal_disparity_px(
    detection: GridDetection,
    baseline_mm: float,
    square_mm: float,
    inner_cols: int = 9,
    inner_rows: int = 6,
) -> float:
    """Approximate expected horizontal disparity from baseline and board scale.

    This uses the detected square size instead of requiring camera focal length
    or paper distance:

        disparity_px ~= observed_square_px * baseline_mm / square_mm

    It is a diagnostic only. Toe-in/angled cameras and lens distortion can move
    the measured value away from this estimate.
    """
    if square_mm <= 0:
        raise ValueError("square_mm must be positive")
    return estimate_square_px(detection, inner_cols, inner_rows) * baseline_mm / square_mm


def estimate_right_eye_offset(
    left_images: Iterable[np.ndarray],
    right_images: Iterable[np.ndarray],
    inner_cols: int = 9,
    inner_rows: int = 6,
    output_scale: tuple[float, float] = (1.0, 1.0),
    baseline_mm: float | None = None,
    square_mm: float = 20.0,
) -> OffsetResult:
    """Estimate right-eye nudge required to overlay the printed grid.

    The returned nudge is in processed eye pixels. Left-eye nudge is assumed
    to stay at zero; the right eye is shifted by median(left - right).
    """
    deltas: list[np.ndarray] = []
    expected_disparities: list[float] = []
    for left, right in zip(left_images, right_images):
        det_l = detect_grid(left, inner_cols=inner_cols, inner_rows=inner_rows)
        det_r = detect_grid(right, inner_cols=inner_cols, inner_rows=inner_rows)
        if det_l is None or det_r is None:
            continue
        if det_l.corners.shape != det_r.corners.shape:
            continue
        deltas.append(np.median(det_l.corners - det_r.corners, axis=0))
        if baseline_mm is not None:
            expected_disparities.append(
                expected_horizontal_disparity_px(
                    det_l,
                    baseline_mm=baseline_mm,
                    square_mm=square_mm,
                    inner_cols=inner_cols,
                    inner_rows=inner_rows,
                )
            )

    if not deltas:
        raise RuntimeError("Could not detect the chessboard in any stereo sample")

    median = np.median(np.vstack(deltas), axis=0)
    sx, sy = output_scale
    dx = float(median[0] * sx)
    dy = float(median[1] * sy)
    return OffsetResult(
        nudge_right_x=int(round(dx)),
        nudge_right_y=int(round(dy)),
        dx=dx,
        dy=dy,
        samples=len(deltas),
        expected_baseline_dx=(
            float(np.median(expected_disparities) * sx)
            if expected_disparities else None
        ),
    )


def update_config_offsets(
    config_path: str | Path,
    result: OffsetResult,
    set_left_zero: bool = True,
    disable_auto_align: bool = False,
) -> None:
    """Write detected right-eye offsets into ``config.yaml``."""
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if raw is None:
        raw = {}
    state = raw.setdefault("calibration_state", {})
    if set_left_zero:
        state["nudge_left_x"] = 0
        state["nudge_left_y"] = 0
    state["nudge_right_x"] = result.nudge_right_x
    state["nudge_right_y"] = result.nudge_right_y
    if disable_auto_align:
        raw.setdefault("stereo", {}).setdefault("alignment", {})["enabled"] = False
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
