import numpy as np

from src.config import StereoCfg
from src.stereo_processor import StereoProcessor


def _make_processor() -> StereoProcessor:
    cfg = StereoCfg()
    cfg.zoom.min = 1.0
    cfg.zoom.max = 5.0
    cfg.zoom.step = 0.1
    cfg.convergence.base_offset = 0
    cfg.convergence.auto_adjust = False
    return StereoProcessor(cfg, eye_width=100, eye_height=100)


def test_joint_zoom_center_moves_horizontal_crop():
    p = _make_processor()
    p.zoom = 2.0

    # Horizontal gradient: brighter values to the right.
    frame = np.tile(np.arange(100, dtype=np.uint8), (100, 1))
    frame = np.dstack([frame, frame, frame])

    p.set_joint_zoom_center(20)
    left_view = p.process_eye(frame, "left")

    p.set_joint_zoom_center(80)
    right_view = p.process_eye(frame, "left")

    assert right_view.mean() > left_view.mean()


def test_joint_zoom_center_moves_vertical_crop():
    p = _make_processor()
    p.zoom = 2.0

    # Vertical gradient: brighter values toward the bottom.
    frame = np.tile(np.arange(100, dtype=np.uint8).reshape(100, 1), (1, 100))
    frame = np.dstack([frame, frame, frame])

    p.set_joint_zoom_center_y(20)
    top_view = p.process_eye(frame, "left")

    p.set_joint_zoom_center_y(80)
    bottom_view = p.process_eye(frame, "left")

    assert bottom_view.mean() > top_view.mean()


def test_convergence_still_offsets_left_and_right_crops():
    p = _make_processor()
    p.zoom = 2.0
    p.base_offset = 10

    frame = np.tile(np.arange(100, dtype=np.uint8), (100, 1))
    frame = np.dstack([frame, frame, frame])

    left_eye = p.process_eye(frame, "left")
    right_eye = p.process_eye(frame, "right")

    # Positive convergence shifts left-eye crop right and right-eye crop left.
    assert left_eye.mean() > right_eye.mean()


def test_process_eye_preserves_output_aspect_before_resize():
    cfg = StereoCfg()
    cfg.aspect_mode = "crop"
    p = StereoProcessor(cfg, eye_width=96, eye_height=108)
    frame = np.zeros((108, 192, 3), dtype=np.uint8)

    # Draw a 40x40 square centered in the camera frame. If the full 16:9
    # frame were squeezed into a 8:9 SBS eye, this would become a rectangle.
    frame[34:74, 76:116] = 255

    out = p.process_eye(frame, "left")
    mask = out[:, :, 0] > 128
    ys, xs = np.where(mask)
    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1

    assert abs(width - height) <= 2


def test_process_eye_full_aspect_keeps_camera_width():
    cfg = StereoCfg()
    cfg.aspect_mode = "full"
    p = StereoProcessor(cfg, eye_width=96, eye_height=108)
    frame = np.zeros((108, 192, 3), dtype=np.uint8)

    frame[:, :20] = 255
    out = p.process_eye(frame, "left")

    # Full mode uses the full 16:9 camera frame, so the left edge remains
    # visible instead of being cropped away for square-pixel geometry.
    assert out[:, 0].mean() > 0
