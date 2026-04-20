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
