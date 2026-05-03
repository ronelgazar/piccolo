import numpy as np

from src.calibration import CalibrationOverlay
from src.config import CalibrationCfg
from src.ui.calibration_tab import CalibrationTab


def test_apply_nudge_shifts_without_wrapping_right_edge_into_left():
    overlay = CalibrationOverlay(CalibrationCfg())
    overlay.nudge_left = 2
    left = np.zeros((2, 5, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    left[:, -1] = 255

    shifted, _ = overlay.apply_nudge(left, right)

    assert shifted[:, :2].sum() == 0
    assert shifted[:, -1].sum() == 0


def test_apply_nudge_large_shift_blanks_image():
    overlay = CalibrationOverlay(CalibrationCfg())
    overlay.nudge_left = 10
    left = np.full((2, 5, 3), 255, dtype=np.uint8)
    right = np.zeros_like(left)

    shifted, _ = overlay.apply_nudge(left, right)

    assert shifted.sum() == 0


def test_overlay_shift_eye_shifts_without_wrapping():
    img = np.zeros((2, 5, 3), dtype=np.uint8)
    img[:, -1] = 255

    shifted = CalibrationTab._shift_eye(img, shift_x=2, shift_y=0)

    assert shifted[:, :2].sum() == 0
    assert shifted[:, -1].sum() == 0


def test_scale_eye_down_preserves_frame_size_and_centers_image():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)

    out = CalibrationOverlay._scale_eye(img, 80)

    assert out.shape == img.shape
    assert out[0].sum() == 0
    assert out[-1].sum() == 0
    assert out[5, 5].sum() > 0


def test_scale_eye_up_preserves_frame_size():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[2:8, 2:8] = 255

    out = CalibrationOverlay._scale_eye(img, 120)

    assert out.shape == img.shape
    assert out.sum() > img.sum()
