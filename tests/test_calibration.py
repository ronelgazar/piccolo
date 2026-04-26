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
