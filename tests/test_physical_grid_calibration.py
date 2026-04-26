import cv2
import numpy as np
import yaml

from src.physical_grid_calibration import (
    OffsetResult,
    detect_grid,
    estimate_right_eye_offset,
    expected_horizontal_disparity_px,
    generate_chessboard_page,
    update_config_offsets,
)


def test_generated_chessboard_is_detectable(tmp_path):
    path = tmp_path / "grid.png"
    generate_chessboard_page(path, inner_cols=9, inner_rows=6, square_mm=20, dpi=120)

    img = cv2.imread(str(path))
    det = detect_grid(img, inner_cols=9, inner_rows=6)

    assert det is not None
    assert det.corners.shape == (54, 2)


def test_estimate_right_eye_offset_from_shifted_grid(tmp_path):
    path = tmp_path / "grid.png"
    generate_chessboard_page(path, inner_cols=9, inner_rows=6, square_mm=20, dpi=120)
    left = cv2.imread(str(path))

    shift_x = 18
    shift_y = -12
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    right = cv2.warpAffine(left, matrix, (left.shape[1], left.shape[0]))

    result = estimate_right_eye_offset([left], [right], inner_cols=9, inner_rows=6)

    assert result.nudge_right_x == -shift_x
    assert result.nudge_right_y == -shift_y
    assert result.samples == 1


def test_expected_horizontal_disparity_uses_detected_square_scale(tmp_path):
    path = tmp_path / "grid.png"
    square_mm = 20.0
    baseline_mm = 42.8
    generate_chessboard_page(path, inner_cols=9, inner_rows=6, square_mm=square_mm, dpi=120)
    img = cv2.imread(str(path))
    det = detect_grid(img, inner_cols=9, inner_rows=6)

    disparity = expected_horizontal_disparity_px(det, baseline_mm, square_mm)

    assert disparity > 0
    assert disparity > baseline_mm


def test_update_config_offsets_can_disable_auto_align(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"stereo": {"alignment": {"enabled": True}}}),
        encoding="utf-8",
    )
    result = OffsetResult(nudge_right_x=3, nudge_right_y=4, dx=3.0, dy=4.0, samples=1)

    update_config_offsets(path, result, disable_auto_align=True)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["calibration_state"]["nudge_right_x"] == 3
    assert raw["calibration_state"]["nudge_right_y"] == 4
    assert raw["stereo"]["alignment"]["enabled"] is False


def test_update_config_offsets_keeps_auto_align_by_default(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"stereo": {"alignment": {"enabled": True}}}),
        encoding="utf-8",
    )
    result = OffsetResult(nudge_right_x=3, nudge_right_y=4, dx=3.0, dy=4.0, samples=1)

    update_config_offsets(path, result)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["calibration_state"]["nudge_right_x"] == 3
    assert raw["calibration_state"]["nudge_right_y"] == 4
    assert raw["stereo"]["alignment"]["enabled"] is True
