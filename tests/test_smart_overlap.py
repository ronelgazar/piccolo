import numpy as np
import cv2

from src.physical_grid_calibration import generate_chessboard_page
from src.smart_overlap import (
    OverlapPair,
    OverlapMetrics,
    compute_align_ok,
    compute_zoom_ok,
    find_chessboard_pairs,
)


def test_overlap_pair_holds_indices_color_and_xy():
    p = OverlapPair(index=2, color=(255, 100, 50), left_xy=(10.0, 20.0), right_xy=(11.0, 21.0))
    assert p.index == 2
    assert p.color == (255, 100, 50)
    assert p.left_xy == (10.0, 20.0)
    assert p.right_xy == (11.0, 21.0)


def test_overlap_metrics_defaults_to_empty():
    m = OverlapMetrics(mode="chessboard", pairs=[], vert_dy_px=0.0, rotation_deg=0.0,
                       zoom_ratio=None, n_inliers=0, n_requested=8,
                       align_ok=False, zoom_ok=False)
    assert m.pairs == []
    assert m.zoom_ratio is None
    assert m.align_ok is False


def test_compute_align_ok_within_thresholds():
    assert compute_align_ok(vert_dy_px=2.0, rotation_deg=0.2, n_inliers=5,
                            max_vert_dy_px=5.0, max_rotation_deg=0.5,
                            min_pairs_for_metrics=4) is True


def test_compute_align_ok_fails_on_low_inliers_even_when_metrics_perfect():
    assert compute_align_ok(vert_dy_px=0.0, rotation_deg=0.0, n_inliers=2,
                            max_vert_dy_px=5.0, max_rotation_deg=0.5,
                            min_pairs_for_metrics=4) is False


def test_compute_zoom_ok_within_tol():
    assert compute_zoom_ok(zoom_ratio=1.01, n_inliers=4,
                           max_zoom_ratio_err=0.02,
                           min_pairs_for_metrics=4) is True


def test_compute_zoom_ok_none_ratio_returns_false():
    assert compute_zoom_ok(zoom_ratio=None, n_inliers=4,
                           max_zoom_ratio_err=0.02,
                           min_pairs_for_metrics=4) is False


def _render_grid_eye(tmp_path):
    """Render the standard 9x6 grid PNG and return a BGR image."""
    out = tmp_path / "grid.png"
    generate_chessboard_page(out, inner_cols=9, inner_rows=6, square_mm=20.0, dpi=150)
    img = cv2.imread(str(out))
    assert img is not None
    return img


def test_find_chessboard_pairs_returns_K_pairs(tmp_path):
    img = _render_grid_eye(tmp_path)
    pairs, zoom_ratio = find_chessboard_pairs(img, img.copy(), pair_count=8)
    assert len(pairs) == 8
    assert all(len(p.left_xy) == 2 and len(p.right_xy) == 2 for p in pairs)
    # Identical inputs -> ratio ~= 1
    assert zoom_ratio is not None
    assert abs(zoom_ratio - 1.0) < 0.05


def test_find_chessboard_pairs_no_grid_returns_empty(tmp_path):
    blank = np.full((480, 640, 3), 200, dtype=np.uint8)
    pairs, zoom_ratio = find_chessboard_pairs(blank, blank.copy(), pair_count=8)
    assert pairs == []
    assert zoom_ratio is None


def test_find_chessboard_pairs_detects_injected_scale(tmp_path):
    img = _render_grid_eye(tmp_path)
    # Resize right eye to 90 % so right squares are smaller -> ratio < 1.
    h, w = img.shape[:2]
    img_r = cv2.resize(img, (int(w * 0.9), int(h * 0.9)))
    img_r = cv2.copyMakeBorder(img_r, 0, h - img_r.shape[0],
                                0, w - img_r.shape[1],
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    _, ratio = find_chessboard_pairs(img, img_r, pair_count=8)
    assert ratio is not None
    assert ratio < 0.95
