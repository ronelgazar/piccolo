import numpy as np

from src.smart_overlap import (
    OverlapPair,
    OverlapMetrics,
    compute_align_ok,
    compute_zoom_ok,
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
