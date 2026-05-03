import numpy as np
import pytest

from src.config import SmartOverlapCfg
from src.stereo_matching import StereoFeatureMatcher
from src.smart_overlap import SmartOverlapAnalyzer


def test_worker_process_returns_payload_with_image_and_metrics():
    pytest.importorskip("PyQt6")
    from src.ui.smart_overlap_worker import SmartOverlapWorker, SmartOverlapResult

    cfg = SmartOverlapCfg()
    matcher = StereoFeatureMatcher(max_features=300, match_ratio=0.75,
                                   ransac_thresh=2.0, frame_w=320, frame_h=240)
    analyzer = SmartOverlapAnalyzer(
        max_vert_dy_px=cfg.max_vert_dy_px, max_rotation_deg=cfg.max_rotation_deg,
        max_zoom_ratio_err=cfg.max_zoom_ratio_err,
        min_pairs_for_metrics=cfg.min_pairs_for_metrics,
        pair_stability_tol_px=cfg.pair_stability_tol_px,
        matcher=matcher,
    )
    w = SmartOverlapWorker(analyzer)

    sbs = np.full((240, 640, 3), 50, dtype=np.uint8)
    result = w._process(sbs, mode="live", pair_count=4)
    assert isinstance(result, SmartOverlapResult)
    assert result.image is not None
    assert result.metrics.mode == "live"


def test_worker_scales_analysis_metrics_back_to_display_size():
    pytest.importorskip("PyQt6")
    from src.smart_overlap import OverlapMetrics, OverlapPair
    from src.ui.smart_overlap_worker import SmartOverlapWorker

    cfg = SmartOverlapCfg()
    analyzer = SmartOverlapAnalyzer(
        max_vert_dy_px=cfg.max_vert_dy_px, max_rotation_deg=cfg.max_rotation_deg,
        max_zoom_ratio_err=cfg.max_zoom_ratio_err,
        min_pairs_for_metrics=cfg.min_pairs_for_metrics,
        pair_stability_tol_px=cfg.pair_stability_tol_px,
        matcher=None,
    )
    worker = SmartOverlapWorker(analyzer)
    metrics = OverlapMetrics(
        mode="chessboard",
        pairs=[
            OverlapPair(index=0, color=(255, 0, 0), left_xy=(10.0, 20.0), right_xy=(12.0, 24.0))
        ],
        vert_dy_px=4.0,
        rotation_deg=0.1,
        zoom_ratio=1.0,
        n_inliers=4,
        n_requested=4,
        align_ok=True,
        zoom_ok=True,
    )

    scaled = worker._scale_metrics(metrics, 2.0)

    assert scaled.pairs[0].left_xy == (20.0, 40.0)
    assert scaled.pairs[0].right_xy == (24.0, 48.0)
    assert scaled.vert_dy_px == 8.0
    assert scaled.align_ok is False
