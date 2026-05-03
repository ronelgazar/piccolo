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
