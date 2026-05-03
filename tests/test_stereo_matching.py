import numpy as np
import cv2

from src.stereo_matching import theil_sen, StereoFeatureMatcher, MatchResult


def test_theil_sen_recovers_slope_and_intercept_on_clean_line():
    rng = np.random.default_rng(0)
    x = rng.uniform(-100, 100, size=80)
    true_slope = 0.05
    true_intercept = 1.2
    y = true_slope * x + true_intercept
    slope, intercept, rms = theil_sen(x, y)
    assert abs(slope - true_slope) < 1e-6
    assert abs(intercept - true_intercept) < 1e-6
    assert rms < 1e-6


def test_theil_sen_robust_to_30_pct_outliers():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.uniform(-100, 100, size=n)
    y = 0.02 * x + 0.5
    # Corrupt 30 % of points with very large residuals
    mask = rng.random(n) < 0.30
    y[mask] += rng.uniform(-50, 50, size=mask.sum())
    slope, intercept, _ = theil_sen(x, y)
    assert abs(slope - 0.02) < 0.01
    assert abs(intercept - 0.5) < 1.0


def test_theil_sen_zero_slope_when_no_x_spread():
    x = np.full(20, 5.0)
    y = np.linspace(0, 10, 20)
    slope, intercept, _ = theil_sen(x, y)
    # All x identical → no useful slope information → fall back to slope 0
    assert slope == 0.0
    assert abs(intercept - float(np.median(y))) < 1e-6


def _synthetic_feature_pair(shift_y: int = 0):
    """Build a 320x240 grey pair with strong corner features."""
    img_l = np.full((240, 320), 128, dtype=np.uint8)
    rng = np.random.default_rng(7)
    # Sprinkle high-contrast 6x6 squares to give SIFT something to match.
    for _ in range(60):
        x = int(rng.integers(20, 300))
        y = int(rng.integers(20, 220))
        img_l[y:y + 6, x:x + 6] = 255
    img_r = np.roll(img_l, shift_y, axis=0)
    return img_l, img_r


def test_feature_matcher_returns_inliers_for_feature_rich_pair():
    img_l, img_r = _synthetic_feature_pair(shift_y=3)
    m = StereoFeatureMatcher(max_features=500, match_ratio=0.75, ransac_thresh=2.0,
                             frame_w=320, frame_h=240)
    result = m.match(img_l, img_r)
    assert isinstance(result, MatchResult)
    assert result.pts_l.shape == result.pts_r.shape
    assert result.pts_l.shape[1] == 2
    assert result.pts_l.shape[0] >= 8


def test_feature_matcher_returns_empty_for_blank_pair():
    blank = np.full((240, 320), 128, dtype=np.uint8)
    m = StereoFeatureMatcher(max_features=500, match_ratio=0.75, ransac_thresh=2.0,
                             frame_w=320, frame_h=240)
    result = m.match(blank, blank.copy())
    assert result.pts_l.shape[0] == 0
    assert result.pts_r.shape[0] == 0
