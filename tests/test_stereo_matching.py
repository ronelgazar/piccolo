import numpy as np

from src.stereo_matching import theil_sen


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
