"""Shared stereo feature matching + Theil-Sen regression.

Extracted from StereoAligner so SmartOverlapAnalyzer can reuse the same
matching pipeline. No behaviour change versus the original private methods
in stereo_align.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


def theil_sen(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Theil-Sen robust linear fit: y = slope*x + intercept.

    Returns (slope, intercept, rms_residual). For n <= 200 uses all unique
    pairs; above that, samples 50_000 random pairs to keep runtime bounded.
    Tolerates ~29 % gross outliers.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    min_sep = 30.0  # minimum x-separation for a useful slope pair

    if n <= 200:
        ii, jj = np.triu_indices(n, k=1)
        dx = x[jj] - x[ii]
        valid = np.abs(dx) > min_sep
        if valid.sum() < 3:
            slope = 0.0
        else:
            slopes = (y[jj[valid]] - y[ii[valid]]) / dx[valid]
            slope = float(np.median(slopes))
    else:
        rng = np.random.default_rng(42)
        n_pairs = min(50_000, n * (n - 1) // 2)
        ai = rng.integers(0, n, n_pairs)
        bi = rng.integers(0, n, n_pairs)
        keep = ai != bi
        ai, bi = ai[keep], bi[keep]
        dx = x[bi] - x[ai]
        valid = np.abs(dx) > min_sep
        if valid.sum() < 3:
            slope = 0.0
        else:
            slopes = (y[bi[valid]] - y[ai[valid]]) / dx[valid]
            slope = float(np.median(slopes))

    intercepts = y - slope * x
    intercept = float(np.median(intercepts))
    residuals = y - (intercept + slope * x)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return slope, intercept, rms


@dataclass(frozen=True)
class MatchResult:
    """Inlier pixel coordinates for a stereo pair after full filtering.

    `pts_l[i]` and `pts_r[i]` are the same scene point seen from each eye.
    Both arrays have shape (N, 2) where N may be 0.
    """
    pts_l: np.ndarray
    pts_r: np.ndarray


class StereoFeatureMatcher:
    """SIFT + cross-check FLANN + grid distribution + F-matrix RANSAC.

    Identical pipeline to the original private methods in StereoAligner.
    Can be used for both auto-alignment (`StereoAligner`) and manual
    diagnostic visualisation (`SmartOverlapAnalyzer`).
    """

    _GRID = 6  # spatial-distribution grid (matches per cell are capped)

    def __init__(
        self,
        max_features: int,
        match_ratio: float,
        ransac_thresh: float,
        frame_w: int,
        frame_h: int,
    ):
        self.match_ratio = float(match_ratio)
        self.ransac_thresh = float(ransac_thresh)
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._sift = cv2.SIFT_create(nfeatures=int(max_features))
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=80)
        self._matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, gray_l: np.ndarray, gray_r: np.ndarray) -> MatchResult:
        gray_l = self._clahe.apply(gray_l)
        gray_r = self._clahe.apply(gray_r)

        kp_l, des_l = self._sift.detectAndCompute(gray_l, None)
        kp_r, des_r = self._sift.detectAndCompute(gray_r, None)
        if des_l is None or des_r is None or len(kp_l) < 10 or len(kp_r) < 10:
            return MatchResult(pts_l=np.empty((0, 2)), pts_r=np.empty((0, 2)))

        pairs = self._cross_check(des_l, des_r)
        if len(pairs) < 8:
            pairs = self._one_way(des_l, des_r)
            if len(pairs) < 8:
                return MatchResult(pts_l=np.empty((0, 2)), pts_r=np.empty((0, 2)))

        pts_l = np.float32([kp_l[a].pt for a, _ in pairs])
        pts_r = np.float32([kp_r[b].pt for _, b in pairs])

        pts_l, pts_r = self._enforce_distribution(pts_l, pts_r)
        if len(pts_l) < 8:
            return MatchResult(pts_l=np.empty((0, 2)), pts_r=np.empty((0, 2)))

        try:
            F, mask = cv2.findFundamentalMat(
                pts_l, pts_r,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
                confidence=0.999,
            )
        except cv2.error:
            F, mask = None, None
        if F is not None and mask is not None:
            inliers = mask.ravel() == 1
            pts_l = pts_l[inliers]
            pts_r = pts_r[inliers]

        return MatchResult(pts_l=pts_l, pts_r=pts_r)

    # --- internals ---------------------------------------------------------

    def _cross_check(self, des_l, des_r):
        try:
            raw_lr = self._matcher.knnMatch(des_l, des_r, k=2)
            raw_rl = self._matcher.knnMatch(des_r, des_l, k=2)
        except cv2.error:
            return []
        good_lr = {m.queryIdx: m.trainIdx
                   for pair in raw_lr if len(pair) == 2
                   for m, n in [pair] if m.distance < self.match_ratio * n.distance}
        good_rl = {m.queryIdx: m.trainIdx
                   for pair in raw_rl if len(pair) == 2
                   for m, n in [pair] if m.distance < self.match_ratio * n.distance}
        return [(l, r) for l, r in good_lr.items() if good_rl.get(r) == l]

    def _one_way(self, des_l, des_r):
        try:
            raw = self._matcher.knnMatch(des_l, des_r, k=2)
        except cv2.error:
            return []
        return [(m.queryIdx, m.trainIdx)
                for pair in raw if len(pair) == 2
                for m, n in [pair] if m.distance < self.match_ratio * n.distance]

    def _enforce_distribution(self, pts_l: np.ndarray, pts_r: np.ndarray):
        g = self._GRID
        cell_w = self.frame_w / g
        cell_h = self.frame_h / g
        buckets: dict[tuple, list[int]] = {}
        for i in range(len(pts_l)):
            gx = min(int(pts_l[i, 0] / cell_w), g - 1)
            gy = min(int(pts_l[i, 1] / cell_h), g - 1)
            buckets.setdefault((gx, gy), []).append(i)
        max_per = max(6, len(pts_l) // (g * g) + 1)
        keep = [i for indices in buckets.values() for i in indices[:max_per]]
        idx = np.array(keep)
        return pts_l[idx], pts_r[idx]
