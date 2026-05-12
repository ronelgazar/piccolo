"""GPU warp_pair matches CPU warp_pair within sub-pixel tolerance."""
from __future__ import annotations

import math
import numpy as np

from tests._cuda_helpers import skip_if_no_cuda


def _make_aligner(frame_w: int = 80, frame_h: int = 60):
    from src.config import AlignmentCfg
    from src.stereo_align import StereoAligner

    cfg = AlignmentCfg(enabled=True)
    aligner = StereoAligner(cfg, frame_w, frame_h)
    aligner._smooth_dy = 4.0
    aligner._smooth_dtheta = math.radians(0.5)
    aligner._build_warp_matrices()
    return aligner


def _smooth_gradient(h: int, w: int, seed: int) -> np.ndarray:
    """Camera-like smooth color gradient (no per-pixel noise).

    Random uint8 noise amplifies bilinear-resampler sub-pixel disagreements
    between CPU and GPU implementations (a half-pixel shift between two
    random uint8 values can differ by tens). Real camera frames are
    spatially correlated, so the per-pixel diff is small even when the
    samplers disagree at the sub-pixel level.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        (xx * 3.0 + seed * 7) % 256,
        (yy * 4.0 + seed * 11) % 256,
        ((xx + yy) * 2.0 + seed * 13) % 256,
    ], axis=-1)
    return base.astype(np.uint8)


def test_warp_pair_gpu_matches_cpu_interior():
    """GPU and CPU warp_pair produce visually equivalent output away from edges.

    Border pixels can differ by up to ~250 because CPU and GPU `warpAffine`
    handle out-of-bounds sampling differently (CPU interpolates near the
    edge using available pixels; GPU treats them as out-of-bounds). This is
    harmless in the live pipeline because `fill_holes_cross_gpu` runs
    immediately after and substitutes the partner eye's pixel where one
    eye is zero. So we test the interior region only — that's the part
    the surgeon actually sees.
    """
    skip_if_no_cuda()
    import cv2

    aligner = _make_aligner()
    frame_l = _smooth_gradient(60, 80, seed=2)
    frame_r = _smooth_gradient(60, 80, seed=5)

    cpu_l, cpu_r = aligner.warp_pair(frame_l.copy(), frame_r.copy())

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(frame_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(frame_r)
    out_l = cv2.cuda_GpuMat(60, 80, cv2.CV_8UC3)
    out_r = cv2.cuda_GpuMat(60, 80, cv2.CV_8UC3)
    aligner.warp_pair_gpu(gpu_l, gpu_r, out_l, out_r)
    gpu_l_arr = out_l.download()
    gpu_r_arr = out_r.download()

    # Restrict comparison to interior (5 px from each edge).
    s = slice(5, -5)
    diff_l = np.abs(cpu_l[s, s].astype(int) - gpu_l_arr[s, s].astype(int)).max()
    diff_r = np.abs(cpu_r[s, s].astype(int) - gpu_r_arr[s, s].astype(int)).max()

    # Tolerance of 5 accounts for fp32 sub-pixel interpolation diffs between
    # CPU's IPP bilinear and CUDA's texture-unit bilinear. Empirically the
    # smooth-gradient case observes max diff of 4 (~1.5% of uint8 range);
    # bumped to 5 for safety. This is imperceptible in the final SBS frame.
    assert diff_l <= 5, f"left interior diff was {diff_l}"
    assert diff_r <= 5, f"right interior diff was {diff_r}"


def test_warp_pair_gpu_writes_to_preallocated_destinations():
    """`out_l` / `out_r` GpuMats must contain the warp result after the call.

    Guards against a future API change where cv2.cuda.warpAffine could
    allocate a new internal buffer instead of writing into the provided dst.
    """
    skip_if_no_cuda()
    import cv2

    aligner = _make_aligner()
    frame_l = _smooth_gradient(60, 80, seed=7)
    frame_r = _smooth_gradient(60, 80, seed=9)

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(frame_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(frame_r)
    # Pre-fill destinations with a sentinel; the warp must overwrite it.
    out_l = cv2.cuda_GpuMat(); out_l.upload(np.full((60, 80, 3), 99, dtype=np.uint8))
    out_r = cv2.cuda_GpuMat(); out_r.upload(np.full((60, 80, 3), 99, dtype=np.uint8))

    used_warp = aligner.warp_pair_gpu(gpu_l, gpu_r, out_l, out_r)
    assert used_warp is True

    out_l_arr = out_l.download()
    out_r_arr = out_r.download()
    # The sentinel value 99 should be gone (the warp wrote real data).
    # Interior pixels (away from border) should match neither the sentinel
    # nor zero — they should be the warped gradient values.
    assert (out_l_arr[10:50, 10:70] != 99).any(), "out_l not written"
    assert (out_r_arr[10:50, 10:70] != 99).any(), "out_r not written"


def test_warp_pair_gpu_disabled_aligner_passthrough():
    """When the aligner is disabled or has no correction, return False."""
    skip_if_no_cuda()
    import cv2
    from src.config import AlignmentCfg
    from src.stereo_align import StereoAligner

    cfg = AlignmentCfg(enabled=False)
    aligner = StereoAligner(cfg, frame_w=40, frame_h=30)

    rng = np.random.default_rng(3)
    frame_l = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    frame_r = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)

    gpu_l = cv2.cuda_GpuMat(); gpu_l.upload(frame_l)
    gpu_r = cv2.cuda_GpuMat(); gpu_r.upload(frame_r)
    out_l = cv2.cuda_GpuMat(30, 40, cv2.CV_8UC3)
    out_r = cv2.cuda_GpuMat(30, 40, cv2.CV_8UC3)
    used_warp = aligner.warp_pair_gpu(gpu_l, gpu_r, out_l, out_r)

    assert used_warp is False, "disabled aligner must report no-warp"
