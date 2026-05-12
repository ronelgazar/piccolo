"""GPU-resident hot path for the live stereo pipeline.

See docs/superpowers/specs/2026-05-10-gpu-pipeline-latency-reduction-design.md.
"""
from __future__ import annotations

import cv2
import numpy as np


def fill_holes_cross_gpu(gpu_l: "cv2.cuda_GpuMat", gpu_r: "cv2.cuda_GpuMat") -> None:
    """In-place equivalent of pipeline_worker._fill_holes_cross.

    For each pixel that is fully zero in one eye and non-zero in the other,
    copy the non-zero pixel into the zero side. Pure GPU; no host round-trip.
    """
    if gpu_l.size() != gpu_r.size() or gpu_l.type() != gpu_r.type():
        return

    # Per-channel zero mask: cv2.cuda.compareWithScalar yields 8U single-channel.
    chans_l = cv2.cuda.split(gpu_l)
    chans_r = cv2.cuda.split(gpu_r)

    zero_l = cv2.cuda.compareWithScalar(chans_l[0], 0, cv2.CMP_EQ)
    for ch in chans_l[1:]:
        zero_l = cv2.cuda.bitwise_and(zero_l, cv2.cuda.compareWithScalar(ch, 0, cv2.CMP_EQ))

    zero_r = cv2.cuda.compareWithScalar(chans_r[0], 0, cv2.CMP_EQ)
    for ch in chans_r[1:]:
        zero_r = cv2.cuda.bitwise_and(zero_r, cv2.cuda.compareWithScalar(ch, 0, cv2.CMP_EQ))

    # copy_to_l = zero_l AND NOT zero_r;  copy_to_r = zero_r AND NOT zero_l
    not_zero_r = cv2.cuda.bitwise_not(zero_r)
    not_zero_l = cv2.cuda.bitwise_not(zero_l)
    copy_to_l = cv2.cuda.bitwise_and(zero_l, not_zero_r)
    copy_to_r = cv2.cuda.bitwise_and(zero_r, not_zero_l)

    # Use GpuMat.copyTo(mask, dst) — the instance method that copies masked
    # pixels entirely on-device (no host round-trip).
    # Snapshot original gpu_l before mutation so the R→L copy doesn't corrupt
    # the source for the subsequent L→R copy.
    gpu_l_orig = gpu_l.clone()
    gpu_r.copyTo(copy_to_l, gpu_l)        # copy R→L where L was zero and R was nonzero
    gpu_l_orig.copyTo(copy_to_r, gpu_r)   # copy original L→R where R was zero and L was nonzero


class GpuPipeline:
    """Owns persistent cuda_GpuMat buffers and runs the GPU-resident hot path.

    Lifecycle:
        pipeline = GpuPipeline(aligner, processor, calibration)
        sbs = pipeline.process(frame_l, frame_r)
        pipeline.release()
    """

    def __init__(self, aligner, processor, calibration):
        self.aligner = aligner
        self.processor = processor
        self.calibration = calibration
        self._gpu_in_l = None
        self._gpu_in_r = None
        self._gpu_warp_l = None
        self._gpu_warp_r = None
        self._gpu_sbs = None
        self._frame_shape: tuple[int, int] | None = None

    def process(self, frame_l: np.ndarray, frame_r: np.ndarray) -> np.ndarray:
        """Run the GPU-resident pipeline; return the composed SBS ndarray."""
        h, w = frame_l.shape[:2]
        self._ensure_buffers(h, w)

        self._gpu_in_l.upload(frame_l)
        self._gpu_in_r.upload(frame_r)

        used_warp = self.aligner.warp_pair_gpu(
            self._gpu_in_l,
            self._gpu_in_r,
            self._gpu_warp_l,
            self._gpu_warp_r,
        )
        if used_warp:
            src_l, src_r = self._gpu_warp_l, self._gpu_warp_r
        else:
            src_l, src_r = self._gpu_in_l, self._gpu_in_r

        fill_holes_cross_gpu(src_l, src_r)
        self.processor.process_pair_gpu(src_l, src_r, self._gpu_sbs)

        eye_l_gpu = cv2.cuda_GpuMat(
            self._gpu_sbs, (0, 0, self.processor.eye_w, self.processor.eye_h)
        )
        eye_r_gpu = cv2.cuda_GpuMat(
            self._gpu_sbs,
            (self.processor.eye_w, 0, self.processor.eye_w, self.processor.eye_h),
        )
        self.calibration.apply_nudge_gpu(eye_l_gpu, eye_r_gpu)

        return self._gpu_sbs.download()

    def release(self) -> None:
        self._gpu_in_l = None
        self._gpu_in_r = None
        self._gpu_warp_l = None
        self._gpu_warp_r = None
        self._gpu_sbs = None
        self._frame_shape = None

    def _ensure_buffers(self, h: int, w: int) -> None:
        if self._frame_shape == (h, w):
            return

        self._gpu_in_l = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
        self._gpu_in_r = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
        self._gpu_warp_l = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
        self._gpu_warp_r = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
        self._gpu_sbs = cv2.cuda_GpuMat(
            self.processor.eye_h, self.processor.eye_w * 2, cv2.CV_8UC3
        )
        self._frame_shape = (h, w)
