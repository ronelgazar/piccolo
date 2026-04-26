"""Generate and apply printed-grid stereo calibration."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera import CameraCapture
from src.config import load_config
from src.physical_grid_calibration import (
    detect_grid,
    draw_detection_overlay,
    estimate_right_eye_offset,
    generate_chessboard_page,
    update_config_offsets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_print = sub.add_parser("print-pattern", help="Create a printable calibration page")
    p_print.add_argument("--output", default="calibration_grid.pdf")
    p_print.add_argument("--cols", type=int, default=9, help="Inner chessboard columns")
    p_print.add_argument("--rows", type=int, default=6, help="Inner chessboard rows")
    p_print.add_argument("--square-mm", type=float, default=20.0)
    p_print.add_argument("--dpi", type=int, default=300)

    p_preview = sub.add_parser("preview", help="Open both cameras and show live grid detection")
    p_preview.add_argument("--config", default="config.yaml")
    p_preview.add_argument("--cols", type=int, default=9, help="Inner chessboard columns")
    p_preview.add_argument("--rows", type=int, default=6, help="Inner chessboard rows")
    p_preview.add_argument("--debug-dir", default="calibration_debug")

    p_apply = sub.add_parser("apply", help="Capture both cameras and write offsets to config.yaml")
    p_apply.add_argument("--config", default="config.yaml")
    p_apply.add_argument("--cols", type=int, default=9, help="Inner chessboard columns")
    p_apply.add_argument("--rows", type=int, default=6, help="Inner chessboard rows")
    p_apply.add_argument("--samples", type=int, default=10)
    p_apply.add_argument("--timeout-sec", type=float, default=60.0)
    p_apply.add_argument("--baseline-mm", type=float, default=42.8)
    p_apply.add_argument("--square-mm", type=float, default=20.0)
    p_apply.add_argument("--preview", action="store_true")
    p_apply.add_argument("--debug-dir", default="calibration_debug")
    p_apply.add_argument("--no-write", action="store_true")
    p_apply.add_argument(
        "--disable-auto-align",
        action="store_true",
        help="Disable stereo.alignment.enabled after writing physical grid offsets",
    )

    args = parser.parse_args()
    if args.cmd == "print-pattern":
        out = generate_chessboard_page(
            args.output,
            inner_cols=args.cols,
            inner_rows=args.rows,
            square_mm=args.square_mm,
            dpi=args.dpi,
        )
        print(f"Wrote {out}")
        return 0

    cfg = load_config(args.config)
    left = CameraCapture(
        cfg.cameras.left.index,
        cfg.cameras.left.width,
        cfg.cameras.left.height,
        backend=cfg.cameras.backend,
        name="cal-left",
    ).start()
    right = CameraCapture(
        cfg.cameras.right.index,
        cfg.cameras.right.width,
        cfg.cameras.right.height,
        backend=cfg.cameras.backend,
        name="cal-right",
    ).start()

    if args.cmd == "preview":
        try:
            _preview_loop(left, right, cfg, args.cols, args.rows, Path(args.debug_dir))
        finally:
            left.stop()
            right.stop()
            cv2.destroyAllWindows()
        return 0

    left_frames = []
    right_frames = []
    debug_dir = Path(args.debug_dir)
    valid_count = 0
    last_status_print = 0.0
    last_seen_left = None
    last_seen_right = None
    try:
        deadline = time.monotonic() + args.timeout_sec
        last_ids = (-1, -1)
        while len(left_frames) < args.samples and time.monotonic() < deadline:
            fl, id_l = left.read_latest_no_copy()
            fr, id_r = right.read_latest_no_copy()
            if fl is None or fr is None or (id_l, id_r) == last_ids:
                time.sleep(0.005)
                continue
            last_ids = (id_l, id_r)
            if cfg.cameras.left.flip_180:
                fl = cv2.rotate(fl, cv2.ROTATE_180)
            if cfg.cameras.right.flip_180:
                fr = cv2.rotate(fr, cv2.ROTATE_180)
            last_seen_left = fl.copy()
            last_seen_right = fr.copy()

            det_l = detect_grid(fl, inner_cols=args.cols, inner_rows=args.rows)
            det_r = detect_grid(fr, inner_cols=args.cols, inner_rows=args.rows)
            if det_l is not None and det_r is not None:
                valid_count += 1
                left_frames.append(fl.copy())
                right_frames.append(fr.copy())

            now = time.monotonic()
            if now - last_status_print > 1.0:
                print(
                    f"Detection: left={'OK' if det_l else '--'} "
                    f"right={'OK' if det_r else '--'} "
                    f"valid={valid_count}/{args.samples}"
                )
                last_status_print = now

            if args.preview:
                overlay_l = draw_detection_overlay(fl, det_l, args.cols, args.rows)
                overlay_r = draw_detection_overlay(fr, det_r, args.cols, args.rows)
                preview = cv2.resize(cv2.hconcat([overlay_l, overlay_r]), (1280, 360))
                cv2.imshow("physical grid calibration", preview)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        left.stop()
        right.stop()
        if args.preview:
            cv2.destroyAllWindows()

    if not left_frames:
        debug_dir.mkdir(parents=True, exist_ok=True)
        if last_seen_left is not None:
            cv2.imwrite(str(debug_dir / "last_left_failed.png"), last_seen_left)
        if last_seen_right is not None:
            cv2.imwrite(str(debug_dir / "last_right_failed.png"), last_seen_right)
        print(
            "No valid stereo chessboard detections were captured. "
            f"Saved latest frames in {debug_dir}."
        )
        print("Run this setup view first:")
        print(
            f"  .\\.venv\\Scripts\\python .\\scripts\\physical_grid_calibration.py "
            f"preview --cols {args.cols} --rows {args.rows}"
        )
        return 2

    sx = (cfg.display.width / 2) / cfg.cameras.left.width
    sy = cfg.display.height / cfg.cameras.left.height
    result = estimate_right_eye_offset(
        left_frames,
        right_frames,
        inner_cols=args.cols,
        inner_rows=args.rows,
        output_scale=(sx, sy),
        baseline_mm=args.baseline_mm,
        square_mm=args.square_mm,
    )
    print(
        "Detected offset: "
        f"right x={result.nudge_right_x:+d}px, "
        f"right y={result.nudge_right_y:+d}px "
        f"from {result.samples} valid sample(s)"
    )
    if result.expected_baseline_dx is not None:
        print(
            "Baseline check: "
            f"{args.baseline_mm:g} mm lens spacing predicts about "
            f"{result.expected_baseline_dx:.1f}px horizontal disparity "
            "at the detected grid scale."
        )
        print(
            "Treat this as a plausibility check only; angled cameras change "
            "the measured disparity."
        )
    if not args.no_write:
        update_config_offsets(
            Path(args.config),
            result,
            disable_auto_align=args.disable_auto_align,
        )
        print(f"Updated {args.config}")
        if args.disable_auto_align:
            print("Disabled stereo.alignment.enabled to avoid double-correcting the grid offsets.")
    return 0


def _read_pair(left, right, cfg, last_ids):
    fl, id_l = left.read_latest_no_copy()
    fr, id_r = right.read_latest_no_copy()
    if fl is None or fr is None or (id_l, id_r) == last_ids:
        return None, None, last_ids
    if cfg.cameras.left.flip_180:
        fl = cv2.rotate(fl, cv2.ROTATE_180)
    if cfg.cameras.right.flip_180:
        fr = cv2.rotate(fr, cv2.ROTATE_180)
    return fl, fr, (id_l, id_r)


def _preview_loop(left, right, cfg, cols: int, rows: int, debug_dir: Path) -> None:
    print("Preview mode. Hold the printed grid so both cameras see it.")
    print("Keys: S saves current frames, ESC exits.")
    last_ids = (-1, -1)
    last_print = 0.0
    while True:
        fl, fr, last_ids = _read_pair(left, right, cfg, last_ids)
        if fl is None or fr is None:
            time.sleep(0.005)
            continue

        det_l = detect_grid(fl, inner_cols=cols, inner_rows=rows)
        det_r = detect_grid(fr, inner_cols=cols, inner_rows=rows)
        now = time.monotonic()
        if now - last_print > 1.0:
            print(f"Detection: left={'OK' if det_l else '--'} right={'OK' if det_r else '--'}")
            last_print = now

        overlay_l = draw_detection_overlay(fl, det_l, cols, rows)
        overlay_r = draw_detection_overlay(fr, det_r, cols, rows)
        preview = cv2.resize(cv2.hconcat([overlay_l, overlay_r]), (1280, 360))
        cv2.imshow("physical grid calibration setup", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return
        if key == ord("s"):
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / "setup_left.png"), fl)
            cv2.imwrite(str(debug_dir / "setup_right.png"), fr)
            print(f"Saved setup frames in {debug_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
