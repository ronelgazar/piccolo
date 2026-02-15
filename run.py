#!/usr/bin/env python3
"""Piccolo – Stereoscopic Surgery Display.

Usage
-----
    python run.py                  # use config.yaml in the project root
    python run.py --config my.yaml # custom config
    python run.py --test           # synthetic test-pattern mode (no cameras)
"""

from __future__ import annotations

import argparse
import sys
import os

# Ensure the project root is on the path so ``src`` can be imported.
sys.path.insert(0, os.path.dirname(__file__))

from src.config import load_config
from src.app import PiccoloApp


def main():
    parser = argparse.ArgumentParser(description="Piccolo – Stereoscopic Surgery Display")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--test", action="store_true", help="Use test patterns instead of cameras")
    parser.add_argument("--windowed", action="store_true", help="Run in a window (not fullscreen)")
    parser.add_argument("--list-monitors", action="store_true", help="List connected monitors and exit")
    parser.add_argument("--monitor", type=str, default=None,
                        help="Monitor index (0,1,...) or 'auto' to target Goovis")
    args = parser.parse_args()

    if args.list_monitors:
        from src.display import _list_monitors, _find_goovis
        monitors = _list_monitors()
        print(f"Detected {len(monitors)} monitor(s):")
        for i, m in enumerate(monitors):
            tag = " [PRIMARY]" if m["is_primary"] else ""
            print(f"  #{i}: {m['name']}  {m['width']}x{m['height']}  "
                  f"at ({m['x']},{m['y']}){tag}")
        goovis = _find_goovis(monitors)
        if goovis:
            print(f"\nGoovis detected: {goovis['name']}  {goovis['width']}x{goovis['height']}")
        else:
            print("\nGoovis not detected.  Is it set to 'Extend' mode in Windows Display settings?")
        return

    cfg = load_config(args.config)
    if args.test:
        cfg.cameras.test_mode = True
    if args.windowed:
        cfg.display.fullscreen = False
    if args.monitor is not None:
        cfg.display.monitor = int(args.monitor) if args.monitor.isdigit() else args.monitor

    app = PiccoloApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
