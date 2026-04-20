import os
import tempfile
import yaml
from src.config import PiccoloCfg, load_config
from src.config_state import save_calibration_state


def _write_yaml(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)


def test_load_missing_calibration_state_uses_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        _write_yaml(path, {"display": {"width": 1920}})
        cfg = load_config(path)
        assert cfg.calibration_state.nudge_left_x == 0
        assert cfg.calibration_state.joint_zoom_center == 50


def test_load_populates_calibration_state():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        _write_yaml(path, {
            "calibration_state": {
                "nudge_left_x": 3,
                "nudge_right_y": -5,
                "joint_zoom_center": 42,
            }
        })
        cfg = load_config(path)
        assert cfg.calibration_state.nudge_left_x == 3
        assert cfg.calibration_state.nudge_right_y == -5
        assert cfg.calibration_state.joint_zoom_center == 42
        assert cfg.calibration_state.nudge_left_y == 0


def test_save_calibration_state_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        _write_yaml(path, {"display": {"width": 1920}})
        cfg = load_config(path)
        cfg.calibration_state.nudge_left_x = 7
        cfg.calibration_state.nudge_right_y = -2
        cfg.calibration_state.convergence_offset = 12
        save_calibration_state(cfg, path)
        cfg2 = load_config(path)
        assert cfg2.calibration_state.nudge_left_x == 7
        assert cfg2.calibration_state.nudge_right_y == -2
        assert cfg2.calibration_state.convergence_offset == 12
        assert cfg2.display.width == 1920


def test_save_does_not_clobber_other_keys():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        _write_yaml(path, {
            "cameras": {"left": {"index": 5, "flip_180": True}},
            "stereo": {"zoom": {"step": 0.07}},
        })
        cfg = load_config(path)
        cfg.calibration_state.nudge_left_x = 9
        save_calibration_state(cfg, path)
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        assert raw["cameras"]["left"]["index"] == 5
        assert raw["cameras"]["left"]["flip_180"] is True
        assert raw["stereo"]["zoom"]["step"] == 0.07
        assert raw["calibration_state"]["nudge_left_x"] == 9
