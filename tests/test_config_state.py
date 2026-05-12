import os
import sys
import tempfile
import yaml
from src.config import PiccoloCfg, SmartOverlapCfg, bundled_config_path, default_config_path, load_config
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
        assert cfg.calibration_state.scale_left_pct == 100
        assert cfg.calibration_state.scale_right_pct == 100


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


def test_default_config_path_uses_exe_directory_when_frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\App\piccolo.exe")

    assert default_config_path() == r"C:\App\config.yaml"


def test_load_config_uses_bundled_config_when_external_missing(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        bundle = os.path.join(tmp, "bundle")
        exe_dir = os.path.join(tmp, "app")
        os.makedirs(bundle)
        os.makedirs(exe_dir)
        _write_yaml(os.path.join(bundle, "config.yaml"), {"display": {"width": 1234}})
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", bundle, raising=False)
        monkeypatch.setattr(sys, "executable", os.path.join(exe_dir, "piccolo.exe"))

        cfg = load_config()

        assert bundled_config_path() == os.path.join(bundle, "config.yaml")
        assert cfg.display.width == 1234


def test_save_calibration_state_creates_complete_external_config_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        cfg = PiccoloCfg()
        cfg.calibration_state.nudge_right_x = 11

        save_calibration_state(cfg, path)

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        assert raw["display"]["width"] == cfg.display.width
        assert raw["stereo"]["aspect_mode"] == cfg.stereo.aspect_mode
        assert raw["calibration_state"]["nudge_right_x"] == 11


def test_smart_overlap_defaults_when_missing(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("display:\n  width: 1920\n", encoding="utf-8")
    cfg = load_config(str(p))
    assert isinstance(cfg.smart_overlap, SmartOverlapCfg)
    assert cfg.smart_overlap.default_mode == "chessboard"
    assert cfg.smart_overlap.pair_count == 8
    assert cfg.smart_overlap.max_vert_dy_px == 5.0
    assert cfg.smart_overlap.max_rotation_deg == 0.5
    assert cfg.smart_overlap.max_zoom_ratio_err == 0.02


def test_smart_overlap_overrides_from_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "smart_overlap:\n"
        "  default_mode: live\n"
        "  pair_count: 12\n"
        "  max_vert_dy_px: 3.5\n",
        encoding="utf-8",
    )
    cfg = load_config(str(p))
    assert cfg.smart_overlap.default_mode == "live"
    assert cfg.smart_overlap.pair_count == 12
    assert cfg.smart_overlap.max_vert_dy_px == 3.5


def test_smart_overlap_mode_and_pair_count_persist(tmp_path):
    p = tmp_path / "config.yaml"
    cfg = load_config(str(p))
    cfg.calibration_state.smart_overlap_mode = "live"
    cfg.calibration_state.smart_overlap_pair_count = 12
    save_calibration_state(cfg, path=str(p))

    cfg2 = load_config(str(p))
    assert cfg2.calibration_state.smart_overlap_mode == "live"
    assert cfg2.calibration_state.smart_overlap_pair_count == 12


def test_performance_config_has_use_gpu_pipeline_default():
    from src.config import PerformanceCfg

    cfg = PerformanceCfg()
    assert hasattr(cfg, "use_gpu_pipeline"), "PerformanceCfg missing use_gpu_pipeline"
    assert cfg.use_gpu_pipeline is True, "default should be True"


def test_performance_config_use_gpu_pipeline_loads_from_yaml(tmp_path):
    import yaml
    from src.config import load_config

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.safe_dump({"performance": {"use_gpu_pipeline": False}}))
    cfg = load_config(str(yaml_path))
    assert cfg.performance.use_gpu_pipeline is False
