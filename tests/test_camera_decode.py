"""TurboJPEG fast-path falls back to OpenCV gracefully."""
from __future__ import annotations


def test_turbojpeg_decode_backend_constructs_without_error():
    from src.camera import CameraCapture

    cap = CameraCapture(
        index=0,
        width=640,
        height=480,
        fps=60,
        backend="opencv",
        name="test-cam",
    )
    cap.decode_backend = "turbojpeg"
    assert hasattr(cap, "_try_init_turbojpeg")
    cap._try_init_turbojpeg()
    assert cap._tj is None or hasattr(cap._tj, "decode")
