import pytest
from PyQt6.QtWidgets import QApplication
from src.config import load_config
from src.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_opens(qapp):
    cfg = load_config()
    cfg.cameras.test_mode = True
    # Don't start the pipeline worker in tests — we only verify the UI shell
    w = MainWindow(cfg, start_worker=False)
    w.show()
    assert w.isVisible()
    central = w.centralWidget()
    assert central.count() == 3
    assert central.tabText(0) == "Live"
    assert central.tabText(1) == "Calibration"
    assert central.tabText(2) == "Settings"
    assert w.worker.raw_frame_requested is False
    central.setCurrentIndex(1)
    assert w.worker.raw_frame_requested is True
    central.setCurrentIndex(0)
    assert w.worker.raw_frame_requested is False
    w.close()
