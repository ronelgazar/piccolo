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
    w = MainWindow(cfg)
    w.show()
    assert w.isVisible()
    central = w.centralWidget()
    assert central.count() == 3
    assert central.tabText(0) == "Live"
    assert central.tabText(1) == "Calibration"
    assert central.tabText(2) == "Settings"
    w.close()
