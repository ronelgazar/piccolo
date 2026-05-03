"""Main Qt window: tabbed assistant UI + pipeline worker."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QMainWindow, QTabWidget

from ..config import PiccoloCfg
from .live_tab import LiveTab
from .calibration_tab import CalibrationTab
from .settings_tab import SettingsTab
from .pipeline_worker import PipelineWorker
from .goovis_window import GoovisWindow


class MainWindow(QMainWindow):
    def __init__(self, cfg: PiccoloCfg, start_worker: bool = True):
        super().__init__()
        self.cfg = cfg
        self._overlay_to_goovis = False
        self._smart_overlap_to_goovis = False
        self.setWindowTitle("Piccolo")
        self.resize(1280, 800)

        self.worker = PipelineWorker(cfg, self)

        self.tabs = QTabWidget(self)
        self.live_tab = LiveTab(self.worker, self)
        self.calibration_tab = CalibrationTab(self.worker, self)
        self.settings_tab = SettingsTab(self.worker, self)
        self.tabs.addTab(self.live_tab, "Live")
        self.tabs.addTab(self.calibration_tab, "Calibration")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)
        self._on_tab_changed(self.tabs.currentIndex())

        # Goovis output window (optional)
        self.goovis: GoovisWindow | None = GoovisWindow(cfg.display)
        if self.goovis.show_on_goovis():
            self.worker.sbs_qimage_ready.connect(self._on_goovis_worker_frame)
            if hasattr(self.calibration_tab, "overlay_mode_changed"):
                self.calibration_tab.overlay_mode_changed.connect(self._on_overlay_mode_changed)
            if hasattr(self.calibration_tab, "overlay_frame_ready"):
                self.calibration_tab.overlay_frame_ready.connect(self._on_goovis_overlay_frame)
            if hasattr(self.calibration_tab, "smart_overlap_mode_changed"):
                self.calibration_tab.smart_overlap_mode_changed.connect(self._on_smart_overlap_mode_changed)
            if hasattr(self.calibration_tab, "smart_overlap_frame_ready"):
                self.calibration_tab.smart_overlap_frame_ready.connect(self._on_goovis_smart_overlap_frame)
        else:
            self.goovis.deleteLater()
            self.goovis = None

        if start_worker:
            self.worker.start()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not event.isAutoRepeat():
            self.worker.input.on_key_down(self._qt_key_name(event))
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if not event.isAutoRepeat():
            self.worker.input.on_key_up(self._qt_key_name(event))
        super().keyReleaseEvent(event)

    @staticmethod
    def _qt_key_name(event: QKeyEvent) -> str:
        txt = event.text().lower()
        if txt and txt.isprintable() and len(txt) == 1:
            return txt
        key = event.key()
        return Qt.Key(key).name.removeprefix("Key_").lower() if key else ""

    def closeEvent(self, event) -> None:
        if self.goovis is not None:
            self.goovis.close()
        self.calibration_tab.stop_background_work()
        if self.worker.isRunning():
            self.worker.stop()
        super().closeEvent(event)

    def _on_overlay_mode_changed(self, active: bool) -> None:
        self._overlay_to_goovis = active

    def _on_smart_overlap_mode_changed(self, active: bool) -> None:
        self._smart_overlap_to_goovis = active

    def _on_tab_changed(self, idx: int) -> None:
        self.worker.raw_frame_requested = self.tabs.widget(idx) is self.calibration_tab

    def _on_goovis_worker_frame(self, image) -> None:
        if self.goovis is None or self._overlay_to_goovis or self._smart_overlap_to_goovis:
            return
        self.goovis.video.set_frame(image)

    def _on_goovis_overlay_frame(self, image) -> None:
        if self.goovis is None or not self._overlay_to_goovis:
            return
        self.goovis.video.set_frame(image)

    def _on_goovis_smart_overlap_frame(self, image) -> None:
        if self.goovis is None or not self._smart_overlap_to_goovis:
            return
        self.goovis.video.set_frame(image)
