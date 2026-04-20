"""Live tab: SBS preview, zoom-center arrow pad, annotations, bottom status strip."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QStackedLayout, QPushButton,
    QLabel, QGroupBox,
)

from .video_widget import VideoWidget
from .annotation_overlay_widget import AnnotationCanvas


class LiveTab(QWidget):
    def __init__(self, worker: PipelineWorker, parent: QWidget | None = None):
        super().__init__(parent)
        self.worker = worker

        root = QVBoxLayout(self)
        main_layout = QHBoxLayout()
        root.addLayout(main_layout, stretch=1)

        # Preview stack: video + annotation canvas on top
        preview_wrap = QWidget(self)
        stack = QStackedLayout(preview_wrap)
        stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self.preview = VideoWidget(preview_wrap)
        self.canvas = AnnotationCanvas(preview_wrap)
        stack.addWidget(self.preview)
        stack.addWidget(self.canvas)
        main_layout.addWidget(preview_wrap, stretch=4)

        # Side panel (right)
        side = QVBoxLayout()
        main_layout.addLayout(side, stretch=1)
        side.addWidget(self._make_zoom_pad())
        side.addWidget(self._make_annotation_group())
        side.addStretch(1)

        # Status strip (bottom)
        root.addWidget(self._make_status_strip())

        # Wire signals
        if worker is not None:
            worker.sbs_qimage_ready.connect(self.preview.set_frame)
            worker.status_tick.connect(self._on_status)

    # ------------------------------------------------------------------

    def _make_zoom_pad(self) -> QGroupBox:
        box = QGroupBox("Zoom Center", self)
        grid = QGridLayout(box)
        btn_up = QPushButton("↑")
        btn_down = QPushButton("↓")
        btn_left = QPushButton("←")
        btn_right = QPushButton("→")
        for b in (btn_up, btn_down, btn_left, btn_right):
            b.setFixedSize(48, 48)
            b.setStyleSheet("font-size: 18px;")
        grid.addWidget(btn_up,    0, 1)
        grid.addWidget(btn_left,  1, 0)
        grid.addWidget(btn_right, 1, 2)
        grid.addWidget(btn_down,  2, 1)
        btn_up.clicked.connect(lambda: self._nudge_center_y(-1))
        btn_down.clicked.connect(lambda: self._nudge_center_y(+1))
        btn_left.clicked.connect(lambda: self._nudge_center_x(-1))
        btn_right.clicked.connect(lambda: self._nudge_center_x(+1))
        return box

    def _make_annotation_group(self) -> QGroupBox:
        box = QGroupBox("Annotations", self)
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        btn_undo = QPushButton("Undo")
        btn_clear = QPushButton("Clear")
        btn_undo.clicked.connect(self.canvas.undo)
        btn_clear.clicked.connect(self.canvas.clear)
        row.addWidget(btn_undo); row.addWidget(btn_clear)
        lay.addLayout(row)
        return box

    def _make_status_strip(self) -> QWidget:
        strip = QWidget(self)
        lay = QHBoxLayout(strip)
        lay.setContentsMargins(8, 4, 8, 4)
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_align = QLabel("Align: --")
        self.lbl_pedal = QLabel("Pedal: OFF")
        for lbl in (self.lbl_fps, self.lbl_align, self.lbl_pedal):
            lbl.setStyleSheet("font-family: monospace;")
        lay.addWidget(self.lbl_fps)
        lay.addSpacing(20)
        lay.addWidget(self.lbl_align)
        lay.addStretch(1)
        lay.addWidget(self.lbl_pedal)
        return strip

    # ------------------------------------------------------------------

    def _nudge_center_x(self, delta: int) -> None:
        if self.worker is None:
            return
        p = self.worker.processor
        if hasattr(p, "joint_zoom_center"):
            p.set_joint_zoom_center(p.joint_zoom_center + delta)

    def _nudge_center_y(self, delta: int) -> None:
        if self.worker is None:
            return
        p = self.worker.processor
        if hasattr(p, "joint_zoom_center_y"):
            p.set_joint_zoom_center_y(p.joint_zoom_center_y + delta)

    def _on_status(self, st: dict) -> None:
        self.lbl_fps.setText(f"FPS: {st['fps']:.0f}")
        dy = st["dy"]
        dt = st["dtheta_deg"]
        conv = "ok" if st["aligner_converged"] else "calibrating"
        self.lbl_align.setText(f"Align: dy={dy:+.1f}px rot={dt:+.2f}° ({conv})")
        mode = st["pedal_mode"]
        mode_names = {"a": "ZOOM", "b": "SIDE", "c": "UP/DOWN"}
        self.lbl_pedal.setText(f"Pedal: {mode_names.get(mode, 'OFF')}")
