"""click-plot main window: menu bar, toolbar, 6 linked plots, status bar."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QMainWindow, QSplitter, QStatusBar

from .constants import DEBOUNCE_MS, NUM_PLOTS
from .dataset import Dataset
from .plot_panel import PlotSlotState

STEP_FRACTION = 0.8  # step forward/back by this fraction of the current window width


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("click-plot")
        self.resize(1400, 900)

        self.menuBar()  # present, empty for now

        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("MainToolBar")

        reset_action = toolbar.addAction("Reset Full Range")
        reset_action.triggered.connect(self._reset_time_range)

        back_action = toolbar.addAction("◀ Step Back")
        back_action.triggered.connect(lambda: self._step_time(-1))

        forward_action = toolbar.addAction("Step Forward ▶")
        forward_action.triggered.connect(lambda: self._step_time(1))

        self._x_range_initialized = False

        self.panels: list[PlotSlotState] = []
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        for i in range(NUM_PLOTS):
            panel = PlotSlotState(
                index=i,
                show_x_labels=(i == NUM_PLOTS - 1),
                on_dataset_loaded=self._on_dataset_loaded,
                on_hover=self._on_hover,
            )
            self.panels.append(panel)
            splitter.addWidget(panel.plot_widget)
        self.setCentralWidget(splitter)

        master = self.panels[0].plot_widget
        for panel in self.panels[1:]:
            panel.plot_widget.setXLink(master)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Right-click a plot to load a dataset.")

        self._resize_debounce = QTimer()
        self._resize_debounce.setSingleShot(True)
        self._resize_debounce.timeout.connect(self._redraw_all)

    def _on_dataset_loaded(self, panel: PlotSlotState, dataset: Dataset) -> None:
        if not self._x_range_initialized:
            master = self.panels[0].plot_widget
            span = dataset.t_sec[-1] - dataset.t_sec[0]
            padding = span * 0.02 if span > 0 else 1.0
            master.setXRange(dataset.t_sec[0] - padding, dataset.t_sec[-1] + padding, padding=0)
            self._x_range_initialized = True
        self.statusBar().showMessage(
            f"Loaded '{dataset.name}' into Plot {panel.index + 1} ({dataset.n_points:,} points)"
        )

    def _on_hover(self, panel: PlotSlotState, t_sec: float, value: Optional[float]) -> None:
        t_us = int(round(t_sec * 1_000_000))
        dt = np.datetime64(t_us, "us")
        if value is None:
            self.statusBar().showMessage(f"Plot {panel.index + 1}: t = {dt}  (no dataset loaded)")
        else:
            self.statusBar().showMessage(f"Plot {panel.index + 1}: t = {dt}   value = {value:.6g}")

    def _reset_time_range(self) -> None:
        loaded = [panel.dataset for panel in self.panels if panel.dataset is not None]
        if not loaded:
            return
        t_min = min(ds.t_sec[0] for ds in loaded)
        t_max = max(ds.t_sec[-1] for ds in loaded)
        self.panels[0].plot_widget.setXRange(t_min, t_max, padding=0.02)

    def _step_time(self, direction: int) -> None:
        master = self.panels[0].plot_widget
        (x0, x1), _ = master.getPlotItem().getViewBox().viewRange()
        shift = STEP_FRACTION * (x1 - x0) * direction
        master.setXRange(x0 + shift, x1 + shift, padding=0)

    def _redraw_all(self) -> None:
        for panel in self.panels:
            panel.redraw()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        super().resizeEvent(event)
        self._resize_debounce.start(DEBOUNCE_MS)
