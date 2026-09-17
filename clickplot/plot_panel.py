"""One plot slot: its PlotWidget, loaded dataset, render mode, and redraw logic."""
from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox

from .constants import DEBOUNCE_MS, DEFAULT_COLUMN_TARGET, PLOT_COLORS
from .dataset import Dataset, DatasetLoadError
from .decimation import bar_spec, decimate_for_view, dot_xy, line_xy
from .time_axis import MicrosecondAxisItem


class _TimeSelectViewBox(pg.ViewBox):
    """A ViewBox with wheel-zoom locked to X, and Ctrl+left-drag time selection.

    - Scroll-wheel always zooms X only: normal ViewBoxes zoom whichever axis
      the wheel event applies to (both, over the plot area; just Y, over the
      Y-axis label). Forcing axis=0 makes the wheel-scale mask only ever
      include X, regardless of where the cursor is.
    - Plain left-drag pans normally (both X and Y, standard PanMode).
    - Ctrl+left-drag instead draws a selection rectangle; on release the X
      range zooms to the selected span while Y is left exactly as it was
      (showAxRect is overridden to drop the rect's Y component).
    """

    def wheelEvent(self, ev, axis=None):
        super().wheelEvent(ev, axis=0)

    def showAxRect(self, ax, **kwargs):
        ax = ax.normalized()
        y0, y1 = self.viewRange()[1]
        self.setRange(xRange=(ax.left(), ax.right()), yRange=(y0, y1), padding=0, **kwargs)
        self.sigRangeChangedManually.emit(self.state["mouseEnabled"])

    def mouseDragEvent(self, ev, axis=None):
        select_time = (
            axis is None
            and ev.button() == Qt.MouseButton.LeftButton
            and bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
        )
        previous_mode = self.state["mouseMode"]
        self.state["mouseMode"] = pg.ViewBox.RectMode if select_time else pg.ViewBox.PanMode
        try:
            super().mouseDragEvent(ev, axis=axis)
        finally:
            self.state["mouseMode"] = previous_mode


class RenderMode(Enum):
    DOT = "Dot"
    LINE = "Line"
    BAR = "Bar"


class PlotSlotState:
    """Owns one of the six vertically-stacked plots and its independent state."""

    def __init__(
        self,
        index: int,
        show_x_labels: bool = True,
        on_dataset_loaded: Optional[Callable[["PlotSlotState", Dataset], None]] = None,
        on_hover: Optional[Callable[["PlotSlotState", float, Optional[float]], None]] = None,
    ):
        self.index = index
        self.on_dataset_loaded = on_dataset_loaded
        self.on_hover = on_hover
        self.dataset: Optional[Dataset] = None
        self.render_mode = RenderMode.LINE
        self.active_item = None
        self.color = PLOT_COLORS[index % len(PLOT_COLORS)]

        self.plot_widget = pg.PlotWidget(
            viewBox=_TimeSelectViewBox(),
            axisItems={"bottom": MicrosecondAxisItem(orientation="bottom")},
        )
        self.plot_widget.setMinimumHeight(80)
        self.plot_widget.setLabel("left", f"Plot {index + 1}")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        if not show_x_labels:
            self.plot_widget.getAxis("bottom").setStyle(showValues=False)

        view_box = self.plot_widget.getPlotItem().getViewBox()
        view_box.setMenuEnabled(False)
        view_box.setMouseEnabled(x=True, y=True)
        view_box.setMouseMode(pg.ViewBox.PanMode)
        view_box.sigXRangeChanged.connect(self._on_view_changed)

        self.plot_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.plot_widget.customContextMenuRequested.connect(self._show_context_menu)

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.redraw)

        self._hover_proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse_moved
        )

    # -- data loading ----------------------------------------------------

    def load_dataset(self, path: str) -> Dataset:
        """Load a dataset into this slot. Raises DatasetLoadError on failure."""
        dataset = Dataset.load(path)
        self.dataset = dataset
        self._clear_active_item()
        self._fit_y_range()
        self.redraw()
        return dataset

    def _fit_y_range(self) -> None:
        if self.dataset is None:
            return
        vmin = float(self.dataset.v.min())
        vmax = float(self.dataset.v.max())
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0
        self.plot_widget.setYRange(vmin, vmax, padding=0.05)

    # -- hover / status reporting --------------------------------------

    def _on_mouse_moved(self, evt) -> None:
        pos = evt[0]
        if self.on_hover is None or not self.plot_widget.sceneBoundingRect().contains(pos):
            return
        view_box = self.plot_widget.getPlotItem().getViewBox()
        mouse_point = view_box.mapSceneToView(pos)
        x = mouse_point.x()

        value = None
        if self.dataset is not None and self.dataset.n_points > 0:
            t_sec = self.dataset.t_sec
            idx = int(np.searchsorted(t_sec, x))
            idx = min(max(idx, 0), self.dataset.n_points - 1)
            if idx > 0 and abs(t_sec[idx - 1] - x) < abs(t_sec[idx] - x):
                idx -= 1
            x = float(t_sec[idx])
            value = float(self.dataset.v[idx])

        self.on_hover(self, x, value)

    # -- render mode -------------------------------------------------------

    def set_render_mode(self, mode: RenderMode) -> None:
        if mode == self.render_mode:
            return
        self.render_mode = mode
        self._clear_active_item()
        self.redraw()

    def _clear_active_item(self) -> None:
        if self.active_item is not None:
            self.plot_widget.removeItem(self.active_item)
            self.active_item = None

    # -- redraw / decimation ------------------------------------------------

    def _on_view_changed(self, *_args) -> None:
        self._debounce_timer.start(DEBOUNCE_MS)

    def _visible_columns(self) -> int:
        view_box = self.plot_widget.getPlotItem().getViewBox()
        width = int(view_box.width())
        return width if width > 1 else DEFAULT_COLUMN_TARGET

    def redraw(self) -> None:
        if self.dataset is None:
            return
        view_box = self.plot_widget.getPlotItem().getViewBox()
        (x0, x1), _ = view_box.viewRange()
        n_columns = self._visible_columns()

        data = decimate_for_view(self.dataset.t_sec, self.dataset.v, x0, x1, n_columns)

        if self.render_mode is RenderMode.LINE:
            x, y = line_xy(data)
            item = self._ensure_line_item()
            item.setData(x, y, connect="all")
        elif self.render_mode is RenderMode.DOT:
            x, y = dot_xy(data)
            item = self._ensure_dot_item()
            item.setData(x=x, y=y)
        else:  # BAR
            spec = bar_spec(data, x0, x1, n_columns)
            if spec.as_bars:
                self._set_bar_item(spec.x, spec.y, spec.width)
            else:
                item = self._ensure_line_item()
                item.setData(spec.x, spec.y, connect="all")

    def _ensure_line_item(self) -> pg.PlotDataItem:
        if not isinstance(self.active_item, pg.PlotDataItem):
            self._clear_active_item()
            self.active_item = pg.PlotDataItem(pen=pg.mkPen(color=self.color, width=1))
            self.plot_widget.addItem(self.active_item)
        return self.active_item

    def _ensure_dot_item(self) -> pg.ScatterPlotItem:
        if not isinstance(self.active_item, pg.ScatterPlotItem):
            self._clear_active_item()
            self.active_item = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(*self.color, 200))
            self.plot_widget.addItem(self.active_item)
        return self.active_item

    def _set_bar_item(self, x, height, width) -> None:
        # BarGraphItem arrays aren't cheaply resizable in place across
        # pyqtgraph versions; rebuild it each redraw (only happens after the
        # debounce interval, not per-frame).
        self._clear_active_item()
        self.active_item = pg.BarGraphItem(x=x, height=height, width=width, brush=pg.mkBrush(*self.color, 200))
        self.plot_widget.addItem(self.active_item)

    # -- context menu --------------------------------------------------

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self.plot_widget)
        load_action = menu.addAction("Load Dataset...")
        style_menu = menu.addMenu("Plot Style")
        group = QActionGroup(style_menu)
        group.setExclusive(True)
        style_actions = {}
        for mode in RenderMode:
            action = style_menu.addAction(mode.value)
            action.setCheckable(True)
            action.setChecked(mode is self.render_mode)
            group.addAction(action)
            style_actions[action] = mode

        chosen = menu.exec(self.plot_widget.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is load_action:
            self._prompt_load_dataset()
        elif chosen in style_actions:
            self.set_render_mode(style_actions[chosen])

    def _prompt_load_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.plot_widget, f"Load Dataset for Plot {self.index + 1}", "", "Datasets (*.npz *.wav)"
        )
        if not path:
            return
        try:
            dataset = self.load_dataset(path)
        except DatasetLoadError as exc:
            QMessageBox.warning(self.plot_widget, "Failed to load dataset", str(exc))
            return
        if self.on_dataset_loaded is not None:
            self.on_dataset_loaded(self, dataset)
