"""One plot slot: its PlotWidget, loaded dataset, overlaid series, and redraw logic."""
from __future__ import annotations

import html
from enum import Enum
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox

from .constants import DEBOUNCE_MS, DEFAULT_COLUMN_TARGET, series_color
from .dataset import Dataset, DatasetLoadError
from .decimation import bar_spec, decimate_series_for_view, dot_xy, line_xy
from .time_axis import MicrosecondAxisItem


class _TimeSelectViewBox(pg.ViewBox):
    """A ViewBox with wheel-zoom locked to X, and Ctrl+left-drag time selection.

    - Scroll-wheel always zooms X only: normal ViewBoxes zoom whichever axis
      the wheel event applies to (both, over the plot area; just Y, over the
      Y-axis label). Forcing axis=0 makes the wheel-scale mask only ever
      include X, regardless of where the cursor is.
    - Plain left-drag pans X only (vertical panning is disabled entirely via
      setMouseEnabled(y=False) in PlotSlotState).
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


class SeriesState:
    """One overlaid value series within a plot: its own style, visibility, and graphics item."""

    def __init__(self, name: str, v: np.ndarray, color: tuple, render_mode: RenderMode = RenderMode.DOT):
        self.name = name
        self.v = v
        self.color = color
        self.render_mode = render_mode
        self.visible = True
        self.item = None


class PlotSlotState:
    """Owns one of the six vertically-stacked plots and its independent state."""

    def __init__(
        self,
        index: int,
        show_x_labels: bool = True,
        on_dataset_loaded: Optional[Callable[["PlotSlotState", Dataset, bool], None]] = None,
        on_hover: Optional[Callable[["PlotSlotState", Optional[float], list], None]] = None,
    ):
        self.index = index
        self.on_dataset_loaded = on_dataset_loaded
        self.on_hover = on_hover
        self.dataset: Optional[Dataset] = None
        self.series: list[SeriesState] = []

        self.plot_widget = pg.PlotWidget(
            viewBox=_TimeSelectViewBox(),
            axisItems={"bottom": MicrosecondAxisItem(orientation="bottom")},
        )
        self.plot_widget.setMinimumHeight(80)
        self.plot_widget.setLabel("left", f"Plot {index + 1}")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        if not show_x_labels:
            self.plot_widget.getAxis("bottom").setStyle(showValues=False)

        self.legend = self.plot_widget.addLegend()

        view_box = self.plot_widget.getPlotItem().getViewBox()
        view_box.setMenuEnabled(False)
        view_box.setMouseEnabled(x=True, y=False)
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
        """Load a dataset into this slot, replacing any current one. Raises DatasetLoadError on failure."""
        dataset = Dataset.load(path)
        self._rebuild_series(dataset, prior=None)
        return dataset

    def reload_dataset(self) -> Dataset:
        """Re-read the currently-loaded file from disk, preserving per-series settings where names match."""
        if self.dataset is None:
            raise DatasetLoadError("No dataset loaded in this plot.")
        dataset = Dataset.load(self.dataset.path)  # build before tearing down; failure leaves plot untouched
        prior = {s.name: (s.render_mode, s.visible, s.color) for s in self.series}
        self._rebuild_series(dataset, prior=prior)
        return dataset

    def _rebuild_series(self, dataset: Dataset, prior: Optional[dict] = None) -> None:
        for series in self.series:
            if series.item is not None:
                self.plot_widget.removeItem(series.item)
        self.legend.clear()

        self.dataset = dataset
        self.series = []
        n = len(dataset.series)
        for i, s in enumerate(dataset.series):
            render_mode, visible, color = (prior or {}).get(s.name, (RenderMode.DOT, True, series_color(i, n)))
            state = SeriesState(s.name, s.v, color, render_mode)
            state.visible = visible
            self.series.append(state)

        self.plot_widget.setLabel("left", dataset.name)
        self._fit_y_range()
        self.redraw()

    def _fit_y_range(self) -> None:
        if not self.series:
            return
        vmin = min(float(s.v.min()) for s in self.series)
        vmax = max(float(s.v.max()) for s in self.series)
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

        if self.dataset is None or self.dataset.n_points == 0:
            self.on_hover(self, None, [])
            return

        t_sec = self.dataset.t_sec
        idx = int(np.searchsorted(t_sec, x))
        idx = min(max(idx, 0), self.dataset.n_points - 1)
        if idx > 0 and abs(t_sec[idx - 1] - x) < abs(t_sec[idx] - x):
            idx -= 1
        nearest_t = float(t_sec[idx])
        values = [(s.name, float(s.v[idx])) for s in self.series if s.visible]
        self.on_hover(self, nearest_t, values)

    # -- per-series visibility / render mode -----------------------------

    def _toggle_series_visible(self, series: SeriesState) -> None:
        series.visible = not series.visible
        if not series.visible:
            self._clear_series_item(series)
        self.redraw()

    def _set_series_render_mode(self, series: SeriesState, mode: RenderMode) -> None:
        if mode == series.render_mode:
            return
        series.render_mode = mode
        self._clear_series_item(series)
        self.redraw()

    def _clear_series_item(self, series: SeriesState) -> None:
        if series.item is not None:
            self.plot_widget.removeItem(series.item)
            self.legend.removeItem(series.item)
            series.item = None

    # -- redraw / decimation ------------------------------------------------

    def _on_view_changed(self, *_args) -> None:
        self._debounce_timer.start(DEBOUNCE_MS)

    def _visible_columns(self) -> int:
        view_box = self.plot_widget.getPlotItem().getViewBox()
        width = int(view_box.width())
        return width if width > 1 else DEFAULT_COLUMN_TARGET

    def redraw(self) -> None:
        if self.dataset is None or not self.series:
            return
        view_box = self.plot_widget.getPlotItem().getViewBox()
        (x0, x1), _ = view_box.viewRange()
        n_columns = self._visible_columns()

        visible_data_list = decimate_series_for_view(
            self.dataset.t_sec, [s.v for s in self.series], x0, x1, n_columns
        )
        for series, vd in zip(self.series, visible_data_list):
            if series.visible:
                self._ensure_series_item(series, vd, x0, x1, n_columns)
            else:
                self._clear_series_item(series)

    def _ensure_series_item(self, series: SeriesState, vd, x0: float, x1: float, n_columns: int) -> None:
        """Create, update in place, or swap the graphics item for one series.

        legend.addItem/removeItem are only called on actual item creation or
        type-swap -- never on a same-type data update -- so redraws (which
        happen every debounce tick during pan/zoom) never duplicate legend rows.
        """
        if series.render_mode is RenderMode.LINE:
            x, y = line_xy(vd)
            kind = pg.PlotDataItem

            def apply(item):
                item.setData(x, y, connect="all")

            def make():
                return pg.PlotDataItem(pen=pg.mkPen(color=series.color, width=1))

        elif series.render_mode is RenderMode.DOT:
            x, y = dot_xy(vd)
            kind = pg.ScatterPlotItem

            def apply(item):
                item.setData(x=x, y=y)

            def make():
                return pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(*series.color, 200))

        else:  # BAR
            spec = bar_spec(vd, x0, x1, n_columns)
            if spec.as_bars:
                kind = pg.BarGraphItem

                def apply(item, spec=spec):
                    item.setOpts(x=spec.x, height=spec.y, width=spec.width)

                def make(spec=spec):
                    return pg.BarGraphItem(x=spec.x, height=spec.y, width=spec.width, brush=pg.mkBrush(*series.color, 150))

            else:
                kind = pg.PlotDataItem

                def apply(item, spec=spec):
                    item.setData(spec.x, spec.y, connect="all")

                def make():
                    return pg.PlotDataItem(pen=pg.mkPen(color=series.color, width=1))

        if series.item is not None and type(series.item) is kind:
            apply(series.item)
            return

        if series.item is not None:
            self.plot_widget.removeItem(series.item)
            self.legend.removeItem(series.item)
            series.item = None

        series.item = make()
        apply(series.item)
        self.plot_widget.addItem(series.item)
        self.legend.addItem(series.item, html.escape(series.name))

    # -- context menu --------------------------------------------------

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self.plot_widget)
        load_action = menu.addAction("Load Dataset...")
        reload_action = menu.addAction("Reload Data")
        reload_action.setEnabled(self.dataset is not None)

        action_map = {}
        if self.series:
            menu.addSeparator()
            for series in self.series:
                series_menu = menu.addMenu(series.name.replace("&", "&&"))

                visible_action = series_menu.addAction("Visible")
                visible_action.setCheckable(True)
                visible_action.setChecked(series.visible)
                action_map[visible_action] = ("visible", series)

                style_menu = series_menu.addMenu("Style")
                group = QActionGroup(style_menu)
                group.setExclusive(True)
                for mode in RenderMode:
                    style_action = style_menu.addAction(mode.value)
                    style_action.setCheckable(True)
                    style_action.setChecked(mode is series.render_mode)
                    group.addAction(style_action)
                    action_map[style_action] = ("style", series, mode)

        chosen = menu.exec(self.plot_widget.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is load_action:
            self._prompt_load_dataset()
        elif chosen is reload_action:
            self._prompt_reload_dataset()
        elif chosen in action_map:
            entry = action_map[chosen]
            if entry[0] == "visible":
                self._toggle_series_visible(entry[1])
            else:
                self._set_series_render_mode(entry[1], entry[2])

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
            self.on_dataset_loaded(self, dataset, False)

    def _prompt_reload_dataset(self) -> None:
        try:
            dataset = self.reload_dataset()
        except DatasetLoadError as exc:
            QMessageBox.warning(self.plot_widget, "Failed to reload dataset", str(exc))
            return
        if self.on_dataset_loaded is not None:
            self.on_dataset_loaded(self, dataset, True)
