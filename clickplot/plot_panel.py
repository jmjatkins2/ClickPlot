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

from .constants import DEBOUNCE_MS, DEFAULT_COLUMN_TARGET, NORMALIZED_Y_RANGE, series_color
from .dataset import Dataset, DatasetLoadError
from .decimation import bar_spec, decimate_series_for_view, dot_xy, line_xy
from .time_axis import MicrosecondAxisItem


class _TimeSelectViewBox(pg.ViewBox):
    """A ViewBox with wheel-zoom locked to X, left-drag time selection, and right-drag pan.

    - Scroll-wheel always zooms X only: normal ViewBoxes zoom whichever axis
      the wheel event applies to (both, over the plot area; just Y, over the
      Y-axis label). Forcing axis=0 makes the wheel-scale mask only ever
      include X, regardless of where the cursor is.
    - Left-drag reports the dragged x-range (in data/view coordinates) via
      on_selection_drag, instead of panning -- PlotSlotState/MainWindow turn
      that into a persistent selection band shown across all plots.
    - Right-drag pans X (Y stays disabled via setMouseEnabled(y=False) in
      PlotSlotState either way), reimplementing the base class's own
      left/middle-button PanMode translate logic but keyed to the right
      button instead, since pyqtgraph's default right-drag behavior is a
      zoom-by-drag we don't want here.
    """

    on_selection_drag = None  # set by PlotSlotState after construction

    def wheelEvent(self, ev, axis=None):
        super().wheelEvent(ev, axis=0)

    def mouseDragEvent(self, ev, axis=None):
        if axis is None and ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            x0 = self.mapToView(ev.buttonDownPos(ev.button())).x()
            x1 = self.mapToView(ev.pos()).x()
            if self.on_selection_drag is not None:
                self.on_selection_drag(min(x0, x1), max(x0, x1), ev.isFinish())
            return
        if axis is None and ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self._pan_by_drag(ev)
            return
        super().mouseDragEvent(ev, axis=axis)

    def _pan_by_drag(self, ev):
        mask = np.array(self.state["mouseEnabled"], dtype=np.float64)
        tr = pg.functions.invertQTransform(self.childGroup.transform())
        dif = (ev.pos() - ev.lastPos()) * -1
        delta = tr.map(dif * mask) - tr.map(pg.Point(0, 0))
        self._resetTarget()
        self.translateBy(x=delta.x(), y=delta.y())
        self.sigRangeChangedManually.emit(self.state["mouseEnabled"])


class RenderMode(Enum):
    DOT = "Dot"
    LINE = "Line"
    BAR = "Bar"


class SeriesState:
    """One overlaid value series within a plot: its own style, visibility, and graphics item.

    Each series normalizes its own finite min/max to -1..1 (see normalize())
    so it independently fills the plot's full vertical range regardless of
    other overlaid series' amplitude -- the transform is derived once from
    the whole series at construction time, not re-fit as the view changes.
    """

    def __init__(self, name: str, v: np.ndarray, color: tuple, render_mode: RenderMode = RenderMode.DOT):
        self.name = name
        self.v = v
        self.color = color
        self.render_mode = render_mode
        self.visible = True
        self.item = None
        self.y_offset, self.y_scale = self._compute_normalization(v)

    @staticmethod
    def _compute_normalization(v: np.ndarray) -> tuple[float, float]:
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return 0.0, 0.0  # all non-finite -- normalize() is a no-op, values stay NaN
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            return vmin, 0.0  # flat series -- renders at 0, the band's center
        return (vmax + vmin) / 2.0, 2.0 / (vmax - vmin)  # maps [vmin, vmax] -> [-1, 1]

    def normalize(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_offset) * self.y_scale


class PlotSlotState:
    """Owns one of the six vertically-stacked plots and its independent state."""

    def __init__(
        self,
        index: int,
        show_x_labels: bool = True,
        on_dataset_loaded: Optional[Callable[["PlotSlotState", Dataset, bool], None]] = None,
        on_hover: Optional[Callable[["PlotSlotState", Optional[float], list], None]] = None,
        on_selection_drag: Optional[Callable[["PlotSlotState", float, float, bool], None]] = None,
    ):
        self.index = index
        self.on_dataset_loaded = on_dataset_loaded
        self.on_hover = on_hover
        self.on_selection_drag = on_selection_drag
        self.dataset: Optional[Dataset] = None
        self.series: list[SeriesState] = []
        self._selecting = False  # True while a left-drag selection gesture is in progress

        view_box = _TimeSelectViewBox()
        self.plot_widget = pg.PlotWidget(
            viewBox=view_box,
            axisItems={"bottom": MicrosecondAxisItem(orientation="bottom")},
        )
        self.plot_widget.setMinimumHeight(80)
        self.plot_widget.setLabel("left", f"Plot {index + 1}")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setYRange(*NORMALIZED_Y_RANGE, padding=0)
        if not show_x_labels:
            self.plot_widget.getAxis("bottom").setStyle(showValues=False)

        self.legend = self.plot_widget.addLegend()

        self.region_item = pg.LinearRegionItem(values=(0, 1), movable=False)
        self.region_item.setVisible(False)
        self.plot_widget.addItem(self.region_item)

        view_box.setMenuEnabled(False)
        view_box.setMouseEnabled(x=True, y=False)
        view_box.on_selection_drag = self._handle_selection_drag
        view_box.sigXRangeChanged.connect(self._on_view_changed)

        self.plot_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.plot_widget.customContextMenuRequested.connect(self._show_context_menu)

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.redraw)

        self._hover_proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse_moved
        )

    # -- time-range selection --------------------------------------------

    def _handle_selection_drag(self, x0: float, x1: float, finished: bool) -> None:
        self._selecting = not finished
        if self.on_selection_drag is not None:
            self.on_selection_drag(self, x0, x1, finished)

    def set_selection_region(self, x0: float, x1: float) -> None:
        self.region_item.setRegion((x0, x1))
        self.region_item.setVisible(True)

    def clear_selection_region(self) -> None:
        self.region_item.setVisible(False)

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
        self.redraw()

    # -- hover / status reporting --------------------------------------

    def _on_mouse_moved(self, evt) -> None:
        if self._selecting:
            return  # let the selection's status-bar readout win during a left-drag
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
        values = [(s.name, float(s.v[idx])) for s in self.series if s.visible and np.isfinite(s.v[idx])]
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
            y = series.normalize(y)
            kind = pg.PlotDataItem

            def apply(item):
                item.setData(x, y, connect="all")

            def make():
                return pg.PlotDataItem(pen=pg.mkPen(color=series.color, width=1))

        elif series.render_mode is RenderMode.DOT:
            x, y = dot_xy(vd)
            y = series.normalize(y)
            kind = pg.ScatterPlotItem

            def apply(item):
                item.setData(x=x, y=y)

            def make():
                return pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(*series.color, 200))

        else:  # BAR
            spec = bar_spec(vd, x0, x1, n_columns)
            norm_y = series.normalize(spec.y)
            if spec.as_bars:
                # Bars grow from the bottom of the normalized band (-1, this
                # series' own minimum) up to its normalized value, so height
                # is always >= 0 and reads as "how far up this series' own
                # range is this sample" -- consistent with line/dot.
                height = norm_y + 1.0
                kind = pg.BarGraphItem

                def apply(item, x=spec.x, height=height, width=spec.width):
                    item.setOpts(x=x, y0=-1.0, height=height, width=width)

                def make(x=spec.x, height=height, width=spec.width):
                    return pg.BarGraphItem(
                        x=x, y0=-1.0, height=height, width=width, brush=pg.mkBrush(*series.color, 150)
                    )

            else:
                kind = pg.PlotDataItem

                def apply(item, x=spec.x, y=norm_y):
                    item.setData(x, y, connect="all")

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
