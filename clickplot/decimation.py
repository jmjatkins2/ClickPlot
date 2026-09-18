"""View-dependent decimation for up to ~2M irregularly-spaced points.

The core algorithm buckets the visible time range into one bucket per pixel
column and keeps each bucket's min/max value (a peak-preserving envelope),
using np.unique + np.minimum/maximum.reduceat on the sorted, visible slice.
This is O(n_visible) and safe to re-run on every pan/zoom/resize.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import BAR_MIN_PIXEL_WIDTH, RAW_POINT_THRESHOLD


@dataclass
class VisibleData:
    """Result of decimating a dataset to the current view."""

    decimated: bool
    # Raw (undecimated) visible slice -- always populated, even when
    # `decimated` is True, since bar mode needs raw spacing to decide
    # whether to draw discrete bars.
    t_raw: np.ndarray
    v_raw: np.ndarray
    # Only populated when `decimated` is True: one entry per bucket.
    bucket_t: np.ndarray | None = None
    bucket_min: np.ndarray | None = None
    bucket_max: np.ndarray | None = None


def _visible_index_range(t_sec: np.ndarray, x0: float, x1: float, margin_frac: float = 0.05):
    span = x1 - x0
    margin = span * margin_frac if span > 0 else 0.0
    i0 = int(np.searchsorted(t_sec, x0 - margin, side="left"))
    i1 = int(np.searchsorted(t_sec, x1 + margin, side="right"))
    return i0, i1


def _bucket_ids_for(t_slice: np.ndarray, x0: float, x1: float, n_columns: int) -> np.ndarray:
    n_columns = max(int(n_columns), 1)
    span = x1 - x0 if x1 > x0 else 1.0
    bucket_w = span / n_columns
    # t_slice is sorted, so the returned array is non-decreasing: np.unique's
    # first-occurrence index is exactly each bucket's start, giving O(n)
    # grouping that naturally skips empty buckets.
    return np.clip(((t_slice - x0) / bucket_w).astype(np.int64), 0, n_columns - 1)


def decimate_series_for_view(
    t_sec: np.ndarray,
    v_arrays: list[np.ndarray],
    x0: float,
    x1: float,
    n_columns: int,
    raw_threshold: int = RAW_POINT_THRESHOLD,
) -> list[VisibleData]:
    """Decimate one or more value series that all share the same t_sec.

    The visible index range and bucket assignment depend only on t_sec/x0/
    x1/n_columns -- never on individual series values -- so they're computed
    once here and reused for every series' min/max reduction, instead of
    recomputing the same search/bucketing work once per series.
    """
    i0, i1 = _visible_index_range(t_sec, x0, x1)
    t_raw = t_sec[i0:i1]

    if t_raw.shape[0] <= raw_threshold:
        return [VisibleData(decimated=False, t_raw=t_raw, v_raw=v[i0:i1]) for v in v_arrays]

    bucket_ids = _bucket_ids_for(t_raw, x0, x1, n_columns)
    _, first_idx = np.unique(bucket_ids, return_index=True)
    bucket_t = t_raw[first_idx]

    results = []
    for v in v_arrays:
        v_raw = v[i0:i1]
        results.append(
            VisibleData(
                decimated=True,
                t_raw=t_raw,
                v_raw=v_raw,
                bucket_t=bucket_t,
                # fmin/fmax (unlike minimum/maximum) ignore NaN when the other
                # operand is finite, so a single NaN sample doesn't null out
                # its whole bucket -- it's simply excluded from the envelope.
                bucket_min=np.fmin.reduceat(v_raw, first_idx),
                bucket_max=np.fmax.reduceat(v_raw, first_idx),
            )
        )
    return results


def line_xy(data: VisibleData) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) for a PlotDataItem in line mode: raw, or a min/max envelope."""
    if not data.decimated:
        return data.t_raw, data.v_raw
    n = data.bucket_t.shape[0]
    x = np.repeat(data.bucket_t, 2)
    y = np.empty(2 * n, dtype=np.float64)
    y[0::2] = data.bucket_min
    y[1::2] = data.bucket_max
    return x, y


def dot_xy(data: VisibleData) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) for a ScatterPlotItem: raw points, or unconnected min+max points."""
    if not data.decimated:
        return data.t_raw, data.v_raw
    x = np.concatenate([data.bucket_t, data.bucket_t])
    y = np.concatenate([data.bucket_min, data.bucket_max])
    return x, y


@dataclass
class BarSpec:
    as_bars: bool  # True: draw discrete BarGraphItem bars. False: envelope fallback.
    x: np.ndarray
    y: np.ndarray
    width: float = 0.0  # only meaningful when as_bars is True


def bar_spec(data: VisibleData, x0: float, x1: float, n_columns: int) -> BarSpec:
    """Decide between discrete bars and a line-envelope fallback.

    Discrete bars are only legible once each bar spans at least
    BAR_MIN_PIXEL_WIDTH pixels; otherwise degrade to the same min/max
    envelope used for line mode.

    The decision depends only on `data.t_raw`'s spacing, not on values --
    since every series of a dataset shares the same t_raw (see
    decimate_series_for_view), all series in one plot always get the same
    as_bars decision at a given zoom level. That's an invariant, not a bug:
    bars and the envelope fallback never coexist across series in one plot.
    """
    n_columns = max(int(n_columns), 1)
    span = x1 - x0 if x1 > x0 else 1.0

    if data.t_raw.shape[0] >= 2:
        median_spacing = float(np.median(np.diff(data.t_raw)))
    else:
        median_spacing = span

    pixels_per_sample = n_columns * median_spacing / span

    if not data.decimated and pixels_per_sample >= BAR_MIN_PIXEL_WIDTH:
        return BarSpec(as_bars=True, x=data.t_raw, y=data.v_raw, width=median_spacing)

    x, y = line_xy(data)
    return BarSpec(as_bars=False, x=x, y=y)
