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


def _bucket_min_max(t_slice: np.ndarray, v_slice: np.ndarray, x0: float, x1: float, n_columns: int):
    n_columns = max(int(n_columns), 1)
    span = x1 - x0 if x1 > x0 else 1.0
    bucket_w = span / n_columns
    bucket_ids = np.clip(((t_slice - x0) / bucket_w).astype(np.int64), 0, n_columns - 1)
    # t_slice is sorted, so bucket_ids is non-decreasing: np.unique's first
    # occurrence index is exactly the bucket's start, giving O(n) grouping
    # that naturally skips empty buckets.
    _, first_idx = np.unique(bucket_ids, return_index=True)
    seg_min = np.minimum.reduceat(v_slice, first_idx)
    seg_max = np.maximum.reduceat(v_slice, first_idx)
    seg_t = t_slice[first_idx]
    return seg_t, seg_min, seg_max


def decimate_for_view(
    t_sec: np.ndarray,
    v: np.ndarray,
    x0: float,
    x1: float,
    n_columns: int,
    raw_threshold: int = RAW_POINT_THRESHOLD,
) -> VisibleData:
    i0, i1 = _visible_index_range(t_sec, x0, x1)
    t_slice = t_sec[i0:i1]
    v_slice = v[i0:i1]

    if t_slice.shape[0] <= raw_threshold:
        return VisibleData(decimated=False, t_raw=t_slice, v_raw=v_slice)

    bucket_t, bucket_min, bucket_max = _bucket_min_max(t_slice, v_slice, x0, x1, n_columns)
    return VisibleData(
        decimated=True,
        t_raw=t_slice,
        v_raw=v_slice,
        bucket_t=bucket_t,
        bucket_min=bucket_min,
        bucket_max=bucket_max,
    )


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
