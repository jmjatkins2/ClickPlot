"""Tunable thresholds and defaults shared across click-plot modules."""

import pyqtgraph as pg

# Below this many visible points, plot at full resolution (no decimation).
RAW_POINT_THRESHOLD = 8_000

# Fallback pixel-column target used before a plot widget has a real width.
DEFAULT_COLUMN_TARGET = 1400

# Minimum pixels a bar must span before we draw discrete bars instead of
# falling back to a min/max line envelope.
BAR_MIN_PIXEL_WIDTH = 3

# Debounce interval (ms) for redraws triggered by pan/zoom/resize bursts.
DEBOUNCE_MS = 20

NUM_PLOTS = 6

# Fixed Y range every plot uses: each series independently normalizes its
# own min/max to -1..1, so all overlaid series share this same visual band
# regardless of their real amplitudes. The 0.05 is padding, matching the
# ~5% padding the old data-driven Y-fit used.
NORMALIZED_Y_RANGE = (-1.05, 1.05)

# Distinct, high-saturation colors for a plot's overlaid series (assigned in
# load order, one per series -- NOT one per plot).
PLOT_COLORS = [
    (220, 30, 30),    # red
    (30, 90, 220),    # blue
    (30, 160, 60),    # green
    (230, 150, 10),   # orange
    (150, 40, 190),   # purple
    (0, 170, 170),    # teal
]


def series_color(index: int, total: int) -> tuple[int, int, int]:
    """Pick a color for the index'th of `total` series in one plot.

    Uses the fixed high-saturation palette for the common case; falls back
    to procedurally-generated hues for datasets with more series than the
    palette has entries, so colors never silently collide.
    """
    if index < len(PLOT_COLORS):
        return PLOT_COLORS[index]
    c = pg.intColor(index, hues=max(total, len(PLOT_COLORS) + 1))
    return (c.red(), c.green(), c.blue())
