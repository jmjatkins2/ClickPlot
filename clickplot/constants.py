"""Tunable thresholds and defaults shared across click-plot modules."""

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

# Distinct, high-saturation colors for the 6 plots' lines/dots/bars.
PLOT_COLORS = [
    (220, 30, 30),    # red
    (30, 90, 220),    # blue
    (30, 160, 60),    # green
    (230, 150, 10),   # orange
    (150, 40, 190),   # purple
    (0, 170, 170),    # teal
]
