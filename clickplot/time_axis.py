"""Datetime X-axis with microsecond-resolution tick labels.

pyqtgraph's DateAxisItem represents X as POSIX seconds (float64) and its
built-in zoom levels format down to whole seconds/milliseconds. We only need
to override tick *label formatting* here -- tick *placement* (tickValues)
still comes from the base class, which already picks sensibly small spacing
when the view is zoomed in tight; we just need to render those fine spacings
with microsecond precision instead of getting rounded off.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pyqtgraph as pg


class MicrosecondAxisItem(pg.DateAxisItem):
    """A DateAxisItem that renders sub-millisecond ticks with microseconds.

    All timestamps are treated as UTC explicitly: the underlying data is
    timezone-naive datetime64, and formatting with a fixed UTC offset avoids
    DST/local-timezone drift between machines.
    """

    def tickStrings(self, values, scale, spacing):
        if spacing >= 1e-3 or not values:
            return super().tickStrings(values, scale, spacing)

        strings = []
        show_micros = spacing < 1e-3
        for v in values:
            dt = datetime.fromtimestamp(v, tz=timezone.utc)
            text = dt.strftime("%H:%M:%S.%f")
            if not show_micros:
                text = text[:-3]
            strings.append(text)
        return strings
