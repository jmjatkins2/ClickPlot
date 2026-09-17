"""QApplication bootstrap and global pyqtgraph configuration."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_API", "pyside6")

import pyqtgraph as pg  # noqa: E402 - QT_API must be set before this import

from .main_window import MainWindow  # noqa: E402


def main() -> int:
    pg.setConfigOptions(antialias=False, background="w", foreground="k")
    app = pg.mkQApp("click-plot")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
