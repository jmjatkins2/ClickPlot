"""Convert a CPOD/FPOD click-train export (cpod-trains.txt) into a
click-plot-loadable .npz dataset.

The datetime for each row is Time (minute resolution) + Start (microseconds
into that minute). Only TrDur_us, NofClx, Clx/s, modalKHz, and avSPL are
carried into the output as separate series sharing that one time axis.

Usage:
    python scripts/convert_cpod_trains.py [--input examples/cpod-trains.txt] [--output examples/cpod-trains.npz] [--shift-seconds X]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VALUE_COLUMNS = ["TrDur_us", "NofClx", "Clx/s", "modalKHz", "avSPL"]


def convert(input_path: Path, output_path: Path, shift_seconds: float = 0.0) -> int:
    df = pd.read_csv(input_path, sep="\t")

    timestamps = (
        pd.to_datetime(df["Time"], format="%d/%m/%Y %H:%M")
        + pd.to_timedelta(df["Start"], unit="us")
        + pd.to_timedelta(shift_seconds, unit="s")
    )
    df = df.assign(_t=timestamps).sort_values("_t")

    t = df["_t"].to_numpy(dtype="datetime64[us]")
    series = {name: df[name].to_numpy(dtype=np.float64) for name in VALUE_COLUMNS}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, t=t, **series)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parent.parent / "examples"
    parser.add_argument("--input", type=Path, default=default_dir / "cpod-trains.txt")
    parser.add_argument("--output", type=Path, default=default_dir / "cpod-trains.npz")
    parser.add_argument(
        "--shift-seconds",
        type=float,
        default=0.0,
        help="Shift all output timestamps by this many seconds (positive or negative).",
    )
    args = parser.parse_args()

    n = convert(args.input, args.output, shift_seconds=args.shift_seconds)
    shift_note = f", shifted {args.shift_seconds:+g}s" if args.shift_seconds else ""
    print(f"Wrote {args.output} ({n:,} points, {len(VALUE_COLUMNS)} series{shift_note})")


if __name__ == "__main__":
    main()
