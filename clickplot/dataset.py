"""Loading and validating click-plot's dataset formats: .npz and .wav.

.npz: an archive holding one shared 't' array (datetime64[us]) plus one or
more named value arrays (float64) -- each non-'t' array is an independent
series overlaid on the same plot, written with
np.savez(path, t=..., series_a=..., series_b=...).

.wav: a PCM audio file (8/16/32-bit). Channel 0 is used (other channels are
discarded) as a single series named 'v', normalized to -1.0..1.0, with
timestamps synthesized from the sample rate starting at the Unix epoch
(there is no absolute time in a WAV file).
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# sample width (bytes) -> (numpy dtype, zero offset, full-scale divisor)
_WAV_SAMPLE_FORMATS = {
    1: (np.uint8, 128.0, 128.0),  # WAV's 8-bit format is unsigned, centered on 128
    2: (np.int16, 0.0, 32768.0),
    4: (np.int32, 0.0, 2147483648.0),
}


class DatasetLoadError(Exception):
    """Raised when a file does not contain a valid click-plot dataset."""


@dataclass
class Series:
    name: str
    v: np.ndarray  # float64, same length/order as the owning Dataset's t_us


@dataclass
class Dataset:
    path: Path
    t_us: np.ndarray  # int64, sorted, exact microsecond timestamps
    t_sec: np.ndarray  # float64 seconds, derived once from t_us (for plotting)
    series: list[Series]  # one or more value series, all sharing t_us/t_sec

    @property
    def n_points(self) -> int:
        return self.t_us.shape[0]

    @property
    def name(self) -> str:
        return self.path.name

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            t_us, named_arrays = cls._read_npz(path)
        elif suffix == ".wav":
            t_us, named_arrays = cls._read_wav(path)
        else:
            raise DatasetLoadError(
                f"'{path.name}': unsupported file type '{suffix or '(none)'}' -- "
                "click-plot loads .npz (t + one or more value arrays) or .wav (PCM audio) files."
            )
        return cls._finalize(path, t_us, named_arrays)

    @classmethod
    def _read_npz(cls, path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        try:
            loaded = np.load(path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as-is
            raise DatasetLoadError(f"Could not read '{path.name}': {exc}") from exc

        if not hasattr(loaded, "files"):
            raise DatasetLoadError(
                f"'{path.name}' is a plain .npy array, not a click-plot dataset: "
                "expected a .npz archive containing a 't' (datetime64) array plus "
                "one or more value arrays, e.g. written with "
                "np.savez(path, t=..., series_a=..., series_b=...)."
            )

        try:
            if "t" not in loaded.files:
                raise DatasetLoadError(
                    f"'{path.name}' is not a click-plot dataset: expected a .npz "
                    f"archive with a 't' array, got {loaded.files!r}."
                )

            t_field = loaded["t"]
            if not np.issubdtype(t_field.dtype, np.datetime64):
                raise DatasetLoadError(
                    f"'{path.name}' array 't' must be datetime64, got {t_field.dtype!r}."
                )

            series_names = [name for name in loaded.files if name != "t"]
            if not series_names:
                raise DatasetLoadError(
                    f"'{path.name}' has no data series (only 't') -- expected at "
                    "least one additional value array."
                )

            named_arrays: dict[str, np.ndarray] = {}
            for name in series_names:
                v_field = loaded[name]
                if not np.issubdtype(v_field.dtype, np.floating):
                    raise DatasetLoadError(
                        f"'{path.name}' array '{name}' must be a float dtype, got {v_field.dtype!r}."
                    )
                named_arrays[name] = np.ascontiguousarray(v_field, dtype=np.float64)

            # Coerce any datetime64 resolution (commonly [ns]) down to [us], the
            # resolution click-plot works in. This truncates, never loses whole
            # microseconds, and matches the display quantum.
            t_us = t_field.astype("datetime64[us]").view(np.int64).copy()
        finally:
            loaded.close()

        return t_us, named_arrays

    @classmethod
    def _read_wav(cls, path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        try:
            with wave.open(str(path), "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
        except (wave.Error, EOFError, OSError) as exc:
            raise DatasetLoadError(f"Could not read '{path.name}' as WAV: {exc}") from exc

        if sample_width not in _WAV_SAMPLE_FORMATS:
            raise DatasetLoadError(
                f"'{path.name}': unsupported WAV sample width ({sample_width * 8}-bit). "
                "click-plot supports 8/16/32-bit PCM WAV files."
            )
        if frame_rate <= 0:
            raise DatasetLoadError(f"'{path.name}' has an invalid sample rate ({frame_rate}).")

        dtype, offset, full_scale = _WAV_SAMPLE_FORMATS[sample_width]
        samples = np.frombuffer(raw, dtype=dtype).reshape(-1, n_channels)
        v = (samples[:, 0].astype(np.float64) - offset) / full_scale  # channel 0 only

        t_us = np.round(np.arange(n_frames, dtype=np.float64) * (1_000_000.0 / frame_rate)).astype(np.int64)
        return t_us, {"v": v}

    @classmethod
    def _finalize(cls, path: Path, t_us: np.ndarray, named_arrays: dict[str, np.ndarray]) -> "Dataset":
        for name, v in named_arrays.items():
            if v.shape[0] != t_us.shape[0]:
                raise DatasetLoadError(
                    f"'{path.name}' array '{name}' length does not match 't'."
                )
        if t_us.shape[0] == 0:
            raise DatasetLoadError(f"'{path.name}' contains no data points.")

        order = np.argsort(t_us, kind="stable")
        if not np.array_equal(order, np.arange(t_us.shape[0])):
            t_us = t_us[order]
            named_arrays = {name: v[order] for name, v in named_arrays.items()}

        t_sec = t_us.astype(np.float64) / 1_000_000.0

        series = [Series(name=name, v=v) for name, v in named_arrays.items()]
        return cls(path=path, t_us=t_us, t_sec=t_sec, series=series)
