# click-plot

Interactive desktop viewer for up to 6 related time-series datasets (up to 2M
points each), built with PySide6 + pyqtgraph.

## Setup

A venv already exists at `venv\`. If you need to recreate it:

```
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

You don't need to activate the venv (PowerShell's script-execution policy may
block `Activate.ps1` anyway) -- just call `venv\Scripts\python.exe` directly,
as in the commands below. If you'd rather activate it, either run
`venv\Scripts\activate.bat` from `cmd.exe`, or in PowerShell run
`Set-ExecutionPolicy -Scope Process RemoteSigned` first (applies only to the
current window).

## Generate example data

Two 1,000,000-point sinusoidal datasets at 1ms spacing, written to `examples/`:

```
venv\Scripts\python.exe scripts\generate_example_data.py
```

## Run

```
venv\Scripts\python.exe -m clickplot
```

## Usage

- The window has 6 vertically-stacked plots sharing one X (datetime) axis.
  Scroll to zoom, or Ctrl+left-drag to select and zoom to a time window, on
  any plot; all 6 stay aligned. Plain left-drag pans X only (vertical
  panning is disabled -- each plot's Y range auto-fits its loaded series).
  Drag the splitter handles between plots to resize them individually.
- Right-click inside any plot to load a `.npz` or `.wav` dataset into that
  plot, reload the current file from disk (useful for a file that's still
  being written to), or configure each loaded series.
- A NumPy dataset file is a `.npz` archive containing one shared `t` array
  (`datetime64[us]`) plus **one or more** named value arrays (`float64`).
  Each value array is a separate **series**, overlaid on the same plot with
  its own color and a legend entry named after the array. Per-series,
  right-click gives a "Visible" checkbox (show/hide) and a Dot / Line / Bar
  style choice (defaults to Dot). Written with e.g.
  `np.savez(path, t=..., channel_a=..., channel_b=...)` -- see
  `scripts/generate_example_data.py` for a reference writer.
- A `.wav` audio file is also loadable directly, as a single series named
  `v`: only 8/16/32-bit PCM WAV is supported (float32, 24-bit, and
  compressed WAV are not). For multi-channel audio, only channel 0 is used.
  Samples are normalized to the -1.0..1.0 range, and since WAV files don't
  carry an absolute timestamp, the time axis is synthesized from the file's
  sample rate starting at the Unix epoch (displayed as elapsed time from
  `00:00:00.000000`).
- Each plot's label (left axis) shows the loaded file's name.

## Tests

```
venv\Scripts\python.exe -m pytest tests
```
