#!/usr/bin/env python3
"""
UTC plotter for FM meteor scatter observations
Loads .npz files and plots power traces versus UTC time.

Designed for: Visualizing long-term FM observation data
Features: UTC time conversion, trigger highlighting, multiple plot options
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import glob
import os
import re
from datetime import datetime, timezone, timedelta


_CHUNK_RE = re.compile(r"^(?P<station>.+)_(?P<stamp>\d{8}_\d{6})_chunk\d+\.npz$")


def index_recording_files(directory):
    """Index completed recorder chunks by their filename timestamp."""
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        name = os.path.basename(path)
        match = _CHUNK_RE.match(name)
        if not match or not os.path.isfile(path) or os.path.islink(path):
            continue
        try:
            stamp = datetime.strptime(match["stamp"], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        stamp = stamp.replace(tzinfo=timezone.utc)
        rows.append({
            "path": path,
            "name": name,
            "station": match["station"],
            "start_ns": int(stamp.timestamp() * 1e9),
        })
    return sorted(rows, key=lambda row: row["start_ns"])


def load_npz_range(rows, start_ns, end_ns):
    """Load every stored sample overlapping [start_ns, end_ns].

    Unlike the web viewer, this deliberately does not bucket the result. The
    caller is local WSL analysis with more memory/CPU available, so a focused
    hour or day is plotted at the recorder's original frame rate.
    """
    fields = ("t_utc_ns", "power_dbfs", "noise_dbfs", "peak_dbfs",
              "snr_db", "trigger")
    parts = {field: [] for field in fields}
    meta = {}
    files_used = []
    candidates = [row for row in rows if row["start_ns"] < end_ns]
    for i, row in enumerate(candidates):
        next_start = (candidates[i + 1]["start_ns"]
                      if i + 1 < len(candidates) else None)
        if next_start is not None and next_start <= start_ns:
            continue
        try:
            with np.load(row["path"]) as chunk:
                times = chunk["t_utc_ns"]
                if len(times) == 0 or times[-1] < start_ns or times[0] > end_ns:
                    continue
                keep = (times >= start_ns) & (times <= end_ns)
                if not np.any(keep):
                    continue
                for field in fields:
                    parts[field].append(chunk[field][keep])
                if not meta:
                    for field in ("station", "center_freq_hz", "threshold_db",
                                  "frame_rate_hz", "gain_db"):
                        if field in chunk.files:
                            meta[field] = chunk[field][0].item() \
                                if hasattr(chunk[field][0], "item") else chunk[field][0]
                files_used.append(row["name"])
        except Exception as exc:
            print(f"Skipping unreadable chunk {row['name']}: {exc}")

    if not parts["t_utc_ns"]:
        return None
    data = {field: np.concatenate(parts[field]) for field in fields}
    order = np.argsort(data["t_utc_ns"])
    for field in fields:
        data[field] = data[field][order]
    data["meta"] = meta
    data["files_used"] = files_used
    return data


def recording_bounds(rows):
    """Return actual (first sample, last sample) timestamps for indexed chunks."""
    if not rows:
        return None
    first = last = None
    for row in rows:
        try:
            with np.load(row["path"]) as chunk:
                times = chunk["t_utc_ns"]
                if len(times):
                    first = int(times[0]) if first is None else min(first, int(times[0]))
                    last = int(times[-1]) if last is None else max(last, int(times[-1]))
        except Exception:
            continue
    return (first, last) if first is not None else None


def _date_to_ns(day_text, hour_text):
    """Convert the GUI's explicit UTC date/hour controls to nanoseconds."""
    dt = datetime.strptime(f"{day_text} {hour_text}", "%Y-%m-%d %H")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def _matplotlib_utc_days(timestamp_ns):
    """Convert recorder nanoseconds to Matplotlib date numbers."""
    epoch = mdates.date2num(datetime(1970, 1, 1, tzinfo=timezone.utc))
    return timestamp_ns.astype(np.float64) / (1e9 * 86400.0) + epoch


class LocalViewer:
    """Tk/Matplotlib local viewer for full-resolution focused NPZ ranges."""

    def __init__(self, directory, show_peaks=True, show_snr=True,
                 highlight_triggers=True):
        try:
            import tkinter as tk
            from tkinter import ttk
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.widgets import RectangleSelector
        except Exception as exc:
            raise RuntimeError(
                "the local GUI needs Tk/Matplotlib GUI support; "
                "install python3-tk (and use WSLg or an X server on WSL)"
            ) from exc

        self.tk = tk
        self.ttk = ttk
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.NavigationToolbar2Tk = NavigationToolbar2Tk
        self.RectangleSelector = RectangleSelector
        self.directory = os.path.abspath(os.path.expanduser(directory))
        self.rows = index_recording_files(self.directory)
        self.bounds = recording_bounds(self.rows)
        if not self.bounds:
            raise RuntimeError(f"no readable recorder NPZ chunks found in {self.directory}")
        self.show_peaks = show_peaks
        self.show_snr = show_snr
        self.highlight_triggers = highlight_triggers

        try:
            self.root = tk.Tk()
        except Exception as exc:
            raise RuntimeError(
                "the local GUI could not open a display; use WSLg or set up an X server"
            ) from exc
        self.root.title("Remote Radio Observatory — local NPZ viewer")
        self.root.geometry("1500x950")

        controls = ttk.Frame(self.root, padding=6)
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="UTC day").pack(side=tk.LEFT)
        self.day_var = tk.StringVar()
        self.day_entry = ttk.Entry(controls, textvariable=self.day_var, width=12)
        self.day_entry.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(controls, text="start hour").pack(side=tk.LEFT)
        self.hour_var = tk.StringVar(value="00")
        self.hour_box = ttk.Combobox(controls, textvariable=self.hour_var,
                                     values=[f"{hour:02d}" for hour in range(24)],
                                     state="readonly", width=4)
        self.hour_box.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(controls, text="window").pack(side=tk.LEFT)
        self.window_var = tk.StringVar(value="1 hour")
        self.window_box = ttk.Combobox(
            controls, textvariable=self.window_var,
            values=("1 hour", "6 hours", "24 hours", "7 days", "all"),
            state="readonly", width=10)
        self.window_box.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Button(controls, text="plot selection",
                   command=self.plot_selection).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="previous day",
                   command=lambda: self.shift_day(-1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="next day",
                   command=lambda: self.shift_day(1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="reset view",
                   command=self.reset_view).pack(side=tk.LEFT, padx=3)
        ttk.Label(controls, text="drag across any plot to zoom").pack(
            side=tk.LEFT, padx=10)
        self.status_var = tk.StringVar(value="loading…")
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT, padx=14)

        self.figure = plt.Figure(figsize=(15, 9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.day_entry.bind("<Return>", lambda _event: self.plot_selection())

        last_ns = self.bounds[1]
        default_start_ns = max(self.bounds[0], last_ns - 3600 * 1_000_000_000)
        default_dt = datetime.fromtimestamp(default_start_ns / 1e9, tz=timezone.utc)
        self.day_var.set(default_dt.strftime("%Y-%m-%d"))
        self.hour_var.set(f"{default_dt.hour:02d}")
        self.plot_selection()

    def shift_day(self, days):
        try:
            day = datetime.strptime(self.day_var.get(), "%Y-%m-%d")
        except ValueError:
            self.status_var.set("day must be YYYY-MM-DD")
            return
        self.day_var.set((day + timedelta(days=days)).strftime("%Y-%m-%d"))
        self.plot_selection()

    def _selection_ns(self):
        window = self.window_var.get()
        if window == "all":
            return self.bounds
        start_ns = _date_to_ns(self.day_var.get(), self.hour_var.get())
        hours = {"1 hour": 1, "6 hours": 6, "24 hours": 24,
                 "7 days": 24 * 7}[window]
        return start_ns, start_ns + hours * 3600 * 1_000_000_000

    def plot_selection(self):
        try:
            start_ns, end_ns = self._selection_ns()
        except (KeyError, ValueError) as exc:
            self.status_var.set(f"invalid selection: {exc}")
            return
        self.status_var.set("loading full-resolution samples…")
        self.root.update_idletasks()
        data = load_npz_range(self.rows, start_ns, end_ns)
        if data is None:
            self.status_var.set("no samples in that UTC window")
            return
        self.draw(data, start_ns, end_ns)

    def draw(self, data, start_ns, end_ns):
        for selector in getattr(self, "selectors", []):
            selector.set_active(False)
        self.selectors = []
        self.figure.clear()
        include_peak = self.show_peaks and "peak_dbfs" in data
        include_snr = self.show_snr and "snr_db" in data
        plot_count = 1 + int(include_peak) + int(include_snr)
        axes = np.atleast_1d(self.figure.subplots(plot_count, 1, sharex=True))
        x = _matplotlib_utc_days(data["t_utc_ns"])
        meta = data.get("meta", {})
        station = meta.get("station", "Unknown")
        freq = meta.get("center_freq_hz")
        title = f"FM Observation — {station}"
        if freq is not None:
            title += f" ({float(freq) / 1e6:.3f} MHz)"
        self.figure.suptitle(title)

        power_axis = axes[0]
        power_axis.plot(x, data["power_dbfs"], label="Target power",
                     linewidth=0.7, color="tab:blue")
        power_axis.plot(x, data["noise_dbfs"], label="Noise floor",
                     linewidth=0.7, color="tab:green", alpha=0.8)
        power_axis.set_ylabel("dBFS")
        power_axis.legend(loc="upper right")

        plot_index = 1
        if include_peak:
            axes[plot_index].plot(x, data["peak_dbfs"], label="Peak power",
                                  linewidth=0.7, color="tab:orange")
            axes[plot_index].set_ylabel("Peak dBFS")
            axes[plot_index].legend(loc="upper right")
            plot_index += 1

        snr_axis = None
        if include_snr:
            snr_axis = axes[plot_index]
            snr_axis.plot(x, data["snr_db"], label="SNR",
                          linewidth=0.7, color="tab:purple")
            snr_axis.set_ylabel("SNR (dB)")
            snr_axis.set_xlabel("Time (UTC)")
            snr_axis.legend(loc="upper right")
            threshold = meta.get("threshold_db")
            if threshold is not None:
                snr_axis.axhline(float(threshold), color="red", linestyle="--",
                                 linewidth=0.8,
                                 label=f"Threshold ({float(threshold):.1f} dB)")
            snr_axis.legend(loc="upper right")

        if self.highlight_triggers:
            trigger = np.asarray(data["trigger"], dtype=bool)
            if np.any(trigger):
                power_axis.scatter(x[trigger], data["power_dbfs"][trigger],
                                   color="red", s=10, zorder=5, label="Trigger")
                if snr_axis is not None:
                    snr_axis.scatter(x[trigger], data["snr_db"][trigger],
                                     color="red", s=10, zorder=5)
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.set_xlim(_matplotlib_utc_days(np.asarray([start_ns, end_ns])))
        axes[-1].xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S", tz=timezone.utc))
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
        self.axes = axes
        self.full_xlim = _matplotlib_utc_days(np.asarray([start_ns, end_ns]))
        self.selectors = [
            self.RectangleSelector(
                axis, self.zoom_selected, useblit=False, button=[1],
                minspanx=1e-9, minspany=0, spancoords="data",
                interactive=False,
                props={"facecolor": "#58a6ff", "alpha": 0.15,
                       "edgecolor": "#58a6ff"},
            )
            for axis in axes
        ]
        self.figure.tight_layout(rect=(0, 0, 1, 0.96))
        self.canvas.draw_idle()
        self.status_var.set(
            f"{len(data['t_utc_ns']):,} raw samples · {len(data['files_used'])} chunk(s) · "
            f"{fmt_utc_ns(start_ns)} → {fmt_utc_ns(end_ns)}")

    def zoom_selected(self, press, release):
        """Zoom every linked panel to a dragged x-range."""
        if press.xdata is None or release.xdata is None:
            return
        left, right = sorted((press.xdata, release.xdata))
        if right - left <= 1e-9:
            return
        for axis in self.axes:
            axis.set_xlim(left, right)
        self.canvas.draw_idle()
        start_ns = int((left - mdates.date2num(
            datetime(1970, 1, 1, tzinfo=timezone.utc))) * 86400 * 1e9)
        end_ns = int((right - mdates.date2num(
            datetime(1970, 1, 1, tzinfo=timezone.utc))) * 86400 * 1e9)
        self.status_var.set(f"zoomed view: {fmt_utc_ns(start_ns)} → "
                           f"{fmt_utc_ns(end_ns)} · press reset view to undo")

    def reset_view(self):
        if not hasattr(self, "axes"):
            return
        for axis in self.axes:
            axis.set_xlim(self.full_xlim)
        self.canvas.draw_idle()
        self.status_var.set("reset to selected UTC window")

    def run(self):
        self.root.mainloop()


def fmt_utc_ns(timestamp_ns):
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")


def launch_local_gui(directory, show_peaks=True, show_snr=True,
                     highlight_triggers=True):
    """Launch the full-resolution local Tk viewer."""
    LocalViewer(directory, show_peaks=show_peaks, show_snr=show_snr,
                highlight_triggers=highlight_triggers).run()


def load_npz_file(filepath):
    """Load and validate .npz file"""
    try:
        data = np.load(filepath)
        print(f"Loaded {filepath}")
        print(f"Available fields: {list(data.keys())}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def convert_timestamps_to_utc(timestamp_ns):
    """Convert nanosecond timestamps to UTC datetime objects"""
    # Convert nanoseconds to seconds
    timestamp_s = timestamp_ns / 1e9
    # Convert to UTC datetime objects
    return [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamp_s]


def load_and_combine_npz_files(filepaths):
    """Load and combine multiple .npz files into a single dataset"""
    try:
        # Load all files and collect data
        all_data = {}
        all_timestamps = []

        for filepath in filepaths:
            data = np.load(filepath)
            # Collect timestamps for sorting
            if 't_utc_ns' in data.files:
                all_timestamps.extend(list(data['t_utc_ns']))

            # Collect all other data
            for key in data.files:
                if key not in all_data:
                    all_data[key] = []
                all_data[key].extend(list(data[key]))
            data.close()

        # Sort all data by timestamp
        if all_timestamps and len(all_timestamps) == len(all_data.get('t_utc_ns', [])):
            sort_indices = np.argsort(all_timestamps)
            # Reorder all arrays by sorted timestamps
            for key in all_data:
                all_data[key] = np.array(all_data[key])[sort_indices]

            print(f"Combined {len(filepaths)} files into dataset with {len(all_data['t_utc_ns'])} total samples (sorted by time)")
            return all_data
        else:
            # Fallback: just concatenate in file order
            for key in all_data:
                all_data[key] = np.array(all_data[key])
            print(f"Combined {len(filepaths)} files into dataset with {len(all_data.get('t_utc_ns', []))} total samples (concatenated order)")
            return all_data

    except Exception as e:
        print(f"Error combining files: {e}")
        return None


def plot_observation(data, show_peaks=True, show_snr=True, highlight_triggers=True,
                    save_plot=False, output_file=None):
    """Create plots from observation data"""

    # Extract data
    timestamps_ns = data['t_utc_ns']
    power_dbfs = data['power_dbfs']
    noise_dbfs = data['noise_dbfs']
    # Handle both npz file object and regular dict
    has_peak_dbfs = ('peak_dbfs' in data.files if hasattr(data, 'files') else 'peak_dbfs' in data)
    has_snr_db = ('snr_db' in data.files if hasattr(data, 'files') else 'snr_db' in data)
    has_trigger = ('trigger' in data.files if hasattr(data, 'files') else 'trigger' in data)
    has_threshold_db = ('threshold_db' in data.files if hasattr(data, 'files') else 'threshold_db' in data)

    peak_dbfs = data['peak_dbfs'] if has_peak_dbfs else None
    snr_db = data['snr_db'] if has_snr_db else None
    trigger = data['trigger'] if has_trigger else None
    station = data['station'][0] if len(data['station']) > 0 else "Unknown"
    center_freq = data['center_freq_hz'][0]

    # Convert timestamps to UTC
    utc_times = convert_timestamps_to_utc(timestamps_ns)

    # Create figure with subplots
    n_plots = 1 + (1 if show_peaks else 0) + (1 if show_snr else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4*n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    plot_index = 0

    # Plot 1: Target power and noise floor
    ax1 = axes[plot_index]
    ax1.plot(utc_times, power_dbfs, label='Target Power', linewidth=0.8, color='blue')
    ax1.plot(utc_times, noise_dbfs, label='Noise Floor', linewidth=0.8, color='green', alpha=0.8)

    if highlight_triggers and trigger is not None:
        # Highlight trigger points
        trigger_times = [utc_times[i] for i in range(len(trigger)) if trigger[i]]
        trigger_power = [power_dbfs[i] for i in range(len(trigger)) if trigger[i]]
        if trigger_times:
            threshold_val = data["threshold_db"][0] if has_threshold_db else 6
            ax1.scatter(trigger_times, trigger_power, color='red', s=20, zorder=5,
                       label=f'Triggers (>{threshold_val} dB)')

    ax1.set_ylabel('Relative Power (dB)')
    ax1.set_title(f'FM Observation - {station} ({center_freq/1e6:.3f} MHz)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    plot_index += 1

    # Plot 2: Peak power (if requested and available)
    if show_peaks and peak_dbfs is not None:
        ax2 = axes[plot_index]
        ax2.plot(utc_times, peak_dbfs, label='Peak Power', linewidth=0.8, color='orange')
        ax2.set_ylabel('Peak Power (dB)')
        ax2.set_title('Peak Power Spectrum')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        plot_index += 1

    # Plot 3: SNR (if requested and available)
    if show_snr and snr_db is not None:
        ax3 = axes[plot_index]
        ax3.plot(utc_times, snr_db, label='SNR', linewidth=0.8, color='purple')

        if highlight_triggers and trigger is not None:
            # Highlight trigger points on SNR plot
            trigger_times = [utc_times[i] for i in range(len(trigger)) if trigger[i]]
            trigger_snr = [snr_db[i] for i in range(len(trigger)) if trigger[i]]
            if trigger_times:
                ax3.scatter(trigger_times, trigger_snr, color='red', s=20, zorder=5,
                           label='Triggers')

        # Add threshold line if available
        if has_threshold_db:
            threshold = data['threshold_db'][0]
            ax3.axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
                       label=f'Threshold ({threshold} dB)')

        ax3.set_ylabel('SNR (dB)')
        ax3.set_xlabel('Time (UTC)')
        ax3.set_title('Signal-to-Noise Ratio')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        plot_index += 1

    # Format x-axis as time
    if plot_index > 0:
        axes[plot_index-1].set_xlabel('Time (UTC)')

    # Improve time formatting
    fig.autofmt_xdate()

    # Add overall stats
    if len(power_dbfs) > 0:
        max_power = np.max(power_dbfs)
        mean_power = np.mean(power_dbfs)
        trigger_count = np.sum(trigger) if trigger is not None else 0
        fig.suptitle(f'Stats: Max={max_power:.1f}dB, Mean={mean_power:.1f}dB, '
                    f'Triggers={trigger_count}/{len(power_dbfs)} '
                    f'({100*trigger_count/len(power_dbfs):.1f}%)',
                    fontsize=10)

    plt.tight_layout()

    if save_plot and output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot FM meteor scatter observation data')
    parser.add_argument('npz_file', nargs='?', help='Specific .npz file to plot, or pattern with wildcards (e.g., MUMBAI_*.npz) to plot all matching files')
    parser.add_argument('--dir', type=str, default='./fm_observations',
                        help='Directory to search for .npz files (default: ./fm_observations)')
    parser.add_argument('--no-peaks', action='store_true',
                        help='Do not show peak power plot')
    parser.add_argument('--no-snr', action='store_true',
                        help='Do not show SNR plot')
    parser.add_argument('--no-highlight', action='store_true',
                        help='Do not highlight trigger points')
    parser.add_argument('--save', action='store_true',
                        help='Save plot as PNG instead of displaying')
    parser.add_argument('--output', type=str, help='Output filename for saved plot')
    parser.add_argument('--list', action='store_true',
                        help='List available .npz files and exit')
    parser.add_argument('--gui', action='store_true',
                        help='Launch the local UTC day/hour focus viewer with Matplotlib zoom/pan')

    args = parser.parse_args()

    if args.gui:
        try:
            launch_local_gui(
                args.dir,
                show_peaks=not args.no_peaks,
                show_snr=not args.no_snr,
                highlight_triggers=not args.no_highlight,
            )
        except RuntimeError as exc:
            parser.error(str(exc))
        return

    # List files if requested
    if args.list:
        pattern = os.path.join(args.dir, "*.npz")
        files = sorted(glob.glob(pattern))
        if files:
            print(f"Found {len(files)} .npz files in {args.dir}:")
            for f in files:
                size_mb = os.path.getsize(f) / (1024*1024)
                print(f"  {os.path.basename(f)} ({size_mb:.2f} MB)")
        else:
            print(f"No .npz files found in {args.dir}")
        return

    # Determine files to load
    if args.npz_file:
        # Check if npz_file contains wildcards
        if '*' in args.npz_file or '?' in args.npz_file:
            # Handle wildcard pattern
            pattern = os.path.join(args.dir, args.npz_file) if not os.path.isabs(args.npz_file) else args.npz_file
            filepaths = sorted(glob.glob(pattern))
            if not filepaths:
                print(f"No files matching pattern '{args.npz_file}' found in {args.dir}")
                return
            print(f"Found {len(filepaths)} files matching pattern '{args.npz_file}'")
            # Load and combine data from multiple files
            data = load_and_combine_npz_files(filepaths)
            if data is None:
                return
            # For naming output file when saving
            base_name = os.path.splitext(os.path.basename(filepaths[0]))[0]
            if len(filepaths) > 1:
                base_name += f"_combined_{len(filepaths)}files"
        else:
            # Single file specified
            filepath = args.npz_file if os.path.isabs(args.npz_file) else os.path.join(args.dir, args.npz_file)
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                return
            data = load_npz_file(filepath)
            if data is None:
                return
            base_name = os.path.splitext(os.path.basename(filepath))[0]
    else:
        # Find latest file
        pattern = os.path.join(args.dir, "*.npz")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"No .npz files found in {args.dir}")
            return
        filepath = files[-1]  # Most recent
        print(f"Using latest file: {os.path.basename(filepath)}")
        data = load_npz_file(filepath)
        if data is None:
            return
        base_name = os.path.splitext(os.path.basename(filepath))[0]

    # Determine output filename for saving
    output_file = None
    if args.save:
        output_file = args.output or f"{base_name}_plot.png"

    # Create plots
    plot_observation(
        data,
        show_peaks=not args.no_peaks,
        show_snr=not args.no_snr,
        highlight_triggers=not args.no_highlight,
        save_plot=args.save,
        output_file=output_file
    )


if __name__ == "__main__":
    main()
