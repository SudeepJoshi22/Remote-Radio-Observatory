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
from datetime import datetime, timezone
import argparse
import os
import glob


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

    args = parser.parse_args()

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