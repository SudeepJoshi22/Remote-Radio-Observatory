#!/usr/bin/env python3
"""Streaming, matched-FFT comparison of three published/local estimator recipes.

This is an estimator comparison, NOT a full MeteorRadio/Echoes emulator or a
meteor classifier. See IQ_COMPARISON.md for source revisions and adaptations.
Only unsigned 8-bit interleaved IQ (rtl_sdr .cu8/.iq) is accepted.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

import dsp


FIELDS = (
    "time_s", "channel_dbfs", "guard_dbfs", "temporal_dbfs", "combined_dbfs",
    "ours_db", "ours_warm", "meteor_signal_dbfs", "meteor_noise_dbfs",
    "meteor_db", "meteor_broadband_ratio", "meteor_pass", "echo_signal_db",
    "echo_noise_db", "echo_db", "rail_fraction",
)
COL = {name: i for i, name in enumerate(FIELDS)}
SOURCES = {
    "MeteorRadio": "https://github.com/rabssm/MeteorRadio/blob/"
                   "c13c2334e39a0f80b98a759b28c0e4a1efcfc481/src/meteor_radar.py",
    "Echoes": "https://sourceforge.net/p/echoes/git/ci/"
              "ecf4400bcf9af0300ea42d24c2405c531fe420c6/tree/trunk/echoes/control.cpp",
}


class EchoReference:
    """Source-style next-scan reference; newest scan has double weight.

    control.cpp enqueues the scan mean, then adds it once more when updating N.
    Detection uses the PREVIOUS update. Warm-up is excluded from scoring here.
    """

    def __init__(self, scans):
        self.values = deque(maxlen=scans)
        self.scans = scans
        self.reference = np.nan

    def push(self, scan_mean):
        old = self.reference if len(self.values) == self.scans else np.nan
        self.values.append(float(scan_mean))
        self.reference = (sum(self.values) + scan_mean) / (len(self.values) + 1)
        return old


def meteor_block(power, detection_mask, noise_mask, threshold=45.0):
    """power has shape (time, frequency); return one result for this block.

    Implements analyse_psd/check_trigger's peak, selected-column median and
    broadband statistic, in float64 rather than upstream float16. Includes
    all bins in the full-spectrum broadband check, as upstream does.
    """
    signal = power[:, detection_mask]
    noise = power[:, noise_mask]
    peak_row, _ = np.unravel_index(np.argmax(signal), signal.shape)
    noise_row, _ = np.unravel_index(np.argmax(noise), noise.shape)
    peak = float(signal.max())
    floor = float(np.median(noise[noise_row]))
    medians = np.median(power, axis=1)
    base = float(np.median(medians))
    # Undefined ratios remain missing, rather than manufacturing enormous SNRs.
    ratio = float(medians.max() / base) if base > 0 else np.nan
    score = float(10 * np.log10(peak / floor)) if floor > 0 and peak > 0 else np.nan
    passed = bool(np.isfinite(score) and np.isfinite(ratio)
                  and score > 10 * np.log10(threshold) and ratio <= 3)
    return peak_row, peak, floor, score, ratio, passed


def mask_for(freq, limits, name):
    lo, hi = limits
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError(f"{name}: limits must be finite and increasing")
    df = freq[1] - freq[0]
    if lo < freq[0] or hi > freq[-1] + df:
        raise ValueError(f"{name}: band {lo:g}..{hi:g} Hz is outside captured bandwidth")
    mask = (freq > lo) & (freq <= hi)  # upstream MeteorRadio convention
    if not mask.any():
        raise ValueError(f"{name}: no FFT bins; increase --nfft or widen the band")
    return mask


def parser():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="rtl_sdr unsigned 8-bit interleaved IQ file")
    p.add_argument("--sample-rate", type=float, required=True)
    p.add_argument("--center-freq", type=float, required=True,
                   help="actual SDR tuned frequency, Hz (not inferred from filename)")
    p.add_argument("--gain", type=float, help="capture gain in dB, recorded as metadata only")
    p.add_argument("--start-utc", help="time of file's first IQ pair, ISO 8601 with timezone")
    p.add_argument("--start", type=float, default=0, help="start offset into file, seconds")
    p.add_argument("--duration", type=float, help="seconds to analyse; default: available file")
    p.add_argument("--output-dir", type=Path, required=True, help="new or empty output directory")
    p.add_argument("--nfft", type=int, default=8192)
    p.add_argument("--window", choices=("hann", "hamming", "rect"), default="hann")
    p.add_argument("--target-offset", type=float, default=0, help="target minus tuned frequency, Hz")
    p.add_argument("--channel-bw", type=float, default=180000)
    p.add_argument("--guard-lo", type=float, default=150000)
    p.add_argument("--guard-hi", type=float, default=400000)
    p.add_argument("--floor-seconds", type=float, default=60)
    p.add_argument("--floor-percentile", type=float, default=10)
    p.add_argument("--our-threshold-db", type=float, default=5)
    p.add_argument("--preset", choices=("fm", "beacon"), default="fm",
                   help="FM ranges are adaptations; beacon uses MeteorRadio's +/-120/500 Hz")
    p.add_argument("--peak-band", nargs=2, type=float, metavar=("LOW_HZ", "HIGH_HZ"),
                   help="peak search offsets relative to target, for MeteorRadio and Echoes")
    p.add_argument("--reference-band", nargs=2, type=float, metavar=("LOW_HZ", "HIGH_HZ"),
                   help="spectral median / Echoes displayed-band offsets relative to target")
    p.add_argument("--block-frames", type=int, default=64,
                   help="FFT frames per bounded-memory block; also MeteorRadio analysis interval")
    p.add_argument("--meteor-threshold", type=float, default=45, help="linear peak/median ratio")
    p.add_argument("--echo-scans", type=int, default=125, help="scans in Echoes background FIFO")
    p.add_argument("--echo-threshold-db", type=float,
                   help="optional differential threshold; no invented default for Echoes")
    p.add_argument("--plot-points", type=int, default=2000)
    p.add_argument("--zoom-seconds", type=float, default=4,
                   help="detail plot duration around highest channel-power frame")
    return p


def validate(a):
    for key in ("sample_rate", "center_freq", "channel_bw", "guard_lo", "guard_hi",
                "floor_seconds", "meteor_threshold", "zoom_seconds"):
        if not np.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be finite and positive")
    if a.nfft < 16 or a.block_frames < 2 or a.echo_scans < 2 or a.plot_points < 10:
        raise ValueError("nfft >= 16, block-frames >= 2, echo-scans >= 2, plot-points >= 10 required")
    if not np.isfinite(a.start) or a.start < 0:
        raise ValueError("--start must be finite and nonnegative")
    if a.duration is not None and (not np.isfinite(a.duration) or a.duration <= 0):
        raise ValueError("--duration must be finite and positive")
    if not 0 <= a.floor_percentile <= 100 or not np.isfinite(a.target_offset):
        raise ValueError("invalid percentile or target offset")
    if a.guard_hi <= a.guard_lo or a.guard_lo <= a.channel_bw / 2:
        raise ValueError("guard bands must lie outside the channel and have positive width")
    for key in ("gain", "our_threshold_db", "echo_threshold_db"):
        value = getattr(a, key)
        if value is not None and not np.isfinite(value):
            raise ValueError(f"--{key.replace('_', '-')} must be finite")
    if a.start_utc:
        dt = datetime.fromisoformat(a.start_utc.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("--start-utc needs a timezone, e.g. 2026-09-24T10:00:00Z")
    if a.input.suffix.lower() == ".npz":
        raise ValueError("summary NPZ files are not raw IQ; provide rtl_sdr .cu8/.iq")


def analyse(a):
    validate(a)
    size = a.input.stat().st_size  # snapshot: appended bytes are not followed
    if size % 2:
        raise ValueError("odd file size: incomplete IQ pair (or wrong sample format)")
    start_sample = round(a.start * a.sample_rate)
    available = size // 2 - start_sample
    wanted = available if a.duration is None else min(available, int(a.duration * a.sample_rate))
    frames = wanted // a.nfft
    if frames < 1:
        raise ValueError("no complete FFT frame in requested interval")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise ValueError("output directory must be empty; choose a new directory")

    freq = dsp.freq_axis(a.nfft, a.sample_rate) - a.target_offset
    channel = np.abs(freq) <= a.channel_bw / 2
    # Validate all boundaries instead of silently clipping requested bandwidth.
    mask_for(freq, (-a.channel_bw / 2, a.channel_bw / 2), "channel")
    mask_for(freq, (-a.guard_hi, -a.guard_lo), "lower guard")
    mask_for(freq, (a.guard_lo, a.guard_hi), "upper guard")
    # Use the recorder's inclusive guard endpoints for exact local replay.
    guard = ((freq >= -a.guard_hi) & (freq <= -a.guard_lo)) | ((freq >= a.guard_lo) & (freq <= a.guard_hi))
    default_peak = (-120, 120) if a.preset == "beacon" else (-a.channel_bw / 2, a.channel_bw / 2)
    default_reference = (-500, 500) if a.preset == "beacon" else (-a.guard_hi, a.guard_hi)
    peak_band = a.peak_band or default_peak
    reference_band = a.reference_band or default_reference
    detection = mask_for(freq, peak_band, "peak band")
    reference = mask_for(freq, reference_band, "reference band")
    if peak_band[0] < reference_band[0] or peak_band[1] > reference_band[1]:
        raise ValueError("reference band must contain the peak band")
    if a.preset == "beacon" and (detection.sum() < 8 or reference.sum() < 32):
        raise ValueError("beacon bands have too few bins at this resolution; use --nfft 131072 "
                         "at 1.024 MS/s (common FFT, not native receiver emulation)")
    window, norm = (np.hamming(a.nfft), None) if a.window == "hamming" else dsp.make_window(a.nfft, a.window)
    norm = float(np.sum(window ** 2)) if norm is None else norm
    rate = a.sample_rate / a.nfft
    floor = dsp.RollingFloor(rate, a.floor_seconds, a.floor_percentile)
    echo = EchoReference(a.echo_scans)
    df = a.sample_rate / a.nfft

    a.output_dir.mkdir(parents=True, exist_ok=True)
    result = np.lib.format.open_memmap(a.output_dir / "metrics.npy", mode="w+", dtype="float64",
                                       shape=(frames, len(FIELDS)))
    start = start_sample / a.sample_rate
    best_power, best_spectrum = -np.inf, None
    best_index = 0
    began = last_progress = time.monotonic()
    with a.input.open("rb") as handle:
        handle.seek(start_sample * 2)
        for offset in range(0, frames, a.block_frames):
            count = min(a.block_frames, frames - offset)
            raw = handle.read(count * a.nfft * 2)
            if len(raw) != count * a.nfft * 2:
                raise ValueError("input shrank or was truncated during analysis")
            samples = dsp.bytes_to_complex(raw).reshape(count, a.nfft)
            fft = np.fft.fftshift(np.fft.fft(samples * window, axis=1), axes=1)
            # Match dsp.psd's float64 power arithmetic even on NumPy versions
            # whose FFT preserves complex64 input precision.
            power = np.abs(fft).astype(np.float64) ** 2 / (a.sample_rate * norm) * df
            rows = np.full((count, len(FIELDS)), np.nan)
            rows[:, COL["time_s"]] = start + (offset + np.arange(count)) / rate
            channel_db = dsp.to_db(power[:, channel].sum(axis=1))
            guard_db = dsp.to_db(power[:, guard].sum(axis=1) * channel.sum() / guard.sum())
            rows[:, COL["channel_dbfs"]] = channel_db
            rows[:, COL["guard_dbfs"]] = guard_db
            # Echoes' logarithmic levels, normalized to full-spectrum mean.
            levels = dsp.to_db(power) - dsp.to_db(power.mean(axis=1))[:, None]
            valid = power.mean(axis=1) > 0
            echo_signal = levels[:, detection].max(axis=1)
            echo_scan_mean = levels[:, reference].mean(axis=1)
            rows[:, COL["echo_signal_db"]] = np.where(valid, echo_signal, np.nan)
            for j in range(count):
                temporal = floor.push(channel_db[j])
                combined = max(guard_db[j], temporal)
                rows[j, COL["temporal_dbfs"]] = temporal
                rows[j, COL["combined_dbfs"]] = combined
                rows[j, COL["ours_db"]] = channel_db[j] - combined
                rows[j, COL["ours_warm"]] = floor.warm
                if valid[j]:
                    noise = echo.push(echo_scan_mean[j])
                    rows[j, COL["echo_noise_db"]] = noise
                    rows[j, COL["echo_db"]] = echo_signal[j] - noise
            mr_row, sig, noise, score, broadband, passed = meteor_block(
                power, detection, reference, a.meteor_threshold)
            for name, value in (("meteor_signal_dbfs", dsp.to_db(sig)),
                                ("meteor_noise_dbfs", dsp.to_db(noise)),
                                ("meteor_db", score), ("meteor_broadband_ratio", broadband),
                                ("meteor_pass", float(passed))):
                rows[mr_row, COL[name]] = value
            u8 = np.frombuffer(raw, dtype=np.uint8).reshape(count, -1)
            rows[:, COL["rail_fraction"]] = np.mean((u8 == 0) | (u8 == 255), axis=1)
            best = int(np.argmax(channel_db))
            if channel_db[best] > best_power:
                best_power = float(channel_db[best])
                best_index = offset + best
                best_spectrum = power[best].copy()
            result[offset:offset + count] = rows
            if time.monotonic() - last_progress > 5:
                print(f"{offset + count:,}/{frames:,} frames; "
                      f"{(offset + count) / rate:.1f} seconds analysed", flush=True)
                last_progress = time.monotonic()
    result.flush()
    np.savez_compressed(a.output_dir / "strongest_frame_spectrum.npz",
                        offset_hz=freq, bin_power=best_spectrum,
                        time_s=result[best_index, COL["time_s"]])
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    info = {"settings": settings, "columns": list(FIELDS), "sources": SOURCES,
            "scope": "Matched-FFT estimator comparison; not native application emulation or meteor classification",
            "preset_note": "FM frequency ranges are adaptations" if a.preset == "fm" else "Beacon frequency ranges on a shared frontend",
            "input_bytes_at_start": size, "frames": frames, "frame_seconds": 1 / rate,
            "analysed_start_s": start, "analysed_duration_s": frames / rate,
            "trailing_samples_not_analysed": wanted - frames * a.nfft,
            "peak_band_hz": list(peak_band), "reference_band_hz": list(reference_band),
            "bin_counts": {"channel": int(channel.sum()), "guard": int(guard.sum()),
                           "peak": int(detection.sum()), "reference": int(reference.sum())},
            "strongest_frame_index": best_index, "elapsed_seconds": time.monotonic() - began,
            "warnings": ["Raw file has no intrinsic frequency, gain, timestamp or dropped-sample metadata.",
                         "Thresholds and score scales differ; exceedances are not confirmed meteors.",
                         "Our temporal history starts at the selected interval, not the original recorder start.",
                         "Rail hits are diagnostics; absence of hits does not rule out analogue overload."]}
    for key, threshold in (("ours_db", a.our_threshold_db),
                           ("meteor_db", 10 * np.log10(a.meteor_threshold)),
                           ("echo_db", a.echo_threshold_db)):
        values = result[:, COL[key]]
        good = np.isfinite(values)
        info[key] = {"valid_measurements": int(good.sum()),
                     "median_db": float(np.median(values[good])) if good.any() else None,
                     "max_db": float(values[good].max()) if good.any() else None,
                     "threshold_db": threshold,
                     "above_threshold": int(np.sum(values > threshold)) if threshold is not None else None}
    info["meteor_blocks_after_broadband_check"] = int(np.nansum(result[:, COL["meteor_pass"]]))
    info["median_reference_minus_channel_db"] = float(np.median(result[:, COL["combined_dbfs"]] - result[:, COL["channel_dbfs"]]))
    info["rail_fraction_mean"] = float(np.mean(result[:, COL["rail_fraction"]]))
    (a.output_dir / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    make_plots(result, info, a)
    print(f"Saved comparison to {a.output_dir.resolve()}")
    return info


def envelope(t, values, limit):
    """Min/median/max for display only; never subtract independent maxima."""
    good = np.isfinite(values)
    t, values = t[good], values[good]
    if not len(t):
        return np.array([]), np.empty((0, 3))
    edges = np.unique(np.linspace(0, len(t), min(limit, len(t)) + 1).astype(int))
    x, y = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        x.append(np.mean(t[lo:hi]))
        y.append((np.min(values[lo:hi]), np.median(values[lo:hi]), np.max(values[lo:hi])))
    return np.asarray(x), np.asarray(y)


def make_plots(data, info, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    methods = [
        ("Our recorder", [("channel_dbfs", "Channel total", "#2563eb"),
                          ("guard_dbfs", "Nearby-band reference", "#16a34a"),
                          ("temporal_dbfs", "Historical percentile", "#9333ea")],
         "ours_db", a.our_threshold_db, "Integrated power / dBFS"),
        ("MeteorRadio-style", [("meteor_signal_dbfs", "Spectral peak", "#2563eb"),
                               ("meteor_noise_dbfs", "Spectral median", "#16a34a")],
         "meteor_db", 10 * np.log10(a.meteor_threshold), "Bin power / dBFS"),
        ("Echoes-style (differential)", [("echo_signal_db", "Normalized spectral peak", "#2563eb"),
                                         ("echo_noise_db", "Smoothed spectral reference", "#16a34a")],
         "echo_db", a.echo_threshold_db, "Spectrum-relative level / dB"),
    ]
    all_t = data[:, COL["time_s"]]
    center = all_t[info["strongest_frame_index"]]
    for name, bounds in (("comparison.png", None),
                         ("strongest_excursion.png", (center - a.zoom_seconds / 2, center + a.zoom_seconds / 2))):
        # Include neighbouring MeteorRadio blocks in detail view: their score
        # represents a whole analysis block, not every 8-ms frame.
        keep = slice(None) if bounds is None else slice(
            max(0, np.searchsorted(all_t, bounds[0]) - a.block_frames),
            min(len(all_t), np.searchsorted(all_t, bounds[1]) + a.block_frames))
        view = data[keep]
        t = view[:, COL["time_s"]]
        xlabel = "Seconds from start of IQ file"
        if a.start_utc:
            origin = datetime.fromisoformat(a.start_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
            t = mdates.date2num(origin) + t / 86400
            xlabel = "UTC (from supplied file start time)"
        fig, axes = plt.subplots(3, 2, figsize=(15, 11), layout="constrained")
        for row, (title, series, score_name, threshold, ylabel) in enumerate(methods):
            for field, label, color in series:
                x, y = envelope(t, view[:, COL[field]], a.plot_points)
                if len(x):
                    axes[row, 0].fill_between(x, y[:, 0], y[:, 2], color=color, alpha=.15)
                    axes[row, 0].plot(x, y[:, 1], color=color, lw=1, label=label,
                                      marker="." if row == 1 else None,
                                      linestyle="none" if row == 1 else "-")
            x, y = envelope(t, view[:, COL[score_name]], a.plot_points)
            if len(x):
                axes[row, 1].fill_between(x, y[:, 0], y[:, 2], color="#d97706", alpha=.25)
                axes[row, 1].plot(x, y[:, 1], color="#b45309", lw=1,
                                 label="Block scores; shade = min/max" if row == 1 else "Score median; shade = min/max",
                                 marker="." if row == 1 else None,
                                 linestyle="none" if row == 1 else "-")
                # Retain maxima as a thin line so brief excursions remain visible.
                axes[row, 1].plot(x, y[:, 2], color="#b45309", lw=.5, alpha=.65,
                                 marker="." if row == 1 else None,
                                 linestyle="none" if row == 1 else "-")
            if threshold is not None:
                axes[row, 1].axhline(threshold, color="#dc2626", ls="--", lw=1,
                                     label=f"Threshold {threshold:.2f} dB")
            if row == 1:
                bad = (view[:, COL["meteor_broadband_ratio"]] > 3) & np.isfinite(view[:, COL["meteor_db"]])
                axes[row, 1].scatter(t[bad], view[bad, COL["meteor_db"]], marker="x", s=18,
                                     color="#dc2626", label="Broadband ratio > 3")
            for col in (0, 1):
                ax = axes[row, col]
                ax.set_title(title + (" | measured levels" if col == 0 else " | detection statistic"))
                ax.set_ylabel(ylabel if col == 0 else "Peak/reference or channel/reference / dB")
                ax.set_xlabel(xlabel)
                ax.grid(alpha=.2)
                handles, _ = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc="best", fontsize=8)
                if a.start_utc:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
                if len(t) > 1 and t[-1] > t[0]:
                    ax.set_xlim(t[0], t[-1])
        fig.suptitle(f"IQ estimator comparison | {a.center_freq / 1e6:g} MHz | {a.preset.upper()} ranges\n"
                     f"Shared {a.window} FFT: {a.nfft} samples, {a.sample_rate / a.nfft:.2f} Hz/bin; "
                     f"MeteorRadio blocks: {a.block_frames * a.nfft / a.sample_rate:.3f} s\n"
                     "Lines = display medians; shading = range. Different score definitions; no automatic winner.", fontsize=12)
        fig.savefig(a.output_dir / name, dpi=150)
        plt.close(fig)


def main():
    try:
        analyse(parser().parse_args())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
