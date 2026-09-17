#!/usr/bin/env python3
"""
Generate a synthetic multi-hour observation for trying the viewer before any
hardware is connected. Writes the exact schema fm_observe.py writes, chunked
the same way, so the viewer's code path is identical for demo and real data --
there is no separate "demo mode" in server.py to keep in sync.

    python3 demo_data.py --hours 24 --out ~/demo_observations

Then:

    python3 server.py --dir ~/demo_observations
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

import numpy as np

FRAME_RATE = 125.0          # matches fs=1.024e6, nfft=8192 at fm_observe.py defaults
CHUNK_SECONDS = 600.0
FLOOR_DBFS = -98.0          # roughly what a bare V4 reads on an empty channel


def _pings(n_frames, rng, rate_per_hour):
    """Vectorised meteor-like excursions: fast rise, exponential decay."""
    hours = n_frames / FRAME_RATE / 3600.0
    n_events = rng.poisson(rate_per_hour * hours)
    starts = rng.uniform(0, n_frames, size=n_events).astype(np.float64)
    taus_s = rng.uniform(0.08, 1.2, size=n_events)          # decay constant
    peaks_db = rng.uniform(6.0, 28.0, size=n_events)        # peak SNR

    excess = np.zeros(n_frames)
    idx = np.arange(n_frames)
    # A handful of events per day, so a per-event window is cheap even though
    # this loop is not vectorised across events.
    for t0, tau, peak in zip(starts, taus_s, peaks_db):
        tau_frames = tau * FRAME_RATE
        window = slice(max(0, int(t0 - 5)), min(n_frames, int(t0 + tau_frames * 8)))
        t = idx[window] - t0
        rise = np.clip(t / (0.02 * FRAME_RATE), 0, 1)       # ~20 ms rise
        decay = np.exp(-np.clip(t, 0, None) / tau_frames)
        decay[t < 0] = 0
        excess[window] += peak * rise * decay
    return excess, n_events


def generate(hours, station, freq_hz, threshold_db, rate_per_hour, seed):
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 * FRAME_RATE)

    hrs = np.arange(n) / FRAME_RATE / 3600.0
    # Slow daily wobble (thermal drift, not meant to be the sidereal signal)
    # plus a longer slow drift, plus frame-to-frame receiver noise.
    drift = 0.6 * np.sin(2 * np.pi * hrs / 24.0) + 0.3 * np.sin(2 * np.pi * hrs / 61.0)
    noise = rng.normal(scale=0.35, size=n)
    floor = FLOOR_DBFS + drift + noise

    excess, n_events = _pings(n, rng, rate_per_hour)
    power = floor + excess
    peak = power + rng.uniform(2.0, 6.0, size=n)   # single strongest bin, always a bit hotter
    snr = power - floor

    # Simple hysteretic flag, just for a plausible trigger overlay -- not a
    # reimplementation of fm_observe.py's Trigger class.
    trigger = snr > threshold_db

    t0 = datetime.now(timezone.utc) - timedelta(hours=hours)
    t_ns = (int(t0.timestamp() * 1e9)
            + (np.arange(n) / FRAME_RATE * 1e9).astype(np.int64))

    return t_ns, power.astype(np.float32), floor.astype(np.float32), \
        peak.astype(np.float32), snr.astype(np.float32), trigger, n_events


def write_chunks(outdir, station, freq_hz, threshold_db, t_ns, power, noise,
                  peak, snr, trigger, clean=True):
    os.makedirs(outdir, exist_ok=True)

    if clean:
        # A re-run with a different --seed/--hours otherwise leaves the
        # previous run's chunks in place too, and their timestamps -- both
        # anchored on "now" -- nearly overlap, so the viewer silently merges
        # two unrelated synthetic datasets into one. This is demo/scratch
        # data, so removing the previous run for THIS station is safe; real
        # fm_observe.py output is never touched by this.
        import glob
        stale = glob.glob(os.path.join(outdir, f"{station}_*_chunk*.npz"))
        for f in stale:
            os.remove(f)
        if stale:
            print(f"removed {len(stale)} chunk(s) from a previous "
                  f"'{station}' demo run")

    n = len(t_ns)
    per_chunk = int(CHUNK_SECONDS * FRAME_RATE)
    meta = {
        "station": station, "center_freq_hz": freq_hz,
        "sample_rate_hz": 1.024e6, "nfft": 8192, "gain_db": 49.6,
        "channel_bw_hz": 180e3, "threshold_db": threshold_db,
        "window": "hann", "frame_rate_hz": FRAME_RATE,
    }
    written = 0
    for i, start in enumerate(range(0, n, per_chunk)):
        end = min(n, start + per_chunk)
        stamp = datetime.fromtimestamp(t_ns[start] / 1e9, tz=timezone.utc)
        name = f"{station}_{stamp.strftime('%Y%m%d_%H%M%S')}_chunk{i:04d}.npz"
        np.savez_compressed(
            os.path.join(outdir, name),
            t_utc_ns=t_ns[start:end], power_dbfs=power[start:end],
            noise_dbfs=noise[start:end], peak_dbfs=peak[start:end],
            snr_db=snr[start:end], trigger=trigger[start:end],
            **{k: np.asarray([v]) for k, v in meta.items()},
        )
        written += 1
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--out", default=os.path.expanduser("~/demo_observations"))
    p.add_argument("--station", default="DEMO")
    p.add_argument("--freq", type=float, default=97.9e6)
    p.add_argument("--threshold-db", type=float, default=6.0)
    p.add_argument("--rate-per-hour", type=float, default=2.5,
                   help="mean injected events per hour")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-existing", action="store_true",
                   help="do not remove this station's chunks from a "
                        "previous demo run before writing new ones")
    args = p.parse_args()

    print(f"generating {args.hours:.1f}h at {FRAME_RATE:.0f} Hz "
          f"({int(args.hours*3600*FRAME_RATE):,} frames)...")
    t_ns, power, noise, peak, snr, trigger, n_events = generate(
        args.hours, args.station, args.freq, args.threshold_db,
        args.rate_per_hour, args.seed)

    n_written = write_chunks(args.out, args.station, args.freq,
                             args.threshold_db, t_ns, power, noise, peak,
                             snr, trigger, clean=not args.keep_existing)

    print(f"injected {n_events} synthetic events")
    print(f"wrote {n_written} chunk(s) to {args.out}")
    print(f"\n    python3 server.py --dir {args.out}")


if __name__ == "__main__":
    main()
