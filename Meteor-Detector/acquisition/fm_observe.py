#!/usr/bin/env python3
"""
Phase 1: the always-on FM meteor scatter recorder.

Design decisions worth knowing before you change anything:

Gap-free capture
    librtlsdr streams continuously into a callback; the callback does nothing
    but hand raw bytes to a queue, and a worker thread does the DSP. Any design
    that reads, processes, then sleeps for a fixed interval is deaf for the
    duration of the sleep -- and an underdense meteor ping lasts 50-500 ms, so
    a duty-cycled recorder loses most of them and mis-shapes the rest.

Two tiers of storage
    Tier 1  power/noise/SNR at ~125 Hz, written as compressed .npz chunks.
            About 100-270 MB/day. This runs for years on an SD card.
    Tier 2  a rolling ring buffer of raw IQ held in RAM. When Tier 1 triggers,
            the seconds either side of the event are dumped to disk. This is
            what lets you go back and prove an event was a meteor -- rise time,
            decay shape, Doppler -- instead of trusting a threshold.

    Storing raw IQ continuously is not an option: 1.024 MS/s of uint8 IQ is
    2.0 MB/s, or 177 GB/day.

Decimation
    The FFT is the decimator. 1.024 MS/s -> 125 power values/s is 8192:1, and
    it is lossless for envelope detection because it INTEGRATES energy rather
    than discarding samples. Never decimate by keeping every Nth sample: a
    20 ms ping can fall entirely in the gap. Both mean and max are kept per
    frame so a short event cannot be averaged into the floor.

Output schema matches the existing plot_npz_utc.py, so that plotter works
against these files unchanged.
"""

import argparse
import ctypes
import json
import os
import queue
import shutil
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

import warnings

import numpy as np

# pyrtlsdr 0.3.x imports pkg_resources, which setuptools deprecated. The pin is
# deliberate (see requirements.txt); the warning is noise on every run.
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")

import dsp

RUNNING = threading.Event()
RUNNING.set()


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Trigger state machine
# --------------------------------------------------------------------------

class Trigger:
    """Hysteretic threshold detector with a refractory period.

    Hysteresis stops a signal hovering at the threshold from chattering into
    hundreds of one-frame events. The refractory period stops the long decay
    tail of an overdense trail from re-triggering as a stream of new events.
    """

    def __init__(self, frame_rate, threshold_db=6.0, hysteresis_db=2.0,
                 min_frames=2, refractory_s=1.0):
        self.threshold = threshold_db
        self.release = threshold_db - hysteresis_db
        self.min_frames = min_frames
        self.refractory = int(refractory_s * frame_rate)
        self.active = False
        self._run = 0
        self._cool = 0
        self.events = 0

    def update(self, snr_db):
        """Feed one frame. Returns (is_flagged, rising_edge, falling_edge)."""
        rising = falling = False
        if self._cool > 0:
            self._cool -= 1
            return False, False, False

        if not self.active:
            if snr_db > self.threshold:
                self._run += 1
                if self._run >= self.min_frames:
                    self.active = True
                    self.events += 1
                    rising = True
            else:
                self._run = 0
        else:
            if snr_db < self.release:
                self.active = False
                self._run = 0
                self._cool = self.refractory
                falling = True

        return self.active, rising, falling


# --------------------------------------------------------------------------
# Chunk writer (Tier 1)
# --------------------------------------------------------------------------

class ChunkWriter:
    """Accumulates per-frame metrics and flushes compressed .npz chunks."""

    FIELDS = ("t_utc_ns", "power_dbfs", "noise_dbfs", "peak_dbfs", "snr_db")

    def __init__(self, outdir, station, meta, chunk_seconds, frame_rate):
        self.outdir = outdir
        self.station = station
        self.meta = meta
        self.limit = max(1, int(chunk_seconds * frame_rate))
        os.makedirs(outdir, exist_ok=True)
        self.index = 0
        self._reset()

    def _reset(self):
        self.t = []
        self.power = []
        self.noise = []
        self.peak = []
        self.snr = []
        self.trig = []

    def add(self, t_ns, power, noise, peak, snr, trig):
        self.t.append(t_ns)
        self.power.append(power)
        self.noise.append(noise)
        self.peak.append(peak)
        self.snr.append(snr)
        self.trig.append(trig)
        return len(self.t) >= self.limit

    def flush(self):
        if not self.t:
            return None
        stamp = datetime.fromtimestamp(self.t[0] / 1e9, tz=timezone.utc)
        name = (f"{self.station}_{stamp.strftime('%Y%m%d_%H%M%S')}"
                f"_chunk{self.index:04d}.npz")
        path = os.path.join(self.outdir, name)
        np.savez_compressed(
            path,
            t_utc_ns=np.asarray(self.t, dtype=np.int64),
            power_dbfs=np.asarray(self.power, dtype=np.float32),
            noise_dbfs=np.asarray(self.noise, dtype=np.float32),
            peak_dbfs=np.asarray(self.peak, dtype=np.float32),
            snr_db=np.asarray(self.snr, dtype=np.float32),
            trigger=np.asarray(self.trig, dtype=bool),
            **{k: np.asarray([v]) for k, v in self.meta.items()},
        )
        n = len(self.t)
        self.index += 1
        self._reset()
        return path, n


# --------------------------------------------------------------------------
# IQ ring buffer (Tier 2)
# --------------------------------------------------------------------------

class IQRing:
    """Rolling buffer of raw uint8 IQ blocks, dumped around a trigger.

    Blocks are stored exactly as librtlsdr delivered them, so a dump is
    byte-identical to what `rtl_sdr` would have written and can be read with
    dsp.bytes_to_complex().
    """

    def __init__(self, outdir, pre_s, post_s, frame_rate, block_bytes,
                 max_events_per_hour=60, min_free_mb=2048):
        self.outdir = outdir
        self.pre = max(1, int(pre_s * frame_rate))
        self.post = max(1, int(post_s * frame_rate))
        self.frame_rate = frame_rate
        self.block_bytes = block_bytes
        self.max_per_hour = max_events_per_hour
        self.min_free_mb = min_free_mb
        os.makedirs(outdir, exist_ok=True)
        self.ring = deque(maxlen=self.pre)
        self.capturing = False
        self._blocks = None
        self._remaining = 0
        self._meta = None
        self._recent = deque()
        self.written = 0
        self.skipped = 0

    def _budget_ok(self, now):
        while self._recent and now - self._recent[0] > 3600:
            self._recent.popleft()
        if len(self._recent) >= self.max_per_hour:
            return False, "event rate cap"
        free_mb = shutil.disk_usage(self.outdir).free / 1e6
        if free_mb < self.min_free_mb:
            return False, f"low disk ({free_mb:.0f} MB free)"
        return True, ""

    def push(self, block, t_ns, snr, trig_rise, trig_fall):
        """Feed every block. Handles pre-roll, capture and write-out."""
        if self.capturing:
            self._blocks.append(block)
            if not self._meta["ended"]:
                if trig_fall:
                    self._meta["ended"] = True
                self._meta["peak_snr"] = max(self._meta["peak_snr"], snr)
                self._meta["duration_s"] = (
                    (t_ns - self._meta["t_start_ns"]) / 1e9)
            else:
                self._remaining -= 1
                if self._remaining <= 0:
                    self._write()
            return

        self.ring.append(block)
        if trig_rise:
            now = time.time()
            allowed, why = self._budget_ok(now)
            if not allowed:
                self.skipped += 1
                log(f"  IQ capture skipped: {why}")
                return
            self._recent.append(now)
            self.capturing = True
            self._blocks = list(self.ring)
            self._remaining = self.post
            self._meta = {
                "t_start_ns": t_ns,
                "peak_snr": snr,
                "duration_s": 0.0,
                "ended": False,
                "pre_s": len(self.ring) / self.frame_rate,
                "post_s": self.post / self.frame_rate,
            }

    def _write(self):
        stamp = datetime.fromtimestamp(self._meta["t_start_ns"] / 1e9,
                                       tz=timezone.utc)
        base = f"event_{stamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        iq_path = os.path.join(self.outdir, base + ".iq")
        with open(iq_path, "wb") as f:
            for b in self._blocks:
                f.write(b)
        meta = dict(self._meta)
        meta.pop("ended", None)
        meta["t_start_utc"] = stamp.isoformat()
        meta["bytes"] = len(self._blocks) * self.block_bytes
        meta["format"] = "uint8 interleaved I,Q (rtl_sdr native)"
        with open(os.path.join(self.outdir, base + ".json"), "w") as f:
            json.dump(meta, f, indent=2)
        log(f"  IQ event saved: {base}.iq "
            f"({meta['bytes']/1e6:.1f} MB, peak SNR {meta['peak_snr']:.1f} dB, "
            f"{meta['duration_s']*1e3:.0f} ms)")
        self.written += 1
        self.capturing = False
        self._blocks = None
        self._meta = None

    def abort(self):
        if self.capturing and self._blocks:
            self._write()


# --------------------------------------------------------------------------
# Main acquisition
# --------------------------------------------------------------------------

def run(args):
    try:
        from rtlsdr import RtlSdr
    except ImportError:
        log("pyrtlsdr is not installed. pip install pyrtlsdr")
        return 2

    fs = args.sample_rate
    nfft = args.nfft
    block_bytes = nfft * 2                 # uint8 I + uint8 Q per sample
    if block_bytes % 512:
        log(f"block size {block_bytes} is not a multiple of 512; "
            f"librtlsdr requires that. Choose an --nfft that is a multiple of 256.")
        return 2
    frame_rate = fs / nfft

    try:
        sdr = RtlSdr(device_index=args.device)
    except Exception as e:
        log(f"could not open device {args.device}: {e}")
        log("Run  rf_check.py --list-devices  to see what is attached.")
        return 2
    sdr.sample_rate = fs
    sdr.center_freq = args.freq

    for meth, val in (("set_agc_mode", False), ("set_manual_gain_enabled", True)):
        if hasattr(sdr, meth):
            try:
                getattr(sdr, meth)(val)
            except Exception as e:
                log(f"warning: {meth}({val}) failed: {e}")

    if isinstance(args.gain, str) and args.gain.lower() == "auto":
        log("refusing automatic gain: AGC erases the power variations that ARE "
            "the signal. Pick a fixed value from:")
        log(f"  {getattr(sdr, 'valid_gains_db', 'unknown')}")
        sdr.close()
        return 2
    sdr.gain = float(args.gain)

    metrics = dsp.ChannelMetrics(fs, nfft, args.freq,
                                 channel_bw=args.channel_bw,
                                 guard_lo=args.guard_lo,
                                 guard_hi=args.guard_hi,
                                 window=args.window)
    floor = dsp.RollingFloor(frame_rate, seconds=args.floor_seconds,
                             percentile=args.floor_percentile)
    trig = Trigger(frame_rate, args.threshold_db, args.hysteresis_db,
                   args.min_frames, args.refractory_s)

    meta = {
        "station": args.station,
        "center_freq_hz": args.freq,
        "sample_rate_hz": fs,
        "nfft": nfft,
        "gain_db": sdr.gain,
        "channel_bw_hz": args.channel_bw,
        "threshold_db": args.threshold_db,
        "window": args.window,
        "frame_rate_hz": frame_rate,
    }
    writer = ChunkWriter(args.output_dir, args.station, meta,
                         args.chunk_seconds, frame_rate)
    ring = (IQRing(os.path.join(args.output_dir, "events"),
                   args.pre_seconds, args.post_seconds, frame_rate,
                   block_bytes, args.max_events_per_hour, args.min_free_mb)
            if args.save_iq else None)

    log("=" * 62)
    log(f"station     {args.station}")
    log(f"frequency   {args.freq/1e6:.4f} MHz")
    log(f"device      index {args.device}")
    log(f"gain        {sdr.gain} dB (fixed, AGC off)")
    for line in metrics.describe().split("\n"):
        log(f"            {line}")
    log(f"floor       {args.floor_percentile:.0f}th pct over "
        f"{args.floor_seconds:.0f} s")
    log(f"trigger     >{args.threshold_db} dB SNR for {args.min_frames} frames, "
        f"release <{args.threshold_db - args.hysteresis_db} dB")
    log(f"tier 1      {args.output_dir}  ({args.chunk_seconds:.0f} s chunks)")
    if ring:
        ram = ring.pre * block_bytes / 1e6
        log(f"tier 2      IQ ring {args.pre_seconds:.0f}s pre / "
            f"{args.post_seconds:.0f}s post  ({ram:.0f} MB RAM)")
    else:
        log("tier 2      disabled (--save-iq to enable)")
    log("=" * 62)

    # --- streaming ---------------------------------------------------------
    q = queue.Queue(maxsize=args.queue_depth)
    stats = {"blocks": 0, "drops": 0}

    def cb(buf, _ctx):
        """librtlsdr callback. Must return fast: copy and hand off, nothing else."""
        try:
            q.put_nowait(bytes(buf))
        except queue.Full:
            stats["drops"] += 1

    t0_ns = None
    frame = 0
    last_report = time.monotonic()

    def worker():
        nonlocal t0_ns, frame, last_report
        while RUNNING.is_set() or not q.empty():
            try:
                raw = q.get(timeout=0.5)
            except queue.Empty:
                continue

            if t0_ns is None:
                t0_ns = time.time_ns()

            samples = dsp.bytes_to_complex(raw)
            if samples.size < nfft:
                continue
            power, noise_inst, peak = metrics.measure(samples)

            # Two noise references: the guard bands give an instantaneous
            # figure through the same front end, the rolling percentile gives
            # a stable one. The higher of the two is the conservative choice.
            nf = floor.push(power)
            noise = max(noise_inst, nf) if nf is not None else noise_inst
            snr = power - noise

            flagged, rise, fall = trig.update(snr)
            t_ns = t0_ns + int(frame * nfft / fs * 1e9)

            if ring is not None:
                ring.push(raw, t_ns, snr, rise, fall)
            if rise:
                log(f"  TRIGGER #{trig.events}  SNR {snr:.1f} dB  "
                    f"power {power:.1f}  floor {noise:.1f}")

            if writer.add(t_ns, power, noise, peak, snr, bool(flagged)):
                res = writer.flush()
                if res:
                    log(f"  chunk written: {os.path.basename(res[0])} "
                        f"({res[1]} frames)")

            frame += 1
            stats["blocks"] += 1

            now = time.monotonic()
            if now - last_report >= args.report_seconds:
                last_report = now
                dr = stats["drops"]
                pct = 100.0 * dr / max(1, stats["blocks"] + dr)
                warm = "" if floor.warm else "  [floor warming up]"
                msg = (f"power {power:7.2f}  floor {noise:7.2f}  "
                       f"snr {snr:5.2f} dB  events {trig.events}  "
                       f"frames {stats['blocks']}{warm}")
                if dr:
                    msg += f"  DROPPED {dr} ({pct:.2f}%)"
                log(msg)

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    def stop(signum, _frame):
        log(f"signal {signum} -- shutting down")
        RUNNING.clear()
        try:
            sdr.cancel_read_async()
        except Exception:
            pass

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        sdr.read_bytes_async(cb, block_bytes)
    except Exception as e:
        if RUNNING.is_set():
            log(f"stream error: {e}")
    finally:
        RUNNING.clear()
        th.join(timeout=10)
        if ring is not None:
            ring.abort()
        res = writer.flush()
        if res:
            log(f"final chunk: {os.path.basename(res[0])} ({res[1]} frames)")
        try:
            sdr.close()
        except Exception:
            pass

    log("=" * 62)
    log(f"frames processed {stats['blocks']}   dropped {stats['drops']}")
    log(f"triggers {trig.events}")
    if ring is not None:
        log(f"IQ events written {ring.written}, skipped {ring.skipped}")
    if stats["drops"]:
        pct = 100.0 * stats["drops"] / max(1, stats["blocks"] + stats["drops"])
        log(f"WARNING: {pct:.2f}% of blocks were dropped. Timing after each")
        log("drop is uncertain. Lower --sample-rate or raise --queue-depth.")
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Always-on FM meteor scatter recorder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-f", "--freq", type=float, default=107.1e6,
                   help="centre frequency, Hz")
    p.add_argument("-s", "--sample-rate", type=float, default=1.024e6,
                   help="Hz. Valid RTL-SDR ranges are 225001-300000 and "
                        "900001-3200000; 1.024e6 leaves room for guard bands")
    p.add_argument("-D", "--device", type=int, default=0,
                   help="dongle index when more than one is attached")
    p.add_argument("-g", "--gain", default="35",
                   help="FIXED tuner gain in dB. 'auto' is rejected by design")
    p.add_argument("--nfft", type=int, default=8192,
                   help="FFT length; sets the frame rate (fs/nfft)")
    p.add_argument("--window", default="hann",
                   choices=["hann", "blackmanharris", "rect"])
    p.add_argument("--channel-bw", type=float, default=180e3,
                   help="integration bandwidth, Hz. One FM channel is ~180 kHz")
    p.add_argument("--guard-lo", type=float, default=150e3,
                   help="inner edge of the noise guard bands, Hz from centre")
    p.add_argument("--guard-hi", type=float, default=400e3,
                   help="outer edge of the noise guard bands, Hz from centre")
    p.add_argument("--floor-seconds", type=float, default=60.0)
    p.add_argument("--floor-percentile", type=float, default=10.0)
    p.add_argument("--threshold-db", type=float, default=6.0)
    p.add_argument("--hysteresis-db", type=float, default=2.0)
    p.add_argument("--min-frames", type=int, default=2,
                   help="consecutive frames above threshold before triggering")
    p.add_argument("--refractory-s", type=float, default=1.0)
    p.add_argument("--station", default="FM_STATION")
    p.add_argument("-o", "--output-dir", default="./fm_observations")
    p.add_argument("--chunk-seconds", type=float, default=600.0)
    p.add_argument("--save-iq", action="store_true",
                   help="enable Tier 2 triggered raw IQ capture")
    p.add_argument("--pre-seconds", type=float, default=3.0)
    p.add_argument("--post-seconds", type=float, default=5.0)
    p.add_argument("--max-events-per-hour", type=int, default=60)
    p.add_argument("--min-free-mb", type=float, default=2048)
    p.add_argument("--queue-depth", type=int, default=256)
    p.add_argument("--report-seconds", type=float, default=30.0)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
