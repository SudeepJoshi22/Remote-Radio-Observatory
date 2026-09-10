#!/usr/bin/env python3
"""
End-to-end test of the acquisition pipeline against a synthetic sky.

No hardware. A fake librtlsdr streams uint8 IQ containing a quiet noise floor
with meteor-like pings injected at known times, and the test asserts that
fm_observe.py finds exactly those pings, writes a well-formed Tier-1 chunk and
a Tier-2 IQ capture, and that the resulting .npz loads in plot_npz_utc.py.

Run this after any change to dsp.py or fm_observe.py:

    python3 test_pipeline.py
"""

import os
import shutil
import sys
import tempfile
import types

import numpy as np

FS = 1.024e6
NFFT = 8192
FRAME_RATE = FS / NFFT           # 125 Hz
BLOCK_BYTES = NFFT * 2

DURATION_S = 20.0
N_BLOCKS = int(DURATION_S * FRAME_RATE)

# Pings at these times, as (start_s, duration_s, peak_snr_db).
PINGS = [(5.0, 0.25, 18.0), (10.0, 0.60, 12.0), (15.0, 0.15, 25.0)]

NOISE_SIGMA = 0.004              # well clear of the 1/255 quantisation step


def _band_limited(n, rng, bw_hz=180e3):
    """Noise confined to one FM channel: spectrally like a real broadcast."""
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(n, d=1.0 / FS)
    X[np.abs(f) > bw_hz / 2] = 0
    y = np.fft.ifft(X)
    return y / np.sqrt(np.mean(np.abs(y) ** 2))


def _envelope(t):
    """Sum of the injected pings at time t: fast rise, exponential decay."""
    amp = 0.0
    for t0, dur, snr_db in PINGS:
        if t0 <= t < t0 + dur * 6:
            rise = min(1.0, (t - t0) / 0.02)          # 20 ms rise
            decay = np.exp(-(t - t0) / dur)
            lin = 10 ** (snr_db / 20.0)
            amp += NOISE_SIGMA * lin * rise * decay
    return amp


def make_fake_rtlsdr():
    """A drop-in stand-in for the rtlsdr module."""
    rng = np.random.default_rng(12345)

    class FakeRtlSdr:
        def __init__(self, device_index=0, **kwargs):
            self.device_index = device_index
            self.sample_rate = FS
            self.center_freq = 100e6
            self.gain = 0.0
            self.valid_gains_db = [0.0, 15.0, 25.0, 35.0, 49.6]
            self._cancel = False

        def set_agc_mode(self, on):      pass
        def set_manual_gain_enabled(self, on): pass
        def close(self):                 pass
        def cancel_read_async(self):     self._cancel = True

        def read_bytes_async(self, cb, num_bytes):
            n = num_bytes // 2
            for blk in range(N_BLOCKS):
                if self._cancel:
                    break
                t = blk * n / FS
                noise = (rng.normal(scale=NOISE_SIGMA, size=n)
                         + 1j * rng.normal(scale=NOISE_SIGMA, size=n))
                a = _envelope(t)
                sig = a * _band_limited(n, rng) if a > 0 else 0.0
                iq = noise + sig

                # Quantise exactly as the dongle does: uint8, zero at 127.5.
                i8 = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
                q8 = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
                out = np.empty(n * 2, dtype=np.uint8)
                out[0::2] = i8.astype(np.uint8)
                out[1::2] = q8.astype(np.uint8)
                cb(out.tobytes(), None)

    mod = types.ModuleType("rtlsdr")
    mod.RtlSdr = FakeRtlSdr
    return mod


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.modules["rtlsdr"] = make_fake_rtlsdr()

    import fm_observe

    outdir = tempfile.mkdtemp(prefix="rro_test_")
    print(f"synthetic sky: {DURATION_S:.0f} s, {len(PINGS)} pings injected")
    for t0, dur, snr in PINGS:
        print(f"   t={t0:5.1f}s  tau={dur*1e3:4.0f} ms  peak SNR {snr:.0f} dB")
    print(f"output: {outdir}\n" + "-" * 62)

    args = types.SimpleNamespace(
        freq=107.1e6, sample_rate=FS, gain="35", device=0,
        nfft=NFFT, window="hann",
        channel_bw=180e3, guard_lo=150e3, guard_hi=400e3,
        floor_seconds=3.0, floor_percentile=10.0,
        threshold_db=6.0, hysteresis_db=2.0, min_frames=2, refractory_s=1.0,
        station="TEST", output_dir=outdir, chunk_seconds=DURATION_S * 2,
        save_iq=True, pre_seconds=1.0, post_seconds=1.5,
        max_events_per_hour=60, min_free_mb=1,
        queue_depth=N_BLOCKS + 64, report_seconds=1e9,
    )

    fm_observe.RUNNING.set()
    rc = fm_observe.run(args)
    print("-" * 62)

    failures = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f"   {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    check("run completed cleanly", rc == 0, f"rc={rc}")

    chunks = sorted(f for f in os.listdir(outdir) if f.endswith(".npz"))
    check("Tier 1 chunk written", len(chunks) >= 1, f"{len(chunks)} file(s)")
    if not chunks:
        shutil.rmtree(outdir, ignore_errors=True)
        return 1

    d = np.load(os.path.join(outdir, chunks[0]))
    required = ["t_utc_ns", "power_dbfs", "noise_dbfs", "peak_dbfs",
                "snr_db", "trigger", "station", "center_freq_hz",
                "threshold_db"]
    missing = [k for k in required if k not in d.files]
    check("schema matches plot_npz_utc.py", not missing,
          f"missing {missing}" if missing else "all 9 fields present")

    n = len(d["power_dbfs"])
    expect = int(DURATION_S * FRAME_RATE)
    check("frame count", abs(n - expect) <= 4, f"{n} frames (expected ~{expect})")

    check("timestamps monotonic", bool(np.all(np.diff(d["t_utc_ns"]) > 0)))

    snr = d["snr_db"]
    trig = d["trigger"]
    # Group flagged frames into events.
    edges = np.diff(np.concatenate(([0], trig.astype(int), [0])))
    starts = np.where(edges == 1)[0]
    n_events = len(starts)
    check("detected every injected ping", n_events == len(PINGS),
          f"found {n_events}, injected {len(PINGS)}")

    if n_events == len(PINGS):
        t0 = d["t_utc_ns"][0]
        detected = [(d["t_utc_ns"][s] - t0) / 1e9 for s in starts]
        for (want, _, want_snr), got in zip(PINGS, detected):
            err = abs(got - want)
            check(f"ping at t={want:.1f}s located", err < 0.15,
                  f"detected {got:.2f}s (error {err*1e3:.0f} ms)")

    peak_snr = float(np.max(snr))
    check("peak SNR in a sane range", 10 < peak_snr < 45,
          f"{peak_snr:.1f} dB")

    quiet = snr[~trig]
    check("quiet-time SNR near zero", abs(float(np.median(quiet))) < 3.0,
          f"median {np.median(quiet):.2f} dB")

    ev_dir = os.path.join(outdir, "events")
    iqs = sorted(f for f in os.listdir(ev_dir)) if os.path.isdir(ev_dir) else []
    n_iq = len([f for f in iqs if f.endswith(".iq")])
    check("Tier 2 IQ captures written", n_iq == len(PINGS),
          f"{n_iq} .iq file(s)")

    if n_iq:
        import dsp
        first = [f for f in iqs if f.endswith(".iq")][0]
        raw = open(os.path.join(ev_dir, first), "rb").read()
        iq = dsp.bytes_to_complex(raw)
        check("captured IQ decodes to sane amplitudes",
              0 < float(np.max(np.abs(iq))) <= 1.5,
              f"{len(iq)} samples, max |z| = {np.max(np.abs(iq)):.3f}")

    # The existing plotter must be able to consume this unchanged. Exercise
    # exactly the field accesses plot_npz_utc.plot_observation() performs, so
    # the check is meaningful even where matplotlib is not installed.
    try:
        dd = np.load(os.path.join(outdir, chunks[0]))
        _ = dd["t_utc_ns"], dd["power_dbfs"], dd["noise_dbfs"]
        _ = dd["peak_dbfs"], dd["snr_db"], dd["trigger"]
        station = dd["station"][0] if len(dd["station"]) > 0 else "Unknown"
        cf = dd["center_freq_hz"][0]
        thr = dd["threshold_db"][0]
        ok_fields = bool(station) and cf > 0 and thr > 0
        check("plot_npz_utc.py field access pattern works", ok_fields,
              f"station={station} freq={cf/1e6:.3f} MHz threshold={thr} dB")
    except Exception as e:
        check("plot_npz_utc.py field access pattern works", False, repr(e))

    plotter = os.path.abspath(os.path.join(here, "plot_npz_utc.py"))
    have_mpl = True
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        have_mpl = False
    if os.path.exists(plotter) and have_mpl:
        import subprocess
        r = subprocess.run(
            [sys.executable, plotter, "--dir", outdir, "--list"],
            capture_output=True, text=True)
        check("plot_npz_utc.py loads the output", r.returncode == 0
              and ".npz" in r.stdout)
    else:
        print("  SKIP  plot_npz_utc.py subprocess check "
              "(matplotlib not installed here)")

    print("-" * 62)
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        print(f"artefacts kept for inspection: {outdir}")
        return 1
    print("ALL CHECKS PASSED")
    print(f"cleaning up {outdir}")
    shutil.rmtree(outdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
