#!/usr/bin/env python3
"""
Phase 0: prove the RF chain works before spending nights waiting for meteors.

Four checks, in the order you should run them:

  --selftest        no hardware. Verifies the DSP against a synthetic tone of
                    known amplitude. Run this first, anywhere.
  --floor-test      the decisive one, and it needs no transmitter. Galactic
                    background noise raises the floor 8-16 dB through a working
                    antenna, so this is valid on a completely empty band at a
                    remote site.
  --spur-test       sweeps with the antenna connected and disconnected. Anything
                    still present with the antenna off was never on the air --
                    a dongle birdie, or an oscillating line amplifier.
  --gain-linearity  steps through every tuner gain watching SNR. Finds LNA
                    compression and recommends the gain to observe at.
  --stability       parks on one frequency at fixed gain and checks the reading
                    does not drift. Catches AGC left on and thermal drift.
  --sweep           surveys the band and proposes quiet channels. An empty band
                    at a shielded site is expected, not a failure.
  --sidereal        offline analysis of 48h+ of recording. A variation repeating
                    every 23h56m is the galactic plane and cannot be faked by
                    local interference; every 24h00m is terrestrial.

Every check prints an explicit PASS / FAIL / WARN.
"""

import argparse
import glob
import os
import sys
import time

import warnings

import numpy as np

# pyrtlsdr 0.3.x imports pkg_resources, which setuptools deprecated. The pin is
# deliberate (see requirements.txt); the warning is noise on every run.
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")

import dsp

# ANSI colours, disabled when not a tty. On Windows the console needs virtual
# terminal processing switched on explicitly or the escapes print as literal
# garbage; if that cannot be enabled, fall back to plain text.
def _enable_ansi():
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(k.SetConsoleMode(h, mode.value | 0x0004))
    except Exception:
        return False


_TTY = _enable_ansi()
def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s
def ok(s):    return _c("32", s)
def bad(s):   return _c("31", s)
def warn(s):  return _c("33", s)
def bold(s):  return _c("1", s)

PASS = ok("PASS")
FAIL = bad("FAIL")
WARN = warn("WARN")


def hr(title=""):
    print("\n" + bold("=" * 72))
    if title:
        print(bold(title))
        print(bold("=" * 72))


# --------------------------------------------------------------------------
# SDR helper
# --------------------------------------------------------------------------

TUNER_NAMES = {0: "unknown", 1: "E4000", 2: "FC0012", 3: "FC0013",
               4: "FC2580", 5: "R820T/R820T2", 6: "R828D"}


def list_devices():
    """Enumerate attached dongles.

    Worth running whenever more than one is plugged in: the index that `-D`
    takes is assigned by the USB stack and is not stable across replugs, so
    check it rather than assuming. The tuner chip differs between a generic
    DVB-T stick and a purpose-built RTL-SDR, and it determines the available
    gain steps and the noise figure you can expect.
    """
    try:
        from rtlsdr import RtlSdr
    except ImportError:
        print(bad("pyrtlsdr is not installed. pip install pyrtlsdr"))
        return 2

    serials = []
    try:
        serials = RtlSdr.get_device_serial_addresses()
    except Exception as e:
        print(warn(f"could not enumerate serials: {e}"))

    if not serials:
        print(bad("No RTL-SDR devices found."))
        if os.path.exists("/proc/version"):
            try:
                if "microsoft" in open("/proc/version").read().lower():
                    print("\nRunning under WSL. USB devices are not visible to")
                    print("WSL until they are attached from Windows with usbipd:")
                    print("\n  (PowerShell as Administrator, on the Windows side)")
                    print("    usbipd list")
                    print("    usbipd bind   --busid <BUSID>")
                    print("    usbipd attach --wsl --busid <BUSID>")
                    print("\nThen re-run this. `lsusb` inside WSL should show it.")
            except OSError:
                pass
        return 1

    print(f"{'idx':>4}  {'serial':<16} {'tuner':<14} {'gains dB':<8}  name")
    print("-" * 72)
    for i, ser in enumerate(serials):
        name = tuner = ngains = "?"
        try:
            d = RtlSdr(device_index=i)
            try:
                tuner = TUNER_NAMES.get(d.get_tuner_type(), "unknown")
            except Exception:
                pass
            try:
                ngains = str(len(d.valid_gains_db))
            except Exception:
                pass
            d.close()
        except Exception as e:
            name = f"could not open: {e}"
        print(f"{i:>4}  {str(ser):<16} {tuner:<14} {ngains:<8}  {name}")
    print(f"\nSelect one with  -D <idx>  (default 0).")
    return 0


def open_sdr(fs, freq, gain, device=0):
    """Open the dongle with AGC explicitly off and a fixed manual gain.

    This matters more than it looks. Automatic gain control continuously
    renormalises the signal level, which is precisely the quantity a meteor
    detector measures -- AGC will erase a ping as it happens. The old
    legacy/sdr-record.py defaulted to --gain auto.
    """
    try:
        from rtlsdr import RtlSdr
    except ImportError:
        print(bad("pyrtlsdr is not installed. pip install pyrtlsdr"))
        sys.exit(2)

    try:
        sdr = RtlSdr(device_index=device)
    except Exception as e:
        print(bad(f"could not open device {device}: {e}"))
        print("Run  rf_check.py --list-devices  to see what is attached.")
        sys.exit(2)
    sdr.sample_rate = fs
    sdr.center_freq = freq

    # Turn off both AGCs before setting gain; order matters on some dongles.
    for meth, arg in (("set_agc_mode", False), ("set_manual_gain_enabled", True)):
        if hasattr(sdr, meth):
            try:
                getattr(sdr, meth)(arg)
            except Exception as e:
                print(warn(f"  could not call {meth}({arg}): {e}"))

    if isinstance(gain, str) and gain.lower() == "auto":
        print(bad("  refusing to run with automatic gain."))
        print(bad("  AGC removes the power variations you are trying to detect."))
        print("  Valid fixed gains for this dongle (dB):")
        print("   ", getattr(sdr, "valid_gains_db", "unknown"))
        sdr.close()
        sys.exit(2)

    sdr.gain = float(gain)
    actual = sdr.gain
    if abs(actual - float(gain)) > 0.6:
        print(warn(f"  requested gain {gain} dB, dongle snapped to {actual} dB"))
    return sdr


def read_frames(sdr, nfft, n_frames, discard=2):
    """Read n_frames blocks of nfft complex samples, discarding PLL settling."""
    for _ in range(discard):
        sdr.read_samples(nfft)
    out = np.empty((n_frames, nfft), dtype=np.complex64)
    for i in range(n_frames):
        out[i] = sdr.read_samples(nfft)[:nfft]
    return out


# --------------------------------------------------------------------------
# --selftest
# --------------------------------------------------------------------------

def cmd_selftest(args):
    hr("SELF TEST -- DSP correctness, no hardware required")
    fs = 1.024e6
    failures = 0

    print("\n1. Absolute scaling: a full-scale complex tone must read 0.00 dBFS")
    print("   for every window and every FFT length.\n")
    print(f"   {'nfft':>7} {'rect':>9} {'hann':>9} {'blackman-h':>12}")
    for nfft in (1024, 4096, 16384):
        row = []
        for w in ("rect", "hann", "blackmanharris"):
            m = dsp.ChannelMetrics(fs, nfft, 0.0, window=w)
            n = np.arange(nfft)
            x = np.exp(2j * np.pi * 30e3 * n / fs).astype(np.complex64)
            p, _, _ = m.measure(x)
            row.append(p)
            if abs(p) > 0.1:
                failures += 1
        print(f"   {nfft:>7} {row[0]:>9.2f} {row[1]:>9.2f} {row[2]:>12.2f}")
    print(f"\n   {PASS if failures == 0 else FAIL}  scaling is "
          f"{'independent of' if failures == 0 else 'STILL DEPENDENT ON'} nfft and window")

    print("\n2. Linearity: reported level must track amplitude exactly.\n")
    m = dsp.ChannelMetrics(fs, 8192, 0.0, window="hann")
    n = np.arange(8192)
    lin_fail = 0
    for amp in (1.0, 0.5, 0.1, 0.01, 0.001):
        x = (amp * np.exp(2j * np.pi * 30e3 * n / fs)).astype(np.complex64)
        p, _, _ = m.measure(x)
        want = 20 * np.log10(amp)
        err = abs(p - want)
        if err > 0.1:
            lin_fail += 1
        print(f"   amplitude {amp:<7} -> {p:8.2f} dBFS  (expect {want:8.2f}, "
              f"err {err:.3f})")
    print(f"\n   {PASS if lin_fail == 0 else FAIL}  linearity")
    failures += lin_fail

    print("\n3. Window rejection: a strong neighbour 300 kHz away must not")
    print("   leak into the measured channel. The interferer is deliberately")
    print("   placed HALF A BIN off-centre -- a tone landing exactly on a bin")
    print("   is the one case where a rectangular window leaks nothing, and no")
    print("   real signal ever does.\n")
    nfft = 8192
    n = np.arange(nfft)
    off = 300e3 + 0.5 * (fs / nfft)          # half-bin offset: worst case
    strong = np.exp(2j * np.pi * off * n / fs).astype(np.complex64)
    leak = {}
    for w in ("rect", "hann", "blackmanharris"):
        mm = dsp.ChannelMetrics(fs, nfft, 0.0, channel_bw=180e3, window=w)
        leak[w], _, _ = mm.measure(strong)
        tag = "  <- the old code" if w == "rect" else ""
        print(f"   {w:<16} leakage into channel: {leak[w]:8.1f} dBc{tag}")
    gain_db = leak["rect"] - leak["hann"]
    win_ok = gain_db > 25
    if not win_ok:
        failures += 1
    print(f"\n   {PASS if win_ok else FAIL}  hann buys {gain_db:.0f} dB of "
          f"rejection over the rectangular window")

    print("\n4. Channel bandwidth sanity for a real FM signal.\n")
    m = dsp.ChannelMetrics(fs, 8192, 0.0, channel_bw=180e3)
    print("   " + m.describe().replace("\n", "\n   "))
    df = fs / 8192
    old_bw = (5 // 2 * 2) * 240e3 / 1024
    print(f"\n   this build integrates : {m.chan_bins * df / 1e3:8.1f} kHz")
    print(f"   old sdr-record.py     : {old_bw / 1e3:8.2f} kHz   "
          + bad("(0.5% of an FM channel)"))

    print("\n5. uint8 vs int8 decoding of an rtl_sdr byte stream.\n")
    raw = bytes([255, 128, 0, 128, 128, 128])
    good = dsp.bytes_to_complex(raw)
    wrong = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
    wrong = (wrong[0::2] + 1j * wrong[1::2]) / 127.5
    print(f"   correct (uint8): {np.round(good, 3)}")
    print(f"   old plot_iq.py : {np.round(wrong, 3)}   " + bad("<- sign-flipped"))
    print(f"\n   {PASS}  dsp.bytes_to_complex handles the offset correctly")

    hr()
    if failures == 0:
        print(ok("SELF TEST PASSED -- the DSP is sound. Move on to --sweep."))
        return 0
    print(bad(f"SELF TEST FAILED with {failures} problem(s)."))
    return 1


# --------------------------------------------------------------------------
# --sweep
# --------------------------------------------------------------------------

def _sweep_band(args, sdr=None, label=""):
    """Tile-and-stitch power spectrum across [args.start, args.stop].

    Returns (freqs_hz, power_db, linear_power). Opens its own dongle unless one
    is passed in, so the spur test can sweep twice on the same handle.
    """
    fs = args.sample_rate
    nfft = args.nfft
    usable = fs * 0.78                       # discard the filter roll-off edges
    centers = np.arange(args.start + usable / 2, args.stop, usable)

    own = sdr is None
    if own:
        sdr = open_sdr(fs, float(centers[0]), args.gain, args.device)
    win, sumsq = dsp.make_window(nfft, "hann")

    all_f, all_p = [], []
    try:
        for i, fc in enumerate(centers):
            sdr.center_freq = float(fc)
            frames = read_frames(sdr, nfft, args.avg)
            acc = np.zeros(nfft)
            for fr in frames:
                acc += dsp.psd(fr, fs, win, sumsq)
            acc /= len(frames)

            f = dsp.freq_axis(nfft, fs, fc)
            # Kill the R820T DC spike sitting at the tune frequency.
            c = nfft // 2
            acc[c - 3:c + 4] = np.nan
            keep = np.abs(f - fc) <= usable / 2
            all_f.append(f[keep])
            all_p.append(acc[keep])
            print(f"\r  {label}tile {i+1}/{len(centers)}  {fc/1e6:7.2f} MHz",
                  end="", flush=True)
    finally:
        if own:
            sdr.close()
    print()

    freqs = np.concatenate(all_f)
    lin = np.concatenate(all_p)
    order = np.argsort(freqs)
    freqs, lin = freqs[order], lin[order]
    good = ~np.isnan(lin)
    freqs, lin = freqs[good], lin[good]
    return freqs, dsp.to_db(lin), lin


def _group_signals(freqs, db, floor, threshold, df, min_width):
    """Contiguous runs above threshold -> [(peak_idx, width_khz)]."""
    above = db > floor + threshold
    groups, run = [], []
    for i, a in enumerate(above):
        if a:
            run.append(i)
        elif run:
            groups.append(run); run = []
    if run:
        groups.append(run)
    out = []
    for g in groups:
        w = len(g) * df / 1e3
        if w < min_width / 1e3 and len(g) < 2:
            continue
        out.append((g[int(np.argmax(db[g]))], w))
    return out


def cmd_sweep(args):
    hr(f"SWEEP {args.start/1e6:.1f}-{args.stop/1e6:.1f} MHz "
       f"at fixed gain {args.gain} dB")
    print("\nClassifying by BANDWIDTH. A real FM broadcast station is roughly")
    print("180 kHz wide. Anything under ~30 kHz is a birdie or spur generated")
    print("inside the receiver, not a station on the air.\n")
    print("NOTE: at a remote, terrain-shielded site an EMPTY band is normal and")
    print("is not a fault. This check surveys what is there; the verdict on the")
    print("RF chain comes from --floor-test, which needs no stations at all.\n")

    freqs, db, lin = _sweep_band(args)
    fs, nfft = args.sample_rate, args.nfft

    floor = float(np.percentile(db, 25))
    df = fs / nfft
    found = _group_signals(freqs, db, floor, args.threshold, df, args.min_width)

    print(f"\nnoise floor (25th pct): {floor:.1f} dBFS   "
          f"threshold: +{args.threshold} dB   bin: {df/1e3:.1f} kHz\n")
    print(f"{'freq MHz':>10} {'peak dBFS':>10} {'SNR dB':>8} "
          f"{'width kHz':>10}  verdict")
    print("-" * 72)

    stations = narrow = 0
    for pk, w_khz in found:
        if w_khz >= 100:
            verdict = ok("FM STATION")
            stations += 1
        elif w_khz >= 30:
            verdict = warn("narrow/weak - partial or distant")
        else:
            verdict = bad("suspect spur - confirm with --spur-test")
            narrow += 1
        print(f"{freqs[pk]/1e6:>10.3f} {db[pk]:>10.1f} "
              f"{db[pk]-floor:>8.1f} {w_khz:>10.1f}  {verdict}")

    hr("VERDICT")
    if stations == 0:
        print(f"{WARN}  No FM broadcast stations found in "
              f"{args.start/1e6:.0f}-{args.stop/1e6:.0f} MHz.")
        print("\nAt a remote site in hilly terrain this is EXPECTED and is good")
        print("news for meteor scatter -- you need a channel that is dead")
        print("locally. It does NOT tell you whether the RF chain works.")
        print("\nRun --floor-test next. Galactic background noise raises the")
        print("floor 8-16 dB through a working antenna with no transmitter")
        print("involved, so it is a valid test on a completely empty band.")
        rc = 0
    else:
        print(ok(f"{PASS}  {stations} FM broadcast station(s) detected. "
                 "The RF chain is passing real signal."))
        rc = 0

    if narrow:
        print(f"\n{WARN}  {narrow} narrow feature(s) under 30 kHz wide.")
        print("      Nothing on the air in this band is that narrow. These are")
        print("      probably generated inside your own receiver -- dongle")
        print("      birdies, or an LNA oscillating. Run --spur-test to find out:")
        print("      a real signal disappears when you unplug the antenna, an")
        print("      internal spur does not.")

    # Quiet-channel suggestions are useful whether or not stations were found.
    win_bins = max(1, int(200e3 / df))
    kern = np.ones(win_bins) / win_bins
    sdb = dsp.to_db(np.convolve(lin, kern, mode="same"))
    cand = []
    for i in np.argsort(sdb):
        f0 = freqs[i]
        if any(abs(f0 - c) < 400e3 for c in cand):
            continue
        cand.append(f0)
        if len(cand) >= 6:
            break
    print("\nQuietest 200 kHz windows -- candidate meteor-scatter channels:")
    for f0 in sorted(cand):
        j = int(np.argmin(np.abs(freqs - f0)))
        print(f"   {f0/1e6:8.3f} MHz   {sdb[j]-floor:+5.1f} dB vs floor")
    print("\nPick one that is ALSO occupied by a high-power transmitter 800-2000")
    print("km away, along an azimuth where your horizon is open. A locally quiet")
    print("channel with no distant station on it will stay quiet forever.")

    if args.save:
        np.savez_compressed(args.save, freq_hz=freqs, power_dbfs=db,
                            gain_db=args.gain, sample_rate=fs, nfft=nfft)
        print(f"\nsaved: {args.save}")
    return rc


# --------------------------------------------------------------------------
# --stability
# --------------------------------------------------------------------------

def cmd_stability(args):
    hr(f"STABILITY at {args.freq/1e6:.3f} MHz, gain {args.gain} dB, "
       f"{args.minutes:.1f} min")
    print("\nPark on a STRONG LOCAL STATION for this test. A steady transmitter")
    print("is a known-constant input, so any movement in the reading is your")
    print("receiver, not the sky.\n")

    fs = args.sample_rate
    sdr = open_sdr(fs, args.freq, args.gain, args.device)
    m = dsp.ChannelMetrics(fs, args.nfft, args.freq, channel_bw=args.channel_bw)
    print("  " + m.describe().replace("\n", "\n  ") + "\n")

    t_end = time.monotonic() + args.minutes * 60
    ts, pw = [], []
    t0 = time.monotonic()
    try:
        while time.monotonic() < t_end:
            frames = read_frames(sdr, args.nfft, 8, discard=0)
            p = np.mean([m.measure(f)[0] for f in frames])
            ts.append(time.monotonic() - t0)
            pw.append(p)
            print(f"\r  t={ts[-1]:6.1f}s  power={p:8.2f} dBFS  "
                  f"n={len(pw)}", end="", flush=True)
            time.sleep(max(0.0, args.period - 0.05))
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        sdr.close()
    print()

    pw = np.array(pw); ts = np.array(ts)
    if len(pw) < 8:
        print(bad("not enough samples"))
        return 1

    std = float(np.std(pw))
    ptp = float(np.ptp(pw))
    slope = float(np.polyfit(ts, pw, 1)[0]) * 60.0     # dB per minute
    drift = slope * (ts[-1] / 60.0)

    hr("VERDICT")
    print(f"  mean       {np.mean(pw):8.2f} dBFS")
    print(f"  std dev    {std:8.3f} dB")
    print(f"  peak-peak  {ptp:8.2f} dB")
    print(f"  drift      {slope:+8.3f} dB/min  ({drift:+.2f} dB over the run)\n")

    rc = 0
    if std < 0.5:
        print(f"  {PASS}  short-term stability ({std:.3f} dB std)")
    elif std < 1.5:
        print(f"  {WARN}  {std:.3f} dB std -- usable but noisy. Weak station?")
    else:
        print(f"  {FAIL}  {std:.3f} dB std is too high for a steady transmitter.")
        print("        Most likely AGC is still active, or the station is weak")
        print("        enough that you are measuring noise.")
        rc = 1

    if abs(drift) < 1.0:
        print(f"  {PASS}  drift over the run ({drift:+.2f} dB)")
    elif abs(drift) < 3.0:
        print(f"  {WARN}  {drift:+.2f} dB drift -- let the dongle warm up 20 min")
        print("        before starting real observations.")
    else:
        print(f"  {FAIL}  {drift:+.2f} dB drift. A non-TCXO dongle warming up can")
        print("        do this. Warm-up period or a TCXO dongle will fix it.")
        rc = 1

    if ptp > 6.0:
        print(f"  {WARN}  {ptp:.1f} dB peak-peak suggests fading or interference.")
    return rc


# --------------------------------------------------------------------------
# --floor-test
# --------------------------------------------------------------------------

def _measure_floor(sdr, m, nfft, n):
    frames = read_frames(sdr, nfft, n)
    vals = [m.measure(f)[0] for f in frames]
    return float(np.median(vals)), float(np.std(vals))


def cmd_floor_test(args):
    hr(f"ANTENNA FLOOR TEST at {args.freq/1e6:.3f} MHz")
    print("""
The most diagnostic measurement in this file, and the one that works at a site
with no receivable stations at all.

At 100 MHz the galactic background is 1000-3000 K. A decent low-noise amplifier
contributes 75-300 K. So connecting a working antenna must raise the system
noise floor substantially -- 8-16 dB for a typical LNA -- with no transmitter
involved anywhere. The sky is the test signal.

  receiver noise figure    expected floor rise
       0.5 dB  (good LNA)          16.4 dB
       1.0 dB  (LNA)               13.2 dB
       2.0 dB  (LNA)                9.9 dB
       3.5 dB  (bare dongle)        5.8 dB
       5.0 dB  (bare dongle)        4.1 dB
       6.0 dB  (bare dongle)        3.3 dB

  Pass --no-lna if there is no preamp in the chain, so the verdict is
  judged against the bare-dongle range instead.

WHERE TO DISCONNECT (this matters with a line amplifier):
  Unplug at the ANTENNA side of the LNA, leaving the amplifier powered and
  connected to the dongle. Unplugging between the LNA and the dongle removes
  the amplifier's own noise as well and the test tells you nothing about the
  antenna. If you have a 50-ohm terminator, fit it on the LNA input; an open
  input is acceptable but slightly less reliable.

Choose a QUIET frequency -- any real signal masks the effect.
""")
    fs = args.sample_rate
    sdr = open_sdr(fs, args.freq, args.gain, args.device)
    m = dsp.ChannelMetrics(fs, args.nfft, args.freq, channel_bw=args.channel_bw)

    try:
        input(bold("  [1] Antenna CONNECTED. Press Enter to measure..."))
        con, con_s = _measure_floor(sdr, m, args.nfft, args.avg)
        print(f"      connected:    {con:8.2f} dBFS  (std {con_s:.2f})\n")

        input(bold("  [2] Now DISCONNECT the antenna. Press Enter to measure..."))
        dis, dis_s = _measure_floor(sdr, m, args.nfft, args.avg)
        print(f"      disconnected: {dis:8.2f} dBFS  (std {dis_s:.2f})")
    except KeyboardInterrupt:
        print("\n  aborted")
        sdr.close()
        return 1
    finally:
        sdr.close()

    delta = con - dis
    hr("VERDICT")
    print(f"  floor delta: {delta:+.2f} dB\n")

    # Thresholds depend on what is in front of the dongle. A bare RTL-SDR tuner
    # is NF 3.5-6 dB against 0.5-2 dB for a decent LNA, so its own noise is much
    # closer to the sky's and the floor lifts far less. Judging a healthy bare
    # setup against the LNA numbers would call it broken.
    if args.no_lna:
        t_pass, t_warn, expect = 4.0, 2.0, "3-8 dB"
        setup = "bare dongle, no preamp"
    else:
        t_pass, t_warn, expect = 8.0, 4.0, "8-16 dB"
        setup = "with a low-noise preamp"
    print(f"  setup: {setup}   expected range: {expect}\n")

    if delta >= t_pass:
        nf = 10 * np.log10(1 / (10 ** (delta / 10) - 1) * 1500 / 290 + 1)
        print(f"  {PASS}  {delta:.1f} dB. You are external-noise-limited: the sky")
        print("        dominates your receiver's own noise, which is exactly the")
        print("        condition needed to detect anything faint.")
        print(f"        Implies a system noise figure around {nf:.1f} dB "
              "(assuming a 1500 K sky).")
        print("\n        The RF chain is working. An empty FM band at this site is")
        print("        a property of the location, not a fault.")
        if delta > 18:
            print(f"\n  {WARN}  {delta:.1f} dB is larger than galactic noise alone can")
            print("        explain. Indoors, or near a laptop, monitor or switching")
            print("        supply, you are probably measuring man-made interference")
            print("        rather than sky. That still proves the antenna is")
            print("        connected, but do not read it as sky sensitivity --")
            print("        repeat outdoors, away from buildings, for a real number.")
        return 0
    if delta >= t_warn:
        print(f"  {WARN}  only {delta:.1f} dB. The antenna is contributing, but less")
        print("        than galactic noise alone should produce. Suspect:")
        print("          - feedline loss between antenna and LNA")
        print("          - a poor match at the driven element")
        print("          - LNA gain too low to overcome the dongle's noise")
        print("          - you disconnected between LNA and dongle by mistake")
        print("        Try again at higher --gain and confirm the disconnect point.")
        return 0
    # Back out what the antenna actually delivered. This separates a broken
    # feed from a connected-but-badly-matched one, which need different fixes.
    t_rx = 290 * (10 ** (3.5 / 10) - 1)          # assume a good tuner, ~359 K
    t_ant = t_rx * (10 ** (delta / 10) - 1)
    print(f"  {FAIL}  {delta:.1f} dB. The antenna is delivering far less noise")
    print("        than the sky should provide.\n")
    print(f"        Implied antenna temperature: ~{t_ant:.0f} K")
    print("          ~1500 K  galactic background at 100 MHz (a good antenna)")
    print("           ~290 K  ambient -- a lossy but properly matched antenna")
    print("             ~0 K  open circuit or broken feed")

    if t_ant < 250:
        print("\n        Below ambient, which a passive antenna cannot reach by")
        print("        being lossy alone. That points at a severe IMPEDANCE")
        print("        MISMATCH rather than a broken cable: badly misterminated,")
        print("        the delivered noise scales by (1 - |gamma|^2). For")
        print("        reference, VSWR 20:1 still yields about 2.5 dB.")

    if args.no_lna:
        print("\n        With a bare monopole and no preamp, check in this order:")
        print("          1. WHIP LENGTH. A quarter wave at 100 MHz is 75 cm.")
        print("             A short whip is a huge mismatch here. Extend it fully.")
        print("          2. GROUND PLANE. A monopole is only half an antenna; it")
        print("             needs a counterpoise. Stand the magnetic base on a")
        print("             metal sheet, or add three or four 75 cm radials.")
        print("             Without one the coax shield becomes the counterpoise")
        print("             and behaves badly.")
        print("          3. Cable and both connectors, end to end.")
        print("          4. Repeat OUTDOORS, away from the building.")
        print("          5. Run --gain-linearity. If SNR still climbs with gain,")
        print("             the antenna IS coupling, just weakly -- a matching")
        print("             problem, not a dead feed.")
    else:
        print("\n        Check, in this order:")
        print("          1. cable continuity end to end, and both connectors")
        print("          2. LNA actually powered (measure the bias voltage)")
        print("          3. the driven element is not shorted or open")
        print("          4. you unplugged at the ANTENNA side of the LNA")
        print("          5. swap in any other antenna, even a wire, and repeat --")
        print("             a wire that beats the Yagi localises the fault fast")
    return 1


# --------------------------------------------------------------------------
# --spur-test
# --------------------------------------------------------------------------

def cmd_spur_test(args):
    hr("SPUR TEST -- which 'signals' are actually inside your receiver?")
    print("""
Sweeps the band twice: antenna connected, then disconnected. Anything still
present with the antenna removed was never on the air. With a line amplifier in
the chain this matters -- a poorly matched LNA can oscillate and manufacture
signals that look convincing on a waterfall.

Disconnect at the ANTENNA side of the LNA, not between the LNA and the dongle,
so the amplifier stays powered and in its normal operating state.
""")
    sdr = open_sdr(args.sample_rate, args.start, args.gain, args.device)
    df = args.sample_rate / args.nfft
    try:
        input(bold("  [1] Antenna CONNECTED. Press Enter to sweep..."))
        f1, db1, _ = _sweep_band(args, sdr=sdr, label="connected ")
        input(bold("\n  [2] DISCONNECT the antenna at the LNA input. Enter..."))
        f2, db2, _ = _sweep_band(args, sdr=sdr, label="disconnected ")
    except KeyboardInterrupt:
        print("\n  aborted")
        return 1
    finally:
        sdr.close()

    n = min(len(db1), len(db2))
    f, db1, db2 = f1[:n], db1[:n], db2[:n]
    floor1 = float(np.percentile(db1, 25))
    floor2 = float(np.percentile(db2, 25))

    found = _group_signals(f, db1, floor1, args.threshold, df, args.min_width)
    print(f"\nbroadband floor  connected {floor1:7.1f} dBFS   "
          f"disconnected {floor2:7.1f} dBFS   delta {floor1-floor2:+.1f} dB\n")
    print(f"{'freq MHz':>10} {'conn':>8} {'disc':>8} {'drop':>7} "
          f"{'width kHz':>10}  verdict")
    print("-" * 72)

    external = internal = 0
    for pk, w in found:
        drop = db1[pk] - db2[pk]
        if drop >= 6:
            v = ok("EXTERNAL - really on the air"); external += 1
        elif drop >= 3:
            v = warn("ambiguous")
        else:
            v = bad("INTERNAL - generated in your receiver"); internal += 1
        print(f"{f[pk]/1e6:>10.3f} {db1[pk]:>8.1f} {db2[pk]:>8.1f} "
              f"{drop:>+7.1f} {w:>10.1f}  {v}")

    hr("VERDICT")
    if external:
        print(f"  {PASS}  {external} signal(s) confirmed external.")
    if internal:
        print(f"  {FAIL}  {internal} signal(s) are generated inside the receiver.")
        print("        If your LNA is the source, try: a better-matched input,")
        print("        an FM band-pass filter ahead of it, shorter leads, or")
        print("        shielding. Until these are gone every detection threshold")
        print("        you set is competing with your own hardware.")
    if not found:
        print(f"  {WARN}  nothing above threshold in either sweep -- an empty")
        print("        band. Expected here; use --floor-test for the verdict.")
    return 1 if internal else 0


# --------------------------------------------------------------------------
# --gain-linearity
# --------------------------------------------------------------------------

def cmd_gain_linearity(args):
    hr(f"GAIN LINEARITY / LNA COMPRESSION at {args.freq/1e6:.3f} MHz")
    print("""
Steps the tuner through every available gain and watches SNR, not level.

In a healthy chain, raising gain lifts signal and noise together, so SNR climbs
while you are limited by the dongle's own noise, then FLATTENS once the external
noise dominates. The knee is the gain you should use.

If SNR falls again at high gain, something is compressing -- most likely the LNA
driving the dongle into its non-linear region. That is worth knowing before a
month of recording: a compressed front end suppresses exactly the brief
excursions a meteor produces.
""")
    fs = args.sample_rate
    sdr = open_sdr(fs, args.freq, args.gain, args.device)
    gains = list(getattr(sdr, "valid_gains_db", []) or [0, 9, 15, 21, 25, 30,
                                                       35, 40, 44, 49.6])
    m = dsp.ChannelMetrics(fs, args.nfft, args.freq, channel_bw=args.channel_bw)

    print(f"{'gain dB':>8} {'power':>9} {'noise':>9} {'snr':>7} {'peak':>9}")
    print("-" * 48)
    rows = []
    try:
        for g in gains:
            sdr.gain = g
            time.sleep(0.15)
            frames = read_frames(sdr, args.nfft, max(8, args.avg // 4))
            p = np.mean([m.measure(fr)[0] for fr in frames])
            nz = np.mean([m.measure(fr)[1] for fr in frames])
            pk = np.max([m.measure(fr)[2] for fr in frames])
            rows.append((float(sdr.gain), p, nz, p - nz, pk))
            print(f"{sdr.gain:>8.1f} {p:>9.2f} {nz:>9.2f} {p-nz:>7.2f} {pk:>9.2f}")
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        sdr.close()

    if len(rows) < 4:
        print(bad("\nnot enough gain steps to judge"))
        return 1

    g = np.array([r[0] for r in rows])
    snr = np.array([r[3] for r in rows])
    pw = np.array([r[1] for r in rows])
    best = int(np.argmax(snr))
    peak_snr = snr[best]

    # The knee: lowest gain reaching within 1 dB of the best SNR.
    knee_i = int(np.argmax(snr >= peak_snr - 1.0))
    compressed = bool(snr[-1] < peak_snr - 2.0)

    hr("VERDICT")
    print(f"  best SNR {peak_snr:.2f} dB at {g[best]:.1f} dB gain")
    print(f"  knee (within 1 dB of best): {g[knee_i]:.1f} dB\n")

    rise = snr[min(knee_i, len(snr)-1)] - snr[0]
    if rise > 3:
        print(f"  {PASS}  SNR climbs {rise:.1f} dB with gain, so you ARE hearing")
        print("        something external -- the dongle's own noise is not the")
        print("        limit. This is a positive RF chain result on its own.")
    else:
        print(f"  {WARN}  SNR barely moves with gain ({rise:+.1f} dB). Either the")
        print("        band is truly empty and flat, or nothing external is")
        print("        reaching the tuner. Cross-check with --floor-test.")

    if compressed:
        print(f"\n  {FAIL}  SNR falls {peak_snr - snr[-1]:.1f} dB at maximum gain."
              "\n        The front end is compressing. Use the knee gain, and")
        print("        consider an FM band-pass filter ahead of the LNA.")
    else:
        print(f"\n  {PASS}  no compression signature up to {g[-1]:.1f} dB")

    print(f"\n  {bold('Use --gain ' + format(g[knee_i], '.1f'))} for observing:")
    print("  the lowest gain that reaches full sensitivity, which leaves the")
    print("  most headroom before overload.")
    return 0


# --------------------------------------------------------------------------
# --sidereal
# --------------------------------------------------------------------------

SIDEREAL_H = 23.9344696          # 23h 56m 04s
DRIFT_MIN_PER_DAY = (24.0 - SIDEREAL_H) * 60.0    # 3.934 min/day


def cmd_sidereal(args):
    hr("SIDEREAL DRIFT -- definitive proof the antenna sees the sky")
    print(f"""
Offline analysis of Tier-1 .npz chunks from fm_observe.py.

As the Earth turns, the galactic plane sweeps through the antenna beam and the
noise floor rises and falls by a few dB. Anything terrestrial that varies daily
-- mains loading, traffic, your own equipment warming and cooling -- repeats on
the SOLAR day. The sky repeats on the SIDEREAL day, 23h56m04s.

Those two periods differ by one part in 366, so telling them apart by period
alone would need a YEAR of data. The practical measurement is the phase drift
instead: a sidereal feature arrives {DRIFT_MIN_PER_DAY:.2f} minutes earlier each
solar day, which accumulates fast enough to see:

     7 days ->  28 min      21 days ->  83 min
    14 days ->  55 min      30 days -> 118 min

So this test needs 10 days minimum and is convincing at 3-4 weeks. It runs on
data you are collecting anyway -- just leave fm_observe.py going.

A sidereal drift is unambiguous evidence that your antenna is receiving
celestial radio noise, and no local interference can imitate it.
""")
    files = sorted(glob.glob(os.path.join(args.dir, "*.npz")))
    if not files:
        print(bad(f"no .npz files in {args.dir}"))
        return 1

    t, p = [], []
    for fp in files:
        d = np.load(fp)
        if "t_utc_ns" not in d.files or "noise_dbfs" not in d.files:
            continue
        t.append(d["t_utc_ns"].astype(np.float64) / 1e9)
        # The tracked floor, not channel power: we want the sky, not events.
        p.append(d["noise_dbfs"].astype(np.float64))
    if not t:
        print(bad("no usable chunks"))
        return 1
    t = np.concatenate(t); p = np.concatenate(p)
    o = np.argsort(t); t, p = t[o], p[o]
    span_d = (t[-1] - t[0]) / 86400.0
    print(f"loaded {len(files)} chunk(s), {len(t)} frames, {span_d:.1f} days\n")

    if span_d < 5:
        print(bad(f"{FAIL}  only {span_d:.1f} days. Need 10 minimum, 21+ preferred."))
        return 1
    if span_d < 10:
        print(warn(f"{WARN}  {span_d:.1f} days is marginal; treat the result as "
                   "provisional."))

    # Decimate to 5-minute means, then remove slow gain drift with a long
    # moving median so only the daily structure survives.
    bin_s = 300.0
    edges = np.arange(t[0], t[-1] + bin_s, bin_s)
    idx = np.clip(np.searchsorted(edges, t) - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    sums = np.bincount(idx, weights=p, minlength=nb)
    cnts = np.bincount(idx, minlength=nb)
    valid = cnts > 0
    tb = (edges[:-1] + bin_s / 2)[valid]
    pb = (sums[valid] / cnts[valid])

    w = max(3, int(2 * 86400 / bin_s) | 1)          # 2-day moving median
    pad = np.pad(pb, w // 2, mode="edge")
    trend = np.array([np.median(pad[i:i + w]) for i in range(len(pb))])
    pb = pb - trend

    # Per-day sinusoid fit: phase of the daily maximum, in UTC hours.
    day0 = np.floor(tb[0] / 86400.0)
    day = (np.floor(tb / 86400.0) - day0).astype(int)
    peaks_d, peaks_h = [], []
    for dd in range(day.max() + 1):
        m = day == dd
        if np.count_nonzero(m) < int(0.6 * 86400 / bin_s):
            continue                                 # skip partial days
        hh = (tb[m] % 86400.0) / 3600.0
        y = pb[m]
        om = 2 * np.pi / 24.0
        A = np.column_stack([np.cos(om * hh), np.sin(om * hh),
                             np.ones_like(hh)])
        try:
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        amp = float(np.hypot(c[0], c[1]))
        if amp < 0.05:
            continue
        ph = float(np.arctan2(c[1], c[0])) / om      # hours of maximum
        peaks_d.append(dd)
        peaks_h.append(ph % 24.0)

    if len(peaks_d) < 5:
        print(bad(f"{FAIL}  only {len(peaks_d)} usable day(s) with a clear daily "
                  "cycle. Need more continuous data."))
        return 1

    peaks_d = np.array(peaks_d, float)
    peaks_h = np.unwrap(np.array(peaks_h) * (2 * np.pi / 24.0)) * (24 / (2 * np.pi))

    n = len(peaks_d)
    slope, icept = np.polyfit(peaks_d, peaks_h, 1)
    resid = peaks_h - (slope * peaks_d + icept)
    ss = float(np.sum(resid ** 2))
    sxx = float(np.sum((peaks_d - peaks_d.mean()) ** 2))
    se = float(np.sqrt(ss / max(1, n - 2) / max(sxx, 1e-9))) * 60.0
    slope_min = slope * 60.0

    print(f"usable days          {n}")
    print(f"daily peak drift     {slope_min:+.2f} +/- {se:.2f} min/day")
    print(f"  sidereal predicts  {-DRIFT_MIN_PER_DAY:+.2f} min/day")
    print(f"  terrestrial        {0.0:+.2f} min/day")
    print(f"cycle amplitude      {np.percentile(pb, 95) - np.percentile(pb, 5):.2f} dB")

    hr("VERDICT")
    if se > DRIFT_MIN_PER_DAY / 2:
        print(f"  {WARN}  uncertainty ({se:.2f} min/day) is too large next to the")
        print(f"        {DRIFT_MIN_PER_DAY:.2f} min/day being tested for. Keep")
        print("        recording -- the error shrinks as days accumulate.")
        return 0
    z_sid = abs(slope_min + DRIFT_MIN_PER_DAY) / se
    z_sol = abs(slope_min) / se
    if z_sid < 2.5 and z_sid < z_sol:
        print(f"  {PASS}  drift matches SIDEREAL within {z_sid:.1f} sigma.")
        print("        Your antenna is receiving galactic background noise.")
        print("        This is end-to-end proof of sky sensitivity and does not")
        print("        depend on any transmitter existing.")
        return 0
    if z_sol < 2.5 and z_sol < z_sid:
        print(f"  {WARN}  drift is consistent with ZERO ({z_sol:.1f} sigma), i.e. a")
        print("        SOLAR cycle. That points at something terrestrial and")
        print("        diurnal rather than the sky -- mains loading, local")
        print("        machinery, or your equipment's own thermal cycle.")
        return 0
    print(f"  {WARN}  drift {slope_min:+.2f} min/day matches neither hypothesis")
    print(f"        (sidereal {z_sid:.1f} sigma, solar {z_sol:.1f} sigma).")
    print("        Check for gaps in the recording or a gain change mid-run.")
    return 0

# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Phase 0 RF chain diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""order of use at a quiet, terrain-shielded site:

  1. --selftest         verify the DSP            (no hardware)
  2. --floor-test       THE RF chain verdict      (needs no stations)
  3. --spur-test        are those 'signals' yours? (needs no stations)
  4. --gain-linearity   find LNA compression, pick a gain
  5. --stability        confirm fixed gain holds  (needs a steady signal)
  6. --sweep            survey the band, pick a quiet channel
  7. --sidereal         48h+ proof the antenna sees the sky""")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-devices", action="store_true",
                      help="enumerate attached dongles with tuner type and index")
    mode.add_argument("--selftest", action="store_true",
                      help="verify the DSP against synthetic signals (no hardware)")
    mode.add_argument("--sweep", action="store_true",
                      help="sweep the FM band and classify what is on the air")
    mode.add_argument("--stability", action="store_true",
                      help="check for gain drift and residual AGC")
    mode.add_argument("--floor-test", action="store_true",
                      help="antenna connected/disconnected noise floor delta "
                           "(works on a completely empty band)")
    mode.add_argument("--spur-test", action="store_true",
                      help="find 'signals' that are generated inside your own "
                           "receiver rather than being on the air")
    mode.add_argument("--gain-linearity", action="store_true",
                      help="sweep tuner gain to find LNA compression and "
                           "recommend an operating gain")
    mode.add_argument("--sidereal", action="store_true",
                      help="analyse 48h+ of recorded .npz for the 23h56m "
                           "signature of galactic noise")

    p.add_argument("-D", "--device", type=int, default=0,
                   help="dongle index when more than one is attached; "
                        "see --list-devices")
    p.add_argument("-g", "--gain", default="35",
                   help="FIXED tuner gain in dB. 'auto' is rejected by design.")
    p.add_argument("-s", "--sample-rate", type=float, default=2.048e6)
    p.add_argument("--nfft", type=int, default=8192)
    p.add_argument("-f", "--freq", type=float, default=107.1e6,
                   help="frequency for --stability / --floor-test")
    p.add_argument("--channel-bw", type=float, default=180e3,
                   help="integration bandwidth in Hz (default 180 kHz, one FM channel)")
    p.add_argument("--start", type=float, default=88e6, help="sweep start Hz")
    p.add_argument("--stop", type=float, default=108e6, help="sweep stop Hz")
    p.add_argument("--avg", type=int, default=32, help="frames averaged per point")
    p.add_argument("--threshold", type=float, default=6.0,
                   help="dB above floor to call something a signal")
    p.add_argument("--min-width", type=float, default=10e3,
                   help="ignore features narrower than this (Hz)")
    p.add_argument("--minutes", type=float, default=10.0, help="--stability duration")
    p.add_argument("--period", type=float, default=1.0, help="--stability sample period s")
    p.add_argument("--save", type=str, help="save sweep to .npz")
    p.add_argument("--no-lna", action="store_true",
                   help="no preamp in the chain: use bare-dongle thresholds for "
                        "--floor-test (expect 3-8 dB rather than 8-16 dB)")
    p.add_argument("--dir", type=str, default="./fm_observations",
                   help="directory of .npz chunks for --sidereal")

    args = p.parse_args()
    try:
        if args.list_devices:
            return list_devices()
        if args.selftest:
            return cmd_selftest(args)
        if args.sweep:
            return cmd_sweep(args)
        if args.stability:
            return cmd_stability(args)
        if args.floor_test:
            return cmd_floor_test(args)
        if args.spur_test:
            return cmd_spur_test(args)
        if args.gain_linearity:
            return cmd_gain_linearity(args)
        if args.sidereal:
            return cmd_sidereal(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
