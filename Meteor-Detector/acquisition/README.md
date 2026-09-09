# Acquisition

The data acquisition chain, rebuilt. Run the checks in order — each one gates
the next, and skipping ahead is how the previous generation of this code spent
months recording the inside of a USB dongle.

## Files

| File | Purpose |
|---|---|
| `dsp.py` | Shared DSP primitives. Imported by **both** the diagnostic tool and the recorder, so what you validate on the bench is what gets written to disk. |
| `rf_check.py` | Phase 0. Four checks that prove the RF chain works. |
| `fm_observe.py` | Phase 1. The always-on recorder. |
| `test_pipeline.py` | End-to-end test against a synthetic sky. No hardware needed. |
| `fm-observe.service` | systemd unit for unattended running on the Pi. |

## Order of operations

Sirsi is remote and terrain-shielded: the FM band is essentially empty, as
confirmed independently in SDR++. **"Can I hear a local station?" is therefore
not a valid test here** — it fails for a site reason and says nothing about the
hardware. Every check below works on a completely empty band.

### 0. Verify the DSP (anywhere, no hardware)

```bash
python3 rf_check.py --selftest
python3 test_pipeline.py
```

### 1. The RF chain verdict — needs no transmitter at all

```bash
python3 rf_check.py --floor-test --freq 107.1e6 --gain 35
```

At 100 MHz the galactic background is 1000–3000 K against an LNA's 75–300 K, so
**connecting a working antenna must raise the noise floor 8–16 dB** whether or
not anything is on the air. The sky is the test signal.

| LNA noise figure | expected floor rise |
|---|---|
| 0.5 dB | 16.4 dB |
| 1.0 dB | 13.2 dB |
| 2.0 dB | 9.9 dB |
| 3.0 dB | 7.9 dB |

**With a line amplifier, disconnect at the *antenna* side of the LNA**, leaving
it powered and connected to the dongle. Unplugging between LNA and dongle
removes the amplifier's own noise too, and the test then measures nothing about
the antenna.

### 2. Are those "signals" even real?

```bash
python3 rf_check.py --spur-test --gain 35
```

Sweeps with the antenna connected and disconnected. Anything still present with
the antenna off was never on the air — a dongle birdie, or an oscillating line
amplifier. The December 2025 survey in `../legacy/quick-sdr-analysis/output.csv` showed
four features 0–23 kHz wide, one of them a single bin; nothing broadcast is that
narrow, so they are all candidates for being self-generated.

### 3. Find the right gain, and check the LNA is not compressing

```bash
python3 rf_check.py --gain-linearity --freq 107.1e6
```

Steps through every tuner gain watching SNR rather than level. SNR should climb
then flatten; the knee is the gain to use. If it falls again at high gain the
front end is compressing, which suppresses exactly the brief excursions a meteor
produces. Prints a recommended `--gain`.

### 4. Confirm the gain holds

```bash
python3 rf_check.py --stability --freq <ANY SIGNAL YOU CAN HEAR> --gain <knee>
```

Needs something steady to watch. Toward Mangalore (bearing 178°) there is a
receivable signal; use that.

### 5. Survey the band and choose a channel

```bash
python3 rf_check.py --sweep --gain <knee> --save sweep.npz
```

An empty band here is expected and is *good* for meteor scatter — you need a
channel that is dead locally. The check reports the quietest 200 kHz windows.

### 6. Record

```bash
python3 fm_observe.py --freq <CHANNEL> --gain <knee> --station SIRSI --save-iq
```

### 7. Prove the antenna sees the sky (weeks, from data you collect anyway)

```bash
python3 rf_check.py --sidereal --dir ~/fm_observations
```

The galactic plane sweeping the beam makes the noise floor rise and fall daily.
The sky repeats on the **sidereal** day (23h56m04s); anything terrestrial
repeats on the **solar** day. Those periods differ by one part in 366, so
separating them *by period* would need a year — but the **phase drift** is
measurable much sooner: a sidereal feature arrives 3.93 minutes earlier each
solar day.

| span | accumulated drift |
|---|---|
| 7 days | 28 min |
| 14 days | 55 min |
| 21 days | 83 min |
| 30 days | 118 min |

Ten days is the minimum, three to four weeks is convincing. A sidereal drift
cannot be imitated by any local interference.

## Site geometry

Forward scatter wants a transmitter **800–2000 km** away, on a channel that is
dead locally, along an azimuth where the horizon is open. Great-circle
distances from Sirsi (14.641 N, 74.831 E):

| Target | km | bearing | note |
|---|---|---|---|
| Mangalore | 197 | 178° S | too close for a beacon, but a useful test signal |
| Thiruvananthapuram | 718 | 161° SSE | workable, close to the open horizon |
| Colombo, Sri Lanka | 1018 | 147° SSE | **optimal range**, open horizon |
| Bhopal | 996 | 15° NNE | optimal range, but likely terrain-blocked |
| Delhi | 1572 | 9° N | optimal range, but likely terrain-blocked |
| Kolkata | 1675 | 56° NE | optimal range, but likely terrain-blocked |

FM was audible toward Mangalore (178°) with a monopole and not elsewhere, which
suggests the **southern horizon is the open one**. That points at
Thiruvananthapuram and Colombo as the candidate transmitters. Before committing,
measure the terrain horizon elevation toward those bearings — meteor scatter
arrives at 5–25° elevation, so a ridge above that in the target direction kills
the path.

## Why it is built this way

**Gap-free capture.** librtlsdr streams into a callback that does nothing but
hand bytes to a queue; a worker thread does the DSP. A recorder that reads,
processes, then sleeps for a fixed interval is deaf for the duration of the
sleep, and an underdense ping lasts 50–500 ms.

**Fixed gain, AGC off.** Automatic gain control continuously renormalises the
signal level — which is exactly the quantity being measured. Both tools refuse
to run with `--gain auto`.

**Full-channel integration.** FM is constant-envelope: the transmitter's total
power is steady, but it is spread across ±75 kHz of deviation and the centre bin
is occupied only during near-silence in the audio. Integrating a narrow slice
measures the station's *program material*, not its presence.

**Correct PSD normalisation.** `|X|² / (fs · Σw²)`. The `Σw²` term makes the
result independent of both window and FFT length; a full-scale tone reads
0 dBFS at any setting, so runs are comparable.

**dBFS, never dBm.** An RTL-SDR has no absolute power calibration. The only
physically meaningful quantity is SNR against a tracked floor.

**Two tiers of storage.** Continuous raw IQ at 1.024 MS/s is 2.0 MB/s, or
177 GB/day — not an option. Tier 1 is a ~125 Hz power series (~100–270 MB/day,
years on an SD card). Tier 2 is a RAM ring buffer of raw IQ, dumped only around
a trigger, so events can be re-examined later.

**Decimate by integrating, never by subsampling.** The FFT is the decimator:
1.024 MS/s → 125 power values/s is 8192:1 and loses nothing that matters for
envelope detection. Keeping every Nth sample instead would alias, and a 20 ms
ping can fall entirely in the gap.

## What to expect

Validate on **aircraft scatter** before meteors — aircraft produce
Doppler-drifting events lasting 10–60 s, many per day, on the same geometry. If
you can see aircraft, sensitivity is sufficient for meteors, and you get
feedback in a day instead of a month.

Then confirm meteors statistically: real rates show a **diurnal peak near
06:00 local** (Earth's apex) and rise during showers (Geminids Dec 13–14,
Quadrantids Jan 3–4). A flat rate, or one that peaks at rush hour, is
interference.

One caveat on the physics: Indian FM transmitters typically run 10–20 kW ERP
against the 100+ kW stations European observers use for this, so expect fewer
pings than the tutorials promise. Aim the Yagi at the distant transmitter's
azimuth with ~10–20° elevation — not at the horizon, and not at the zenith.
