# Remote Radio Observatory

Radio meteor detection by FM forward scatter, using an RTL-SDR and a Yagi-Uda
antenna at Sirsi, Karnataka. Designed to run unattended on a Raspberry Pi 5.

A distant FM transmitter, several hundred kilometres beyond the horizon, is
normally inaudible. When a meteor ionises a trail at 85–105 km altitude, that
trail briefly reflects VHF and the station appears for a fraction of a second.
Recording channel power continuously and looking for those excursions is the
whole method.

## Layout

```
Meteor-Detector/
  acquisition/          the live pipeline  <- everything current is here
    dsp.py                shared DSP primitives
    rf_check.py           Phase 0: prove the RF chain works
    fm_observe.py         Phase 1: the always-on recorder
    plot_npz_utc.py       plots the recorder output vs UTC
    test_pipeline.py      end-to-end test, no hardware needed
    fm-observe.service    systemd unit for the Pi
  legacy/               first generation, superseded, kept for reference
  radar_config_sample   site parameters
  install.sh
  MeteorRadioFM/        submodule: fork of rabssm/MeteorRadio
```

Everything superseded now lives under `Meteor-Detector/legacy/` — the old SigMF
recorder and its two viewers, the raw-IQ tools, the `rtl_power` surveys, and the
exploratory scripts. Nothing was deleted; see
[`Meteor-Detector/legacy/README.md`](Meteor-Detector/legacy/README.md) for what
each piece did and why it was retired. If you have old `.sigmf-data` recordings,
the viewers in there are still the only things that read them.

## Setup

```bash
cd Meteor-Detector
./install.sh
source venv/bin/activate
```

## Getting started

Sirsi is remote and terrain-shielded — the FM band is essentially empty, which
is *good* for meteor scatter but means "can I hear a local station?" is not a
valid hardware test. Every check below works on an empty band. Full rationale in
[`Meteor-Detector/acquisition/README.md`](Meteor-Detector/acquisition/README.md).

```bash
cd Meteor-Detector/acquisition

# 0. Verify the DSP. No hardware required.
python3 rf_check.py --selftest
python3 test_pipeline.py

# 1. THE RF chain verdict. Galactic noise raises the floor 8-16 dB through a
#    working antenna, with no transmitter involved. Disconnect at the ANTENNA
#    side of the line amplifier.
python3 rf_check.py --floor-test --freq 107.1e6 --gain 35

# 2. Are those "signals" real, or is your own receiver making them?
python3 rf_check.py --spur-test --gain 35

# 3. Find the operating gain; check the LNA is not compressing.
python3 rf_check.py --gain-linearity --freq 107.1e6

# 4. Survey the band, pick a quiet channel.
python3 rf_check.py --sweep --gain <knee> --save sweep.npz

# 5. Record.
python3 fm_observe.py --freq <CHANNEL> --gain <knee> --station SIRSI --save-iq

# 6. After 2-4 weeks: prove the antenna sees the sky.
python3 rf_check.py --sidereal --dir ~/fm_observations
```

Then plot:

```bash
python3 plot_npz_utc.py --dir ~/fm_observations --list
python3 plot_npz_utc.py --dir ~/fm_observations
```

## Design notes

**Fixed gain, always.** Automatic gain control continuously renormalises the
signal level — which is exactly the quantity being measured. Both `rf_check.py`
and `fm_observe.py` refuse to run with `--gain auto`.

**Integrate the whole channel.** FM is constant-envelope: transmitted power is
steady but spread across ±75 kHz of deviation, and the centre bin is occupied
only during near-silence in the audio. A narrow measurement tracks the
station's program material, not its presence.

**dBFS, never dBm.** An RTL-SDR has no absolute power calibration. Only SNR
against a tracked noise floor is physically meaningful.

**Two tiers of storage.** Raw IQ at 1.024 MS/s is 177 GB/day. Tier 1 is a
~125 Hz power series (~100–270 MB/day); Tier 2 is a RAM ring buffer of raw IQ
dumped only around a trigger.

**Decimate by integrating, never by subsampling.** The FFT is the decimator.
Keeping every Nth sample aliases, and a 20 ms ping can fall entirely between
kept points.

## Validating detections

Before trusting any meteor count:

1. **Aircraft scatter** first — Doppler-drifting events lasting 10–60 s, many
   per day, on the same geometry. Seeing aircraft proves sensitivity in a day
   rather than a month.
2. **The diurnal curve** — real meteor rates peak near 06:00 local (Earth's
   apex) and rise during showers (Geminids Dec 13–14, Quadrantids Jan 3–4). A
   flat rate, or one peaking at rush hour, is interference.

## Hardware

RTL-SDR dongle, line amplifier, Yagi-Uda for the FM band, Raspberry Pi 5.

Forward scatter wants a transmitter 800–2000 km away on a locally dead channel,
along an azimuth where the horizon is open. FM has been audible toward Mangalore
(bearing 178°) and not elsewhere, which suggests the southern horizon is the
open one — putting **Thiruvananthapuram (718 km, 161°)** and **Colombo
(1018 km, 147°)** in range. Aim the Yagi at roughly 10–20° elevation, not at the
horizon and not at the zenith.

Site parameters live in `Meteor-Detector/radar_config_sample`.
