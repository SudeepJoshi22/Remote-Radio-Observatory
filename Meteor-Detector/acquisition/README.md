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

## Bench test: bare antenna into the SDR, no preamp

> A copy-pasteable checklist with fill-in-the-blank results lives in
> [`RUN_SHEET.md`](RUN_SHEET.md).

Worth doing before committing to the full setup. It establishes a **control**:
every number measured here becomes the reference for judging whether the LNA and
the Yagi actually improve things.

```bash
python3 rf_check.py --selftest                       # software intact?
python3 rf_check.py --floor-test --no-lna -g 49.6    # antenna connected?
python3 rf_check.py --spur-test -g 49.6 --save bare_spurs.npz
python3 rf_check.py --gain-linearity -f 107.1e6
python3 rf_check.py --sweep -g 49.6 --save bare_sweep.npz
```

**Pass `--no-lna`.** A bare RTL-SDR tuner is NF 3.5–6 dB against 0.5–2 dB for a
preamp, so its own noise sits much closer to the sky's and the floor lifts far
less. Expect **3–8 dB, centred near 5**, not the 8–16 dB of a preamped chain.
Without the flag the tool judges against the LNA range and calls a healthy bare
setup a WARN.

Use **maximum gain** (`-g 49.6`) here. With no preamp you need every dB, and
with nothing amplifying ahead of it there is little risk of overloading.

Three things this baseline buys you:

1. **A spur fingerprint.** With no LNA, every narrow feature `--spur-test` finds
   is the dongle's own. Save it. If new spurs appear once the LNA is fitted, the
   amplifier is the source — that is otherwise very hard to pin down.
2. **A floor-delta reference.** Adding a good LNA should push the delta from
   ~5 dB toward 10–15 dB. If it does not improve, the LNA is not helping and may
   be hurting.
3. **A gain reference.** Compare the knee `--gain-linearity` reports before and
   after the preamp.

**Location matters more than usual.** Indoors beside a laptop, monitor or
switching supply, the floor delta can reach 15–25 dB from man-made interference
alone. That proves the antenna is connected, but it is not sky sensitivity — the
tool warns when the delta exceeds 18 dB for this reason. Repeat outdoors, away
from buildings, for a number that means something.

**Positive control at Sirsi:** FM has been audible toward Mangalore (bearing
178°) with a monopole. Point the antenna south and check `--sweep` finds a
feature ~180 kHz wide there. That is the one signal known to be receivable at
this site, so seeing it end-to-end validates the whole chain.

### Drivers

The RTL-SDR needs no kernel driver in the usual sense — librtlsdr talks to it
through libusb in userspace. What it does need is for nothing *else* to claim
the device first.

**Linux / Raspberry Pi OS.** `../install.sh` handles all of this. The critical
part is that the kernel sees an RTL2832U and loads `dvb_usb_rtl28xxu`, treating
it as a DVB-T television tuner. librtlsdr then cannot claim the USB interface
and everything fails with:

```
usb_claim_interface error -6
Failed to open rtlsdr device #0
```

This is the most common RTL-SDR problem on Linux. The installer writes
`/etc/modprobe.d/blacklist-rtlsdr.conf`, unloads the module, installs udev rules
for non-root access, and adds you to `plugdev`. **After a first install, unplug
and replug the dongle** so it re-enumerates without the DVB driver attached.

To check by hand:

```bash
lsmod | grep dvb                    # should print nothing
sudo modprobe -r dvb_usb_rtl28xxu   # if it does
rtl_test -t                         # should find and open the device
```

**WSL2 (Windows Subsystem for Linux).** WSL2 has no direct USB access, so the
dongle must be forwarded from Windows with `usbipd-win`. Zadig is *not* needed
on this route — usbipd hands the raw device through and the Linux side owns the
driver.

```powershell
winget install usbipd                    # once, PowerShell

usbipd list                              # find the BUSID
usbipd bind   --busid <BUSID>            # once per device, as Administrator
usbipd attach --wsl --busid <BUSID>      # after every reboot or replug
```

`lsusb` inside WSL should then list it, and `install.sh` applies normally. Two
things to expect:

- **udev may not run** without systemd enabled, so device permissions can need
  `sudo` even after the rules are installed.
- **USB-over-IP adds latency.** If `fm_observe.py` reports dropped blocks, drop
  `--sample-rate` to `250e3`. The recorder counts drops and prints the
  percentage precisely so this is visible rather than silently corrupting
  timing. WSL is fine for the Phase 0 checks; do long recordings on the Pi.

**Native Windows** (not WSL) is a different route: `librtlsdr.dll` and
`libusb-1.0.dll` on `PATH`, plus **Zadig** binding **WinUSB** to
"Bulk-In, Interface (Interface 0)".

### Troubleshooting

**`undefined symbol: rtlsdr_set_dithering` on import**

```
AttributeError: /usr/lib/x86_64-linux-gnu/librtlsdr.so:
undefined symbol: rtlsdr_set_dithering
```

pyrtlsdr 0.4.0 and later bind that symbol unconditionally at import, and it
exists only in the **rtl-sdr-blog fork** of librtlsdr — not in the mainline
Osmocom build that Debian and Ubuntu package. `requirements.txt` therefore pins
`pyrtlsdr>=0.3.0,<0.4`, which works against the packaged library. If pip has
pulled a newer one:

```bash
pip install "pyrtlsdr>=0.3.0,<0.4" "setuptools<81"
```

The `setuptools` bound is needed too: pyrtlsdr 0.3.0 imports `pkg_resources`,
which setuptools 81 removed.

*Optional upgrade path.* If you later want pyrtlsdr 0.5.0 — no deprecated
dependencies, plus official V4 and dithering support — build the blog fork
instead of pinning:

```bash
sudo apt install build-essential cmake libusb-1.0-0-dev
git clone https://github.com/rtlsdrblog/rtl-sdr-blog
cd rtl-sdr-blog && mkdir build && cd build
cmake ../ -DINSTALL_UDEV_RULES=ON && make && sudo make install && sudo ldconfig
pip install "pyrtlsdr>=0.5" "setuptools"
```

This matters mainly for **HF below 24 MHz**, where the V4 uses an upconverter.
At the ~100 MHz this project works at, the tuner is direct and the packaged
library is fine, so it is not worth destabilising a working rig mid-experiment.

**`usb_claim_interface error -6` / `Failed to open rtlsdr device #0`**

The kernel's DVB-T driver has the dongle. See *Drivers* above; after
blacklisting, unplug and replug so it re-enumerates.

**No devices found under WSL** — the dongle is not attached. Re-run
`usbipd attach --wsl --busid <BUSID>` from PowerShell; the attachment does not
survive a reboot or replug.

### Using more than one dongle

A generic DVB-T stick and a purpose-built RTL-SDR both enumerate as RTL2832U
devices and both work here, but they usually carry different tuner chips, which
changes the available gain steps and the noise figure. List what is attached:

```bash
python3 rf_check.py --list-devices
```

```
 idx  serial           tuner          gains dB  name
   0  00000001         R820T/R820T2   29
   1  00000001         R828D          29
```

Then pass `-D <idx>` to any check. The index comes from the USB stack and is
**not stable across replugs**, so re-check it rather than assuming.

Comparing two dongles is worth doing properly: run `--floor-test --no-lna` and
`--gain-linearity` on each. The one with the larger floor delta and the higher
SNR plateau has the better noise figure, and that is the one to put on the
antenna.

Note the blacklist in `install.sh` stops these dongles working as actual DVB-T
television receivers. If you need one back for TV, delete
`/etc/modprobe.d/blacklist-rtlsdr.conf` and reboot.

`--selftest` and `test_pipeline.py` need no dongle at all, so run those first to
confirm the Python side works before fighting drivers.

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
