# Handover — Remote Radio Observatory

Written 2026-09-18, distilling one long working session that took this repo
from "months of null results, unsure if it's the software or the hardware" to
a verified DSP pipeline, a diagnosed and partially-fixed antenna, and a
real (if still narrow) candidate signal on the air. Everything below is either
verified in this session against real hardware/data, or clearly marked as not
yet done.

## The project, in one paragraph

Detecting meteors by FM forward scatter: a distant FM transmitter, normally
below the horizon, briefly reflects off a meteor's ionised trail and appears
for a fraction of a second. An RTL-SDR + directional antenna at Sirsi,
Karnataka watches one channel continuously; a spike in received power is a
candidate ping. The system is meant to end up on a headless Raspberry Pi 5,
recording unattended, viewable remotely.

## Where things stand right now

- **Repo**: rebuilt, documented, tested. `master` at `53070c4`.
- **DSP**: verified correct against synthetic signals (`rf_check.py --selftest`,
  `test_pipeline.py`) — not yet run against a real recording session.
- **RF chain**: verified working. A bare monopole (no LNA) on the actual RTL-SDR
  Blog V4 showed the sky is reaching the receiver (floor test **PASS, 5.43 dB**,
  after fixing the antenna).
- **A candidate real signal exists**: ~97.9 MHz, survived antenna disconnect,
  moved ~79 kHz between two captures (consistent with FM deviation) — **not yet
  confirmed by ear on SDR++**, which is the next action.
- **`fm_observe.py` (the actual recorder) has not been run yet.** Everything so
  far is `rf_check.py` diagnostics. This is the biggest gap before real data
  exists.
- **A local interactive viewer exists** and works against synthetic data; not
  yet tried against a real recording (because none exists yet).
- Two loose ends left deliberately unresolved — see **Outstanding items**.

## Repo map

```
Remote-Radio-Observatory/
├── README.md                          top-level quick start
├── HANDOVER.md                        this file
├── requirements.txt                   pinned deps (see "Environment" below)
└── Meteor-Detector/
    ├── install.sh                     venv + DVB-T blacklist + udev rules
    ├── radar_config_sample            site metadata (lat/lon/elev, stub)
    ├── MeteorRadioFM/                 git submodule (fork), uninitialized
    ├── acquisition/                   <-- the live pipeline, everything current
    │   ├── dsp.py                     shared, verified DSP primitives
    │   ├── rf_check.py                Phase 0: 7 diagnostic checks
    │   ├── fm_observe.py              Phase 1: the recorder (not yet run for real)
    │   ├── plot_npz_utc.py            static one-shot plot
    │   ├── viewer/                    interactive local web viewer (new, stage 1)
    │   ├── test_pipeline.py           synthetic end-to-end test
    │   ├── fm-observe.service         systemd unit for the Pi
    │   ├── README.md                  full rationale, ordered run sequence
    │   └── RUN_SHEET.md                bench checklist — blanks NOT filled in;
    │                                   real numbers live in this handover instead
    └── legacy/                        first generation, nothing deleted, all
                                        superseded — see legacy/README.md
```

## The physics established (so decisions aren't re-derived from scratch)

- **dBFS, never dBm.** An RTL-SDR has no absolute power calibration. The only
  meaningful quantity is SNR against a tracked floor.
- **An FM broadcast channel is ~180 kHz wide**, constant-envelope. The old
  code integrated 0.94 kHz — 0.5% of the channel — which measured the
  station's *program material*, not its presence.
- **Galactic background noise (1000–3000 K at 100 MHz) is the RF-chain test
  signal**, because it requires no transmitter at all:
  - with a good LNA (NF 0.5–2 dB): connecting the antenna should raise the
    floor **8–16 dB**
  - **bare dongle, no LNA** (NF 3.5–6 dB): expect only **3–8 dB**, because the
    receiver's own noise is much closer to the sky's — this is the number that
    matters for the current setup
  - **> 18 dB is suspicious**: bigger than the sky can produce, almost always
    man-made interference (laptop, charger) rather than sky sensitivity
- **Sidereal proof.** The galactic plane sweeps the beam on a **23h56m04s**
  period; anything terrestrial is on the 24h00m solar day. Separating them *by
  period* needs a full year (they differ by 1 part in 366) — the tractable
  measurement is the **phase drift**, 3.93 min/day, detectable in 10 days,
  convincing in 3–4 weeks. (`rf_check.py --sidereal`, not yet run — needs
  weeks of real recording first.)
- **Site geometry.** From Sirsi (14.641 N, 74.831 E), optimal forward-scatter
  range is 800–2000 km. FM is audible toward **Mangalore (197 km, bearing
  178° S)** with a monopole and nowhere else tried — meaning the **southern
  horizon is the open one**. In-range candidates along that heading:
  **Thiruvananthapuram (718 km, 161°)** and **Colombo, Sri Lanka (1018 km,
  147°)**. Northern candidates (Delhi, Bhopal, Kolkata) are geometrically
  optimal but likely terrain-blocked — unverified.
- **Validate on aircraft before meteors.** Aircraft scatter gives Doppler-drifting
  events lasting 10–60 s, many per day, same geometry — a same-week sensitivity
  check instead of a months-long wait.
- **Indian FM transmitters run ~10–20 kW ERP**, well under the 100+ kW European
  observers typically use for this technique — expect fewer pings than tutorials
  promise.

## Why the old software could never have found a meteor

Found by testing the DSP against synthetic signals of known amplitude, not by
inspection:

| Bug | Old `sdr-record.py` / `plot_iq.py` | Effect |
|---|---|---|
| Integration bandwidth | 0.94 kHz of a 180 kHz channel | measured audio content, not carrier presence |
| PSD normalization | divided by `(fs/nfft)` instead of multiplying | +60.2 dB error at nfft=1024, scaled with `--fft-size` — no two runs comparable |
| Off-by-one bins | `integration_bins=5` → actually 4 bins | any odd value silently wrong |
| No window function | rectangular | 112 dB worse rejection of an off-bin interferer than Hann |
| Gain | defaulted to `auto` | AGC actively erases the power variation being measured |
| `plot_iq.py` sample decode | read `rtl_sdr`'s **unsigned** 8-bit output as **signed** `int8` | every sample above 127 wrapped negative; DC offset (127.5) never removed — never decoded correctly |
| `plot_iq.py` decimation | `[::1000]` | genuine subsampling — a 20 ms ping can fall entirely in the gap |
| `plot_iq.py` CSV export | `time_data`/`amp_data` rebound as locals in `update()` | export always dumped the t=0 window regardless of the UI slider |

All fixed in the new pipeline; originals preserved read-only under `legacy/`
with this same table in `legacy/README.md`.

## The new acquisition pipeline

**`dsp.py`** — one DSP module, imported by every other tool, so validated math
equals recorded math. Correct PSD: `|X|² / (fs · Σw²)`. Verified: a full-scale
tone reads **0.00 dBFS at every window and every FFT length** (was previously
dependent on both).

**`rf_check.py`** — seven checks, ordered because each gates the next. Full
detail and exact commands in `acquisition/README.md`; skip straight to
**Bench results** below for what they actually returned.

1. `--selftest` — DSP correctness, no hardware
2. `--floor-test` — the RF-chain verdict; works on a dead band (`--no-lna` for
   a bare dongle)
3. `--spur-test` — antenna connected vs disconnected; separates real signals
   from things the receiver manufactures itself
4. `--gain-linearity` — finds the LNA compression knee (meaningless on an
   empty channel — see bug below)
5. `--stability` — AGC/drift check, needs a steady real signal
6. `--sweep` — band survey, classifies by bandwidth, proposes quiet channels
7. `--sidereal` — phase-drift proof the antenna sees the sky, needs 10+ days
   of real `fm_observe.py` output

**`fm_observe.py`** — the actual recorder, **not yet run against real
hardware**. Gap-free async capture (`read_bytes_async`, no duty-cycle sleep),
fixed gain enforced, hysteretic trigger with refractory period. Two tiers:

- **Tier 1**: `.npz` chunks, `t_utc_ns / power_dbfs / noise_dbfs / peak_dbfs /
  snr_db / trigger` + scalar metadata. At defaults (1.024 MS/s, nfft 8192):
  **125 Hz, 144 files/day, 10.8M points/field/day**.
- **Tier 2**: raw IQ ring buffer in RAM, dumped ±N seconds only around a
  trigger (`--save-iq`).

**`viewer/`** — new this session. A Flask+Plotly local web page for browsing
Tier-1 output: pan/zoom, recorder-trigger markers, and a **client-side
threshold slider** that instantly recomputes which points would have been
flagged at a different SNR threshold — the manual-verification step before
committing a real `--threshold-db`. `demo_data.py` generates a synthetic
dataset in the exact recorder schema, so it's usable with zero real data.
Decimates by min/max-per-bucket, never plain subsampling, for the same reason
`plot_iq.py`'s old `[::1000]` was wrong. Tested (`test_viewer.py`, 10/10) —
this is the tool's own test, separate from `test_pipeline.py`.

Three-stage remote-access plan: **stage 1** (this local server) is done;
**stage 2** (same code, `--host 0.0.0.0`, browse from the Pi's LAN IP) needs
zero new code; **stage 3** (view from anywhere) is deliberately **not
decided** — revisit once real recordings exist and the actual need is known
(Cloudflare Tunnel vs Tailscale were the two candidates discussed).

## Bench test results — the actual numbers obtained this session

Hardware: **RTL-SDR Blog V4** (R828D tuner, 29 gain steps, max 49.6 dB),
bare monopole (no LNA), WSL2 on Windows, dongle forwarded via `usbipd`.

1. **First floor test — FAIL.**
   ```
   connected:    -46.58 dBFS   disconnected: -47.95 dBFS
   delta: +1.36 dB   (expected 3-8 dB bare)
   ```
   Backed out an implied antenna temperature of ~130 K — *below ambient*,
   which a passive antenna cannot reach by being merely lossy; pointed at
   **impedance mismatch**, not a broken cable. Led to two suspects: whip
   too short (quarter-wave at 100 MHz is 75 cm) and **no ground plane** (a
   monopole needs a counterpoise; magnetic base on a non-conductive surface
   makes the coax shield an accidental one).

2. **After fixing the antenna — PASS.**
   ```
   connected:    -42.57 dBFS   disconnected: -47.99 dBFS
   delta: +5.43 dB   implied system NF ≈ 4.9 dB
   ```
   The disconnected floor was near-identical across both runs (−47.95 vs
   −47.99) — clean confirmation the receiver hadn't changed, only the
   antenna's coupling had. **This is the answer to the months-old question:
   the RF chain works.** *(What exactly was changed on the antenna — whip
   length, ground plane, or both — was not recorded. Worth writing down before
   the Yagi build.)*

3. **`--gain-linearity` on 107.1 MHz (empty channel) — misleading, now fixed.**
   Original run reported "SNR falls 2.2 dB at max gain, front end compressing"
   and recommended `--gain 0.0`. **That was a tool bug**: on an empty channel
   both "signal" and "noise" bins are just receiver noise, so the ratio
   tracks the tuner's own response shape and wanders a couple of dB for no
   physical reason — `--gain 0.0` would have left the receiver ADC-noise
   -limited. Fixed to detect the flat case (SNR range < 3 dB) and report
   **INCONCLUSIVE** instead of a verdict. **Needs re-running against a real
   receivable signal** (97.9 MHz once confirmed) for a real answer.

4. **`--sweep`, 88–108 MHz, gain 49.6 dB.** One feature stood out:
   **~97.84–97.9 MHz, ~21.8 dB above floor, ~50–75 kHz wide** (narrower than
   full 180 kHz broadcast, consistent with a weak/distant station or one at
   low modulation during capture). Its peak **moved ~79 kHz between the sweep
   and the later spur-test capture** — a fixed spur cannot do that; FM
   deviation is ±75 kHz, so a wandering peak of that size is exactly what a
   modulated carrier looks like.

5. **`--spur-test`, gain 49.6 dB — 10 signals survived antenna disconnect,
   but "survived disconnect" ≠ "is a broadcast station."** Re-classified by
   width after the fact:
   ```
     94.592   +13.8 dB   0.8 kHz   narrowband EMI
     97.890    +7.7 dB   0.8 kHz   narrowband EMI (shoulder of 97.9?)
     97.895    +9.7 dB   0.5 kHz   narrowband EMI (shoulder of 97.9?)
     97.919   +22.4 dB  51.0 kHz   CANDIDATE STATION  <-- the same 97.9 feature
     98.560    +9.3 dB   0.5 kHz   narrowband EMI
     98.816   +13.4 dB   0.5 kHz   narrowband EMI
    100.000   +25.3 dB   1.0 kHz   narrowband EMI  (exactly on 100 MHz)
    100.800   +16.0 dB   0.8 kHz   narrowband EMI
    102.784   +13.8 dB   0.5 kHz   narrowband EMI
    107.008   +12.9 dB   0.5 kHz   narrowband EMI
   ```
   **6 of the 9 EMI lines are exact multiples of 64 kHz**, and one sits at
   precisely 100.000 MHz — a clock/switching-supply harmonic (laptop, charger,
   or USB link), not radio. Tool now classifies drop+width together (station /
   candidate / EMI / internal) instead of calling everything "on the air."
   **These EMI lines matter operationally**: one landing inside a chosen
   observing channel will trigger the detector on interference, not sky.

**Immediate next action or diagnostic tool run**: tune **97.9 MHz, WFM** in
SDR++. Listening for audio/RDS settles in ten seconds whether this is a real
station (and gives station ID → transmitter location → path), versus checking
against 100.000 MHz side-by-side as a reference for what EMI sounds/looks
like.

## Environment / driver gotchas (WSL2 + this specific dongle)

- **WSL2 has no native USB.** Dongle must be forwarded from Windows:
  `usbipd bind`/`usbipd attach --wsl` (PowerShell, as Administrator) — **does
  not survive reboot or replug**, must be re-run each session.
- **`pyrtlsdr` version is pinned `>=0.3.0,<0.4`.** 0.4.0+ binds
  `rtlsdr_set_dithering` unconditionally at import; that symbol exists only in
  the rtl-sdr-blog fork of librtlsdr, not the mainline Osmocom build Ubuntu
  ships. Without the pin: `AttributeError: undefined symbol:
  rtlsdr_set_dithering` on the very first import.
- **`setuptools<81` is pinned too** — pyrtlsdr 0.3.0 imports `pkg_resources`,
  which setuptools 81 removed.
- **`install.sh` blacklists the kernel's DVB-T driver**
  (`dvb_usb_rtl28xxu`), which otherwise claims the dongle before librtlsdr can
  open it (`usb_claim_interface error -6`). This is reversible — delete
  `/etc/modprobe.d/blacklist-rtlsdr.conf` and reboot — but it means **the same
  dongle can't simultaneously be a TV tuner and an SDR** without that toggle.
  Relevant since the user plans to test a DVB-T stick later too.
- **Multiple dongles**: `rf_check.py --list-devices` enumerates; `-D <idx>` on
  any check selects one. Index is **not stable across replugs**.
- **Confirmed hardware**: RTL-SDR Blog V4, Rafael Micro R828D tuner, 29 gain
  steps (0.0 to 49.6 dB), bias-tee capable (`rtlsdr_set_bias_tee` present in
  the installed library — not yet used, but the LNA could eventually be
  powered up the coax instead of a separate supply). V4's HF upconverter is
  irrelevant at ~100 MHz — the tuner is direct there, mainline librtlsdr is
  fine, no need for the blog-fork build.

## Outstanding items — noticed, not yet actioned

1. **A branch never got cleaned up.** `rebuild-acquisition-pipeline` still
   exists both locally and on `origin`, identical to `master` — created
   before "commit to master, no branches" was said. Delete with:
   ```bash
   git branch -d rebuild-acquisition-pipeline
   git push origin --delete rebuild-acquisition-pipeline
   ```
2. **`origin` is still an HTTPS remote with no stored credentials.** Every
   push this session went via an explicit `git@github.com:...` SSH URL
   instead of `git push`, because HTTPS fails with no username. Worth fixing
   once, permanently:
   ```bash
   git remote set-url origin git@github.com:SudeepJoshi22/Remote-Radio-Observatory.git
   ```
3. **`RUN_SHEET.md`'s fill-in-the-blank results were never filled in.** The
   real numbers from the bench session live only in this handover (see
   above) and in chat history. Worth transcribing into the run sheet itself,
   or treating this handover as the record instead.
4. **What exactly fixed the antenna (whip length vs ground plane vs both)
   was not recorded** at the time. Reproducing it deliberately on the Yagi
   build depends on knowing which one mattered.
5. **The GitHub MCP server has been failing to connect all session**
   (`400: Authorization header is badly formatted`) — unrelated to the SSH
   push path above; if any GitHub-integrated tooling is expected to work,
   this needs separate attention.

## What to do next, in order

1. **Confirm 97.9 MHz by ear** in SDR++ (WFM, listen for audio/RDS). Settles
   whether the candidate signal is real and gives a station ID.
2. **Run `fm_observe.py` for real, for the first time.** Everything validated
   so far is `rf_check.py` diagnostics against live samples, not the actual
   recorder. The plan is now an overnight run, on the Yagi, on a headless Pi
   — full ordered command sequence in `acquisition/RUN_SHEET.md`'s "First
   overnight acquisition" section (added after this handover was written).
   Includes a real, sourced candidate-station list for the 750–1000 km band
   (97.9 MHz / Colombo / China Radio International as the leading match,
   with backups) — supersedes the "start short (an hour)" note this bullet
   used to carry.
3. **Point the new viewer at that real recording**
   (`viewer/server.py --dir ~/fm_observations`) — the first real test of the
   viewer against real data rather than synthetic.
4. **Re-run `--gain-linearity` and `--stability` on 97.9 MHz** (or whichever
   frequency is confirmed live) now that there's a real signal to measure
   against, replacing the earlier inconclusive/misleading runs.
5. **Build the Yagi setup, reproducing whatever fixed the monopole**
   deliberately rather than by accident, and re-run `--floor-test` on it —
   expect an improvement over the monopole's 5.43 dB once directional gain is
   in play.
6. **Once weeks of real Tier-1 data exist**, run `--sidereal` for the
   unfakeable proof the antenna sees the sky, and decide the stage-3
   remote-access approach based on actual usage patterns.
7. **Later**: repeat the whole Phase-0 sequence with the DVB-T dongle for
   comparison (remember to reverse the DVB blacklist first, or keep the two
   dongles' roles separate).
