# Bench Run Sheet — bare antenna, no LNA

Print this or keep it open while testing. Commands are copy-pasteable in order;
each one gates the next.

**Setup used when this sheet was written:** RTL-SDR Blog V4 (R828D, 29 gain
steps, max 49.6 dB), monopole, no preamp, WSL2 over usbipd.

```bash
cd ~/github/Remote-Radio-Observatory/Meteor-Detector/acquisition
source ../venv/bin/activate
mkdir -p ~/rro-logs
```

---

## 0. Software sanity — no dongle needed

Do this before touching usbipd. If it fails, the problem is not your hardware.

```bash
python3 rf_check.py --selftest
python3 test_pipeline.py
```

- [ ] `SELF TEST PASSED`
- [ ] `ALL CHECKS PASSED`

## 0b. Attach the dongle (WSL only)

PowerShell **as Administrator**, on the Windows side. Does **not** survive a
reboot or replug — re-run `attach` each time.

```powershell
usbipd list
usbipd bind   --busid <BUSID>      # once per device, ever
usbipd attach --wsl --busid <BUSID>
```

```bash
python3 rf_check.py --list-devices
```

- [ ] one row listed, tuner named

---

## 1. Floor test — **the measurement that matters**

Interactive: prompts you to connect, then disconnect.

```bash
python3 rf_check.py --floor-test --no-lna -g 49.6 -f 107.1e6 2>&1 | tee ~/rro-logs/01-floor.txt
```

| Result | Verdict |
|---|---|
| **≥ 4 dB** | PASS — antenna is feeding the receiver. ~5 dB is normal bare. |
| 2–4 dB | WARN — contributing, but weakly. Short or badly matched monopole. |
| **< 2 dB** | FAIL — antenna contributing nothing. This is the answer if it happens. |
| **> 18 dB** | You are hearing the laptop, not the sky. Move outdoors and repeat. |

- Unplug at the **antenna connector**, not the USB.
- Do it **outdoors, away from the building**, if at all possible. Indoors,
  switching-supply noise inflates the delta and the number means little.

**Result: __________ dB**

If it fails, the number still tells you *which* fault. The tool back-computes
the delivered antenna temperature:

| Implied T_ant | Meaning |
|---|---|
| ~1500 K | a good antenna seeing the sky |
| ~290 K | connected but lossy — matched, just inefficient |
| **< 250 K** | **impedance mismatch**, not a broken cable |
| ~0 K | open circuit / broken feed |

A bare monopole is the usual culprit for the mismatch case. Two things fix it:

1. **Whip length** — a quarter wave at 100 MHz is **75 cm**. A short whip is a
   severe mismatch at this frequency.
2. **Ground plane** — a monopole is only half an antenna. It needs a
   counterpoise: stand the magnetic base on a metal sheet, or add three or four
   75 cm radials. Without one the coax shield acts as the counterpoise and
   behaves badly.

For scale, even VSWR 20:1 still yields about 2.5 dB, so a delta below that means
the match is worse than 20:1.

## 2. Spur fingerprint

Interactive, same connect/disconnect. Sweeps twice, so give it a minute.

```bash
python3 rf_check.py --spur-test -g 49.6 2>&1 | tee ~/rro-logs/02-spurs.txt
```

Anything marked **INTERNAL** is the dongle generating it, not a signal on the
air. With no LNA fitted, this list is provably the V4's own — it is the baseline
for blaming the amplifier later if new spurs appear.

**Internal spurs found: __________________**

## 3. Gain curve

Automatic; steps through all 29 gains.

```bash
python3 rf_check.py --gain-linearity -f 107.1e6 2>&1 | tee ~/rro-logs/03-gain.txt
```

- SNR rising **> 3 dB** across the range → something external is reaching the
  tuner. That is a positive RF-chain result on its own.
- SNR **falling > 2 dB** at max gain → front end compressing. Unlikely without
  an LNA; expected once one is fitted.
- Prints a **recommended gain** — use it for everything afterwards.

**Recommended gain: __________ dB**

## 4. Band survey

```bash
python3 rf_check.py --sweep -g 49.6 --save ~/rro-logs/bare_sweep.npz 2>&1 | tee ~/rro-logs/04-sweep.txt
```

Point the monopole **south, bearing 178° toward Mangalore** — the one direction
with proven signal at Sirsi.

- A real FM station is **~180 kHz wide**. Anything under 30 kHz is a spur.
- An empty band here is *expected* and is good for meteor scatter.

**Stations found (freq / width): __________________**

## 5. Stability — only if step 4 found a real station

```bash
python3 rf_check.py --stability -f <THAT_FREQ> -g 49.6 --minutes 10 2>&1 | tee ~/rro-logs/05-stability.txt
```

- std **< 0.5 dB** → PASS, gain genuinely fixed
- std **> 1.5 dB** → AGC still active, or you are watching noise
- drift **< 1 dB** over the run → PASS

Skip if the band is empty; it needs a steady signal to mean anything.

**std: ________ dB   drift: ________ dB/min**

---

## Reference: known-good numbers from this rig

Measured on the V4 at 107.1 MHz, 1.024 MS/s, gain 49.6 dB, 12 frames:

```
channel power  -46.07 dBFS   (std 0.23 dB)
guard noise    -46.63 dBFS
SNR              0.56 dB      <- empty channel, as expected
```

The **0.23 dB standard deviation** is the tell that AGC is off and gain is
fixed. If yours is much larger, something is wrong before you measure anything
else.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `undefined symbol: rtlsdr_set_dithering` | pyrtlsdr ≥ 0.4 against mainline librtlsdr. `pip install "pyrtlsdr>=0.3.0,<0.4" "setuptools<81"` |
| `No module named 'pkg_resources'` | setuptools ≥ 81. `pip install "setuptools<81"` |
| `usb_claim_interface error -6` | Kernel DVB driver holds the dongle. Run `install.sh`, then unplug/replug. |
| No devices found (WSL) | Not attached. Re-run `usbipd attach --wsl --busid <BUSID>`. |
| Device vanished mid-session | usbipd attachment dropped. Re-attach. |

## What to send back

`01-floor.txt` and `02-spurs.txt` decide everything else.
