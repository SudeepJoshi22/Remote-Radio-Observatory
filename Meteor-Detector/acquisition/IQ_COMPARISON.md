# Compare the three estimators on the same IQ recording

`compare_iq.py` reads raw RTL-SDR **unsigned 8-bit interleaved I,Q** (`.cu8` or
`.iq`), without loading the entire recording into RAM. It compares our recorder,
a MeteorRadio-style estimator, and an Echoes-style differential estimator.
It does not change acquisition or require an SDR to be connected.

This is a **matched-FFT estimator comparison**, not an emulation of the complete
three applications. The FM settings below adapt the two upstream spectral
estimators to explicitly stated bands. They have not been validated as meteor
detectors at Sirsi. The program does not select a winner or classify meteors.

## Run on the Raspberry Pi or a computer with a copy of the recording

Use the project's Python environment (NumPy and Matplotlib are needed):

```bash
cd ~/github/Remote-Radio-Observatory/Meteor-Detector/acquisition
../venv/bin/python compare_iq.py \
  ~/iq_capture/sirsi_1033MHz_1024ksps_gain49p6.cu8 \
  --sample-rate 1024000 --center-freq 103300000 --gain 49.6 \
  --output-dir ~/iq_comparison_1033
```

If your environment is elsewhere, replace `../venv/bin/python` with that Python.
The script produces static PNGs without a desktop/display server. Use a new or
empty output directory on each run; it refuses to overwrite existing results.

Start with a short section if desired:

```bash
../venv/bin/python compare_iq.py \
  ~/iq_capture/sirsi_1033MHz_1024ksps_gain49p6.cu8 \
  --sample-rate 1024000 --center-freq 103300000 --gain 49.6 \
  --start 600 --duration 120 --output-dir ~/iq_comparison_minutes10to12
```

`--start` is seconds from the beginning of the raw file. Without `--start-utc`,
graphs use elapsed seconds. If you KNOW when the first IQ pair was captured, add
`--start-utc 2026-09-24T10:00:00Z` (replace with the actual time). A file's creation
or modification timestamp is not a reliable first-sample timestamp.

The file size is snapshotted at launch: if capture is still writing, only the
complete frames already present are analysed. It does not follow a growing file.
An incomplete final FFT frame is discarded and reported. Do not truncate or
replace the input while analysis runs.

## Outputs

* `comparison.png`: all three level/score plots for the selected interval.
* `strongest_excursion.png`: detail around the largest **channel-power** frame.
  This selection does not imply the event is a meteor. Use `--zoom-seconds` to
  change the detail window; neighbouring MeteorRadio blocks are included.
* `metrics.npy`: memory-mappable float64 table, one row per shared FFT frame.
* `summary.json`: column names, capture parameters, actual bands, pinned sources,
  counts and limitations. Threshold exceedances are measurements, not events.
* `strongest_frame_spectrum.npz`: full bin-power spectrum for that frame, frequency
  offsets relative to target, and file-relative time. This NPZ **does** retain
  spectral data, unlike the recorder's summary-only NPZ chunks.

Upload the PNGs and `summary.json` for review; you do not have to upload the raw
hour-long recording. Keep the raw recording for subsequent analysis.

An hour at 1.024 MS/s contains about 7.37 GB of IQ. Default working FFT blocks
contain only 64 frames (0.512 s). The derived table is about 58 MB/hour and stays
on disk via a memory map. Plot memory scales with the much smaller derived
table, not with the raw IQ. Processing reads every complete FFT frame; plotting
reduces points only after the estimators have run.

## What is implemented

All three recipes receive identical non-overlapping FFT frames. Defaults match
our recorder: 8192 samples, Hann window, 125 Hz bins, 8 ms per frame at 1.024 MS/s.
Window power normalization is from `dsp.py`. Optional `--nfft` and `--window`
change the shared front end for all three, and are recorded in the summary.

### Our recorder

Channel power is integrated over +/-90 kHz. Reference power is integrated over
[-400,-150] and [150,400] kHz, then scaled by the ratio of bin counts. The other
reference is `dsp.RollingFloor` (10th percentile over 60 seconds, updated about
twice per second). The score is channel dBFS minus the larger reference.
Both references are saved separately. Default threshold line: 5 dB, adjustable
with `--our-threshold-db`. Hysteresis/refractory/event counting are NOT replayed.

History starts at the selected interval. `ours_warm` identifies when a full
reference window has been acquired; the original recording may have had earlier
history that is unavailable. Our plotted score is retained during this warm-up,
as in the recorder. Default settings reproduce `ChannelMetrics`/`RollingFloor`
for the same input frames, subject to the missing prehistory.

### MeteorRadio-style

Source: `rabssm/MeteorRadio`, revision
`c13c2334e39a0f80b98a759b28c0e4a1efcfc481`, `src/meteor_radar.py`,
`analyse_psd()` and `check_trigger()`.

For each block, take the maximum bin power across detection frequencies and
times. Find the time column containing the maximum power in the wider reference
band; the median of that column across reference frequencies is the noise
estimate. Those two maxima need not occur at the same time. Score:
`10 log10(peak / median)`. Default threshold is upstream's linear 45 (~16.53 dB).

The separate broadband ratio is the maximum of the full-spectrum per-scan
medians divided by their median across the block. A new candidate must have a
ratio <=3 as in upstream. Red crosses identify ratios >3. No event state machine,
holdoff, post-capture duration calculation or RF resampling is emulated.

One result is stored at the detection peak's frame per block; other rows contain
NaN for MeteorRadio fields. It represents that entire block, not an 8-ms decision.
`--block-frames` controls the block length and thus changes this method's result.
It does not change our or Echoes' values. The partial final block is included.

Differences from native upstream: shared sample rate/window/FFT, no padded
ShortTimeFFT columns, configured block length instead of SDR callback size,
float64 powers instead of float16. These are explicit experimental controls.

### Echoes-style (differential mode)

Source: Echoes revision `ecf4400bcf9af0300ea42d24c2405c531fe420c6`,
`trunk/echoes/radio.cpp` and `control.cpp`.

Bin levels are `10 log10(bin power / mean full-spectrum bin power)`, with scale
gain 1 and offset 0. S is the maximum level inside the detection band. Each
scan's reference is the arithmetic mean of the **logarithmic** levels in the
configured displayed/reference band. A FIFO smooths these scan means. The
inspected code includes the newest scan twice in the next-scan average; this
weighting is reproduced. The current S is compared with the previous update of
N. Scoring waits for `--echo-scans` prior scans (default 125).

No display gain calibration, notches, scan dropping/max-hold, adaptive threshold
state machine, event merging, noise-limit inhibition, or Ebrow classification is
reproduced. This is the differential score, not automatic mode. There is no
assumed Echoes threshold; optionally supply `--echo-threshold-db`.

**Its spectrum-relative levels are not ADC-referenced dBFS**, so its left panel
has its own label and scale. Scores from the three methods are also not directly
equivalent definitions of physical S/N. Larger numbers do not establish a better
detector.

## FM versus beacon settings

Default `--preset fm` uses a peak band +/-90 kHz and reference band +/-400 kHz
for MeteorRadio/Echoes. These **adaptations** compare spectral peak prominence
inside the FM channel against a wider background. The reference includes the
channel; it does NOT reuse our guard-only reference or temporal percentile.
These ranges can contain neighbouring stations and must not be assumed clean.

`--peak-band LOW HIGH` and `--reference-band LOW HIGH` override those two bands
in Hz relative to the target. Echoes' reference band represents its displayed
band; the upstream application has user-configurable ranges, not one universal
meteor setting. `--target-offset` is target frequency minus SDR tuned frequency.
All bands must fit inside the captured spectrum.

`--preset beacon` uses MeteorRadio's +/-120 Hz peak and +/-500 Hz reference
ranges for both spectral methods. At 1.024 MS/s use finer resolution, e.g.:

```bash
../venv/bin/python compare_iq.py \
  ~/iq_capture/sirsi_1033MHz_1024ksps_gain49p6.cu8 \
  --sample-rate 1024000 --center-freq 103300000 --gain 49.6 \
  --preset beacon --nfft 131072 --window hamming --block-frames 4 \
  --echo-scans 8 --output-dir ~/iq_comparison_beacon_ranges
```

This examines only a narrow part of the FM signal. It is NOT a GRAVES recording
and does not turn broadcast FM into a beacon. Our integration method also uses
the changed shared FFT in this run. The script rejects under-resolved beacon
bands rather than silently using only one central bin. No DC-bin removal is
silently applied; a centre-frequency spur can affect the peak methods.

## How to read the plots

Lines show display-bucket medians, shading covers min/max, and score maxima are
retained as a thin line. MeteorRadio block results use points to avoid inventing
continuous slopes between block peaks. Scores are computed BEFORE plotting aggregation: never
subtract independent peak signal and peak noise traces. A smooth reference is
not by itself evidence of accuracy. Inspect spectral content and independently
identify candidate events before comparing missed detections or false alarms.

`rail_fraction` counts I/Q byte values at 0 or 255. It can flag clipping concerns,
but absence of rail hits does not rule out tuner overload. IQ timestamps assume
uninterrupted samples; raw files cannot reveal where driver drops occurred.

## Verification

```bash
python3 -m unittest -v test_compare_iq.py
```

Tests check local DSP replay, upstream estimator arithmetic, reference-column
selection, broadband rejection, Echoes weighting/lag, streaming independence,
invalid input handling, partial frames, timestamps and real plot generation.
Synthetic tests validate implementation, not astronomical classification.
