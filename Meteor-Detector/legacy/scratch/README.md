# Scratch

Exploratory scripts kept for reference. **None of these are part of the
acquisition pipeline**, and despite the `test_` prefixes none of them are tests
— the real test is `../acquisition/test_pipeline.py`.

| File | What it was for |
|---|---|
| `check_calculation.py` | Working out the dBFS reference level against a live dongle. The conclusion — pyrtlsdr returns I,Q each in [-1,1], so full-scale complex power is 1.0, and an RTL-SDR has no absolute calibration anyway — is now encoded in `../acquisition/dsp.py`. |
| `test_simple.py` | Smallest possible "does the dongle respond" check. |
| `test_npz_values.py` | Ad-hoc dump of dBFS ranges in a `.npz`. |
| `waterfall-plot-test.py` | Spectrogram layout experiment on synthetic data. No SDR involved. |
| `find_curvature.py` | Central-angle-from-distance geodesy helper. Unconnected to the rest; useful when working out forward-scatter geometry. |

Note that these scripts used a Hann window while the old recorder used none, so
what was validated here was never what got recorded. That is why `dsp.py` now
exists and is imported by both the diagnostics and the recorder.
