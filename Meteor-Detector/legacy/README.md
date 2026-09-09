# Legacy

The first generation of this project, kept for reference. **None of this is on
the live path** — that is `../acquisition/`.

Nothing here is deleted because some of it still has value: if you have old
`.sigmf-data` recordings, the viewers below are the only things that read them.

## The old acquisition pipeline

| File | What it did | Why it was retired |
|---|---|---|
| `sdr-record.py` | Recorded SigMF (`.sigmf-data` float32 + `.sigmf-meta`) | Superseded by `../acquisition/fm_observe.py`. Its FFT path integrated 0.94 kHz of a 180 kHz FM channel, defaulted to `--gain auto`, used no window function, and reported a level that scaled with `--fft-size`. |
| `record.sh` | `rtl_sdr` raw IQ dump to `output.iq` | Raw IQ is 43–177 GB/day; the two-tier scheme in `fm_observe.py` replaces it. |
| `README_FM_OBSERVER.md` | Documented `fm_observe_npz.py` | **That script no longer exists.** `.gitignore`'s `/fm_*` rule swallowed it and it was never committed, so it was lost. This file is the only surviving record of what it did — its `.npz` field list is why `fm_observe.py` writes a schema `plot_npz_utc.py` can still read. |

## Viewers

| File | Reads | Note |
|---|---|---|
| `sdr-view-recordings.py` | SigMF | Tk desktop viewer. Looks for `recordings/` relative to the working directory. |
| `sdr-viewer/` | SigMF | Flask + Plotly web viewer, port 5001. Looks for `sdr-viewer/recordings/`, which never matched where `sdr-record.py` wrote — symlink one to the other if you need it. Runs with `debug=True` bound to `0.0.0.0`; don't expose it. |
| `plot_iq.py` | raw `.iq` | Interactive slider viewer. Bugs fixed before retirement: `int8`→`uint8` with DC offset removal, hardcoded `output2.iq`→CLI argument, a closure bug that made CSV export always dump the t=0 window, and `[::1000]` subsampling replaced with block max-hold. |
| `sdr-mini-gui.py` | live SDR | Click-a-frequency spectrum explorer. Still genuinely useful for looking around the band by hand. Defaults to `gain='auto'`. |
| `sdr-waterfall.py` | live SDR | One-shot capture and spectrogram. |

## `quick-sdr-analysis/`

Three `rtl_power` band-survey scripts and one saved scan. Superseded by
`rf_check.py --sweep`, which classifies signals by bandwidth rather than just
listing bins.

`output.csv` is worth keeping: a real 105–108 MHz survey from Sirsi on
2025-12-22. Re-reading it with the current understanding, every feature in it is
0–23 kHz wide — far too narrow for the ~180 kHz of an FM broadcast signal — and
both rows show the symmetric hump of the RTL-SDR's own passband rather than sky
noise. It is evidence of a receiver producing its own artifacts on an empty
band, which is what `rf_check.py --spur-test` now tests for directly.

Note the two scripts disagree with the data: `analyze_dbm_power.sh` scans
98.1–98.5 MHz, but the committed CSV is 105–108 MHz from a different run.

## `scratch/`

Exploratory scripts, described in `scratch/README.md`. Despite the `test_`
prefixes none of them are tests — the real one is
`../acquisition/test_pipeline.py`.
