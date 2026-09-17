# Viewer

Stage 1 of the plan: an interactive local web page for browsing `fm_observe.py`
output, so a day's data can be scrolled through and candidate meteor pings
eyeballed *before* committing to a `--threshold-db` value in the recorder.

`plot_npz_utc.py`, one directory up, is the static alternative -- one file,
one matplotlib window, no interactivity. This is for the case that motivated
it: sitting down with a full day of chunks, panning and zooming, and dragging
a threshold slider against the real noise floor to see what it would and
would not have caught.

## Try it without any hardware

```bash
python3 demo_data.py --hours 24 --out ~/demo_observations
python3 server.py --dir ~/demo_observations
```

Open **http://localhost:5002**. `demo_data.py` writes the exact schema
`fm_observe.py` writes -- there is no separate "demo mode" in `server.py` to
keep in sync with the real one.

## With real data

```bash
python3 server.py --dir ~/fm_observations
```

## What's on the page

- Two linked plots: channel power + noise floor on top, SNR below, sharing the
  time axis. Pan and zoom either one (Plotly), the other follows.
- Red dots mark frames `fm_observe.py` actually flagged as a trigger.
- A **threshold slider** recomputes, client-side, which points would have been
  flagged at a different SNR threshold -- instantly, without re-recording or
  restarting anything. This is the manual-verification step: drag it against
  real data to see where a threshold should sit before setting
  `--threshold-db` for the next recording run.
- Preset buttons for last 1h / 6h / 24h / 7d / all.

## Why min/max buckets, not a plain average

A day of data at the recorder's defaults is 10.8M points per field
(125 Hz × 86400 s) -- too many for a browser to plot directly. The server
reduces it to at most a few thousand points per request, but by taking the
**min and max within each bucket**, never a mean. A meteor ping is a brief
spike; averaging it into a bucket with mostly quiet neighbours would erase it
from the plot precisely where you need to see it. This is the same principle
`dsp.py` and the fixed `legacy/plot_iq.py` follow for the same reason -- see
the main `acquisition/README.md` under "Decimate by integrating, never by
subsampling."

Zoom in far enough (below a few thousand samples in view) and the server
switches to returning every raw sample -- no bucketing artefacts once you're
looking closely at a candidate.

## Files

| File | What |
|---|---|
| `server.py` | Flask app: indexes chunk files by filename, loads the ones overlapping a requested time range, downsamples, serves JSON |
| `index.html` | The page: Plotly.js chart, range buttons, threshold slider |
| `demo_data.py` | Synthetic dataset generator, same schema as `fm_observe.py` |
| `test_viewer.py` | Smoke test: generates data, hits the Flask app directly, checks bucketing preserves injected pings |

## Stages 2 and 3

This is a plain Flask dev server, so:

- **Stage 2 (Pi, same network):** run the identical command on the Pi --
  `python3 server.py --dir ~/fm_observations --host 0.0.0.0` -- and browse to
  `http://<pi-lan-ip>:5002` from any device on the same network. No code
  change.
- **Stage 3 (from anywhere):** put a tunnel or reverse proxy in front of the
  same server (Cloudflare Tunnel, Tailscale, etc.) rather than rewriting it.
  Not yet decided which -- revisit once real recordings are flowing.

Flask's built-in server prints a warning that it is a development server, not
for production. For a single viewer on your own network that is fine; if this
ever needs to hold up under real concurrent load, put it behind `gunicorn` or
similar first.
