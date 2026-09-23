# Viewer

An interactive web page for browsing `fm_observe.py`
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
- A UTC focus selector: choose a calendar day, starting hour, and 1-hour,
  6-hour, or 24-hour window. Focus windows use a denser response than the
  multi-day overview, making it practical to inspect an older day's ping.
- Plotly pan/zoom requests that selected UTC range again. A close view therefore
  returns the original 125 Hz samples instead of merely magnifying the day
  response.
- A dated list of completed NPZ chunks and IQ event sidecars, with one-click
  downloads. Downloads are limited to filenames currently present in the
  recorder's strict indexes; `.npz.tmp` files are never listed or served.

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
| `index.html` | The page: Plotly.js chart, server-backed zoom, file list, downloads, range buttons, threshold slider |
| `wsgi.py` | Gunicorn entry point; reads `RRO_DATA_DIR` |
| `rro-viewer.service` | Production Pi service on port 5002 |
| `demo_data.py` | Synthetic dataset generator, same schema as `fm_observe.py` |
| `test_viewer.py` | Smoke test: generates data, hits the Flask app directly, checks bucketing preserves injected pings |

## Pi service and private remote viewing

For a Pi that records continuously, install the dependencies from the repository
requirements and install `rro-viewer.service` as documented in that unit. It
runs Gunicorn with one worker and four threads on the same port as the old
viewer:

```bash
sudo cp rro-viewer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rro-viewer
```

It binds `0.0.0.0:5002`, so the LAN address remains
`http://<pi-lan-ip>:5002`. Install Tailscale on the Pi and on the Windows host,
but do not run a second Tailscale daemon inside WSL2. On the Pi, publish only
the local viewer through Tailscale Serve:

```bash
sudo tailscale up
sudo tailscale serve --bg http://127.0.0.1:5002
tailscale serve status
```

Open the private HTTPS URL shown by `tailscale serve status` in the normal
Windows browser. Tailscale ACLs/tailnet membership control who can reach it;
the viewer itself has no public listener. Raspberry Pi Connect remains useful
for acquisition commands.

Tailscale's Windows/WSL2 integration warns against installing Tailscale again
inside WSL2 because the two network stacks can conflict. Start WSL only when
you want local analysis or a manual archive download.
