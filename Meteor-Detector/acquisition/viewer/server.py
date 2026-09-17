#!/usr/bin/env python3
"""
Local (and later remote) viewer for fm_observe.py Tier-1 .npz chunks.

    python3 server.py --dir ~/fm_observations
    python3 server.py --dir ~/demo_observations       # try it without hardware

Then open http://localhost:5002 in a browser.

This is deliberately a web server rather than a desktop GUI: the exact same
code, run unmodified on the Raspberry Pi, is stage 2 of the plan (browse
http://<pi-ip>:5002 from any device on the LAN), and stage 3 (viewing from
anywhere) is a tunnel/proxy placed in front of this, not a rewrite.

Downsampling follows the same rule as the rest of this project: decimate by
taking extremes over a window, never by keeping every Nth sample. Each bucket
returned to the browser carries both the min and the max of every field, so a
brief ping cannot be averaged into the noise floor the way naive subsampling
would lose it -- see dsp.py and legacy/plot_iq.py's history for why that
matters here specifically.
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)

# Set by main() from --dir. Read-only after startup.
DATA_DIR = None

_NAME_RE = re.compile(r"^(?P<station>.+)_(?P<stamp>\d{8}_\d{6})_chunk\d+\.npz$")


def _index():
    """List chunk files with the time range implied by their filename.

    Filenames alone are enough to decide which files overlap a requested
    window without opening them -- important once a directory holds weeks of
    chunks. The end time is a lower bound (start of the last frame, not the
    end of the chunk); good enough for selecting files, not for exact ranges.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.npz"))):
        name = os.path.basename(path)
        m = _NAME_RE.match(name)
        if not m:
            continue
        try:
            stamp = datetime.strptime(m["stamp"], "%Y%m%d_%H%M%S")
            stamp = stamp.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        rows.append({"path": path, "station": m["station"],
                     "start_ns": int(stamp.timestamp() * 1e9)})
    return rows


def _load_range(rows, start_ns, end_ns):
    """Concatenate every field across chunks overlapping [start_ns, end_ns].

    Inclusive at both ends deliberately: /api/summary's end_ns is the
    timestamp of the actual last recorded frame, and the "all" / "last Nh"
    buttons pass that straight through as end_ns here. An exclusive upper
    bound would silently drop that one newest sample on every such request.

    A chunk's filename only gives its start; a chunk is included whenever its
    start precedes the window's end, then trimmed by the actual t_utc_ns
    values after loading -- cheap, since a query touches at most a day or two
    of 10-minute chunks even for a "last 7 days" view.
    """
    fields = ("t_utc_ns", "power_dbfs", "noise_dbfs", "peak_dbfs", "snr_db",
              "trigger")
    parts = {f: [] for f in fields}
    meta = None
    touched = 0

    candidates = [r for r in rows if r["start_ns"] < end_ns]
    candidates.sort(key=lambda r: r["start_ns"])
    # Chunks are ~10 min; also keep the one immediately before the window in
    # case it runs into it.
    for i, r in enumerate(candidates):
        nxt = candidates[i + 1]["start_ns"] if i + 1 < len(candidates) else None
        if nxt is not None and nxt <= start_ns:
            continue
        try:
            d = np.load(r["path"])
        except Exception:
            continue
        t = d["t_utc_ns"]
        if len(t) == 0 or t[-1] < start_ns or t[0] > end_ns:
            continue
        keep = (t >= start_ns) & (t <= end_ns)
        if not np.any(keep):
            continue
        for f in fields:
            parts[f].append(d[f][keep])
        if meta is None:
            meta = {k: d[k][0] for k in
                    ("station", "center_freq_hz", "threshold_db",
                     "frame_rate_hz", "gain_db") if k in d.files}
        touched += 1

    if not parts["t_utc_ns"]:
        return None, meta, touched

    out = {f: np.concatenate(parts[f]) for f in fields}
    order = np.argsort(out["t_utc_ns"])
    for f in fields:
        out[f] = out[f][order]
    return out, meta, touched


def _bucketize(data, max_points):
    """Reduce to at most max_points buckets, each carrying min AND max.

    Below max_points samples, return them verbatim -- full resolution once
    zoomed in enough for it to matter.
    """
    t = data["t_utc_ns"]
    n = len(t)
    if n <= max_points:
        return {
            "t_utc_ns": t.tolist(),
            "power_min": data["power_dbfs"].tolist(),
            "power_max": data["power_dbfs"].tolist(),
            "noise_min": data["noise_dbfs"].tolist(),
            "noise_max": data["noise_dbfs"].tolist(),
            "snr_min": data["snr_db"].tolist(),
            "snr_max": data["snr_db"].tolist(),
            "trigger": data["trigger"].tolist(),
            "raw": True,
        }

    edges = np.linspace(0, n, max_points + 1).astype(int)
    edges = np.unique(edges)

    def mm(arr):
        lo = np.empty(len(edges) - 1)
        hi = np.empty(len(edges) - 1)
        for i in range(len(edges) - 1):
            seg = arr[edges[i]:edges[i + 1]]
            lo[i] = seg.min()
            hi[i] = seg.max()
        return lo, hi

    p_lo, p_hi = mm(data["power_dbfs"])
    nz_lo, nz_hi = mm(data["noise_dbfs"])
    s_lo, s_hi = mm(data["snr_db"])
    trig = np.array([data["trigger"][edges[i]:edges[i + 1]].any()
                     for i in range(len(edges) - 1)])
    t_bucket = t[edges[:-1]]

    return {
        "t_utc_ns": t_bucket.tolist(),
        "power_min": p_lo.tolist(), "power_max": p_hi.tolist(),
        "noise_min": nz_lo.tolist(), "noise_max": nz_hi.tolist(),
        "snr_min": s_lo.tolist(), "snr_max": s_hi.tolist(),
        "trigger": trig.tolist(),
        "raw": False,
    }


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/api/summary")
def summary():
    rows = _index()
    if not rows:
        return jsonify({"count": 0, "start_ns": None, "end_ns": None,
                        "stations": []})
    rows.sort(key=lambda r: r["start_ns"])
    stations = sorted(set(r["station"] for r in rows))
    # Peek the last chunk for its true end time (start + frame count / rate).
    last = np.load(rows[-1]["path"])
    end_ns = int(last["t_utc_ns"][-1]) if len(last["t_utc_ns"]) else rows[-1]["start_ns"]
    return jsonify({"count": len(rows), "start_ns": rows[0]["start_ns"],
                    "end_ns": end_ns, "stations": stations,
                    "dir": DATA_DIR})


@app.route("/api/data")
def data():
    try:
        start_ns = int(request.args["start_ns"])
        end_ns = int(request.args["end_ns"])
    except (KeyError, ValueError):
        return jsonify({"error": "start_ns and end_ns (int, UTC ns) required"}), 400
    max_points = int(request.args.get("max_points", 2000))
    max_points = max(50, min(max_points, 20000))

    rows = _index()
    if not rows:
        return jsonify({"error": f"no chunk files found in {DATA_DIR}"}), 404

    merged, meta, touched = _load_range(rows, start_ns, end_ns)
    if merged is None:
        return jsonify({"error": "no data in that range", "files_checked": touched}), 404

    bucketed = _bucketize(merged, max_points)
    bucketed["meta"] = {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in (meta or {}).items()}
    bucketed["files_used"] = touched
    bucketed["raw_points"] = int(len(merged["t_utc_ns"]))
    return jsonify(bucketed)


def main():
    global DATA_DIR
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="./fm_observations",
                   help="directory of .npz chunks from fm_observe.py")
    p.add_argument("--port", type=int, default=5002)
    p.add_argument("--host", default="127.0.0.1",
                   help="use 0.0.0.0 to allow other devices on the LAN "
                        "(stage 2: run this on the Pi)")
    args = p.parse_args()

    DATA_DIR = os.path.abspath(os.path.expanduser(args.dir))
    if not os.path.isdir(DATA_DIR):
        print(f"warning: {DATA_DIR} does not exist yet -- it will show as "
              f"empty until fm_observe.py (or demo_data.py) writes into it")

    print(f"serving {DATA_DIR}")
    print(f"open http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    sys.exit(main())
