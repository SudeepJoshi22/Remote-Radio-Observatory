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
import tempfile
import zipfile
from datetime import datetime, timezone

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)

# Set by main() or wsgi.py from --dir/RRO_DATA_DIR. Read-only after startup.
# Having a useful import-time default is important when Gunicorn imports the
# WSGI module instead of calling main().
DATA_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get("RRO_DATA_DIR", "./fm_observations")))

_NAME_RE = re.compile(r"^(?P<station>.+)_(?P<stamp>\d{8}_\d{6})_chunk\d+\.npz$")
_EVENT_RE = re.compile(r"^event_(?P<stamp>\d{8}_\d{6}_\d{3})\.(?P<kind>iq|json)$")


def _parse_stamp(value, fmt):
    try:
        return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _event_index():
    """Return only complete, known event sidecars in the data directory."""
    rows = {}
    event_dirs = (DATA_DIR, os.path.join(DATA_DIR, "events"))
    paths = []
    for directory in event_dirs:
        paths.extend(glob.glob(os.path.join(directory, "event_*.iq")))
        paths.extend(glob.glob(os.path.join(directory, "event_*.json")))
    for path in paths:
        name = os.path.basename(path)
        m = _EVENT_RE.match(name)
        if not m or not os.path.isfile(path) or os.path.islink(path):
            continue
        stamp = _parse_stamp(m["stamp"], "%Y%m%d_%H%M%S_%f")
        if stamp is None:
            continue
        base = name.rsplit(".", 1)[0]
        row = rows.setdefault(base, {"base": base, "start_ns": int(stamp.timestamp() * 1e9),
                                     "date": stamp.isoformat(), "iq": None, "json": None})
        row[m["kind"]] = {
            "name": name,
            "path": path,
            "size": os.path.getsize(path),
            "download_url": "/download/" + name,
        }
    return sorted(rows.values(), key=lambda r: r["start_ns"], reverse=True)


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
        if not m or not os.path.isfile(path) or os.path.islink(path):
            continue
        try:
            stamp = datetime.strptime(m["stamp"], "%Y%m%d_%H%M%S")
            stamp = stamp.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        rows.append({"path": path, "station": m["station"],
                     "start_ns": int(stamp.timestamp() * 1e9)})
    return rows


def _public_chunk(row):
    """Convert an internal chunk row into the stable API/file-list shape."""
    path = row["path"]
    name = os.path.basename(path)
    return {
        "name": name,
        "type": "npz",
        "station": row["station"],
        "start_ns": row["start_ns"],
        "date": datetime.fromtimestamp(row["start_ns"] / 1e9,
                                        tz=timezone.utc).isoformat(),
        "size": os.path.getsize(path),
        "download_url": "/download/" + name,
    }


def _chunks_for_range(rows, start_ns, end_ns):
    """Return indexed chunks containing at least one sample in a time range."""
    selected = []
    candidates = sorted((r for r in rows if r["start_ns"] < end_ns),
                       key=lambda r: r["start_ns"])
    for i, row in enumerate(candidates):
        next_start = (candidates[i + 1]["start_ns"]
                      if i + 1 < len(candidates) else None)
        if next_start is not None and next_start <= start_ns:
            continue
        try:
            with np.load(row["path"]) as chunk:
                t = chunk["t_utc_ns"]
                if len(t) and t[-1] >= start_ns and t[0] <= end_ns:
                    selected.append(row)
        except Exception:
            # A bad/incomplete file is not downloadable or plot data.
            continue
    return selected


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

    candidates = sorted((r for r in rows if r["start_ns"] < end_ns),
                        key=lambda r: r["start_ns"])
    # Chunks are ~10 min; also keep the one immediately before the window in
    # case it runs into it.
    for i, r in enumerate(candidates):
        nxt = candidates[i + 1]["start_ns"] if i + 1 < len(candidates) else None
        if nxt is not None and nxt <= start_ns:
            continue
        try:
            # Closing the NpzFile matters on long-running Gunicorn workers:
            # otherwise every zoom can leave a descriptor open until GC.
            d = np.load(r["path"])
            with d:
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
        except Exception:
            continue

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
    try:
        with np.load(rows[-1]["path"]) as last:
            end_ns = (int(last["t_utc_ns"][-1])
                      if len(last["t_utc_ns"]) else rows[-1]["start_ns"])
    except Exception:
        end_ns = rows[-1]["start_ns"]
    return jsonify({"count": len(rows), "start_ns": rows[0]["start_ns"],
                    "end_ns": end_ns, "stations": stations,
                    "dir": DATA_DIR})


@app.route("/api/files")
def files():
    """List downloadable completed chunks and their matching IQ events.

    The allowlist is built from the same strict filename indexes used by the
    download route. Temporary recorder files and arbitrary paths never appear.
    """
    try:
        limit = max(1, min(int(request.args.get("limit", 500)), 5000))
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        return jsonify({"error": "limit and offset must be integers"}), 400

    chunks = [_public_chunk(r) for r in
              sorted(_index(), key=lambda r: r["start_ns"], reverse=True)]
    events = _event_index()
    all_files = chunks + [
        {"name": item[kind]["name"], "type": kind,
         "start_ns": item["start_ns"], "date": item["date"],
         "size": item[kind]["size"],
         "download_url": item[kind]["download_url"],
         "event_base": item["base"]}
        for item in events for kind in ("iq", "json") if item[kind]
    ]
    all_files.sort(key=lambda row: row["start_ns"], reverse=True)
    return jsonify({"files": all_files[offset:offset + limit],
                    "total": len(all_files), "offset": offset,
                    "limit": limit})


@app.route("/download/<path:filename>")
def download(filename):
    """Download a file only when its exact basename is in a recording index."""
    if filename != os.path.basename(filename):
        return jsonify({"error": "only indexed recording filenames are allowed"}), 404

    allowed = {os.path.basename(row["path"]) for row in _index()}
    event_paths = {item[kind]["name"]: item[kind]["path"]
                   for item in _event_index()
                   for kind in ("iq", "json") if item[kind]}
    allowed.update(event_paths)
    if filename not in allowed:
        return jsonify({"error": "recording is not indexed or is incomplete"}), 404
    directory = os.path.dirname(event_paths[filename]) if filename in event_paths else DATA_DIR
    return send_from_directory(directory, filename, as_attachment=True,
                               conditional=True)


@app.route("/download-range")
def download_range():
    """ZIP all completed NPZ chunks overlapping the selected plot window."""
    try:
        start_ns = int(request.args["start_ns"])
        end_ns = int(request.args["end_ns"])
    except (KeyError, ValueError):
        return jsonify({"error": "start_ns and end_ns (int, UTC ns) required"}), 400
    if end_ns <= start_ns:
        return jsonify({"error": "end_ns must be greater than start_ns"}), 400

    rows = _chunks_for_range(_index(), start_ns, end_ns)
    if not rows:
        return jsonify({"error": "no completed NPZ chunks in that range"}), 404

    fd, archive_path = tempfile.mkstemp(prefix="rro-npz-", suffix=".zip")
    try:
        with os.fdopen(fd, "wb") as output:
            with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED,
                                 allowZip64=True) as archive:
                for row in rows:
                    archive.write(row["path"], arcname=os.path.basename(row["path"]))
    except Exception:
        try:
            os.unlink(archive_path)
        except FileNotFoundError:
            pass
        raise

    start_text = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).strftime("%Y%m%d_%H%M")
    end_text = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).strftime("%Y%m%d_%H%M")
    response = send_file(archive_path, as_attachment=True,
                         download_name=f"rro_npz_{start_text}_{end_text}.zip",
                         mimetype="application/zip", conditional=True)
    response.call_on_close(lambda: os.unlink(archive_path)
                           if os.path.exists(archive_path) else None)
    return response


@app.route("/api/data")
def data():
    try:
        start_ns = int(request.args["start_ns"])
        end_ns = int(request.args["end_ns"])
    except (KeyError, ValueError):
        return jsonify({"error": "start_ns and end_ns (int, UTC ns) required"}), 400
    try:
        max_points = int(request.args.get("max_points", 2000))
    except ValueError:
        return jsonify({"error": "max_points must be an integer"}), 400
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
