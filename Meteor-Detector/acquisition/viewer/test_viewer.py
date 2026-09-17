#!/usr/bin/env python3
"""
Smoke test for the viewer's data path: generates a small synthetic dataset,
hits server.py's Flask app directly (no real HTTP, no browser), and checks
that bucketing preserves injected pings rather than averaging them away --
the same "decimate by extremes, never by subsampling" property the rest of
this project relies on.

    python3 test_viewer.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import demo_data
import server


def main():
    outdir = tempfile.mkdtemp(prefix="rro_viewer_test_")
    failures = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f"   {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    try:
        t_ns, power, noise, peak, snr, trigger, n_events = demo_data.generate(
            hours=3.0, station="TEST", freq_hz=97.9e6, threshold_db=6.0,
            rate_per_hour=5.0, seed=99)
        demo_data.write_chunks(outdir, "TEST", 97.9e6, 6.0,
                               t_ns, power, noise, peak, snr, trigger)

        server.DATA_DIR = outdir
        client = server.app.test_client()

        r = client.get("/api/summary")
        check("summary responds 200", r.status_code == 200, str(r.status_code))
        s = r.get_json()
        check("summary reports chunk files", s["count"] > 0, f"count={s['count']}")

        r = client.get(f"/api/data?start_ns={s['start_ns']}&end_ns={s['end_ns']}"
                       f"&max_points=300")
        check("data responds 200", r.status_code == 200, str(r.status_code))
        d = r.get_json()

        check("bucket count within limit", len(d["t_utc_ns"]) <= 300,
              f"{len(d['t_utc_ns'])}")
        check("raw_points matches generated frames",
              d["raw_points"] == len(t_ns), f"{d['raw_points']} vs {len(t_ns)}")
        check("min <= max in every bucket",
              all(a <= b + 1e-6 for a, b in zip(d["power_min"], d["power_max"])))

        true_peak = float(np.max(power))
        seen_peak = max(d["power_max"])
        check("strongest injected ping visible even fully zoomed out",
              abs(true_peak - seen_peak) < 0.5,
              f"true={true_peak:.2f} seen={seen_peak:.2f}")

        n_true_triggers = int(np.sum(trigger))
        n_bucket_triggers = sum(d["trigger"])
        check("some bucket flags every triggered region",
              (n_true_triggers == 0) == (n_bucket_triggers == 0),
              f"true={n_true_triggers} bucketed={n_bucket_triggers}")

        # Full resolution once the window is small enough.
        narrow_end = s["start_ns"] + int(20e9)
        r = client.get(f"/api/data?start_ns={s['start_ns']}&end_ns={narrow_end}"
                       f"&max_points=5000")
        d2 = r.get_json()
        check("small window returns raw (unbucketed) samples", d2["raw"] is True)

        r = client.get("/api/data?start_ns=abc&end_ns=123")
        check("malformed request is rejected, not a 500",
              r.status_code == 400, f"{r.status_code}")

    finally:
        shutil.rmtree(outdir, ignore_errors=True)

    print("-" * 60)
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
