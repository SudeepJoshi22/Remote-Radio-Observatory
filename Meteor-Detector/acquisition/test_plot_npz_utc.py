#!/usr/bin/env python3
"""Headless checks for the local full-resolution NPZ range loader."""

import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plot_npz_utc import index_recording_files, load_npz_range, recording_bounds


def main():
    outdir = tempfile.mkdtemp(prefix="rro_local_viewer_test_")
    try:
        t0 = 1_700_000_000_000_000_000
        times = t0 + np.arange(20, dtype=np.int64) * 8_000_000
        stamp = datetime.fromtimestamp(t0 / 1e9, tz=timezone.utc)
        name = f"TEST_{stamp.strftime('%Y%m%d_%H%M%S')}_chunk0000.npz"
        np.savez_compressed(
            os.path.join(outdir, name),
            t_utc_ns=times,
            power_dbfs=np.arange(20, dtype=np.float32),
            noise_dbfs=np.zeros(20, dtype=np.float32),
            peak_dbfs=np.ones(20, dtype=np.float32),
            snr_db=np.arange(20, dtype=np.float32),
            trigger=np.zeros(20, dtype=bool),
            station=np.asarray(["TEST"]),
            center_freq_hz=np.asarray([97.9e6]),
            threshold_db=np.asarray([6.0]),
        )
        rows = index_recording_files(outdir)
        assert len(rows) == 1
        assert recording_bounds(rows) == (int(times[0]), int(times[-1]))
        selected = load_npz_range(rows, int(times[5]), int(times[12]))
        assert selected is not None
        assert selected["t_utc_ns"][0] == times[5]
        assert selected["t_utc_ns"][-1] == times[12]
        assert len(selected["t_utc_ns"]) == 8
        assert selected["files_used"] == [name]
        print("LOCAL FULL-RESOLUTION VIEWER LOADER PASSED")
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    main()
