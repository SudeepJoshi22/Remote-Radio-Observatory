#!/usr/bin/env python3
"""Regression test for atomic Tier-1 chunk publication."""

import os
import json
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fm_observe import ChunkWriter, IQRing


def main():
    outdir = tempfile.mkdtemp(prefix="rro_chunk_writer_test_")
    try:
        writer = ChunkWriter(outdir, "TEST", {"station": "TEST"},
                             chunk_seconds=1, frame_rate=2)
        writer.add(1_700_000_000_000_000_000, -80, -90, -70, 10, True)
        writer.add(1_700_000_000_500_000_000, -81, -90, -71, 9, False)
        result = writer.flush()
        assert result is not None
        path, count = result
        assert count == 2
        assert path.endswith(".npz") and os.path.isfile(path)
        assert not os.path.exists(path + ".tmp")
        with np.load(path) as data:
            assert len(data["t_utc_ns"]) == 2
            assert bool(data["trigger"][0])

        events = os.path.join(outdir, "events")
        ring = IQRing(events, pre_s=1, post_s=1, frame_rate=10,
                      block_bytes=4, max_events_per_hour=60, min_free_mb=1,
                      periodic_interval_s=1, periodic_seconds=0.2,
                      meta={"station": "TEST"})
        block = b"\x80\x81\x82\x83"
        for i in range(25):
            ring.push(block, 1_700_000_000_000_000_000 + i * 100_000_000,
                      0.0, False, False)
        ring.abort()

        snapshots = sorted(name for name in os.listdir(events)
                           if name.startswith("sample_") and name.endswith(".iq"))
        assert len(snapshots) >= 2
        assert ring.periodic_written == len(snapshots)
        assert not any(name.endswith(".tmp") for name in os.listdir(events))
        for name in snapshots:
            assert os.path.getsize(os.path.join(events, name)) > 0
            meta_name = name[:-3] + ".json"
            with open(os.path.join(events, meta_name)) as f:
                assert json.load(f)["kind"] == "periodic_snapshot"
        assert not any(name.startswith("event_") for name in os.listdir(events))
        print("ATOMIC CHUNK WRITER PASSED")
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    main()
