#!/usr/bin/env python3
"""Regression test for atomic Tier-1 chunk publication."""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fm_observe import ChunkWriter, ContinuousIQWriter


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
        iq = ContinuousIQWriter(events, duration_s=2.5, chunk_seconds=1,
                                frame_rate=10, block_bytes=4,
                                sample_rate_hz=10, min_free_mb=1,
                                meta={"station": "TEST"})
        block = b"\x80\x81\x82\x83"
        for i in range(25):
            iq.push(block, 1_700_000_000_000_000_000 + i * 100_000_000)
        iq.close(1_700_000_000_250_000_000)

        chunks = sorted(name for name in os.listdir(events)
                        if name.startswith("continuous_") and name.endswith(".iq"))
        assert len(chunks) == 3
        assert iq.written == len(chunks)
        assert not any(name.endswith(".tmp") for name in os.listdir(events))
        assert sum(os.path.getsize(os.path.join(events, name)) for name in chunks) == 100
        for name in chunks:
            assert os.path.getsize(os.path.join(events, name)) > 0
            meta_name = name[:-3] + ".json"
            with open(os.path.join(events, meta_name)) as f:
                assert json.load(f)["kind"] == "continuous_iq"
        assert not any(name.startswith("event_") for name in os.listdir(events))
        print("ATOMIC CHUNK WRITERS PASSED")
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    main()
