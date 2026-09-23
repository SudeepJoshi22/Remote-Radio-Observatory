#!/usr/bin/env python3
"""Regression test for atomic Tier-1 chunk publication."""

import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fm_observe import ChunkWriter


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
        print("ATOMIC CHUNK WRITER PASSED")
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    main()
