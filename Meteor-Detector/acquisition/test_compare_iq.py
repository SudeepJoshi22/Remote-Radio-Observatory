#!/usr/bin/env python3
"""Numerical/streaming checks; these do not validate meteor classification."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import compare_iq as comparison
import dsp


class EstimatorTests(unittest.TestCase):
    def test_meteor_selects_reference_at_wider_band_peak(self):
        # The strongest reference-band point is at a DIFFERENT time from
        # the detection peak. Catch accidentally taking both from one row.
        power = np.ones((3, 7))
        power[0, 3] = 100
        power[1, :] = 2
        power[1, 0] = 200
        detected = np.array([False, False, True, True, True, False, False])
        row, peak, noise, score, ratio, passed = comparison.meteor_block(
            power, detected, np.ones(7, bool))
        self.assertEqual(row, 0)
        self.assertEqual(peak, 100)
        self.assertEqual(noise, 2)
        self.assertAlmostEqual(score, 10 * np.log10(50))
        self.assertEqual(ratio, 2)
        self.assertTrue(passed)

    def test_broadband_veto_is_separate(self):
        power = np.ones((8, 9))
        power[3] = 10
        power[3, 4] = 1000
        detected = np.arange(9) == 4
        _, _, _, score, ratio, passed = comparison.meteor_block(power, detected, np.ones(9, bool))
        self.assertAlmostEqual(score, 20)
        self.assertEqual(ratio, 10)
        self.assertFalse(passed)

    def test_zero_spectrum_has_no_defined_meteor_score(self):
        result = comparison.meteor_block(np.zeros((3, 5)), np.ones(5, bool), np.ones(5, bool))
        self.assertTrue(np.isnan(result[3]))
        self.assertFalse(result[-1])

    def test_echo_previous_reference_and_source_weighting(self):
        state = comparison.EchoReference(2)
        self.assertTrue(np.isnan(state.push(-10)))
        self.assertTrue(np.isnan(state.push(-4)))
        self.assertAlmostEqual(state.push(2), (-10 - 4 - 4) / 3)
        self.assertAlmostEqual(state.push(9), (-4 + 2 + 2) / 3)


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "test.cu8"
        self.fs = 1024000
        rng = np.random.default_rng(340)
        frames = 200
        n = frames * 8192
        t = np.arange(n) / self.fs
        samples = .008 * (rng.normal(size=n) + 1j * rng.normal(size=n))
        # Narrow tone appears in-frame; broadband disturbance spans several
        # frames. They are synthetic RF features, not labelled real meteors.
        samples[90 * 8192:94 * 8192] += .15 * np.exp(2j * np.pi * 20000 * t[90 * 8192:94 * 8192])
        samples[140 * 8192:144 * 8192] *= 5
        interleaved = np.empty(2 * n, dtype=np.uint8)
        interleaved[0::2] = np.clip(np.rint(samples.real * 127.5 + 127.5), 0, 255)
        interleaved[1::2] = np.clip(np.rint(samples.imag * 127.5 + 127.5), 0, 255)
        interleaved.tofile(self.raw)

    def tearDown(self):
        self.temp.cleanup()

    def args(self, output, *extra):
        return comparison.parser().parse_args([
            str(self.raw), "--sample-rate", str(self.fs), "--center-freq", "103300000",
            "--floor-seconds", "0.12", "--echo-scans", "4",
            "--output-dir", str(self.root / output), *extra])

    def run_analysis(self, args, plots=False):
        with contextlib.redirect_stdout(io.StringIO()):
            if plots:
                info = comparison.analyse(args)
            else:
                with patch.object(comparison, "make_plots"):
                    info = comparison.analyse(args)
        return np.load(args.output_dir / "metrics.npy"), info

    def test_ours_replays_existing_dsp(self):
        a = self.args("replay")
        result, _ = self.run_analysis(a)
        metric = dsp.ChannelMetrics(self.fs, 8192, 103300000)
        floor = dsp.RollingFloor(125, seconds=.12)
        samples = dsp.bytes_to_complex(self.raw.read_bytes()).reshape(-1, 8192)
        for i, frame in enumerate(samples):
            power, noise, _ = metric.measure(frame)
            temporal = floor.push(power)
            np.testing.assert_allclose(result[i, [1, 2, 3, 4, 5]],
                                       [power, noise, temporal, max(noise, temporal),
                                        power - max(noise, temporal)], atol=2e-6)

    def test_read_block_size_does_not_change_ours_or_echo(self):
        first, _ = self.run_analysis(self.args("first", "--block-frames", "13"))
        second, _ = self.run_analysis(self.args("second", "--block-frames", "64"))
        np.testing.assert_allclose(first[:, :7], second[:, :7], atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(first[:, 12:], second[:, 12:], atol=1e-10, equal_nan=True)
        # MeteorRadio is intentionally a block statistic (including partial tail).
        self.assertEqual(np.isfinite(first[:, 9]).sum(), 16)
        self.assertEqual(np.isfinite(second[:, 9]).sum(), 4)

    def test_echo_normalization_and_noise_are_independent_of_absolute_gain(self):
        a = self.args("echo")
        result, _ = self.run_analysis(a)
        samples = dsp.bytes_to_complex(self.raw.read_bytes())[:8192]
        window, norm = dsp.make_window(8192)
        p = dsp.psd(samples, self.fs, window, norm) * self.fs / 8192
        level = 10 * np.log10(p / p.mean())
        freq = dsp.freq_axis(8192, self.fs)
        mask = (freq > -90000) & (freq <= 90000)
        self.assertAlmostEqual(result[0, 12], level[mask].max(), places=6)
        np.testing.assert_allclose(10 * np.log10(37 * p / (37 * p).mean()), level, atol=1e-10)

    def test_offsets_partial_tail_and_plot_outputs(self):
        result, info = self.run_analysis(self.args("plots", "--start", ".08", "--duration", ".333",
                                                  "--start-utc", "2026-09-24T02:00:00Z"), plots=True)
        self.assertEqual(len(result), 41)
        self.assertAlmostEqual(result[0, 0], .08)
        self.assertGreater(info["trailing_samples_not_analysed"], 0)
        for name in ("comparison.png", "strongest_excursion.png", "summary.json", "strongest_frame_spectrum.npz"):
            self.assertGreater((self.root / "plots" / name).stat().st_size, 100)

    def test_bad_input_and_protection(self):
        for extra in (("--duration", "-1"), ("--preset", "beacon"), ("--guard-hi", "700000"),
                      ("--start-utc", "2026-09-24T00:00:00")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                comparison.analyse(self.args("bad", *extra))
        self.raw.write_bytes(b"123")
        with self.assertRaisesRegex(ValueError, "odd file size"):
            comparison.analyse(self.args("odd"))


if __name__ == "__main__":
    unittest.main()
