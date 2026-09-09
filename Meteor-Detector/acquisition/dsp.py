"""
Shared DSP primitives for the Remote Radio Observatory.

Both the diagnostic tool (rf_check.py) and the recorder (fm_observe.py) import
from here. That is deliberate: in the previous generation of this code the
validation scripts used a Hann window while the recorder used none, so what was
checked on the bench was never what got written to disk.

Reference level
---------------
An RTL-SDR has no absolute power calibration. Everything here is dBFS, where
0 dBFS is a full-scale complex tone (|z| == 1). Do not report these numbers as
dBm; the only physically meaningful quantity is SNR against a tracked floor.
"""

import numpy as np

# librtlsdr hands back unsigned 8-bit interleaved I,Q with zero at 127.5.
_U8_ZERO = 127.5
_U8_SCALE = 127.5


def bytes_to_complex(raw):
    """Convert a raw librtlsdr uint8 byte stream to complex64 in [-1, 1].

    This is the conversion pyrtlsdr's read_samples() applies internally. Use it
    when reading .iq files written by rtl_sdr, or bytes from read_bytes_async.

    Reading rtl_sdr output as int8 (a mistake in the old legacy/plot_iq.py) wraps every
    sample above 127 to negative and leaves the 127.5 DC offset in place.
    """
    u8 = np.frombuffer(raw, dtype=np.uint8)
    if u8.size % 2:
        u8 = u8[:-1]
    iq = np.empty(u8.size // 2, dtype=np.complex64)
    iq.real = (u8[0::2].astype(np.float32) - _U8_ZERO) / _U8_SCALE
    iq.imag = (u8[1::2].astype(np.float32) - _U8_ZERO) / _U8_SCALE
    return iq


def make_window(nfft, kind="hann"):
    """Analysis window plus its noise-power normalisation factor sum(w**2).

    Windowing is not cosmetic here. With a rectangular window a strong station
    a few hundred kHz away leaks into the quiet channel at about -73 dBc; with
    Hann that drops to about -114 dBc. That 41 dB is the difference between
    seeing a meteor ping and seeing the neighbouring transmitter.
    """
    if kind == "hann":
        w = np.hanning(nfft)
    elif kind == "blackmanharris":
        n = np.arange(nfft)
        a = (0.35875, 0.48829, 0.14128, 0.01168)
        w = (a[0] - a[1] * np.cos(2 * np.pi * n / (nfft - 1))
             + a[2] * np.cos(4 * np.pi * n / (nfft - 1))
             - a[3] * np.cos(6 * np.pi * n / (nfft - 1)))
    elif kind == "rect":
        w = np.ones(nfft)
    else:
        raise ValueError(f"unknown window: {kind}")
    return w.astype(np.float32), float(np.sum(w.astype(np.float64) ** 2))


def psd(samples, fs, window, win_sumsq):
    """Two-sided power spectral density estimate, linear, fftshifted.

        S(f_k) = |FFT(x * w)_k|**2 / (fs * sum(w**2))

    The sum(w**2) term is what makes the result independent of both the window
    and the FFT length. The old process_fft_power() divided by (fs / nfft)
    instead, which is the reciprocal of the N normalisation: it inflated every
    reading by N**2 (+60.2 dB at nfft=1024) and made the number scale with
    --fft-size, so no two runs at different settings were comparable.
    """
    nfft = window.size
    x = samples[:nfft] * window
    spec = np.fft.fftshift(np.fft.fft(x, n=nfft))
    return (np.abs(spec).astype(np.float64) ** 2) / (fs * win_sumsq)


def freq_axis(nfft, fs, center_hz=0.0):
    """Absolute frequency for each fftshifted bin."""
    return np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs)) + center_hz


def band_mask(freqs, f_lo, f_hi):
    """Boolean mask selecting bins in [f_lo, f_hi]."""
    return (freqs >= f_lo) & (freqs <= f_hi)


def band_power(psd_lin, mask, fs, nfft):
    """Total power in a band, linear full-scale units.

    Integrates the density over the band: sum(S_k) * df, df = fs / nfft.
    A full-scale complex tone inside the band integrates to 1.0 (0 dBFS)
    regardless of window choice or FFT length -- verified by
    `rf_check.py --selftest`.
    """
    return float(np.sum(psd_lin[mask]) * (fs / nfft))


def to_db(x, floor=1e-30):
    """Linear power to dB, guarded against log(0)."""
    return 10.0 * np.log10(np.maximum(x, floor))


class ChannelMetrics:
    """Per-frame power measurement for one channel plus its guard bands.

    The channel width matters more than anything else in this file. An FM
    broadcast signal occupies about 180 kHz and is constant-envelope: the
    transmitter's total power is steady, but it is spread across the deviation
    and the centre bin is only occupied during near-silence in the audio.
    Integrating a narrow slice (the old default worked out to 0.94 kHz) measures
    the station's *program material*, not its presence.
    """

    def __init__(self, fs, nfft, center_hz, channel_bw=180e3,
                 guard_lo=150e3, guard_hi=400e3, window="hann"):
        self.fs = float(fs)
        self.nfft = int(nfft)
        self.center_hz = float(center_hz)
        self.window, self.win_sumsq = make_window(self.nfft, window)

        # Baseband frequency offsets from the tuned centre.
        f = freq_axis(self.nfft, self.fs, 0.0)
        self.freqs = f

        half = channel_bw / 2.0
        self.chan_mask = band_mask(f, -half, half)

        # Guard bands sit outside the channel skirts but inside Nyquist. They
        # give an instantaneous noise reference measured through the same
        # front-end, gain setting and window as the channel itself.
        nyq = self.fs / 2.0
        g_hi = min(guard_hi, nyq * 0.98)
        if g_hi <= guard_lo:
            raise ValueError(
                f"sample rate {fs/1e6:.3f} MS/s is too low for guard bands at "
                f"{guard_lo/1e3:.0f}-{guard_hi/1e3:.0f} kHz; raise --sample-rate")
        self.guard_mask = (band_mask(f, -g_hi, -guard_lo)
                           | band_mask(f, guard_lo, g_hi))

        self.chan_bins = int(np.count_nonzero(self.chan_mask))
        self.guard_bins = int(np.count_nonzero(self.guard_mask))
        if self.chan_bins == 0 or self.guard_bins == 0:
            raise ValueError("channel or guard band selects zero bins")

        # Scale guard power to an equivalent noise power in the channel
        # bandwidth, so SNR compares like with like.
        self._guard_to_chan = self.chan_bins / self.guard_bins

    def describe(self):
        df = self.fs / self.nfft
        return (f"fs={self.fs/1e6:.3f} MS/s  nfft={self.nfft}  "
                f"bin={df:.1f} Hz  frame={self.fs/self.nfft:.1f} Hz\n"
                f"channel: {self.chan_bins} bins "
                f"({self.chan_bins*df/1e3:.1f} kHz)   "
                f"guard: {self.guard_bins} bins "
                f"({self.guard_bins*df/1e3:.1f} kHz)")

    def measure(self, samples):
        """One frame -> (power_dbfs, noise_dbfs, peak_dbfs).

        power  total power in the channel
        noise  guard-band power scaled to the channel bandwidth
        peak   strongest single bin anywhere in the frame, for overload checks
        """
        s = psd(samples, self.fs, self.window, self.win_sumsq)
        df = self.fs / self.nfft
        chan = float(np.sum(s[self.chan_mask]) * df)
        guard = float(np.sum(s[self.guard_mask]) * df) * self._guard_to_chan
        peak = float(np.max(s) * df)
        return to_db(chan), to_db(guard), to_db(peak)


class RollingFloor:
    """Rolling percentile noise floor.

    A meteor ping is a brief excursion, so a percentile over a window many times
    longer than a ping tracks slow drift (temperature, ionosphere, the station's
    own fading) while being almost unmoved by the events you want to keep. The
    10th percentile over 60 s is a good starting point; raise the window if the
    site has slow interference, lower it if gain drifts quickly.
    """

    def __init__(self, frame_rate, seconds=60.0, percentile=10.0,
                 update_hz=2.0):
        self.n = max(16, int(frame_rate * seconds))
        self.percentile = percentile
        self._buf = np.zeros(self.n, dtype=np.float32)
        self._count = 0
        self._idx = 0
        self._value = None
        self._every = max(1, int(frame_rate / update_hz))
        self._since = 0

    def push(self, x):
        self._buf[self._idx] = x
        self._idx = (self._idx + 1) % self.n
        self._count = min(self._count + 1, self.n)
        self._since += 1
        if self._value is None or self._since >= self._every:
            self._since = 0
            self._value = float(np.percentile(self._buf[:self._count],
                                              self.percentile))
        return self._value

    @property
    def value(self):
        return self._value

    @property
    def warm(self):
        """True once the window is full enough for the percentile to mean much."""
        return self._count >= self.n
