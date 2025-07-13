import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, decimate
from matplotlib.gridspec import GridSpec
from rtlsdr import RtlSdr
import argparse

# === Command-line arguments ===
parser = argparse.ArgumentParser(description="SDR IQ Snapshot and Spectrogram")
parser.add_argument('--freq', type=float, default=100e6, help='Center frequency in Hz (default: 100 MHz)')
parser.add_argument('--samp_rate', type=float, default=2.4e6, help='Sample rate in Hz (default: 2.4 MSPS)')
parser.add_argument('--gain', type=float, default=20, help='Gain (default: 20 dB)')
parser.add_argument('--duration', type=int, default=60, help='Capture duration in seconds (default: 60s)')
parser.add_argument('--decim', type=int, default=10, help='Decimation factor (default: 10)')
args = parser.parse_args()

# === SDR Setup ===
print("[*] Initializing SDR...")
sdr = RtlSdr()
sdr.sample_rate = args.samp_rate
sdr.center_freq = args.freq
sdr.gain = args.gain

Fs = sdr.sample_rate / args.decim  # effective sample rate after decimation
chunk_duration = 1  # second
chunk_samples = int(sdr.sample_rate * chunk_duration)
num_chunks = int(args.duration / chunk_duration)

# === Safe IQ capture with decimation ===
print(f"[*] Capturing {args.duration}s of samples in {num_chunks} chunks with decimation factor {args.decim}...")
iq_chunks = []

for i in range(num_chunks):
    print(f"    Capturing chunk {i+1}/{num_chunks}...")
    chunk = sdr.read_samples(chunk_samples)
    chunk_decimated = decimate(chunk.real, args.decim) + 1j * decimate(chunk.imag, args.decim)
    iq_chunks.append(chunk_decimated)

sdr.close()
iq_signal = np.concatenate(iq_chunks)
print("[*] Capture and decimation complete.")

# === Signal Parameters ===
L = len(iq_signal)
t = np.arange(L) / Fs
nperseg = 1024
noverlap = nperseg // 2

# === Spectrogram ===
f, t_spec, Sxx = spectrogram(iq_signal, fs=Fs, window='hann',
                             nperseg=nperseg, noverlap=noverlap, scaling='density')
Sxx_dB = 10 * np.log10(Sxx + 1e-12)
power_spectrum_dB = 10 * np.log10(np.mean(Sxx, axis=1) + 1e-12)

# === Plotting Setup ===
fig = plt.figure(figsize=(14, 14))
gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 0.05], hspace=0.6)

# 1. Magnitude vs Time
ax0 = fig.add_subplot(gs[0])
ax0.plot(t, np.abs(iq_signal))
ax0.set_title('Signal Magnitude (Time Domain)')
ax0.set_xlabel('Time (s)')
ax0.set_ylabel('Magnitude')
ax0.grid(True)

# 2. Power Spectrum
ax1 = fig.add_subplot(gs[1])
ax1.plot(f, power_spectrum_dB)
ax1.set_title('Power Spectrum (Averaged from Spectrogram)')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Power (dB)')
ax1.grid(True)
ax1.set_xlim(f[0], f[-1])
ax1.set_ylim(Sxx_dB.min(), Sxx_dB.max())

# 3. Waterfall Display
ax2 = fig.add_subplot(gs[2], sharex=ax1)
im = ax2.imshow(
    np.flipud(Sxx_dB.T),
    extent=[f[0], f[-1], t_spec[0], t_spec[-1]],
    aspect='auto',
    cmap='plasma',
    origin='lower',
    vmin=Sxx_dB.min(), vmax=Sxx_dB.max()
)
ax2.set_title('Waterfall Display (Time Upwards, Frequency →)')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Time (s)')
ax2.set_xlim(f[0], f[-1])

# 4. Colorbar below
cax = fig.add_subplot(gs[3])
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Power (dB)')

plt.tight_layout()
plt.show()
