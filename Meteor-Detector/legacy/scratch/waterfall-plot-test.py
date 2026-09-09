import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from matplotlib.gridspec import GridSpec

# === Signal Parameters ===
Fs = 100
L = 1000
nperseg = 128
noverlap = 64
t = np.arange(L) / Fs

# === Signal Generation ===
f_i1 = 5
f_i2 = 30
f_q1 = 15
i_component = 0.7 * np.sin(2 * np.pi * f_i1 * t) + 0.4 * np.sin(2 * np.pi * f_i2 * t) + 0.5 * np.random.randn(L)
q_component = 0.8 * np.sin(2 * np.pi * f_q1 * t) + 0.6 * np.random.randn(L)
iq_signal = i_component + 1j * q_component

# === Spectrogram ===
f, t_spec, Sxx = spectrogram(iq_signal.real, fs=Fs, window='hann',
                             nperseg=nperseg, noverlap=noverlap, scaling='density')
Sxx_dB = 10 * np.log10(Sxx + 1e-12)
power_spectrum_dB = 10 * np.log10(np.mean(Sxx, axis=1) + 1e-12)

# === Plotting Setup ===
fig = plt.figure(figsize=(12, 13))
gs = GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 0.05], hspace=0.6)

# 1. I Component
ax0 = fig.add_subplot(gs[0])
ax0.plot(t, i_component)
ax0.set_title('I Component (Time Domain)')
ax0.set_xlabel('Time (s)')
ax0.set_ylabel('Amplitude')
ax0.grid(True)

# 2. Q Component
ax1 = fig.add_subplot(gs[1])
ax1.plot(t, q_component)
ax1.set_title('Q Component (Time Domain)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

# 3. Power Spectrum
ax2 = fig.add_subplot(gs[2])
ax2.plot(f, power_spectrum_dB)
ax2.set_title('Power Spectrum (Averaged from Spectrogram)')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Power (dB)')
ax2.grid(True)
ax2.set_xlim(f[0], f[-1])
ax2.set_ylim(Sxx_dB.min(), Sxx_dB.max())

# 4. Waterfall Display
ax3 = fig.add_subplot(gs[3], sharex=ax2)
im = ax3.imshow(
    np.flipud(Sxx_dB.T),
    extent=[f[0], f[-1], t_spec[0], t_spec[-1]],
    aspect='auto',
    cmap='plasma',
    origin='lower',
    vmin=Sxx_dB.min(), vmax=Sxx_dB.max()
)
ax3.set_title('Waterfall Display (Time Upwards, Frequency →)')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Time (s)')
ax3.set_xlim(f[0], f[-1])

# 5. Colorbar below the waterfall
cax = fig.add_subplot(gs[4])
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Power (dB)')

plt.tight_layout()
plt.show()
