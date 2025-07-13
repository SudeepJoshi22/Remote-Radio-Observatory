import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
import os

# === Configuration ===
filename = 'output2.iq'    # Your SDR IQ file
sample_rate = 2.56e6       # Hz
dtype = np.int8            # IQ format: int8
window_duration = 2.0      # Seconds per viewing window
decimate = 1000            # Downsample factor for display

# === Compute total duration from file size ===
file_size_bytes = os.path.getsize(filename)
total_samples = file_size_bytes // 2  # 2 bytes per IQ sample
total_duration = total_samples / sample_rate

# === Max-hold buffer
max_hold_trace = None

# === Function to read one time window of data ===
def read_iq_window(start_time):
    num_complex_samples = int(window_duration * sample_rate)
    num_raw_samples = num_complex_samples * 2
    offset_bytes = int(start_time * sample_rate * 2)

    with open(filename, 'rb') as f:
        f.seek(offset_bytes)
        raw = np.frombuffer(f.read(num_raw_samples), dtype=dtype)

    if len(raw) < 2:
        return np.array([]), np.array([]), np.array([])

    if len(raw) % 2 != 0:
        raw = raw[:-1]  # Ensure I/Q pairs

    iq = raw[::2] + 1j * raw[1::2]
    magnitude = np.abs(iq)
    db = 20 * np.log10(magnitude + 1e-6)
    time = np.arange(len(db)) / sample_rate + start_time

    # Downsample for time-domain plot
    time_ds = time[::decimate]
    db_ds = db[::decimate]

    return time_ds, db_ds, iq

# === Initial data ===
start_time = 0.0
time_data, amp_data, iq_data = read_iq_window(start_time)
max_hold_trace = amp_data.copy()

# === Plot setup ===
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
line, = ax.plot(time_data, amp_data, lw=0.5, label="Current")
line_max, = ax.plot(time_data, max_hold_trace, lw=0.5, label="Max Hold", color='r')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (dBFS)')
ax.set_title('IQ Signal Viewer (dBFS)')
ax.grid(True)
ax.invert_yaxis()
ax.legend()

# === Slider setup ===
ax_slider = plt.axes([0.15, 0.22, 0.7, 0.03])
time_slider = Slider(
    ax=ax_slider,
    label='Start Time (s)',
    valmin=0.0,
    valmax=max(0.0, total_duration - window_duration),
    valinit=start_time,
    valstep=0.1,
)

# === Waterfall axes
ax_waterfall = plt.axes([0.15, 0.05, 0.25, 0.15])
ax_waterfall.set_title("Waterfall (FFT)")
ax_waterfall.axis("off")

# === CSV Export button
ax_button_export = plt.axes([0.5, 0.1, 0.1, 0.05])
btn_export = Button(ax_button_export, 'Export CSV')

# === Toggle buttons
ax_check = plt.axes([0.65, 0.1, 0.2, 0.1])
check = CheckButtons(ax_check, ['Show Max-Hold', 'Show FFT'], [True, True])

# === Update function ===
def update(val):
    global max_hold_trace
    start = time_slider.val
    time_data, amp_data, iq_data = read_iq_window(start)

    if len(time_data) == 0:
        print("No data")
        return

    # Update current trace
    line.set_xdata(time_data)
    line.set_ydata(amp_data)

    # Update max hold
    if check.get_status()[0]:  # Show Max-Hold
        max_hold_trace = np.maximum(max_hold_trace, amp_data)
        line_max.set_xdata(time_data)
        line_max.set_ydata(max_hold_trace)
        line_max.set_visible(True)
    else:
        line_max.set_visible(False)

    # Update waterfall (FFT)
    if check.get_status()[1]:  # Show FFT
        ax_waterfall.clear()
        ax_waterfall.set_title("Waterfall (FFT)")
        if len(iq_data) > 0:
            fft = np.fft.fftshift(np.fft.fft(iq_data))
            fft_db = 20 * np.log10(np.abs(fft) + 1e-6)
            freqs = np.fft.fftshift(np.fft.fftfreq(len(fft), d=1/sample_rate)) / 1e6  # MHz
            ax_waterfall.plot(freqs, fft_db, lw=0.5)
            ax_waterfall.set_xlabel("Freq (MHz)")
            ax_waterfall.set_ylabel("dBFS")
    else:
        ax_waterfall.axis("off")

    ax.set_xlim(time_data[0], time_data[-1])
    ax.set_ylim(np.max(amp_data) + 5, np.min(amp_data) - 5)
    fig.canvas.draw_idle()

# === Export callback ===
def export_csv(event):
    fname = f"iq_trace_{time_slider.val:.1f}s.csv"
    np.savetxt(fname, np.column_stack((time_data, amp_data)), delimiter=",", header="Time(s),Amplitude(dBFS)", comments='')
    print(f"Exported: {fname}")

# === Attach events ===
time_slider.on_changed(update)
btn_export.on_clicked(export_csv)
check.on_clicked(lambda _: update(time_slider.val))

# === Initial plot render
update(start_time)
plt.show()
