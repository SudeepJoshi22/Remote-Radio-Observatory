import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from rtlsdr import RtlSdr
from scipy.signal import firwin, lfilter
from datetime import datetime, timezone, timedelta
import sys
import collections
import threading
import time
import os

# --- Shared Buffers for Threading ---
plot_data_buffer = collections.deque()
plot_time_buffer = collections.deque()
stop_event = threading.Event()
plot_buffer_lock = threading.Lock()

# --- FFT Power Processing Function ---
def process_fft_power(samples, fft_size, integration_bins, sample_rate):
    """Calculates a single integrated power spectral density value from a chunk of IQ samples."""
    # 1. Calculate FFT
    fft_result = np.fft.fft(samples, n=fft_size)
    fft_shifted = np.fft.fftshift(fft_result)
    
    # 2. Isolate the bins of interest
    center_bin = fft_size // 2
    start_bin = center_bin - integration_bins // 2
    end_bin = center_bin + integration_bins // 2
    signal_bins = fft_shifted[start_bin:end_bin]
    
    # 3. Calculate power
    power = np.sum(np.abs(signal_bins)**2)
    
    # 4. Normalize by FFT bin width to get Power Spectral Density (PSD)
    # This makes the measurement independent of sample rate and FFT size.
    psd = power / (sample_rate / fft_size)
    
    # 5. Convert to dB
    power_db = 10 * np.log10(psd + 1e-12)
    return power_db

# --- SDR Recording and Processing Thread ---
def sdr_record_and_process_thread(
    sdr_instance: RtlSdr, frequency: float, sample_rate: float, gain: float,
    duration_s: float, output_prefix: str, continuous: bool,
    processing_method: str, downsample_factor: int, downsample_method: str,
    fft_size: int, integration_bins: int, is_plotting_enabled: bool
):
    output_file = None
    recordings_dir = "recordings"
    os.makedirs(recordings_dir, exist_ok=True)

    # --- Setup filenames and open data file ---
    data_filename = os.path.join(recordings_dir, f"{output_prefix}.sigmf-data")
    meta_filename = os.path.join(recordings_dir, f"{output_prefix}.sigmf-meta")
    output_file = open(data_filename, 'wb')
    
    # --- Determine mode-specific parameters ---
    raw_chunk_size = 65536 # Use a consistent raw chunk size for reading
    if processing_method == 'fft_power':
        num_segments = raw_chunk_size // fft_size
        num_output_samples_per_segment = fft_size // downsample_factor
        num_total_output_samples = num_segments * num_output_samples_per_segment
        effective_sample_rate = (sample_rate / raw_chunk_size) * num_total_output_samples
        description = f"SDR FFT Power Spectral Density (dB/Hz) recording at {frequency/1e6:.3f} MHz"
        print(f"  [Recorder] FFT Power Spectral Density (dB/Hz) data will be saved to: {data_filename}")
    else: # amplitude mode
        effective_sample_rate = sample_rate / downsample_factor
        description = f"SDR magnitude (dB) recording at {frequency/1e6:.3f} MHz"
        print(f"  [Recorder] Amplitude (dB) data will be saved to: {data_filename}")

    # --- SigMF Metadata Initialization (Common for both modes) ---
    metadata = {
        "global": {
            "datatype": "f32_le", "sample_rate": effective_sample_rate,
            "start_time": datetime.now(timezone.utc).isoformat(timespec='milliseconds') + 'Z',
            "core:hw": "RTL-SDR", "core:freq_center": frequency, "core:gain": sdr_instance.gain,
            "core:description": description
        },
        "captures": [{"core:sample_start": 0, "core:tuner_freq": frequency, "core:gain": sdr_instance.gain, "core:sample_count": 0}],
        "annotations": []
    }
    with open(meta_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"  [Recorder] Metadata saved to: {meta_filename}")

    recorded_samples_count = 0
    start_time_recording = time.monotonic()

    try:
        while not stop_event.is_set():
            if not continuous and (time.monotonic() - start_time_recording) >= duration_s:
                break

            try:
                samples = sdr_instance.read_samples(raw_chunk_size)
            except Exception as e:
                print(f"\n[Recorder] Error reading samples: {e}. Stopping.")
                stop_event.set()
                break

            timestamp = datetime.now(timezone.utc)

            if processing_method == 'fft_power':
                num_segments = len(samples) // fft_size
                processed_values = []
                for i in range(num_segments):
                    segment = samples[i*fft_size : (i+1)*fft_size]
                    # Pass sample_rate to the processing function for normalization
                    power_db = process_fft_power(segment, fft_size, integration_bins, sample_rate)
                    processed_values.append(power_db)
                
                num_output_samples_per_segment = fft_size // downsample_factor
                final_chunk = np.repeat(processed_values, num_output_samples_per_segment).astype(np.float32)
                print(f"\r  [Recorder] Time: {timestamp.strftime('%H:%M:%S')}, Power: {np.mean(processed_values):.2f} dB/Hz", end="", flush=True)

            else: # amplitude mode
                power_db_chunk = 20 * np.log10(np.abs(samples) + 1e-12)
                trim_len = len(power_db_chunk) // downsample_factor * downsample_factor
                reshaped = power_db_chunk[:trim_len].reshape(-1, downsample_factor)
                if downsample_method == 'avg':
                    final_chunk = reshaped.mean(axis=1).astype(np.float32)
                elif downsample_method == 'max':
                    final_chunk = reshaped.max(axis=1).astype(np.float32)
                print(f"\r  [Recorder] Samples written: {recorded_samples_count + len(final_chunk):,}", end="", flush=True)

            output_file.write(final_chunk.tobytes())
            recorded_samples_count += len(final_chunk)

            if is_plotting_enabled:
                current_chunk_time_start = timestamp - timedelta(seconds=len(samples) / sample_rate)
                new_timestamps = [current_chunk_time_start + timedelta(seconds=i / effective_sample_rate) for i in range(len(final_chunk))]
                with plot_buffer_lock:
                    plot_data_buffer.extend(final_chunk)
                    plot_time_buffer.extend(new_timestamps)

    finally:
        metadata["captures"][0]["core:sample_count"] = recorded_samples_count
        with open(meta_filename, 'w') as f: json.dump(metadata, f, indent=4)
        if output_file: output_file.close()
        print("\n[Recorder] Recording thread finished.")

# --- Matplotlib Plotting Update Function ---
def live_plot_update(frame, line, ax_time):
    with plot_buffer_lock:
        if not plot_data_buffer:
            return line,
        # Limit buffer size to avoid excessive memory usage on long runs
        max_len = 20000 # Display last ~20000 points
        if len(plot_data_buffer) > max_len:
            for _ in range(len(plot_data_buffer) - max_len):
                plot_data_buffer.popleft()
                plot_time_buffer.popleft()
        
        plot_data = np.array(list(plot_data_buffer))
        plot_times = mdates.date2num(list(plot_time_buffer))

    line.set_ydata(plot_data)
    line.set_xdata(plot_times)
    ax_time.relim()
    ax_time.autoscale_view()
    fig = ax_time.get_figure()
    fig.autofmt_xdate()
    return line,

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record and process data from an RTL-SDR.")
    # General arguments
    parser.add_argument('-f', '--freq', type=float, required=True, help="Center frequency in Hz")
    parser.add_argument('-s', '--sample-rate', type=float, required=True, help="Sample rate in Hz")
    parser.add_argument('-g', '--gain', type=str, default='auto', help="Gain in dB (e.g., '30' or 'auto')")
    parser.add_argument('-o', '--output', type=str, help="Output filename prefix")
    parser.add_argument('--plot', action='store_true', help="Enable real-time plotting")
    # Duration arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--duration', type=float, help="Recording duration in seconds")
    group.add_argument('-c', '--continuous', action='store_true', help="Record continuously")
    # Processing arguments
    parser.add_argument('--processing-method', type=str, choices=['amplitude', 'fft_power'], default='amplitude', help="Processing method. Default: amplitude")
    # Amplitude-specific arguments
    parser.add_argument('--downsample-method', type=str, choices=['avg', 'max'], default='avg', help="Downsampling method for amplitude processing. Default: avg")
    parser.add_argument('--downsample', type=int, default=1000, help="Downsample factor for both processing methods. Default: 1000")
    # FFT-specific arguments
    parser.add_argument('--fft-size', type=int, default=1024, help="FFT size for fft_power processing. Default: 1024")
    parser.add_argument('--integration-bins', type=int, default=5, help="Number of FFT bins to integrate for fft_power. Default: 5")

    args = parser.parse_args()

    try:
        gain_val = float(args.gain)
    except ValueError:
        gain_val = 'auto' if args.gain.lower() == 'auto' else parser.error("Invalid gain")

    output_prefix_val = args.output or f"sdr_capture_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    sdr_instance = None
    try:
        sdr_instance = RtlSdr()
        sdr_instance.sample_rate = args.sample_rate
        sdr_instance.center_freq = args.freq
        sdr_instance.gain = gain_val
        print(f"SDR initialized: Sample Rate={sdr_instance.sample_rate/1e6:.3f} MS/s, Freq={sdr_instance.center_freq/1e6:.2f} MHz, Gain={sdr_instance.gain}")

        thread_args = (sdr_instance, args.freq, args.sample_rate, gain_val, args.duration, output_prefix_val, args.continuous, 
                       args.processing_method, args.downsample, args.downsample_method, args.fft_size, args.integration_bins, args.plot)
        recording_thread = threading.Thread(target=sdr_record_and_process_thread, args=thread_args)
        recording_thread.daemon = True
        recording_thread.start()

        if args.plot:
            fig, ax_time = plt.subplots(1, 1, figsize=(12, 6))
            line, = ax_time.plot([], [], lw=1)
            ax_time.set_title(f"Real-time {args.processing_method.replace('_', ' ').title()} (UTC)")
            ax_time.set_xlabel("Time (UTC)")
            ax_time.set_ylabel("Power (dB)" if args.processing_method == 'fft_power' else "Magnitude (dB)")
            ax_time.grid(True, linestyle=':')
            ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ani = animation.FuncAnimation(fig, live_plot_update, fargs=(line, ax_time), interval=200, blit=True, cache_frame_data=False)
            plt.show()
            stop_event.set()
        else:
            # In non-plot mode, we need a way to gracefully stop
            while recording_thread.is_alive():
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    except Exception as e:
        print(f"\n[Main] Error: {e}")
    finally:
        stop_event.set()
        if 'recording_thread' in locals() and recording_thread.is_alive(): recording_thread.join(2)
        if sdr_instance: sdr_instance.close()
        print("SDR closed. Exiting.")
        sys.exit(0)

