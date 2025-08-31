import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from rtlsdr import RtlSdr
from datetime import datetime, timezone
import sys
import collections
import threading
import time
import os

# --- Shared Buffers for Threading ---
plot_power_buffer = collections.deque()
plot_time_buffer = collections.deque()
stop_event = threading.Event()
plot_buffer_lock = threading.Lock()

# --- New FFT Processing Function ---
def process_power_spectrum(samples, fft_size, integration_bins):
    """
    Performs FFT and calculates integrated power in a specific band.
    """
    # 1. Perform FFT
    fft_result = np.fft.fft(samples, n=fft_size)
    fft_shifted = np.fft.fftshift(fft_result)
    
    # 2. Isolate the signal bins around the center frequency
    center_bin = fft_size // 2
    start_bin = center_bin - integration_bins // 2
    end_bin = center_bin + integration_bins // 2
    signal_bins = fft_shifted[start_bin:end_bin]

    # 3. Calculate total power in the selected bins
    power = np.sum(np.abs(signal_bins)**2)
    
    return power

# --- SDR Recording and Processing Thread ---
def sdr_power_thread(
    sdr_instance: RtlSdr,
    fft_size: int,
    integration_bins: int,
    output_file_handle,
    is_plotting_enabled: bool
):
    """
    Continuously reads SDR data, processes it for power, writes to file,
    and adds to plot buffers.
    """
    try:
        while not stop_event.is_set():
            try:
                samples = sdr_instance.read_samples(fft_size)
            except Exception as e:
                print(f"\n[Recorder] Error reading samples: {e}. Stopping.")
                stop_event.set()
                break

            # Process the chunk of samples
            power = process_power_spectrum(samples, fft_size, integration_bins)
            timestamp = datetime.now(timezone.utc)

            # Save data to CSV file
            output_file_handle.write(f"{timestamp.isoformat()},{power}\n")

            # Add to plot buffers if plotting is enabled
            if is_plotting_enabled:
                with plot_buffer_lock:
                    plot_power_buffer.append(power)
                    plot_time_buffer.append(timestamp)
            
            print(f"\r  [Recorder] Time: {timestamp.strftime('%H:%M:%S')}, Relative Power: {power:.2f}", end="", flush=True)
            
            # A small sleep can be added here if the loop runs too fast,
            # but it's often better to let read_samples block.
            # time.sleep(0.01)

    except Exception as e:
        print(f"\n[Recorder] Critical error in recording thread: {e}")
        stop_event.set()
    finally:
        print("\n[Recorder] Recording thread finished.")

# --- Matplotlib Plotting Update Function ---
def live_plot_update(frame, line, ax_time):
    """
    Update function for Matplotlib's FuncAnimation.
    """
    with plot_buffer_lock:
        if not plot_power_buffer:
            return line,

        plot_data = np.array(list(plot_power_buffer))
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
    parser = argparse.ArgumentParser(
        description="Record and plot integrated power from an RTL-SDR for meteor detection."
    )
    parser.add_argument('-f', '--freq', type=float, required=True, help="Center frequency in Hz (e.g., 88.5e6)")
    parser.add_argument('-s', '--sample-rate', type=float, required=True, help="Sample rate in Hz (e.g., 240e3)")
    parser.add_argument('-g', '--gain', type=str, default='auto', help="Gain in dB (e.g., '30' or 'auto')")
    parser.add_argument('--fft-size', type=int, default=1024, help="FFT size (number of points). Default: 1024.")
    parser.add_argument('--integration-bins', type=int, default=5, help="Number of FFT bins to integrate for power. Default: 5.")
    parser.add_argument('-o', '--output', type=str, help="Output CSV filename (default: meteor_power_YYYYMMDDTHHMMSSZ.csv)")
    parser.add_argument('--plot', action='store_true', help="Enable real-time plotting of power data.")

    args = parser.parse_args()

    try:
        gain_val = float(args.gain)
    except ValueError:
        if args.gain.lower() == 'auto':
            gain_val = 'auto'
        else:
            parser.error(f"Invalid gain value '{args.gain}'. Must be a number or 'auto'.")

    output_filename = args.output
    if output_filename is None:
        output_filename = f"meteor_power_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"

    recordings_dir = "recordings"
    os.makedirs(recordings_dir, exist_ok=True)
    full_output_path = os.path.join(recordings_dir, output_filename)

    sdr_instance = None
    recording_thread = None
    output_file = None

    try:
        sdr_instance = RtlSdr()
        sdr_instance.sample_rate = args.sample_rate
        sdr_instance.center_freq = args.freq
        sdr_instance.gain = gain_val

        print(f"SDR initialized: Sample Rate={sdr_instance.sample_rate/1e6:.3f} MS/s, "
              f"Center Freq={sdr_instance.center_freq/1e6:.2f} MHz, Gain={sdr_instance.gain}")
        print(f"FFT Size: {args.fft_size}, Integration Bins: {args.integration_bins}")

        output_file = open(full_output_path, 'w')
        output_file.write("timestamp,power\n") # Write CSV header
        print(f"Saving data to: {full_output_path}")

        recording_thread = threading.Thread(
            target=sdr_power_thread,
            args=(sdr_instance, args.fft_size, args.integration_bins, output_file, args.plot)
        )
        recording_thread.daemon = True
        recording_thread.start()

        if args.plot:
            fig, ax_time = plt.subplots(1, 1, figsize=(12, 6))
            line, = ax_time.plot([], [], lw=1, color='blue')
            ax_time.set_title("Real-time Meteor Scatter Power (UTC)")
            ax_time.set_xlabel("Time (UTC)")
            ax_time.set_ylabel("Relative Power")
            ax_time.grid(True, linestyle=':', alpha=0.7)
            ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            fig.autofmt_xdate()

            ani = animation.FuncAnimation(
                fig, live_plot_update, fargs=(line, ax_time),
                interval=500, blit=True, cache_frame_data=False
            )
            plt.show()
            stop_event.set()
        else:
            print("Recording in background. Press Ctrl+C to stop.")
            while not stop_event.is_set():
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Main] Application interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\n[Main] Failed to initialize SDR or run application: {e}")
    finally:
        stop_event.set()
        if recording_thread and recording_thread.is_alive():
            recording_thread.join(timeout=2)
        if sdr_instance:
            sdr_instance.close()
            print("SDR device closed.")
        if output_file:
            output_file.close()
            print("Data file closed.")
        sys.exit(0)

