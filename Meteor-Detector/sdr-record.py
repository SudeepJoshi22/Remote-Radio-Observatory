import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates # For UTC time axis formatting
from matplotlib.backend_bases import MouseButton # Still needed for GUI click
from matplotlib.widgets import Slider, Button, TextBox # Still needed for GUI widgets
from rtlsdr import RtlSdr
from scipy.signal import firwin, lfilter # Still needed for filtering
from datetime import datetime, timezone, timedelta
import sys
import collections
import threading # For running SDR acquisition in a separate thread
import time # For time.sleep
import os # Added for directory creation

# --- SDR Configuration (Global defaults, can be overridden by CLI args) ---
sample_rate = 2.048e6  # Raw samples per second
center_freq = 98.3e6   # Initial center frequency
gain = 'auto'          # SDR gain
fft_size = 1024        # FFT size for spectrum (not directly used in amplitude plot, but part of SDR context)

# --- Global Variables for Plotting and Interaction (GUI specific) ---
# These are only relevant when --plot is enabled
selected_freq = None   # Frequency selected by clicking on spectrum
current_display_center_freq = center_freq # Center of currently displayed spectrum
v_line = None          # Vertical line for selected frequency (on spectrum plot)

# --- Shared Buffers for Threading (for --plot option) ---
# These deques will hold downsampled amplitude data and their timestamps for plotting.
# Maxlen is set to None to allow "expanding" plot as requested. Be cautious with very long runs.
plot_amplitude_buffer = collections.deque() # Now holds dB values
plot_time_buffer = collections.deque()
stop_event = threading.Event() # Event to signal threads to stop
plot_buffer_lock = threading.Lock() # NEW: Lock to protect access to plot_amplitude_buffer and plot_time_buffer

# --- Low-pass Filter Function (from previous versions, kept for completeness) ---
def lowpass_filter(data, cutoff_freq, fs, order=5):
    """
    Applies a low-pass FIR filter to the input data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    if not (0 < normal_cutoff < 1):
        return data
    b = firwin(order + 1, normal_cutoff, pass_zero=True)
    y = lfilter(b, [1.0], data)
    return y

# --- SDR Recording and Processing Function (runs in a separate thread) ---
def sdr_record_and_process_thread(
    sdr_instance: RtlSdr,
    frequency: float,
    sample_rate: float,
    gain: float,
    duration_s: float,
    output_prefix: str,
    continuous: bool,
    downsample_factor: int,
    downsample_method: str,
    plot_amplitude_buffer: collections.deque, # Passed from main thread
    plot_time_buffer: collections.deque,     # Passed from main thread
    stop_event: threading.Event,             # Passed from main thread
    is_plotting_enabled: bool,               # Flag to know if plotting is active
    plot_buffer_lock: threading.Lock         # NEW: Passed lock
):
    """
    Records SDR data, processes it (amplitude, downsample), writes to file,
    and optionally adds to plot buffers. Runs in a separate thread.
    """
    data_file = None
    
    # Generate default output prefix if not provided (should be handled by main, but safety)
    if output_prefix is None:
        output_prefix = f"sdr_amplitude_capture_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    # Define the recordings directory
    recordings_dir = "recordings"
    
    # Create the recordings directory if it doesn't exist
    os.makedirs(recordings_dir, exist_ok=True)

    # Construct full file paths within the recordings directory
    data_filename = os.path.join(recordings_dir, f"{output_prefix}.sigmf-data")
    meta_filename = os.path.join(recordings_dir, f"{output_prefix}.sigmf-meta")

    # Calculate the effective sample rate after downsampling
    effective_sample_rate = sample_rate / downsample_factor
    if effective_sample_rate < 1:
        print(f"Error: Downsample factor {downsample_factor} is too high for sample rate {sample_rate}.")
        print(f"Effective sample rate would be {effective_sample_rate} Hz. Exiting thread.")
        stop_event.set() # Signal main thread to stop
        return

    try:
        # SDR configuration (already done in main, but good to re-confirm for thread)
        sdr_instance.sample_rate = sample_rate
        sdr_instance.center_freq = frequency
        sdr_instance.gain = gain 
        actual_gain = sdr_instance.gain

        start_time_utc = datetime.now(timezone.utc).isoformat(timespec='milliseconds') + 'Z'

        # --- SigMF Metadata Initialization ---
        direct_sampling_status = None
        if hasattr(sdr_instance, 'direct_sampling'):
            try:
                direct_sampling_status = sdr_instance.direct_sampling
            except Exception as e:
                print(f"[W] Thread: Error retrieving 'direct_sampling': {e}. Skipping metadata field.")
        else:
            print("[W] Thread: SDR object has no 'direct_sampling' attribute. Skipping metadata field.")

        metadata = {
            "global": {
                "datatype": "f32_le",  # Still float 32-bit, but now represents dB magnitude
                "sample_rate": effective_sample_rate, # This is the effective sample rate of the saved amplitude data
                "start_time": start_time_utc,
                "core:hw": "RTL-SDR",
                "core:freq_center": frequency,
                "core:gain": actual_gain,
                "core:direct_samp": direct_sampling_status,
                "core:description": f"SDR magnitude (dB) recording at {frequency/1e6:.3f} MHz (raw {sample_rate/1e6:.3f} MS/s, downsample {downsample_factor})" + 
                                    (f" for {duration_s:.1f} seconds" if not continuous else " (continuous)"),
                "core:raw_sample_rate": sample_rate # Add raw sample rate for context
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:tuner_freq": frequency,
                    "core:gain": actual_gain,
                    "core:sample_count": 0 # Will be updated later with amplitude sample count
                }
            ],
            "annotations": []
        }

        with open(meta_filename, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"  [Recorder] Initial metadata saved to: {meta_filename}")

        data_file = open(data_filename, 'wb')
        print(f"  [Recorder] Magnitude (dB) data will be saved to: {data_filename}")

        recorded_amplitude_samples_count = 0
        raw_chunk_size_samples = 65536 # Read 65536 raw IQ samples at a time

        # --- Recording Loop ---
        start_time_recording = time.monotonic() # For duration tracking

        while not stop_event.is_set(): # Loop until stop_event is set
            if not continuous and (time.monotonic() - start_time_recording) >= duration_s:
                print("\n[Recorder] Fixed duration reached. Stopping recording.")
                break

            try:
                raw_samples_chunk = sdr_instance.read_samples(raw_chunk_size_samples)
            except Exception as e:
                print(f"\n[Recorder] Error reading samples: {e}. Stopping recording.")
                stop_event.set() # Signal other threads to stop
                break
            '''            
            # Calculate amplitude (magnitude)
            amplitude_chunk = np.abs(raw_samples_chunk)
            
            # Convert to dB (add a small epsilon to avoid log10(0))
            power_db_chunk = 20 * np.log10(amplitude_chunk + 1e-12)

            # Downsample the dB values
            downsampled_power_db = power_db_chunk[::downsample_factor].astype(np.float32)
            
            # Write to file
            data_file.write(downsampled_power_db.tobytes())
            recorded_amplitude_samples_count += len(downsampled_power_db)
            '''

            # Calculate amplitude (magnitude)
            amplitude_chunk = np.abs(raw_samples_chunk)
            
            # Convert to dB (add a small epsilon to avoid log10(0))
            power_db_chunk = 20 * np.log10(amplitude_chunk + 1e-12)

            # --- Downsampling (Pooling) ---
            trim_len = len(power_db_chunk) // downsample_factor * downsample_factor
            reshaped = power_db_chunk[:trim_len].reshape(-1, downsample_factor)
            
            if downsample_method == 'avg':
                downsampled_power_db = reshaped.mean(axis=1).astype(np.float32)
            elif downsample_method == 'max':
                downsampled_power_db = reshaped.max(axis=1).astype(np.float32)
            elif downsample_method == 'rms':
                downsampled_power_db = np.sqrt((reshaped**2).mean(axis=1)).astype(np.float32)
            else:
                raise ValueError(f"Unsupported downsampling method: {downsample_method}")

            # Write to file
            data_file.write(downsampled_power_db.tobytes())
            recorded_amplitude_samples_count += len(downsampled_power_db)


            # Add to plot buffers if plotting is enabled
            if is_plotting_enabled:
                # Generate timestamps for the downsampled amplitude data
                current_chunk_time_start = datetime.now(timezone.utc) - timedelta(seconds=len(raw_samples_chunk) / sample_rate)
                new_plot_timestamps = [current_chunk_time_start + timedelta(seconds=i / effective_sample_rate)
                                       for i in range(len(downsampled_power_db))]
                
                # Acquire lock before modifying shared buffers
                with plot_buffer_lock:
                    plot_amplitude_buffer.extend(downsampled_power_db) # Now extending with dB values
                    plot_time_buffer.extend(new_plot_timestamps)

            # Print progress
            if continuous:
                print(f"\r  [Recorder] Samples: {recorded_amplitude_samples_count:,} ({recorded_amplitude_samples_count/effective_sample_rate:.1f} s)", end="", flush=True)
            else:
                progress_percent = (time.monotonic() - start_time_recording) / duration_s * 100
                print(f"\r  [Recorder] Progress: {progress_percent:.1f}% ({recorded_amplitude_samples_count:,} amplitude samples)", end="", flush=True)
            
            time.sleep(0.001) # Small sleep to yield CPU

        # --- Finalize Metadata ---
        print("\n[Recorder] Finalizing recording...")
        metadata["captures"][0]["core:sample_count"] = recorded_amplitude_samples_count
        metadata["global"]["core:duration"] = recorded_amplitude_samples_count / effective_sample_rate
        with open(meta_filename, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"  [Recorder] Final metadata (with {recorded_amplitude_samples_count} magnitude (dB) samples) saved to: {meta_filename}")
        print("[Recorder] Recording thread finished.")

    except Exception as e:
        print(f"\n[Recorder] Critical error in recording thread: {e}")
        stop_event.set() # Signal main thread to stop
    finally:
        if data_file:
            data_file.close()
            print("[Recorder] Data file closed.")
        # SDR device closure is handled in the main thread's finally block
        # to ensure it's always closed even if this thread crashes.


# --- Matplotlib Plotting Update Function (for FuncAnimation) ---
def live_plot_update(frame, line, ax_time, plot_amplitude_buffer, plot_time_buffer, plot_buffer_lock): # NEW: lock passed
    """
    Update function for Matplotlib's FuncAnimation.
    Reads from shared buffers and updates the plot.
    """
    # Acquire lock before reading from shared buffers
    with plot_buffer_lock:
        if not plot_amplitude_buffer:
            return line, # Return line as a tuple for blitting

        # Convert deques to NumPy arrays for plotting and numerical operations
        # Make copies of the data while holding the lock
        plot_data = np.array(list(plot_amplitude_buffer))
        plot_times = mdates.date2num(list(plot_time_buffer)) # mdates.date2num expects a list of datetime objects

    line.set_ydata(plot_data)
    line.set_xdata(plot_times)

    # Dynamically expand X-axis limits
    ax_time.set_xlim(plot_times[0], plot_times[-1])
    
    # Dynamically adjust Y-axis limits for dB values
    if len(plot_data) > 0:
        # Use a more robust way to get min/max for dB, avoiding inf/-inf from log10(0)
        finite_data = plot_data[np.isfinite(plot_data)]
        if len(finite_data) > 0:
            min_val = np.min(finite_data)
            max_val = np.max(finite_data)
            # Add some padding to the limits
            ax_time.set_ylim(min_val - 5, max_val + 5)
        else:
            ax_time.set_ylim(-100, 0) # Default dB range if no valid data
    else:
        ax_time.set_ylim(-100, 0) # Default dB range if buffer is empty

    # Adjust auto-formatting for date labels
    fig.autofmt_xdate()

    return line, # Return line as a tuple for blitting


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record SDR magnitude (dB) data from RTL-SDR dongle and save in SigMF format with downsampling. "
                    "Optionally plot in real-time."
    )
    parser.add_argument(
        '-f', '--freq', type=float, required=True,
        help="Center frequency in Hz (e.g., 100e6 for 100 MHz)"
    )
    parser.add_argument(
        '-s', '--sample_rate', type=float, required=True,
        help="Raw sample rate in Hz (e.g., 2.048e6 for 2.048 MS/s)"
    )
    parser.add_argument(
        '-g', '--gain', type=str, default='auto',
        help="Gain in dB (e.g., '30' or 'auto'). Note: 'auto' is a string."
    )
    parser.add_argument(
        '-d', '--duration', type=float,
        help="Recording duration in seconds. Mutually exclusive with --continuous."
    )
    parser.add_argument(
        '-c', '--continuous', action='store_true',
        help="Record continuously until interrupted (Ctrl+C). Mutually exclusive with --duration."
    )
    parser.add_argument(
        '--downsample-method', type=str, choices=['avg', 'max', 'rms'], default='avg',
        help="Downsampling method: 'avg' = average pooling, 'max' = max pooling, 'rms' = RMS pooling. Default: avg."
    )
    parser.add_argument(
        '--downsample', type=int, default=1000,
        help="Factor by which to downsample amplitude data (e.g., 1000 means 1 sample per 1000 raw samples). Default: 1000."
    )
    parser.add_argument(
        '-o', '--output', type=str, 
        help="Output filename prefix for .sigmf-data and .sigmf-meta files (default: sdr_amplitude_capture_YYYYMMDDTHHMMSSZ)"
    )
    parser.add_argument(
        '--plot', action='store_true',
        help="Enable real-time plotting of magnitude (dB) data."
    )

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.duration is not None and args.continuous:
        parser.error("Arguments --duration and --continuous are mutually exclusive. Please choose one.")
    if args.duration is None and not args.continuous:
        parser.error("Either --duration or --continuous must be specified.")
    if args.downsample < 1:
        parser.error("--downsample factor must be a positive integer (>= 1).")

    # Convert gain argument to float if it's not 'auto'
    try:
        gain_val = float(args.gain)
    except ValueError:
        if args.gain.lower() == 'auto':
            gain_val = 'auto'
        else:
            print(f"Error: Invalid gain value '{args.gain}'. Must be a number or 'auto'.")
            sys.exit(1)

    # Set output prefix if not provided
    output_prefix_val = args.output
    if output_prefix_val is None:
        output_prefix_val = f"sdr_amplitude_capture_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    # Note: The UserWarning "pkg_resources is deprecated" comes from the rtlsdr library itself,
    # not from this code. It's an upstream warning and does not affect the functionality of this script.
    sdr_instance = None
    recording_thread = None

    try:
        sdr_instance = RtlSdr()
        sdr_instance.sample_rate = args.sample_rate
        sdr_instance.center_freq = args.freq
        sdr_instance.gain = gain_val # Set initial gain

        print(f"SDR initialized: Raw Sample Rate={sdr_instance.sample_rate/1e6:.3f} MS/s, "
              f"Center Freq={sdr_instance.center_freq/1e6:.2f} MHz, Gain={sdr_instance.gain:.1f} dB")

        # Start the SDR recording and processing in a separate thread
        recording_thread = threading.Thread(
            target=sdr_record_and_process_thread,
            args=(sdr_instance, args.freq, args.sample_rate, gain_val, 
                  args.duration, output_prefix_val, args.continuous, 
                  args.downsample, args.downsample_method, plot_amplitude_buffer, plot_time_buffer, stop_event, args.plot, plot_buffer_lock)
            )
        recording_thread.daemon = True # Allow main program to exit even if thread is running
        recording_thread.start()

        if args.plot:
            # --- Matplotlib Figure and Axes Setup for Plotting ---
            fig, ax_time = plt.subplots(1, 1, figsize=(12, 6))
            line, = ax_time.plot([], [], lw=1, color='blue')
            ax_time.set_title("Real-time Magnitude (dB) (UTC)")
            ax_time.set_xlabel("Time (UTC)")
            ax_time.set_ylabel("Magnitude (dB)") # Updated label
            ax_time.grid(True, linestyle=':', alpha=0.7)

            # Configure UTC time axis formatting
            ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
            ax_time.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()

            # Start the Matplotlib animation
            ani = animation.FuncAnimation(
                fig, live_plot_update, fargs=(line, ax_time, plot_amplitude_buffer, plot_time_buffer, plot_buffer_lock), # NEW: Pass the lock
                interval=500, blit=True, cache_frame_data=False
            )
            
            # Show the plot window (this blocks the main thread)
            plt.show()

            # If plt.show() returns, it means the plot window was closed.
            # Signal the recording thread to stop.
            stop_event.set()
            print("[Main] Plot window closed. Signalling recording thread to stop.")
        else:
            # If not plotting, just wait for the recording thread to finish
            # or for a KeyboardInterrupt
            print("No plot requested. Recording in background. Press Ctrl+C to stop.")
            recording_thread.join() # Wait for the recording thread to complete

    except KeyboardInterrupt:
        print("\n[Main] Application interrupted by user (Ctrl+C).")
        stop_event.set() # Signal the recording thread to stop
    except Exception as e:
        print(f"\n[Main] Failed to initialize SDR or run application: {e}")
        print("Please ensure your RTL-SDR dongle is connected and drivers are installed.")
        print("You might need to install 'pyrtlsdr' (`pip install pyrtlsdr`) and its backend (librtlsdr).")
    finally:
        # Ensure SDR is closed, regardless of how the script exits
        if sdr_instance:
            sdr_instance.close()
            print("SDR device closed.")
        # Wait for the recording thread to finish its cleanup if it's still alive
        if recording_thread and recording_thread.is_alive():
            recording_thread.join(timeout=5) # Give it a few seconds to clean up
            if recording_thread.is_alive():
                print("[Main] Warning: Recording thread did not terminate cleanly.")
        sys.exit(0)

