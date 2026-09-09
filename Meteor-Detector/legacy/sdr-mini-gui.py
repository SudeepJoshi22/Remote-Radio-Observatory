import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Slider, Button, TextBox
from rtlsdr import RtlSdr
from scipy.signal import firwin, lfilter
from datetime import datetime, timedelta
import sys
import collections

# --- SDR Configuration ---
# These values are typical for RTL-SDR dongles. Adjust as needed.
sample_rate = 2.048e6  # Samples per second (e.g., 2.048 MS/s)
center_freq = 98.3e6   # Initial center frequency (e.g., 98.3 MHz FM band)
gain = 'auto'          # SDR gain (e.g., 'auto' or a specific dB value like 20)
fft_size = 1024        # Number of samples for each FFT calculation

# --- Global Variables for Plotting and Interaction ---
selected_freq = None   # Stores the frequency selected by clicking on the spectrum
sdr = None             # RTL-SDR object, initialized later
display_bandwidth = sample_rate # Initial displayed bandwidth (full SDR bandwidth)
current_display_center_freq = center_freq # The center of the currently displayed spectrum
v_line = None          # Vertical line for selected frequency

# --- Low-pass Filter Function ---
def lowpass_filter(data, cutoff_freq, fs, order=5):
    """
    Applies a low-pass FIR filter to the input data.
    This is used to extract a baseband signal (centered around 0 Hz)
    from the mixed-down complex samples.

    Args:
        data (np.array): The input signal data (real values).
        cutoff_freq (float): The cutoff frequency of the filter (Hz).
        fs (float): The sampling rate of the data (Hz).
        order (int): The order of the FIR filter.

    Returns:
        np.array: The filtered signal data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    # Ensure cutoff frequency is within valid range (0, 1) normalized to Nyquist
    if not (0 < normal_cutoff < 1):
        return data # Return original data if filter parameters are invalid
    # Create the filter coefficients for a low-pass filter (pass_zero=True)
    b = firwin(order + 1, normal_cutoff, pass_zero=True)
    # Apply the filter
    y = lfilter(b, [1.0], data)
    return y

# --- Matplotlib Figure and Axes Setup ---
# Create a figure with two subplots: one for spectrum, one for time domain
# Adjust subplot parameters to make space for the slider and text box at the bottom
# and to increase vertical spacing between subplots.
fig, (ax_spectrum, ax_time) = plt.subplots(2, 1, figsize=(12, 8))
# Adjusted bottom to make space for 4 control widgets
plt.subplots_adjust(bottom=0.35, hspace=0.4)

# Initialize empty plot lines that will be updated in real-time
line_spectrum, = ax_spectrum.plot([], [], lw=1, color='blue')
line_time, = ax_time.plot([], [], lw=1, color='green')

# Configure the spectrum subplot
ax_spectrum.set_title("Real-time Spectrum (Click to select frequency)")
ax_spectrum.set_xlabel("Frequency (MHz)")
ax_spectrum.set_ylabel("Magnitude (dB)")
ax_spectrum.grid(True, linestyle=':', alpha=0.7)

# Add the vertical line for selected frequency
v_line = ax_spectrum.axvline(x=0, color='red', linestyle='--', lw=1.5, visible=False)


# Configure the time domain subplot
ax_time.set_title("Selected Frequency Signal (Time Domain, UTC)")
ax_time.set_xlabel("Time (s)")
ax_time.set_ylabel("Amplitude")
ax_time.grid(True, linestyle=':', alpha=0.7)

# Calculate initial frequency bins for the spectrum plot
freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1/sample_rate)) + center_freq
line_spectrum.set_xdata(freqs)

# Set initial X-axis data for the time domain plot
line_time.set_xdata(np.linspace(0, fft_size / sample_rate, fft_size))
line_time.set_ydata(np.zeros(fft_size))

# --- Control Widgets Setup ---
# Define axes for the new controls in a grid-like fashion at the bottom
# [left, bottom, width, height] - normalized coordinates

# Frequency Input Box
ax_freq_input = plt.axes([0.1, 0.25, 0.35, 0.03])
freq_input_box = TextBox(
    ax_freq_input,
    'Center Freq (MHz): ',
    initial=f"{center_freq/1e6:.2f}"
)

# Sample Rate Input Box
ax_sample_rate_input = plt.axes([0.55, 0.25, 0.35, 0.03])
sample_rate_input_box = TextBox(
    ax_sample_rate_input,
    'Sample Rate (MS/s): ',
    initial=f"{sample_rate/1e6:.3f}"
)

# Bandwidth Slider
ax_bandwidth_slider = plt.axes([0.1, 0.2, 0.35, 0.03])
bandwidth_slider = Slider(
    ax_bandwidth_slider,
    'Display BW (kHz)', # Changed label to clarify it's display bandwidth
    10e3,
    sample_rate, # Max display bandwidth is the SDR's sample rate
    valinit=sample_rate,
    valstep=1e3
)

# Gain Slider - Placeholder initialization. Actual min/max/valinit set after SDR init.
ax_gain_slider = plt.axes([0.55, 0.2, 0.35, 0.03])
# The actual Slider object will be created after SDR initialization to use supported_gains
gain_slider = None # Initialize as None, will be replaced with actual Slider object

# --- Function to update spectrum X-axis limits ---
def update_spectrum_xlim():
    """
    Updates the X-axis limits of the spectrum plot based on
    current_display_center_freq and display_bandwidth.
    """
    global current_display_center_freq, display_bandwidth
    half_bandwidth = display_bandwidth / 2
    ax_spectrum.set_xlim(current_display_center_freq - half_bandwidth,
                         current_display_center_freq + half_bandwidth)
    fig.canvas.draw_idle()

# --- Slider Callbacks ---
def on_bandwidth_change(val):
    global display_bandwidth
    display_bandwidth = val
    update_spectrum_xlim()
    bandwidth_slider.label.set_text(f'Display BW ({val/1e3:.0f} kHz)')

bandwidth_slider.on_changed(on_bandwidth_change)

def on_gain_change(val):
    global sdr, gain
    if sdr:
        try:
            # rtlsdr will automatically pick the closest supported gain
            sdr.gain = val
            gain = sdr.gain # Update global 'gain' to actual set value
            print(f"[i] SDR Gain set to: {sdr.gain:.1f} dB")
        except Exception as e:
            print(f"Error setting gain: {e}")
            # Revert slider value if setting failed
            if gain_slider: # Check if slider exists before setting value
                gain_slider.set_val(gain)
    else:
        print("SDR not initialized. Cannot change gain.")

# --- Text Box Callbacks ---
def on_freq_submit(text):
    global sdr, center_freq, freqs, current_display_center_freq
    try:
        new_freq_mhz = float(text)
        new_center_freq_hz = new_freq_mhz * 1e6

        if not (24e6 <= new_center_freq_hz <= 1.7e9):
            print(f"Warning: Frequency {new_freq_mhz:.2f} MHz is outside typical RTL-SDR range (24 MHz - 1.7 GHz).")

        if sdr:
            sdr.center_freq = new_center_freq_hz
            center_freq = new_center_freq_hz
            print(f"[i] SDR center frequency updated to: {sdr.center_freq/1e6:.2f} MHz")

            # Recalculate frequencies for the spectrum plot based on new center_freq
            freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1/sample_rate)) + center_freq
            line_spectrum.set_xdata(freqs)

            current_display_center_freq = center_freq
            update_spectrum_xlim()
            fig.canvas.draw_idle()
        else:
            print("SDR not initialized. Cannot change frequency.")
    except ValueError:
        print(f"Invalid frequency input: '{text}'. Please enter a number in MHz.")
        freq_input_box.set_val(f"{center_freq/1e6:.2f}")

freq_input_box.on_submit(on_freq_submit)

def on_sample_rate_submit(text):
    # Declare all global variables that will be REASSIGNED at the very top
    global bandwidth_slider, ax_bandwidth_slider
    # Declare other global variables that will be read/modified
    global sdr, sample_rate, freqs, display_bandwidth, fig # fig is also global

    try:
        new_sample_rate_msps = float(text)
        new_sample_rate_hz = new_sample_rate_msps * 1e6

        # Basic validation for reasonable sample rate (e.g., 250 kHz to 2.8 MHz for RTL-SDR)
        if not (250e3 <= new_sample_rate_hz <= 2.8e6):
            print(f"Warning: Sample rate {new_sample_rate_msps:.3f} MS/s is outside typical RTL-SDR range (0.25 - 2.8 MS/s).")

        if sdr:
            sdr.sample_rate = new_sample_rate_hz
            sample_rate = new_sample_rate_hz # Update global variable
            print(f"[i] SDR Sample Rate updated to: {sdr.sample_rate/1e6:.3f} MS/s")

            # Recalculate frequencies for the spectrum plot based on new sample_rate
            freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1/sample_rate)) + center_freq
            line_spectrum.set_xdata(freqs)

            # --- Recreate bandwidth slider to update its max value ---
            # Disconnect old callbacks to prevent memory leaks.
            bandwidth_slider.on_changed(None) # Use of the old bandwidth_slider object
            
            # Remove the old slider axes
            ax_bandwidth_slider.clear() 
            fig.delaxes(ax_bandwidth_slider) # Remove the axes from the figure

            # If current display_bandwidth exceeds new sample_rate, adjust it
            if display_bandwidth > sample_rate:
                display_bandwidth = sample_rate # Update global
            
            # Recreate the axes for the bandwidth slider (reassigns ax_bandwidth_slider)
            # Use the existing ax_bandwidth_slider variable to reassign the new axes
            ax_bandwidth_slider = plt.axes([0.1, 0.2, 0.35, 0.03])
            
            # Recreate the slider with the new sample_rate as its max value (reassigns bandwidth_slider)
            bandwidth_slider = Slider(
                ax_bandwidth_slider,
                'Display BW (kHz)',
                10e3,
                sample_rate, # Use the new sample_rate as the max
                valinit=display_bandwidth, # Use the potentially clamped display_bandwidth
                valstep=1e3
            )
            bandwidth_slider.on_changed(on_bandwidth_change) # Reconnect callback
            
            update_spectrum_xlim()
            fig.canvas.draw_idle()
        else:
            print("SDR not initialized. Cannot change sample rate.")
    except ValueError:
        print(f"Invalid sample rate input: '{text}'. Please enter a number in MS/s.")
        sample_rate_input_box.set_val(f"{sample_rate/1e6:.3f}")

sample_rate_input_box.on_submit(on_sample_rate_submit)

# --- Click Handler for Frequency Selection ---
def onclick(event):
    """
    Handles mouse clicks on the spectrum plot to select a center frequency.
    This also updates the display center frequency.
    """
    global selected_freq, current_display_center_freq, v_line # Added v_line to globals
    if event.inaxes == ax_spectrum and event.button is MouseButton.LEFT:
        selected_freq = event.xdata
        current_display_center_freq = selected_freq # Update display center to clicked freq
        print(f"[i] Frequency selected: {selected_freq/1e6:.2f} MHz") # Formatted output
        
        # Update the vertical line position and make it visible
        v_line.set_xdata([selected_freq, selected_freq])
        v_line.set_visible(True)

        update_spectrum_xlim() # Update spectrum view to center on new selected_freq
        fig.canvas.draw_idle() # Ensure immediate redraw of the line
        
# Connect the click handler to the figure's button press event
fig.canvas.mpl_connect('button_press_event', onclick)

# --- Animation Update Function ---
def update(frame):
    """
    This function is called repeatedly by FuncAnimation to update the plots.
    It reads samples from the SDR, performs FFT, and updates the time-domain plot
    for the selected frequency.
    """
    global sdr, selected_freq, current_display_center_freq, display_bandwidth, freqs, v_line

    if sdr is None:
        return line_spectrum, line_time, v_line # Return v_line as well for blitting

    try:
        # Read samples from the SDR. Read more samples than fft_size for filtering.
        # The rtlsdr library returns complex I/Q samples.
        samples = sdr.read_samples(fft_size * 2) # Read twice the FFT size for better filtering context
    except Exception as e:
        print(f"Error reading samples from SDR: {e}")
        return line_spectrum, line_time, v_line # Return v_line as well for blitting

    # --- Spectrum Plot Update ---
    # Take only fft_size samples for the spectrum calculation
    spectrum = np.fft.fftshift(np.fft.fft(samples[:fft_size]))
    # Calculate power in dB (add a small epsilon to prevent log(0) errors)
    power = 20 * np.log10(np.abs(spectrum) + 1e-12)
    line_spectrum.set_ydata(power) # Update the Y-data of the spectrum line

    # Dynamically adjust Y-axis limits for the spectrum
    ax_spectrum.set_ylim(power.min() - 5, power.max() + 5) # Add some padding
    # The X-axis limits are handled by update_spectrum_xlim() and onclick()

    # --- Time-domain Plot Update for Selected Frequency ---
    if selected_freq is not None:
        # Calculate the frequency offset from the SDR's center frequency
        offset = selected_freq - sdr.center_freq

        # Create a time vector for mixing
        t = np.arange(len(samples)) / sdr.sample_rate
        # Mix the received samples down to baseband for the selected frequency
        # This shifts the selected frequency to 0 Hz (DC)
        mixed = samples * np.exp(-2j * np.pi * offset * t)

        # Apply a low-pass filter to the real part of the mixed signal
        # This isolates the signal around the new 0 Hz (baseband)
        # A 5 kHz cutoff will give a 10 kHz bandwidth around the selected frequency
        filtered = lowpass_filter(mixed.real, 5e3, sdr.sample_rate) # 5 kHz cutoff for 10 kHz BW

        # Take only fft_size samples for display
        filtered = filtered[:fft_size]

        # Safety check: ensure filtered data has the expected length
        if len(filtered) < fft_size:
            print("Warning: Filtered data length mismatch. Skipping time domain update.")
            return line_spectrum, line_time, v_line # Return v_line as well for blitting

        # Get current UTC time for the title (still using for display)
        now = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]

        # Update the Y-data of the time-domain line
        line_time.set_ydata(filtered)
        # Update the X-data (time axis) for the time domain plot (relative time)
        line_time.set_xdata(np.linspace(0, len(filtered)/sdr.sample_rate, len(filtered)))

        # Dynamically adjust X and Y axis limits for the time domain plot
        ax_time.set_xlim(0, len(filtered) / sdr.sample_rate)
        ax_time.set_ylim(filtered.min() - 0.1, filtered.max() + 0.1) # Add some padding
        # Fix for text overlap: use two lines for the title
        ax_time.set_title(f"Time Domain (UTC: {now})\nSelected: {selected_freq/1e6:.2f} MHz") # Formatted output
    else:
        # If no frequency is selected, reset plot
        dummy_times = np.linspace(0, fft_size / sample_rate, fft_size)
        line_time.set_ydata(np.zeros(len(dummy_times)))
        line_time.set_xdata(dummy_times)
        ax_time.set_xlim(dummy_times[0], dummy_times[-1])
        ax_time.set_ylim(-0.1, 0.1)
        ax_time.set_title("Time Domain (UTC) - Click spectrum to select frequency")
        v_line.set_visible(False) # Hide the vertical line if no frequency is selected


    # Return the updated plot lines. This is crucial for blitting optimization.
    return line_spectrum, line_time, v_line # Return v_line as well for blitting

# --- Main Execution ---
if __name__ == "__main__":
    # Note: The UserWarning "pkg_resources is deprecated" comes from the rtlsdr library itself,
    # not from this code. It's an upstream warning and does not affect the functionality of this script.
    try:
        sdr = RtlSdr()
        # Initialize SDR parameters
        sdr.sample_rate = sample_rate
        sdr.center_freq = center_freq
        
        # --- Configure Gain Slider based on supported gains ---
        # Get supported gains from the SDR device
        supported_gains = []
        try:
            supported_gains = sdr.supported_gains
        except AttributeError:
            print("[E] 'supported_gains' attribute not found. Using default gain range.")
            print("    This might happen with certain pyrtlsdr versions or backends.")
            
        # Determine initial gain slider properties
        min_gain_val = 0
        max_gain_val = 50
        initial_gain_val = 25 # Default mid-range if no supported gains found

        if supported_gains:
            min_gain_val = min(supported_gains)
            max_gain_val = max(supported_gains)
            if gain == 'auto':
                initial_gain_val = (min_gain_val + max_gain_val) / 2
            elif gain in supported_gains:
                initial_gain_val = gain
            else:
                initial_gain_val = min(supported_gains, key=lambda x:abs(x-gain))
                print(f"[W] Initial gain {gain} dB not directly supported. Setting to closest: {initial_gain_val:.1f} dB")
            print(f"[i] SDR Supported Gains (dB): {supported_gains}")
        else:
            print("[W] No supported gains found or attribute error. Gain slider using default range (0-50 dB).")

        # Create the Gain Slider *after* determining its min/max/initial values
        # Removed 'global gain_slider' from here as it's declared at module level
        gain_slider = Slider(
            ax_gain_slider,
            'Gain (dB)',
            min_gain_val,
            max_gain_val,
            valinit=initial_gain_val,
            valstep=1
        )
        gain_slider.on_changed(on_gain_change) # Connect the callback

        sdr.gain = initial_gain_val # Set SDR gain
        gain = initial_gain_val # Update global gain

        print(f"SDR initialized: Sample Rate={sdr.sample_rate/1e6:.3f} MS/s, "
              f"Center Freq={sdr.center_freq/1e6:.2f} MHz, Gain={sdr.gain:.1f} dB")

        update_spectrum_xlim()

        # The 'ani' object needs to be accessible in the update function for its interval.
        # It's typically created here.
        ani = animation.FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)

        plt.show()

    except Exception as e:
        print(f"Failed to initialize SDR or run application: {e}")
        print("Please ensure your RTL-SDR dongle is connected and drivers are installed.")
        print("You might need to install 'pyrtlsdr' and its dependencies (e.g., librtlsdr).")
    finally:
        if sdr:
            sdr.close()
            print("SDR closed.")
        sys.exit(0)

