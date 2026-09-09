#!/usr/bin/env python3
"""
Check the calculation logic step by step
"""
import numpy as np
from rtlsdr import RtlSdr

def test_calculation():
    print("Testing calculation logic...")

    # Initialize SDR
    sdr = RtlSdr()
    sdr.sample_rate = 240000
    sdr.center_freq = 107.1e6
    sdr.gain = 15

    try:
        # Read samples
        samples = sdr.read_samples(65536)
        print(f"Read {len(samples)} samples")
        print(f"Sample dtype: {samples.dtype}")
        print(f"Sample range - real: [{np.min(samples.real):.3f}, {np.max(samples.real):.3f}]")
        print(f"Sample range - imag: [{np.min(samples.imag):.3f}, {np.max(samples.imag):.3f}]")
        print(f"Sample magnitude range: [{np.min(np.abs(samples)):.3f}, {np.max(np.abs(samples)):.3f}]")

        # Check if samples are normalized
        max_mag = np.max(np.abs(samples))
        print(f"Maximum magnitude: {max_mag:.3f}")
        print(f"Are samples normalized to [-1,1]? {max_mag <= 1.0}")

        # Apply window and compute FFT
        window = np.hanning(len(samples))
        windowed = samples * window
        fft_result = np.fft.fft(windowed, n=2048)
        fft_shifted = np.fft.fftshift(fft_result)
        power_spectrum = np.abs(fft_shifted) ** 2

        print(f"Power spectrum range: [{np.min(power_spectrum):.3e}, {np.max(power_spectrum):.3e}]")

        # Our dBFS calculation
        # dBFS = 10 * log10(power / power_max)
        # For normalized complex samples: max magnitude = sqrt(2), max power = 2
        power_max = 2.0  # Theoretical maximum for normalized complex samples
        power_db = 10 * np.log10((power_spectrum + 1e-12) / power_max)

        print(f"Our dBFS calculation - Min: {np.min(power_db):.3f}, Max: {np.max(power_db):.3f}")

        # Alternative: if samples are actually in full-scale integer range
        # RTL-SDR typically gives us values in [-128, 127] for 8-bit, then converted to float
        # Let's check what the actual scaling might be
        alt_power_db = 10 * np.log10(power_spectrum + 1e-12)
        print(f"Raw power (no normalization) - Min: {np.min(alt_power_db):.3f}, Max: {np.max(alt_power_db):.3f}")

        # Check what the actual max value is in our data
        actual_max_power = np.max(power_spectrum)
        if actual_max_power > 0:
            measured_max_db = 10 * np.log10(actual_max_power + 1e-12)
            print(f"Measured max power: {measured_max_db:.3f} dB (raw)")

            # If we assume this represents full scale, then:
            corrected_db = 10 * np.log10((power_spectrum + 1e-12) / actual_max_power)
            print(f"If treating measured max as full scale - Min: {np.min(corrected_db):.3f}, Max: {np.max(corrected_db):.3f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sdr.close()

if __name__ == "__main__":
    test_calculation()