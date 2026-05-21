#!/usr/bin/env python3
"""
Simple test to verify SDR reading and basic processing works
"""
import numpy as np
from rtlsdr import RtlSdr
import time

def main():
    print("Testing basic SDR functionality...")

    # Initialize SDR
    sdr = RtlSdr()
    sdr.sample_rate = 240000
    sdr.center_freq = 107.1e6
    sdr.gain = 10  # Low gain to avoid overload
    print(f"SDR configured: {sdr.center_freq/1e6:.1f} MHz, {sdr.sample_rate/1e3:.0f} kHz, gain={sdr.gain}")

    try:
        # Read a small block of samples
        print("Reading samples...")
        samples = sdr.read_samples(240)  # 1ms worth
        print(f"Got {len(samples)} samples")
        print(f"Sample dtype: {samples.dtype}")
        print(f"Sample range: {np.min(np.abs(samples)):.6f} to {np.max(np.abs(samples)):.6f}")

        # Simple FFT test
        window = np.hanning(len(samples))
        windowed = samples * window
        fft_result = np.fft.fft(windowed)
        fft_shifted = np.fft.fftshift(fft_result)
        power = np.abs(fft_shifted) ** 2
        power_db = 10 * np.log10(power + 1e-12)

        print(f"FFT power range: {np.min(power_db):.1f} to {np.max(power_db):.1f} dB")
        print(f"Mean power: {np.mean(power_db):.1f} dB")

        # Success!
        print("Basic SDR test PASSED")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sdr.close()
        print("SDR closed")

if __name__ == "__main__":
    main()