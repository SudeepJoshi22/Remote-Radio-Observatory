#!/usr/bin/env python3
"""
Test script to check the actual dBFS values in NPZ files
"""
import numpy as np
import sys
import os

def test_npz_file(filename):
    print(f"Testing file: {filename}")
    if not os.path.exists(filename):
        print(f"File {filename} does not exist")
        return

    data = np.load(filename)
    print(f"Available fields: {list(data.keys())}")

    if 'power_dbfs' in data.files:
        power_vals = data['power_dbfs']
        print(f"Power dBFS - Min: {np.min(power_vals):.3f}, Max: {np.max(power_vals):.3f}, Mean: {np.mean(power_vals):.3f}")
        print(f"All positive values: {np.sum(power_vals > 0)}/{len(power_vals)} ({100*np.sum(power_vals > 0)/len(power_vals):.1f}%)")

    if 'noise_dbfs' in data.files:
        noise_vals = data['noise_dbfs']
        print(f"Noise dBFS - Min: {np.min(noise_vals):.3f}, Max: {np.max(noise_vals):.3f}, Mean: {np.mean(noise_vals):.3f}")
        print(f"All positive values: {np.sum(noise_vals > 0)}/{len(noise_vals)} ({100*np.sum(noise_vals > 0)/len(noise_vals):.1f}%)")

    if 'snr_db' in data.files:
        snr_vals = data['snr_db']
        print(f"SNR dB - Min: {np.min(snr_vals):.3f}, Max: {np.max(snr_vals):.3f}, Mean: {np.mean(snr_vals):.3f}")

    # Show first few values for inspection
    if 'power_dbfs' in data.files:
        print(f"First 10 power values: {data['power_dbfs'][:10]}")
    if 'noise_dbfs' in data.files:
        print(f"First 10 noise values: {data['noise_dbfs'][:10]}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_npz_file(sys.argv[1])
    else:
        # Find most recent file
        import glob
        files = sorted(glob.glob("fm_observations/*.npz"))
        if files:
            test_npz_file(files[-1])
        else:
            print("No NPZ files found")