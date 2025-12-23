python3 -c "
import pandas as pd
import numpy as np

df = pd.read_csv('output.csv', header=None)

for idx, row in df.iterrows():
    low_hz = row[2]
    high_hz = row[3]
    step_hz = row[4]
    
    powers = pd.to_numeric(row[6:], errors='coerce').values
    
    freqs_mhz = np.arange(low_hz, high_hz, step_hz) / 1e6
    
    # Trim if lengths mismatch slightly
    min_len = min(len(freqs_mhz), len(powers))
    freqs_mhz = freqs_mhz[:min_len]
    powers = powers[:min_len]
    
    # Noise floor: 10th percentile (robust estimate, ignores strong signals)
    noise_floor = np.percentile(powers, 10)
    
    print(f'Frequency range: {low_hz/1e6:.3f} - {high_hz/1e6:.3f} MHz')
    print(f'Bin size: {step_hz/1000:.2f} kHz | Bins: {len(powers)}')
    print(f'Estimated noise floor (10th percentile): {noise_floor:.1f} dB\n')
    print('Freq (MHz)     Power (dB)')
    print('-------------------------')
    
    for f, p in zip(freqs_mhz, powers):
        print(f'{f:8.4f}       {p:6.1f}')
" | less
