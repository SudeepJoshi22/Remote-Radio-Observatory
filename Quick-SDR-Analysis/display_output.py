import pandas as pd
import numpy as np
df = pd.read_csv('output.csv', header=None)
for idx, row in df.iterrows():
    start_date = row[0]
    end_date = row[1]
    low_hz = row[2]
    high_hz = row[3]
    step_hz = row[4]
    powers = pd.to_numeric(row[6:], errors='coerce')
    freqs_mhz = np.arange(low_hz, high_hz, step_hz) / 1e6
    print(f'Scan from {start_date} to {end_date}')
    print(f'Frequency range: {low_hz/1e6:.1f} - {high_hz/1e6:.1f} MHz, bin size ~{step_hz/1e3:.1f} kHz')
    print(f'Noise floor approx: {powers.min():.1f} dB')
    top_idx = powers.nlargest(10).index
    print('Top 10 strongest signals:')
    for i in top_idx:
        print(f'  {freqs_mhz[i-6]:8.3f} MHz : {powers[i]:.1f} dB')
    print('\n') 
