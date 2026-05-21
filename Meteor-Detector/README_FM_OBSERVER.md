# FM Meteor Scatter Observer

Simple tools for long-term FM meteor scatter observation using RTL-SDR.

## Files

1. `fm_observe_npz.py` - Long-term observer that saves power traces in compressed .npz chunks
2. `plot_npz_utc.py` - Plotter that visualizes the saved data versus UTC time

## Dependencies

- Python 3.x
- pyrtlsdr
- numpy
- matplotlib

Install with:
```bash
pip install pyrtlsdr numpy matplotlib
```

## Usage

### 1. Long-term Observation

```bash
python fm_observe_npz.py \
    --frequency 107.1e6 \        # Center frequency in Hz
    --sample-rate 240000 \       # Sample rate in Hz
    --gain 35 \                  # SDR gain in dB
    --interval-ms 20 \           # Measurement interval in ms
    --chunk-seconds 600 \        # Save chunk every N seconds (600 = 10 min)
    --threshold-db 6 \           # SNR trigger threshold in dB
    --station "FM_STATION" \     # Station label
    --output-dir ./fm_observations # Output directory
```

Example for 24-hour run:
```bash
python fm_observe_npz.py --frequency 107.1e6 --gain 35 --interval-ms 20
```

### 2. Plotting Results

```bash
# Plot the most recent file
python plot_npz_utc.py

# Plot a specific file
python plot_npz_utc.py ./fm_observations/FM_STATION_20260521_120000_chunk0000.npz

# List available files
python plot_npz_utc.py --list

# Save plot as PNG instead of displaying
python plot_npz_utc.py --save
```

## Output Format

The observer saves data in compressed `.npz` files with the following arrays:

- `t_utc_ns`: UTC timestamps in nanoseconds since epoch
- `power_dbfs`: Target bin power in dB (relative power, 10*log10(power))
- `noise_dbfs`: Noise floor power in dB (relative power, 10*log10(power))
- `peak_dbfs`: Peak power in dB (relative power, 10*log10(power)) across FFT spectrum
- `snr_db`: Signal-to-noise ratio in dB (target - noise)
- `trigger`: Boolean flag when SNR exceeds threshold
- Plus metadata: station, frequency, sample rate, gain, etc.

## Design Notes

- **No raw IQ storage**: Only processes and saves derived metrics to minimize storage
- **Efficient storage**: .npz compression reduces storage by ~70-90% vs CSV
- **Configurable resolution**: Default 20ms intervals = ~4.3M samples/day
- **Robust to interruption**: Saves chunks periodically and handles Ctrl+C gracefully
- **UTC timestamps**: All time data stored in UTC for consistency

## Expected Storage

With default settings (20ms intervals):
- ~4.32 million samples per day
- ~100-200 MB per day (compressed .npz)
- Much smaller than raw IQ (~108 GB/2 hours at 2.56 MS/s)

## Tips for FM Meteor Scatter

1. **Frequency selection**: Choose FM frequencies that are normally quiet at your location
2. **Best times**: Late night/pre-dawn hours often show increased meteor activity
3. **Threshold tuning**: Start with lower thresholds (3-6 dB) to see all events, then adjust
4. **Verification**: Use known signals (local FM stations) to verify your system responds correctly

## Safety

- Does not transmit - only receives
- Compatible with standard RTL-SDR dongles
- Low CPU and memory footprint suitable for Raspberry Pi