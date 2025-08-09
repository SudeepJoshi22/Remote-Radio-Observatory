import os
import json
import numpy as np
from flask import Flask, jsonify, send_from_directory, abort
from datetime import datetime, timedelta, timezone

app = Flask(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(ROOT_DIR, 'recordings')

@app.route('/api/plot_data/<string:prefix>')
def get_plot_data(prefix):
    """
    Processes a recording file and sends a downsampled dataset ready for plotting.
    """
    try:
        meta_filepath = os.path.join(RECORDINGS_DIR, f"{prefix}.sigmf-meta")
        data_filepath = os.path.join(RECORDINGS_DIR, f"{prefix}.sigmf-data")

        if not os.path.exists(meta_filepath) or not os.path.exists(data_filepath):
            return jsonify({"error": "Data or metadata file not found."}), 404

        with open(meta_filepath, 'r', encoding='utf-8-sig') as f:
            metadata = json.loads(f.read().strip())

        signal_data = np.fromfile(data_filepath, dtype=np.float32)
        
        if signal_data.size == 0:
            return jsonify({"error": "Data file is empty."}), 400

        max_points = 2000
        if signal_data.size > max_points:
            step = len(signal_data) // max_points
            downsampled_data = signal_data[::step]
        else:
            downsampled_data = signal_data
        
        g = metadata.get('global', {})
        start_time_str = g.get('start_time')
        sample_rate = g.get('sample_rate')

        if not start_time_str or not sample_rate:
            return jsonify({"error": "Metadata is missing 'start_time' or 'sample_rate'."}), 400

        if start_time_str.upper().endswith('Z'):
            start_time_str = start_time_str[:-1]
        
        start_dt = datetime.fromisoformat(start_time_str).replace(tzinfo=timezone.utc)
        
        timestamps = []
        original_sample_period_s = 1.0 / sample_rate
        downsampled_period_s = original_sample_period_s * (len(signal_data) / len(downsampled_data))
        
        for i in range(len(downsampled_data)):
            time_delta = timedelta(seconds=(i * downsampled_period_s))
            current_dt = start_dt + time_delta
            ts = current_dt.isoformat()
            timestamps.append(ts)

        return jsonify({
            "timestamps": timestamps,
            "values": downsampled_data.tolist(),
            "metadata": g
        })

    except Exception as e:
        print(f"Error processing plot data for {prefix}: {e}")
        return jsonify({"error": "An error occurred on the server while processing the data."}), 500


@app.route('/api/recordings')
def list_recordings():
    """ This function now only lists the available recordings. """
    if not os.path.isdir(RECORDINGS_DIR):
        return jsonify([])

    recordings_info = []
    for filename in os.listdir(RECORDINGS_DIR):
        if filename.endswith('.sigmf-meta'):
            prefix = filename.replace('.sigmf-meta', '')
            meta_filepath = os.path.join(RECORDINGS_DIR, filename)
            try:
                with open(meta_filepath, 'r', encoding='utf-8-sig') as f:
                    metadata = json.loads(f.read().strip())
                
                start_time = metadata.get('global', {}).get('start_time')
                if start_time:
                    # --- FIX: Clean up the timestamp before sending it ---
                    # This removes timezone info that can confuse some browsers,
                    # ensuring new Date() in JavaScript works reliably.
                    clean_time = start_time.split('+')[0].replace('Z', '')
                    recordings_info.append({'prefix': prefix, 'start_time': clean_time})

            except Exception as e:
                print(f"Error reading metadata for {filename}: {e}")
                continue
    
    recordings_info.sort(key=lambda x: x['start_time'], reverse=True)
    return jsonify(recordings_info)

@app.route('/')
def serve_index():
    return send_from_directory(ROOT_DIR, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

