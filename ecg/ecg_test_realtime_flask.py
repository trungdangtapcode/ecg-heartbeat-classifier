import serial
import numpy as np
from collections import deque
import threading
import time
from flask import Flask, jsonify
import sys

# Serial port configuration
SERIAL_PORT = 'COM6'  # Replace with your port (e.g., '/dev/ttyUSB0' on Linux)
BAUD_RATE = 9600      # Match your device’s baud rate
TIMEOUT = 1           # Seconds
FS = 100              # Sampling frequency in Hz
MAX_POINTS = 1000     # 10 seconds of data at 100 Hz
FIXED_LENGTH = int(1.5 * FS)  # Fixed length for extracted beats (1.5 seconds)

# Shared data for the latest beat
latest_beat = None
latest_beat_time = None
lock = threading.Lock()

# Flask app setup
app = Flask(__name__)

def extract_beat(signal, fs):
    """Extract the first ECG beat from a signal window based on R-peaks."""
    if len(signal) == 0:
        return None
    signal = np.array(signal)
    # Normalize to [0, 1]
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    if signal_max == signal_min:
        return None
    normalized_signal = (signal - signal_min) / (signal_max - signal_min)
    # Find local maxima
    diff = np.diff(normalized_signal)
    local_max_indices = [i for i in range(1, len(diff)) if diff[i-1] > 0 and diff[i] < 0]
    # Select R-peaks with normalized value >= 0.8
    r_peak_candidates = [i for i in local_max_indices if normalized_signal[i] >= 0.8]
    if not r_peak_candidates:
        return None
    # Compute median R-R interval
    rr_intervals = np.diff(r_peak_candidates)
    T = np.median(rr_intervals) if len(rr_intervals) > 0 else 0
    if T == 0:
        return None
    # Extract the first beat
    r = r_peak_candidates[0]
    L = int(np.round(1.2 * T))
    segment = normalized_signal[r : r + L]
    if len(segment) < FIXED_LENGTH:
        pad_length = FIXED_LENGTH - len(segment)
        padded_segment = np.pad(segment, (0, pad_length), 'constant')
    else:
        padded_segment = segment[:FIXED_LENGTH]
    return padded_segment

def ecg_reader():
    """Read ECG data from serial port and update the latest beat."""
    global latest_beat, latest_beat_time
    # Initialize serial connection
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud")
    except serial.SerialException as e:
        print(f"Failed to connect to {SERIAL_PORT}: {e}")
        sys.exit(1)
    
    y_data = deque(maxlen=MAX_POINTS)  # ECG values
    sample_count = 0
    last_extraction_time = time.time()
    
    try:
        while True:
            # Read all available serial data
            while ser.in_waiting:
                try:
                    line_raw = ser.readline().decode('utf-8').strip()
                    if line_raw:
                        y = float(line_raw)
                        y_data.append(y)
                        sample_count += 1
                except (ValueError, UnicodeDecodeError):
                    print(f"Invalid data: {line_raw}")
            
            # Extract a beat every second
            current_time = time.time()
            if current_time - last_extraction_time >= 1.0:
                signal = list(y_data)
                beat = extract_beat(signal, FS)
                if beat is not None:
                    with lock:
                        latest_beat = beat.tolist()
                        latest_beat_time = np.arange(len(beat)) / FS
                        latest_beat_time = latest_beat_time.tolist()
                last_extraction_time = current_time
            time.sleep(0.01)  # Prevent CPU overuse
    except KeyboardInterrupt:
        print("ECG reader stopped")
    finally:
        ser.close()
        print("Serial connection closed")

@app.route('/api/current_beat', methods=['GET'])
def get_current_beat():
    """Return the latest extracted ECG beat as JSON."""
    with lock:
        if latest_beat is None or latest_beat_time is None:
            return jsonify({
                'status': 'error',
                'message': 'No beat data available yet'
            }), 404
        return jsonify({
            'status': 'success',
            'time': latest_beat_time,
            'amplitude': latest_beat
        })

if __name__ == '__main__':
    # Start ECG reader in a separate thread
    reader_thread = threading.Thread(target=ecg_reader, daemon=True)
    reader_thread.start()
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)