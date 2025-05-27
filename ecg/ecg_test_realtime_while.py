import serial
import numpy as np
from collections import deque
import threading
import time
from flask import Flask, jsonify
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Serial port configuration
SERIAL_PORT = 'COM6'  # Replace with your port (e.g., '/dev/ttyUSB0' on Linux)
BAUD_RATE = 9600      # Match your device’s baud rate
TIMEOUT = 1           # Seconds
FS = 100              # Sampling frequency in Hz
MAX_POINTS = 1000     # 10 seconds of data at 100 Hz
FIXED_LENGTH = int(1.5 * FS)  # Fixed length for extracted beats (1.5 seconds)

# Shared data for real-time plot and API
y_data = deque(maxlen=MAX_POINTS)  # ECG values
x_data = deque(maxlen=MAX_POINTS)  # Sample indices
latest_beat = None
latest_beat_time = None
lock = threading.Lock()
sample_count = 0
last_extraction_time = time.time()

# Flask app setup
app = Flask(__name__)

# Matplotlib plot setup
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
line1, = ax1.plot([], [], lw=2, color='b')  # Real-time ECG plot
line2, = ax2.plot([], [], lw=2, color='r')  # Extracted beat plot
ax1.set_title("Real-Time ECG Plot")
ax1.set_xlabel("Sample Number")
ax1.set_ylabel("ECG Value")
ax1.grid(True)
ax2.set_title("Latest Extracted ECG Beat")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Normalized Amplitude")
ax2.set_xlim(0, FIXED_LENGTH / FS)
ax2.set_ylim(0, 1)
ax2.grid(True)

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
    # Select R-peaks with normalized value >= 0.9
    r_peak_candidates = [i for i in local_max_indices if normalized_signal[i] >= 0.8]
    if not r_peak_candidates:
        return None
    # Compute median R-R interval
    rr_intervals = np.diff(r_peak_candidates)
    T = np.median(rr_intervals) if len(rr_intervals) > 0 else 0
    if T == 0:
        return None
    # Extract the first beat
    r = r_peak_candidates[0]  # First R-peak as requested
    L = int(np.round(1.2 * T))
    segment = normalized_signal[r : r + L]
    if len(segment) < FIXED_LENGTH:
        pad_length = FIXED_LENGTH - len(segment)
        padded_segment = np.pad(segment, (0, pad_length), 'constant')
    else:
        padded_segment = segment[:FIXED_LENGTH]
    return padded_segment

from scipy.interpolate import interp1d
def interpolate_to_fixed_length(y, target_len=187, duration=1.5):
    original_len = len(y)
    # Original time values based on input length
    x_original = np.linspace(0, duration, original_len)
    # Target time values for fixed length
    x_target = np.linspace(0, duration, target_len)

    # Interpolation function
    f = interp1d(x_original, y, kind='linear')  # or 'cubic' for smoother result
    y_resampled = f(x_target)

    return y_resampled

def ecg_reader():
    """Read ECG data from serial port and update shared data."""
    global sample_count, last_extraction_time, latest_beat, latest_beat_time
    print("Starting ECG reader thread...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud")
    except serial.SerialException as e:
        print(f"Failed to connect to {SERIAL_PORT}: {e}")
        sys.exit(1)
    
    try:
        while True:
            while ser.in_waiting:
                try:
                    line_raw = ser.readline().decode('utf-8').strip()
                    if line_raw:
                        y = float(line_raw)
                        with lock:
                            y_data.append(y)
                            x_data.append(sample_count)
                            sample_count += 1
                except (ValueError, UnicodeDecodeError):
                    print(f"Invalid data: {line_raw}")
            
            
            current_time = time.time()
            if current_time - last_extraction_time >= 1.0:
                with lock:
                    signal = list(y_data)
                beat = extract_beat(signal, FS)
                beat = interpolate_to_fixed_length(beat) if beat is not None else None
                print(len(beat))
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

def init_plot():
    """Initialize plots with empty data."""
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

def update_plot(frame):
    """Update the real-time and extracted beat plots."""
    with lock:
        if y_data:
            line1.set_data(list(x_data), list(y_data))
            ax1.relim()
            ax1.autoscale_view()
        if latest_beat is not None:
            line2.set_data(latest_beat_time, latest_beat)
            ax2.relim()
            ax2.autoscale_view()
    return line1, line2

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

def run_flask():
    """Run the Flask app."""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    # Start ECG reader in a separate thread
    reader_thread = threading.Thread(target=ecg_reader, daemon=True)
    reader_thread.start()
    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    # Run matplotlib animation in the main thread
    try:
        ani = animation.FuncAnimation(
            fig, update_plot, init_func=init_plot, blit=True, interval=100, cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        print("Plotting stopped by user")