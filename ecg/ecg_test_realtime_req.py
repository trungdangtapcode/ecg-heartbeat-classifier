import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import threading
import queue
import requests
from scipy.interpolate import interp1d
import time
import sys

# Serial port configuration (adjust as needed)
SERIAL_PORT = 'COM3'  # Replace with your serial port
BAUD_RATE = 9600
FS = 100  # Sampling frequency in Hz
MAX_POINTS = 1000  # 10 seconds of data
FIXED_LENGTH = int(1.5 * FS) 

# Initialize serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
except serial.SerialException as e:
    print(f"Serial connection failed: {e}")
    sys.exit(1)

# Data buffers
y_data = deque(maxlen=MAX_POINTS)
x_data = deque(maxlen=MAX_POINTS)
sample_count = 0
last_extraction_time = time.time()
results_queue = queue.Queue()

# Set up figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
line1, = ax1.plot([], [], lw=2, color='blue', label='ECG')
line2, = ax2.plot([], [], lw=2, color='red', label='Extracted Beat')
ax1.set_title('Real-Time ECG Signal')
ax1.set_xlabel('Sample')
ax1.set_ylabel('Amplitude')
ax1.legend()
ax1.grid(True)
ax2.set_title('Extracted ECG Beat (Interpolated to 187 Points)')
ax2.set_xlabel('Normalized Time')
ax2.set_ylabel('Normalized Amplitude')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(True)
ax3.set_title('Classification Results')
ax3.axis('off')

def init():
    """Initialize the plots."""
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

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
    r = r_peak_candidates[0]  # First R-peak as requested
    L = int(np.round(1.2 * T))
    segment = normalized_signal[r : r + L]
    if len(segment) < FIXED_LENGTH:
        pad_length = FIXED_LENGTH - len(segment)
        padded_segment = np.pad(segment, (0, pad_length), 'constant')
    else:
        padded_segment = segment[:FIXED_LENGTH]
    return padded_segment


def interpolate_to_187(signal):
    """Interpolate the signal to 187 points."""
    if len(signal) < 2:
        return None
    x_original = np.linspace(0, 1, len(signal))
    interpolator = interp1d(x_original, signal, kind='linear')
    x_new = np.linspace(0, 1, 187)
    return interpolator(x_new)

def classify_beat(signal):
    """Send the signal to the server for classification."""
    url = "http://192.168.1.1:5000/classify"
    data = {"signal": signal.tolist()}
    try:
        response = requests.post(url, json=data, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Classification request failed: {e}")
        return None

def classify_in_thread(signal, results_queue):
    """Run classification in a separate thread."""
    result = classify_beat(signal)
    if result:
        results_queue.put(result)

def update(frame):
    """Update the plots with new data and classification."""
    global sample_count, last_extraction_time

    # Read serial data
    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                y = float(line)
                y_data.append(y)
                x_data.append(sample_count)
                sample_count += 1
        except (ValueError, UnicodeDecodeError):
            continue

    # Update real-time ECG plot
    if y_data:
        line1.set_data(x_data, y_data)
        ax1.relim()
        ax1.autoscale_view()

    # Extract and classify beat every second
    current_time = time.time()
    if current_time - last_extraction_time >= 1.0:
        beat = extract_beat(list(y_data), FS)
        if beat is not None:
            beat_interpolated = interpolate_to_187(beat)
            if beat_interpolated is not None:
                # Update beat plot
                time_beat = np.linspace(0, 1, 187)
                line2.set_data(time_beat, beat_interpolated)
                ax2.relim()
                ax2.autoscale_view()
                # Classify in a separate thread
                threading.Thread(target=classify_in_thread, args=(beat_interpolated, results_queue)).start()
        last_extraction_time = current_time

    # Update classification results
    if not results_queue.empty():
        results = results_queue.get()
        ax3.clear()
        ax3.axis('off')
        text = "Classification Results:\n"
        for res in results.get('results', []):
            model = res.get('model', 'Unknown')
            prediction = res.get('prediction', 'N/A')
            text += f"{model}: {prediction}\n"
            if 'probabilities' in res:
                probs = [f"{p:.3f}" for p in res['probabilities']]
                text += f"  Prob: {', '.join(probs)}\n"
            elif 'error' in res:
                text += f"  Error: {res['error']}\n"
        ax3.text(0.1, 0.9, text, fontsize=10, verticalalignment='top', horizontalalignment='left')

    return line1, line2

# Start animation
ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=50, cache_frame_data=False)

try:
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    ser.close()
    print("Serial connection closed")