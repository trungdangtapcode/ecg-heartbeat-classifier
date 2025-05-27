import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import sys
import time

# Serial port configuration
SERIAL_PORT = 'COM6'  # Replace with your port (e.g., '/dev/ttyUSB0' on Linux)
BAUD_RATE = 9600      # Match your device’s baud rate
TIMEOUT = 1           # Seconds
FS = 100              # Sampling frequency in Hz
MAX_POINTS = int(2.0 * FS)     # 10 seconds of data at 100 Hz

# WINDOW_LENGTH = int(10.0 * FS)  # 10 seconds of data
FIXED_LENGTH = int(1.5 * FS)  # Fixed length for extracted beats (1.5 seconds)

# Initialize serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud")
except serial.SerialException as e:
    print(f"Failed to connect to {SERIAL_PORT}: {e}")
    sys.exit(1)

# Data buffers
y_data = deque(maxlen=MAX_POINTS)  # ECG values
x_data = deque(maxlen=MAX_POINTS)  # Sample indices
sample_count = 0
last_extraction_time = time.time()

# Create figure with two subplots
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

def init():
    """Initialize both plots with empty data."""
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

def update(frame):
    """Update the real-time plot and extract a beat every second."""
    global sample_count, last_extraction_time
    # Read all available serial data
    while ser.in_waiting:
        try:
            line_raw = ser.readline().decode('utf-8').strip()
            if line_raw:
                y = float(line_raw)
                y_data.append(y)
                x_data.append(sample_count)

                sample_count += 1
        except (ValueError, UnicodeDecodeError):
            print(f"Invalid data: {line_raw}")
    
    # Check if it's time to extract a beat
    current_time = time.time()
    if current_time - last_extraction_time >= 1.0:
        signal = list(y_data)  # Latest data (up to 10 seconds)
        beat = extract_beat(signal, FS)
        beat = interpolate_to_fixed_length(beat) if beat is not None else None
        print("beat len:",len(beat) if beat is not None else "None")
        if beat is not None:
            time_beat = np.arange(len(beat)) / FS
            line2.set_data(time_beat, beat)
            ax2.relim()
            ax2.autoscale_view()
        last_extraction_time = current_time
    
    # Update real-time plot
    if y_data:
        line1.set_data(x_data, y_data)
        ax1.relim()
        ax1.autoscale_view()
    
    return line1, line2

# Create animation
ani = animation.FuncAnimation(
    fig, update, init_func=init, blit=True, interval=100, cache_frame_data=False
)

try:
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    print("Plotting stopped by user")
finally:
    ser.close()
    print("Serial connection closed")