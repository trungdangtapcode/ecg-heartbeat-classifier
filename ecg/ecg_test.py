import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import sys

# Serial port configuration
SERIAL_PORT = 'COM3'  # Replace with your port (e.g., '/dev/ttyUSB0' on Linux)
BAUD_RATE = 9600      # Match your Arduino’s baud rate
TIMEOUT = 1           # Seconds

# Initialize serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud")
except serial.SerialException as e:
    print(f"Failed to connect to {SERIAL_PORT}: {e}")
    sys.exit(1)

# Data buffer (deque for efficient fixed-size buffer)
MAX_POINTS = 1000     # Show 1000 points (e.g., 10 seconds at 100 Hz)
y_data = deque(maxlen=MAX_POINTS)  # ECG values
x_data = deque(maxlen=MAX_POINTS)  # Sample indices
sample_count = 0

# Create plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2, color='b')
ax.set_title("Real-Time ECG Plot")
ax.set_xlabel("Sample Number")
ax.set_ylabel("ECG Value")
ax.grid(True)

def init():
    """Initialize the plot with empty data."""
    line.set_data([], [])
    return line,

def update(frame):
    """Update the plot with all available serial data."""
    global sample_count
    while ser.in_waiting:  # Read all available data
        try:
            line_raw = ser.readline().decode('utf-8').strip()
            if line_raw:
                y = float(line_raw)  # Convert to float
                y_data.append(y)
                x_data.append(sample_count)
                sample_count += 1
        except (ValueError, UnicodeDecodeError):
            print(f"Invalid data: {line_raw}")
    
    if y_data:
        line.set_data(x_data, y_data)
        ax.relim()          # Recalculate limits
        ax.autoscale_view() # Auto-scale the view
    
    return line,

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