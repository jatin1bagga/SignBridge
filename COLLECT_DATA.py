import serial
import threading
import pandas as pd
import time

# ✅ Buffers for each glove
left_data = []
right_data = []   
stop_event = threading.Event()

# ✅ Duration for recording
record_duration = 5  # seconds

def read_glove(port_name, glove_name, buffer):
    try:
        ser = serial.Serial(port_name, 115200, timeout=1)
        print(f"[{glove_name}] Connected to {port_name}")
        while not stop_event.is_set():
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if "," in line and not line.startswith("rst:"):  # ✅ only keep valid CSV rows
                    print(f"[{glove_name}] {line}")
                    buffer.append(line)
    except Exception as e:
        print(f"❌ Error on {glove_name}: {e}")

# ✅ COM ports (change as needed)
left_port = "COM8"
right_port = "COM5" 
 
# ✅ Start both threads
left_thread = threading.Thread(target=read_glove, args=(left_port, "LeftGlove", left_data))
right_thread = threading.Thread(target=read_glove, args=(right_port, "RightGlove", right_data))
left_thread.start()
right_thread.start()

# ✅ Wait for specified time 
print(f"⏳ Recording for {record_duration} seconds...")
time.sleep(record_duration)
stop_event.set()
left_thread.join()
right_thread.join()
print("✅ Recording complete.")

# ✅ Save individual files
with open("LeftGlove_data.csv", "w") as f:
    for line in left_data:
        f.write(line + "\n")

with open("RightGlove_data.csv", "w") as f:
    for line in right_data:
        f.write(line + "\n")

# ✅ Combine both CSVs column-wise (left + right)
# ✅ Combine both CSVs safely
try:
    # Read manually and split by comma
    left_rows = []
    with open("LeftGlove_data.csv", "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) > 1:  # Avoid blank or corrupted lines
                left_rows.append(parts)

    right_rows = []
    with open("RightGlove_data.csv", "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) > 1:
                right_rows.append(parts)

    # Trim to shortest length to avoid mismatch
    min_len = min(len(left_rows), len(right_rows))
    left_rows = left_rows[:min_len]
    right_rows = right_rows[:min_len]

    # Combine row-wise
    combined_rows = [l + r for l, r in zip(left_rows, right_rows)]

    # Save to CombinedGlove.csv
    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv("CombinedGlove.csv", index=False, header=False)
    print("✅ Combined CSV saved as CombinedGlove.csv")
except Exception as e:
    print(f"❌ Error while combining CSVs: {e}")
