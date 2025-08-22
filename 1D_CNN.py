import serial
import threading

def read_glove(port_name, glove_name):
    try:
        ser = serial.Serial(port_name, 115200, timeout=1)
        print(f"[{glove_name}] Connected to {port_name}")
        with open(f"{glove_name}_data.csv", "w") as f:
            while True:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    print(f"[{glove_name}] {line}")
                    f.write(line + "\n")
                    f.flush()  # ✅ Write data immediately to file
    except Exception as e:
        print(f"❌ Error on {glove_name}: {e}")

# ✅ Update these based on your actual ports
left_port = "COM5"
right_port = "COM7"

# ✅ Start both readers in separate threads
threading.Thread(target=read_glove, args=(left_port, "LeftGlove")).start()
threading.Thread(target=read_glove, args=(right_port, "RightGlove")).start()
