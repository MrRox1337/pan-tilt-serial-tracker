## Author: Aman Mishra (Updated)
## Date: November 2025
## Module: PDE 4446 - Robot Sensing and Control
## Lecturers: Dr. Judhi Prasetyo and Dr. Sameer Kishore
## Description: Real-time object tracking using color segmentation, Kalman Filtering, and Serial Control.

import cv2
import numpy as np
import serial
import time

# --- Configuration ---
# CHANGE THIS to match Arduino's port ('COM8' on Windows, '/dev/ttyACM0' on Linux/Mac)
SERIAL_PORT = 'COM8' 
BAUD_RATE = 9600
SERIAL_WRITE_INTERVAL = 0.10  # 100 milliseconds in seconds

# Servo Control Variables
# Sensitivity: How much the servo moves per frame based on error (Lower = smoother/slower, Higher = twitchy/fast)
PAN_SENSITIVITY = 0.012
TILT_SENSITIVITY = 0.012

# Deadband Threshold (Normalized error range where no adjustment is made)
DEADBAND_THRESHOLD = 0.05 

# Invert controls if the servos move the wrong way
PAN_DIR = 1 
TILT_DIR = -1

# Current Position State (Starts at Center 0.0)
current_pan = 0.0
current_tilt = 0.0

# --- KALMAN FILTER CLASS ---
class KalmanFilter:
    def __init__(self):
        # 4 dynamic params (x, y, dx, dy), 2 measurement params (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                             [0, 1, 0, 1], 
                                             [0, 0, 1, 0], 
                                             [0, 0, 0, 1]], np.float32)

    def predict(self, coordX, coordY):
        ''' estimates the position of the object '''
        # Correct the state with the latest measurement
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        # Predict the next state
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y

# --- SERIAL SETUP ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) # Wait for Arduino to reset
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"Error connecting to serial port: {e}")
    ser = None

def nothing(x):
    """Dummy function required for trackbar creation."""
    pass

def send_to_arduino(pan, tilt):
    """Sends the pan and tilt values to Arduino via Serial."""
    if ser and ser.is_open:
        command = f"{pan:.4f} {tilt:.4f}\n"
        ser.write(command.encode('utf-8'))

def clamp(n, minn, maxn):
    """Helper to keep values between min and max."""
    return max(min(maxn, n), minn)

def main():
    global current_pan, current_tilt

    # Initialize webcam
    cap = cv2.VideoCapture(1)

    # Initialize Kalman Filter
    kf = KalmanFilter()

    # Create a window named 'Mask & Controls'
    cv2.namedWindow("Mask & Controls")
    cv2.resizeWindow("Mask & Controls", 400, 500)

    # --- Create Trackbars ---
    cv2.createTrackbar("Hue Min", "Mask & Controls", 155, 179, nothing)
    cv2.createTrackbar("Hue Max", "Mask & Controls", 179, 179, nothing)
    cv2.createTrackbar("Sat Min", "Mask & Controls", 161, 255, nothing)
    cv2.createTrackbar("Sat Max", "Mask & Controls", 255, 255, nothing)
    cv2.createTrackbar("Val Min", "Mask & Controls", 100, 255, nothing)
    cv2.createTrackbar("Val Max", "Mask & Controls", 255, 255, nothing)

    cv2.createTrackbar("Kernel Size", "Mask & Controls", 5, 20, nothing)
    cv2.createTrackbar("Iterations", "Mask & Controls", 2, 10, nothing)

    print("Tracker Started. Press 'q' or 'ESC' to exit.")

    last_serial_send_time = 0

    while True:
        # 1. Read Frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        screen_center_x = width // 2
        screen_center_y = height // 2

        # 2. Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 3. Get trackbar positions
        h_min = cv2.getTrackbarPos("Hue Min", "Mask & Controls")
        h_max = cv2.getTrackbarPos("Hue Max", "Mask & Controls")
        s_min = cv2.getTrackbarPos("Sat Min", "Mask & Controls")
        s_max = cv2.getTrackbarPos("Sat Max", "Mask & Controls")
        v_min = cv2.getTrackbarPos("Val Min", "Mask & Controls")
        v_max = cv2.getTrackbarPos("Val Max", "Mask & Controls")
        
        k_size = cv2.getTrackbarPos("Kernel Size", "Mask & Controls")
        iters = cv2.getTrackbarPos("Iterations", "Mask & Controls")

        if k_size < 1: k_size = 1
        if k_size % 2 == 0: k_size += 1

        # 4. Create Mask
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # 5. Morphological Operations
        kernel = np.ones((k_size, k_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)

        # 6. Contour Detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = 0, 0
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 500:
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2)

                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    # Calculate raw centroid
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # --- KALMAN FILTER PREDICTION ---
                    # Predict the next position based on the current measurement
                    pred_x, pred_y = kf.predict(cx, cy)

                    # Draw Raw (Red) and Predicted (Blue) positions
                    cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1) # Raw (Red)
                    cv2.circle(frame, (pred_x, pred_y), 7, (255, 0, 0), -1) # Predicted (Blue)
                    
                    # Draw vector to PREDICTED position (Green)
                    cv2.arrowedLine(frame, (screen_center_x, screen_center_y), (pred_x, pred_y), (0, 255, 0), 3)

                    # --- TRACKING LOGIC (Using Predicted Coordinates) ---
                    # We use pred_x/pred_y for smoother control
                    error_x = (pred_x - screen_center_x) / (width / 2)
                    error_y = (pred_y - screen_center_y) / (height / 2)
                    
                    # --- DEADZONE ---
                    pan_update = 0.0
                    if abs(error_x) > DEADBAND_THRESHOLD:
                        pan_update = (error_x * PAN_SENSITIVITY * PAN_DIR)
                        
                    tilt_update = 0.0
                    if abs(error_y) > DEADBAND_THRESHOLD:
                        tilt_update = (error_y * TILT_SENSITIVITY * TILT_DIR)

                    current_pan += pan_update
                    current_tilt += tilt_update

                    current_pan = clamp(current_pan, -1.0, 1.0)
                    current_tilt = clamp(current_tilt, -1.0, 1.0)

                    # --- THROTTLED SERIAL WRITE ---
                    current_time = time.time()
                    if (current_time - last_serial_send_time) >= SERIAL_WRITE_INTERVAL:
                        send_to_arduino(current_pan, current_tilt)
                        last_serial_send_time = current_time

                    # Display Data
                    cv2.putText(frame, f"Err X: {error_x:.2f} Y: {error_y:.2f}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Servo: {current_pan:.2f} {current_tilt:.2f}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    pan_status = "Adjusting" if pan_update != 0.0 else "Deadband"
                    tilt_status = "Adjusting" if tilt_update != 0.0 else "Deadband"
                    cv2.putText(frame, f"Status: {pan_status}", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw Center Crosshair
        cv2.circle(frame, (screen_center_x, screen_center_y), 5, (255, 0, 0), -1)

        cv2.imshow("Mask & Controls", mask)
        cv2.imshow("Object Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    if ser:
        ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()