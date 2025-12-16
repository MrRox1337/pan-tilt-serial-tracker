## Author: Aman Mishra (Updated)
## Date: November 2025
## Module: PDE 4446 - Robot Sensing and Control
## Lecturers: Dr. Judhi Prasetyo and Dr. Sameer Kishore
## Description: Real-time object tracking with Kalman Filtering and "Coasting" (Blind Tracking).

import cv2
import numpy as np
import serial
import time

# --- Configuration ---
SERIAL_PORT = 'COM8' 
BAUD_RATE = 9600
SERIAL_WRITE_INTERVAL = 0.10

# Servo Control Variables
PAN_SENSITIVITY = 0.012
TILT_SENSITIVITY = 0.012
DEADBAND_THRESHOLD = 0.05 

# Coasting Configuration (Blind Tracking)
MAX_COAST_FRAMES = 30  # How many frames to keep moving after object is lost (~1 second)

PAN_DIR = 1 
TILT_DIR = -1

current_pan = 0.0
current_tilt = 0.0

# --- KALMAN FILTER CLASS (Refactored) ---
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

    def predict(self):
        ''' Predicts the next state based on previous velocity (Blind) '''
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y

    def correct(self, coordX, coordY):
        ''' Corrects the state with a new actual measurement '''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)

# --- SERIAL SETUP ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"Error connecting to serial port: {e}")
    ser = None

def nothing(x):
    pass

def send_to_arduino(pan, tilt):
    if ser and ser.is_open:
        command = f"{pan:.4f} {tilt:.4f}\n"
        ser.write(command.encode('utf-8'))

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def main():
    global current_pan, current_tilt

    cap = cv2.VideoCapture(1)
    kf = KalmanFilter()

    cv2.namedWindow("Mask & Controls")
    cv2.resizeWindow("Mask & Controls", 400, 500)

    # Default Trackbars
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
    frames_without_detection = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        screen_center_x = width // 2
        screen_center_y = height // 2

        # Color & Masking
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
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

        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        kernel = np.ones((k_size, k_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- KEY CHANGE: PREDICT EVERY FRAME ---
        # We ask the filter: "Based on previous speed, where is it now?"
        pred_x, pred_y = kf.predict()

        target_x, target_y = None, None
        tracking_status = "Scanning"

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                # 1. MEASUREMENT: We found the object!
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 2. CORRECT: Tell the filter the ACTUAL location
                    kf.correct(cx, cy)
                    
                    # Reset lost counter
                    frames_without_detection = 0
                    target_x, target_y = pred_x, pred_y
                    tracking_status = "Locked"

                    # Visualization
                    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) # Red = Raw
                    cv2.circle(frame, (pred_x, pred_y), 5, (255, 0, 0), -1) # Blue = Filtered

        # --- COASTING LOGIC ---
        if target_x is None:
            # We didn't see the object. Should we coast?
            if frames_without_detection < MAX_COAST_FRAMES:
                # Yes! Use the pure prediction to lead the camera
                frames_without_detection += 1
                target_x, target_y = pred_x, pred_y
                tracking_status = f"Coasting ({frames_without_detection})"
                
                # Visual Warning
                cv2.circle(frame, (pred_x, pred_y), 10, (0, 255, 255), 2) # Yellow Empty Circle
                cv2.putText(frame, "LOST SIGNAL - PREDICTING", (pred_x + 15, pred_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                tracking_status = "Lost - Returning to Center"
                # Safety Cutoff Triggered: Reset servos to center
                current_pan = 0.0
                current_tilt = 0.0

        # --- SERVO CONTROL ---
        pan_update = 0.0
        tilt_update = 0.0

        if target_x is not None:
            # Calculate error based on whatever target we have (Locked or Coasting)
            error_x = (target_x - screen_center_x) / (width / 2)
            error_y = (target_y - screen_center_y) / (height / 2)

            if abs(error_x) > DEADBAND_THRESHOLD:
                pan_update = (error_x * PAN_SENSITIVITY * PAN_DIR)
            if abs(error_y) > DEADBAND_THRESHOLD:
                tilt_update = (error_y * TILT_SENSITIVITY * TILT_DIR)

            current_pan += pan_update
            current_tilt += tilt_update
            
            # Draw Vector
            cv2.arrowedLine(frame, (screen_center_x, screen_center_y), (target_x, target_y), (0, 255, 0), 2)

        current_pan = clamp(current_pan, -1.0, 1.0)
        current_tilt = clamp(current_tilt, -1.0, 1.0)

        # Serial Write
        current_time = time.time()
        if (current_time - last_serial_send_time) >= SERIAL_WRITE_INTERVAL:
            send_to_arduino(current_pan, current_tilt)
            last_serial_send_time = current_time

        # UI Info
        cv2.putText(frame, f"Status: {tracking_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Servo: {current_pan:.2f} {current_tilt:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.circle(frame, (screen_center_x, screen_center_y), 5, (255, 0, 0), -1)
        cv2.imshow("Mask & Controls", mask)
        cv2.imshow("Object Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if ser: ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()