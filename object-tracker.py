## Author: Aman Mishra (Updated)
## Date: November 2025
## Module: PDE 4446 - Robot Sensing and Control
## Description: Real-time object tracking using color segmentation and Serial Control.

import cv2
import numpy as np
import serial
import time

# --- Configuration ---
# CHANGE THIS to match your Arduino's port (e.g., 'COM3' on Windows, '/dev/ttyACM0' on Linux/Mac)
SERIAL_PORT = 'COM8' 
BAUD_RATE = 9600

# Servo Control Variables
# Sensitivity: How much the servo moves per frame based on error (Lower = smoother/slower, Higher = twitchy/fast)
PAN_SENSITIVITY = 0.05
TILT_SENSITIVITY = 0.05

# Invert controls if the servos move the wrong way
# Set to 1 for standard, -1 to invert direction
PAN_DIR = -1 
TILT_DIR = 1

# Current Position State (Starts at Center 0.0)
current_pan = 0.0
current_tilt = 0.0

# Initialize Serial Communication
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
        # Format: "val1 val2\n" -> "0.00 0.00\n"
        command = f"{pan:.4f} {tilt:.4f}\n"
        ser.write(command.encode('utf-8'))

def clamp(n, minn, maxn):
    """Helper to keep values between min and max."""
    return max(min(maxn, n), minn)

def main():
    global current_pan, current_tilt

    # Initialize webcam
    cap = cv2.VideoCapture(1)

    # Create a window named 'Mask & Controls'
    cv2.namedWindow("Mask & Controls")
    cv2.resizeWindow("Mask & Controls", 400, 500)

    # --- Create Trackbars ---
    # Default values set for a specific object (adjust as needed)
    cv2.createTrackbar("Hue Min", "Mask & Controls", 0, 179, nothing)
    cv2.createTrackbar("Hue Max", "Mask & Controls", 10, 179, nothing)
    cv2.createTrackbar("Sat Min", "Mask & Controls", 100, 255, nothing)
    cv2.createTrackbar("Sat Max", "Mask & Controls", 255, 255, nothing)
    cv2.createTrackbar("Val Min", "Mask & Controls", 115, 255, nothing)
    cv2.createTrackbar("Val Max", "Mask & Controls", 255, 255, nothing)

    cv2.createTrackbar("Kernel Size", "Mask & Controls", 5, 20, nothing)
    cv2.createTrackbar("Iterations", "Mask & Controls", 2, 10, nothing)

    print("Tracker Started. Press 'q' or 'ESC' to exit.")

    while True:
        # 1. Read Frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Get dimensions
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
        object_detected = False

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 500:
                object_detected = True
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2)

                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
                    cv2.arrowedLine(frame, (screen_center_x, screen_center_y), (cx, cy), (0, 255, 0), 3)

                    # --- TRACKING LOGIC ---
                    
                    # Calculate normalized error (-1.0 to 1.0) relative to screen size
                    # Error X: -1 (Left) to +1 (Right)
                    error_x = (cx - screen_center_x) / (width / 2)
                    
                    # Error Y: -1 (Top) to +1 (Bottom)
                    error_y = (cy - screen_center_y) / (height / 2)

                    # Update Servo Positions based on error (Proportional Control)
                    # We subtract/add a fraction of the error to the current position
                    current_pan += (error_x * PAN_SENSITIVITY * PAN_DIR)
                    current_tilt += (error_y * TILT_SENSITIVITY * TILT_DIR)

                    # Clamp values to valid Arduino input range (-1.0 to 1.0)
                    current_pan = clamp(current_pan, -1.0, 1.0)
                    current_tilt = clamp(current_tilt, -1.0, 1.0)

                    # Send to Arduino
                    send_to_arduino(current_pan, current_tilt)

                    # Display Data on Screen
                    cv2.putText(frame, f"Err X: {error_x:.2f} Y: {error_y:.2f}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Servo: {current_pan:.2f} {current_tilt:.2f}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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