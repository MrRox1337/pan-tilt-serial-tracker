## Author: Aman Mishra
## Date: December 2025
## Module: PDE 4446 - Motion Control and Sensing
## Lecturers: Dr. Judhi Prasetyo and Dr. Sameer Kishore
## Description: Real-time object tracking with Kalman Filtering, Coasting, and Data Association (Gating).
##
## Detailed Overview:
## This script implements a robust 2-axis servo tracker for a colored object.
## It overcomes common tracking issues using three advanced techniques:
## 1. Kalman Filtering: Smooths out jittery measurements and estimates velocity.
## 2. Coasting (Blind Tracking): Continues moving in the last known direction if the object is briefly obstructed.
## 3. Data Association (Gating): Prevents the tracker from snapping to a different object (e.g., a second ball)
##    by ignoring detections that are too far from the predicted position.

import cv2
import numpy as np
import serial
import time

# --- Configuration Constants ---
SERIAL_PORT = 'COM8' 
BAUD_RATE = 9600
SERIAL_WRITE_INTERVAL = 0.10  # 10Hz update rate

# Servo Control Constants
PAN_SENSITIVITY = 0.006
TILT_SENSITIVITY = 0.006
DEADBAND_THRESHOLD = 0.05 
PAN_DIR = 1 
TILT_DIR = -1

# Tracking Constants
MAX_COAST_FRAMES = 30     # ~1 second at 30fps
MAX_TRACKING_GATE = 150   # Max pixel jump allowed between frames


# --- CLASSES ---

class KalmanFilter:
    """
    A wrapper around OpenCV's KalmanFilter for 2D object tracking.
    State Vector: [x, y, dx, dy] (Position and Velocity)
    Measurement Vector: [x, y] (We only measure position)
    """
    def __init__(self):
        # 4 dynamic params (state), 2 measurement params
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Measurement Matrix (H): Maps state to measurement.
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                              [0, 1, 0, 0]], np.float32)
        
        # Transition Matrix (A): Models the physics of motion (Constant Velocity).
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                             [0, 1, 0, 1], 
                                             [0, 0, 1, 0], 
                                             [0, 0, 0, 1]], np.float32)

    def predict(self):
        ''' Predicts the next state based on current state and velocity. '''
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[1])

    def correct(self, x, y):
        ''' Corrects the state with the actual measurement. '''
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)


class ServoController:
    """
    Handles Serial communication and Servo position state.
    """
    def __init__(self, port, baud):
        self.pan = 0.0
        self.tilt = 0.0
        self.last_send_time = 0
        self.connected = False
        
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2) # Allow Arduino reset
            print(f"Connected to Arduino on {port}")
            self.connected = True
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            self.ser = None

    def update_and_transmit(self, target_x, target_y, screen_w, screen_h):
        """Calculates error, updates state, and sends to Arduino (throttled)."""
        screen_center_x = screen_w // 2
        screen_center_y = screen_h // 2

        # Calculate normalized error (-1.0 to 1.0)
        error_x = (target_x - screen_center_x) / (screen_w / 2)
        error_y = (target_y - screen_center_y) / (screen_h / 2)

        # Apply Deadband and Sensitivity
        if abs(error_x) > DEADBAND_THRESHOLD:
            self.pan += (error_x * PAN_SENSITIVITY * PAN_DIR)
        if abs(error_y) > DEADBAND_THRESHOLD:
            self.tilt += (error_y * TILT_SENSITIVITY * TILT_DIR)

        # Clamp values
        self.pan = max(min(1.0, self.pan), -1.0)
        self.tilt = max(min(1.0, self.tilt), -1.0)

        self._transmit()

    def reset_to_center(self):
        """Resets servos to (0,0) and transmits immediately."""
        self.pan = 0.0
        self.tilt = 0.0
        self._transmit()

    def _transmit(self):
        """Internal method to send data if throttling interval has passed."""
        current_time = time.time()
        if (current_time - self.last_send_time) >= SERIAL_WRITE_INTERVAL:
            if self.connected and self.ser.is_open:
                command = f"{self.pan:.4f} {self.tilt:.4f}\n"
                self.ser.write(command.encode('utf-8'))
            self.last_send_time = current_time
    
    def close(self):
        if self.connected and self.ser:
            self.ser.close()


class VisionSystem:
    """
    Handles OpenCV Windows, Trackbars, and Image Processing Pipeline.
    """
    def __init__(self):
        self.window_name = "Mask & Controls"
        cv2.namedWindow(self.window_name)
        cv2.resizeWindow(self.window_name, 400, 500)
        self._create_trackbars()

    def _nothing(self, x): pass

    def _create_trackbars(self):
        cv2.createTrackbar("Hue Min", self.window_name, 155, 179, self._nothing)
        cv2.createTrackbar("Hue Max", self.window_name, 179, 179, self._nothing)
        cv2.createTrackbar("Sat Min", self.window_name, 103, 255, self._nothing)
        cv2.createTrackbar("Sat Max", self.window_name, 255, 255, self._nothing)
        cv2.createTrackbar("Val Min", self.window_name, 100, 255, self._nothing)
        cv2.createTrackbar("Val Max", self.window_name, 255, 255, self._nothing)
        cv2.createTrackbar("Kernel Size", self.window_name, 5, 20, self._nothing)
        cv2.createTrackbar("Iterations", self.window_name, 2, 10, self._nothing)

    def process_frame(self, frame):
        """
        Takes a raw frame, applies HSV/Morph filters, and finds object candidates.
        Returns: (processed_mask, list_of_candidates)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get Trackbar Values
        h_min = cv2.getTrackbarPos("Hue Min", self.window_name)
        h_max = cv2.getTrackbarPos("Hue Max", self.window_name)
        s_min = cv2.getTrackbarPos("Sat Min", self.window_name)
        s_max = cv2.getTrackbarPos("Sat Max", self.window_name)
        v_min = cv2.getTrackbarPos("Val Min", self.window_name)
        v_max = cv2.getTrackbarPos("Val Max", self.window_name)
        k_size = cv2.getTrackbarPos("Kernel Size", self.window_name)
        iters = cv2.getTrackbarPos("Iterations", self.window_name)

        if k_size < 1: k_size = 1
        if k_size % 2 == 0: k_size += 1

        # Create Mask
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological Ops
        kernel = np.ones((k_size, k_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)

        # Extract Candidates
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        if contours:
            for cnt in contours:
                if cv2.contourArea(cnt) > 500:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        candidates.append({'cnt': cnt, 'cx': cx, 'cy': cy})
        
        return mask, candidates


# --- MAIN LOGIC ---

def main():
    # 1. Initialization
    cap = cv2.VideoCapture(1)
    kf = KalmanFilter()
    servo = ServoController(SERIAL_PORT, BAUD_RATE)
    vision = VisionSystem()

    print("Tracker Started. Press 'q' or 'ESC' to exit.")

    # State Variables
    frames_without_detection = MAX_COAST_FRAMES 
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        screen_center_x, screen_center_y = w // 2, h // 2

        # 2. Vision Processing
        mask, candidates = vision.process_frame(frame)

        # 3. Kalman Prediction Step (Always run this!)
        pred_x, pred_y = kf.predict()

        matched_candidate = None
        target_x, target_y = None, None
        tracking_status = "Scanning"

        # 4. Data Association (Intelligent Selection Logic)
        if candidates:
            # Case A: We are Tracking/Coasting -> Use Gating
            if frames_without_detection < MAX_COAST_FRAMES:
                # Find candidate closest to Prediction
                best = min(candidates, key=lambda c: (c['cx']-pred_x)**2 + (c['cy']-pred_y)**2)
                dist = ((best['cx']-pred_x)**2 + (best['cy']-pred_y)**2)**0.5
                
                if dist < MAX_TRACKING_GATE:
                    matched_candidate = best
                # Else: Ignore (it's likely a distractor)

            # Case B: Lost/Initializing -> Pick Largest
            else:
                matched_candidate = max(candidates, key=lambda c: cv2.contourArea(c['cnt']))

        # 5. Measurement Update (Correction)
        if matched_candidate:
            cx, cy = matched_candidate['cx'], matched_candidate['cy']
            kf.correct(cx, cy)
            
            frames_without_detection = 0
            # Drive servos using Prediction (smoother) or Actual (cx, cy)
            target_x, target_y = pred_x, pred_y 
            tracking_status = "Locked"

            # Visuals
            cv2.drawContours(frame, [matched_candidate['cnt']], -1, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) # Raw
            cv2.circle(frame, (pred_x, pred_y), 5, (255, 0, 0), -1) # Pred

        # 6. Coasting Logic
        if target_x is None:
            if frames_without_detection < MAX_COAST_FRAMES:
                frames_without_detection += 1
                target_x, target_y = pred_x, pred_y
                tracking_status = f"Coasting ({frames_without_detection})"
                
                # Visuals
                cv2.circle(frame, (pred_x, pred_y), 10, (0, 255, 255), 2)
                cv2.putText(frame, "LOST - PREDICTING", (pred_x+15, pred_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.circle(frame, (pred_x, pred_y), MAX_TRACKING_GATE, (50, 50, 50), 1)
            else:
                tracking_status = "Lost - Returning to Center"
                servo.reset_to_center()

        # 7. Servo Control
        if target_x is not None:
            servo.update_and_transmit(target_x, target_y, w, h)
            cv2.arrowedLine(frame, (screen_center_x, screen_center_y), (target_x, target_y), (0, 255, 0), 2)

        # 8. UI Overlay
        cv2.putText(frame, f"Status: {tracking_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Servo: {servo.pan:.2f} {servo.tilt:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.circle(frame, (screen_center_x, screen_center_y), 5, (255, 0, 0), -1)
        
        cv2.imshow(vision.window_name, mask)
        cv2.imshow("Object Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    servo.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()