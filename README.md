# Intelligent 2-Axis Pan-Tilt Object Tracker

## 1. Project Overview

This software controls a 2-DOF (Degrees of Freedom) robotic pan-tilt mechanism to
autonomously track a specific colored object in real-time. Unlike simple trackers
that rely solely on immediate sensor data, this system implements a **Kalman Filter** to model the physical state (position and velocity) of the target. This allows the
robot to "understand" motion, enabling it to track objects even when they are
partially occluded or moving rapidly.

## 2. Key Algorithms & Features

### A. Robust Color Segmentation (VisionSystem Class)

- **HSV Transformation:** Converts RGB webcam feed to Hue-Saturation-Value space to
  isolate color from lighting intensity.
- **Morphological Filtering:** Applies 'Opening' (Erosion->Dilation) to remove noise
  and 'Closing' (Dilation->Erosion) to solidify the target blob.

### B. State Estimation (KalmanFilter Class)

- **Model:** Constant Velocity Model.
- **State Vector:** $[x, y, v_x, v_y]$
- **Prediction:** Estimates where the object _should_ be in the current frame based
  on previous velocity.
- **Correction:** Refines the estimate using the actual centroid measured by the
  vision system.

### C. Occlusion Handling ("Coasting")

- When the object is lost (e.g., passes behind an obstacle), the system stops
  correcting the Kalman Filter but continues _predicting_.
- The servos are driven by this prediction for up to 30 frames (~1 second),
  allowing the camera to "lead" the target until it reappears.

### D. Data Association ("Gating")

- To handle multiple similar objects (e.g., two red balls), the system uses a
  spatial gate.
- **Logic:** Only measurements within `MAX_TRACKING_GATE` pixels of the predicted
  location are accepted. Distant distractors are mathematically rejected.

### E. Control Logic (ServoController Class)

- **Proportional Control:** Calculates error based on the distance from the screen center.
- **Deadband Threshold:** A 5% zone around the center where no movement commands are
  sent to prevent mechanical oscillation/jitter.

## 3. Hardware Requirements

- **Microcontroller:** Arduino Uno (or compatible) running StandardFirmata or a
  simple serial listener sketch.
- **Actuators:** 2x Micro Servos (SG90 or MG996R) mounted in a Pan-Tilt bracket.
- **Camera:** Standard USB Webcam.

## 4. Software Dependencies

Ensure you have a Python 3.x environment with the following libraries:

```bash
pip install opencv-python numpy pyserial
```

## 5. Configuration Guide

Adjust the constants at the top of the script to match your hardware:

- `SERIAL_PORT`: 'COMx' (Windows) or '/dev/ttyUSBx' (Linux/Mac).
- `PAN_SENSITIVITY` / `TILT_SENSITIVITY`: Increase for faster response, decrease for smoothness.
- `PAN_DIR` / `TILT_DIR`: Set to -1 if servos move in the opposite direction.

## 6. Operation

1. Connect the Arduino via USB.
2. Run the script: `python object-tracker.py`
3. A "Mask & Controls" window will appear. Adjust the `Hue`, `Sat`, and `Val` sliders
   until your target object is white and the background is black.
4. The "Object Tracker" window will show the live feed:
   - **Yellow Contour:** The detected object.
   - **Red Dot:** Raw measurement.
   - **Blue Dot:** Kalman Prediction.
   - **Green Arrow:** Vector driving the servos.
5. Press 'q' or 'ESC' to exit safely.

## 7. Video Demonstration

[![Video demonstration with explanation (5:56)](https://img.youtube.com/vi/ZYJHcxjyrwg/hqdefault.jpg)](https://youtu.be/ZYJHcxjyrwg)

## 7. Troubleshooting

- **Jittering Servos:** Decrease `SENSITIVITY` or increase `DEADBAND_THRESHOLD`.
- **Lags behind object:** Increase `SENSITIVITY`.
- **Drifts when lost:** The "Coasting" feature is working. If it drifts too far,
  reduce `MAX_COAST_FRAMES`.
- **Snaps to wrong object:** Reduce `MAX_TRACKING_GATE`.
