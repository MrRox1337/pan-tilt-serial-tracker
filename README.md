# Red Object Tracker using OpenCV

This project implements a real-time computer vision application that detects, tracks, and visualizes a specific red object using a webcam. It utilizes **OpenCV** for color space filtering, morphological operations, and contour analysis to robustly identify the object and calculate its spatial offset from the center of the camera frame.

## 📋 Requirements

The primary goal of this tracker is to:

1.  **Isolate a specific object** (a red cube/sphere) from a video feed based on color.
2.  **Filter out noise** and imperfect lighting conditions using morphological transformations.
3.  **Track the object's position** by calculating the centroid of the largest detected contour.
4.  **Visualize the offset** by drawing a vector from the center of the screen to the object.
5.  **Allow real-time tuning** via a GUI with trackbars for HSV thresholds and morphological kernel sizes.

## 🛠️ Implementation Plan

The pipeline follows a standard computer vision workflow:

1.  **Input:** Capture video frames from the webcam.
2.  **Preprocessing:** Convert frames from BGR (standard OpenCV format) to HSV (Hue, Saturation, Value) color space, which is more robust to lighting changes.
3.  **Thresholding:** Create a binary mask where pixels falling within the specified "Red" HSV range are white, and others are black.
4.  **Morphology:** Apply "Opening" (to remove background noise) and "Closing" (to fill holes within the object) to ensure a solid mask.
5.  **Analysis:** Find contours in the mask, select the largest one (assumed to be the target object), and calculate its centroid (center of mass).
6.  **Visualization:** Draw the contour, the centroid, and a vector line indicating distance and direction from the image center.

## 🚀 Installation & Usage

### Prerequisites

Ensure you have Python installed, then install the required dependencies:

```
pip install opencv-python numpy
```

### Running the Tracker

Run the script directly from your terminal:

```
python red_object_tracker.py
```

-   **Controls:** Use the trackbars in the "Mask & Controls" window to tune the detection.
-   **Exit:** Press `q` or `ESC` to stop the program.

## 🧠 Methodology & Code Analysis

### 1\. Color Space Filtering (RGB to HSV)

We convert the input frame to HSV because separating color intensity (Value) from color information (Hue) makes the tracker less sensitive to shadows.

```
# Convert the frame to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define the range of the color to track
lower_bound = np.array([h_min, s_min, v_min])
upper_bound = np.array([h_max, s_max, v_max])

# Create a binary mask
mask = cv2.inRange(hsv, lower_bound, upper_bound)
```

### 2\. Morphological Operations

Raw color masking often results in "salt and pepper" noise. We use **Erosion** and **Dilation** to clean the mask.

-   **Opening:** Erosion followed by Dilation. Removes small noise blobs.
-   **Closing:** Dilation followed by Erosion. Fills small holes inside the object.

```
# Create a kernel (structuring element) based on trackbar input
kernel = np.ones((k_size, k_size), np.uint8)

# Remove noise (Opening)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)

# Fill holes (Closing)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
```

### 3\. Contour Detection & Largest Object Selection

To prevent tracking small background distractions, we find all contours and filter for the largest one.

```
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Identify the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw it on the frame
    if cv2.contourArea(largest_contour) > 500:
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2)
```

### 4\. Centroid Calculation & Vector Visualization

We calculate the **Image Moments** to find the center of mass (cx, cy). A vector is then drawn from the screen center to this point.

```
M = cv2.moments(largest_contour)
if M["m00"] != 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Draw the centroid
    cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)

    # Draw vector from screen center (screen_center_x, screen_center_y) to object
    cv2.arrowedLine(frame, (screen_center_x, screen_center_y), (cx, cy), (0, 255, 0), 3)
```

## 🎛️ Fine-Tuning Guide

The application launches with default values tuned for a standard red object. Use the trackbars to adjust:

| Parameter         | Description                                                                     |
| ----------------- | ------------------------------------------------------------------------------- |
| **Hue (Min/Max)** | The "Color" type. Red wraps around 0 and 180.                                   |
| **Sat (Min/Max)** | Saturation. Higher values = more vivid colors. Lower values = whiter/grayer.    |
| **Val (Min/Max)** | Brightness. Lower values = darker. Higher values = brighter.                    |
| **Kernel Size**   | Increases the smoothing effect. Larger kernel = smoother blobs but less detail. |
| **Iterations**    | How aggressively to apply the smoothing.                                        |

---

_Generated for Red Object Tracker Project_
