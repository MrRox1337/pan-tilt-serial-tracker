import cv2
import numpy as np

def nothing(x):
    """Dummy function required for trackbar creation."""
    pass

def main():
    # Initialize webcam
    # '0' is usually the default camera. Change to '1' if using an external USB cam.
    cap = cv2.VideoCapture(0)

    # Create a window named 'Mask & Controls'
    cv2.namedWindow("Mask & Controls")
    cv2.resizeWindow("Mask & Controls", 400, 500)

    # --- Create Trackbars ---
    # Hue is 0-179 in OpenCV. Saturation and Value are 0-255.
    # Default values are set to approximate the RED color in your image.
    cv2.createTrackbar("Hue Min", "Mask & Controls", 0, 179, nothing)
    cv2.createTrackbar("Hue Max", "Mask & Controls", 10, 179, nothing)
    cv2.createTrackbar("Sat Min", "Mask & Controls", 145, 255, nothing)
    cv2.createTrackbar("Sat Max", "Mask & Controls", 255, 255, nothing)
    cv2.createTrackbar("Val Min", "Mask & Controls", 115, 255, nothing)
    cv2.createTrackbar("Val Max", "Mask & Controls", 255, 255, nothing)

    # Trackbars for Morphological Operations
    # Kernel Size: Controls the size of the structuring element (e.g., 5x5)
    # Iterations: How many times to apply the operation
    cv2.createTrackbar("Kernel Size", "Mask & Controls", 5, 20, nothing)
    cv2.createTrackbar("Iterations", "Mask & Controls", 2, 10, nothing)

    print("Tracker Started. Press 'q' or 'ESC' to exit.")

    while True:
        # 1. Read Frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Mirror the frame (optional, feels more natural)
        frame = cv2.flip(frame, 1)
        
        # Get dimensions for finding the screen center
        height, width = frame.shape[:2]
        screen_center_x = width // 2
        screen_center_y = height // 2

        # 2. Convert to HSV Color Space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 3. Get current positions of all trackbars
        h_min = cv2.getTrackbarPos("Hue Min", "Mask & Controls")
        h_max = cv2.getTrackbarPos("Hue Max", "Mask & Controls")
        s_min = cv2.getTrackbarPos("Sat Min", "Mask & Controls")
        s_max = cv2.getTrackbarPos("Sat Max", "Mask & Controls")
        v_min = cv2.getTrackbarPos("Val Min", "Mask & Controls")
        v_max = cv2.getTrackbarPos("Val Max", "Mask & Controls")
        
        k_size = cv2.getTrackbarPos("Kernel Size", "Mask & Controls")
        iters = cv2.getTrackbarPos("Iterations", "Mask & Controls")

        # Ensure kernel size is at least 1 and odd (required by OpenCV)
        if k_size < 1: k_size = 1
        if k_size % 2 == 0: k_size += 1

        # 4. Create HSV Mask
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # 5. Morphological Operations
        # Create the structuring element (kernel)
        kernel = np.ones((k_size, k_size), np.uint8)
        
        # 'Opening': Erosion followed by Dilation. Removes noise (small dots).
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
        
        # 'Closing': Dilation followed by Erosion. Fills holes inside the object.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)

        # 6. Contour Detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Variable to store the largest contour center
        cx, cy = 0, 0

        if contours:
            # Find the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Filter out very small noise that might still exist
            if cv2.contourArea(largest_contour) > 500:
                # Draw the contour on the main frame
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2)

                # 7. Calculate Centroid (Moments)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Draw Centroid (Red Dot)
                    cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
                    
                    # Draw Vector: From Screen Center (Blue) to Object Centroid (Red)
                    # Using Arrowed Line for visualization
                    cv2.arrowedLine(frame, (screen_center_x, screen_center_y), (cx, cy), (0, 255, 0), 3)
                    
                    # Optional: Display coordinates
                    cv2.putText(frame, f"Offset: {cx - screen_center_x}, {cy - screen_center_y}", 
                                (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw a crosshair at the center of the screen for reference
        cv2.circle(frame, (screen_center_x, screen_center_y), 5, (255, 0, 0), -1)

        # 8. Show Results
        # Show the binary mask (black and white)
        cv2.imshow("Mask & Controls", mask)
        # Show the actual video with overlays
        cv2.imshow("Object Tracker", frame)

        # Exit condition
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()