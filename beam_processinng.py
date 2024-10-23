import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "/Users/yorkgong/Downloads/Laser_Photo_PNG/WIN_20241023_16_43_54_Pro.png"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to get the bright regions
_, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables to store the largest contour information
max_contour = None
max_area = 0
ellipse = None

# Iterate over contours to find the largest bright spot
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

# If a suitable contour is found, fit an ellipse
if max_contour is not None and len(max_contour) >= 5:
    ellipse = cv2.fitEllipse(max_contour)
    cv2.ellipse(image, ellipse, (0, 255, 0), 2)
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    cv2.circle(image, center, 5, (0, 0, 255), -1)

    # Add text to show the major axis, minor axis, and center
    text = f"Major: {max(ellipse[1]):.2f} px, Minor: {min(ellipse[1]):.2f} px"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    center_text = f"Center: ({ellipse[0][0]:.2f}, {ellipse[0][1]:.2f})"
    cv2.putText(image, center_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Display the result
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Bright Spot with Ellipse Fit')
plt.axis('off')
plt.show()