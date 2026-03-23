# Stone Area Measurement Tool

## Overview

This project is an interactive image-processing tool built with Python, OpenCV, NumPy, Matplotlib, and PIL. It allows the user to:

1. Select a reference region containing **5 mm × 5 mm squares**.
2. Detect the squares automatically and calculate the **pixel-to-real-world scale**.
3. Select a stone region in the image.
4. Calculate the **stone area** in both **pixels** and **square millimeters (mm²)**.

The tool is designed for images where a known square reference is available for calibration.

---

## Features

- Interactive mouse-based region selection
- Automatic square detection in a selected calibration region
- Pixel-to-millimeter scale conversion
- Blue background removal using HSV color segmentation
- Stone contour extraction and area calculation
- Visualization of thresholded image and detected stone contour

---

## Requirements

Install the following Python libraries before running the script:

```bash
pip install opencv-python numpy matplotlib pillow
