import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Global variables
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
img_bgr = None
img_bgr_resized = None
img_copy = None
scale = None
processing_step = 1  # 1: Square recognition, 2: Stone area calculation
square_side_length_pixels = None  # Store square side length (pixels)
square_side_length_mm = 5  # Actual square side length is 5mm

# Screen size
screen_width = 1920
screen_height = 1080


# Resize image to fit screen
def resize_image(image, width, height):
    scale_width = width / image.shape[1]
    scale_height = height / image.shape[0]
    scale = min(scale_width, scale_height)
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, scale


# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, img_bgr_resized, img_copy, scale, processing_step

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        img_copy = img_bgr_resized.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_draw = img_copy.copy()
            cv2.rectangle(img_draw, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img_draw)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        cv2.rectangle(img_copy, (ix, iy), (fx, fy), (0, 255, 0), 2)
        cv2.imshow('image', img_copy)

        # Map back to original image size using scale factor
        x1, y1 = int(ix / scale), int(iy / scale)
        x2, y2 = int(fx / scale), int(fy / scale)
        selected_region = img_bgr[y1:y2, x1:x2]

        if processing_step == 1:  # Square recognition step
            detect_squares_original(selected_region)  # Use original square detection method
            processing_step = 2
            print("5mm×5mm square recognition completed. Please select stone area to calculate area")
        elif processing_step == 2:  # Stone area calculation step
            calculate_stone_area(selected_region)


# Original square detection function (with side length constraints)
def detect_squares_original(image):
    global square_side_length_pixels

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for smoothing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarization using Otsu's method
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    side_lengths = []

    # Create white background image with same size as binary image
    height, width = binary.shape
    img_with_squares = np.ones((height, width, 3), dtype=np.uint8) * 255

    for cnt in contours:
        # Calculate bounding rectangle of contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Calculate aspect ratio to check if square
        aspect_ratio = float(w) / float(h)
        if 0.8 < aspect_ratio < 1.2:  # Original tolerance setting
            side_length = h  # Original code uses height as side length

            # Add side length constraint: must be within ±20% of average side length
            if len(side_lengths) > 0:
                avg_side = np.mean(side_lengths)
                if not (0.8 * avg_side < side_length < 1.2 * avg_side):
                    continue  # Skip squares not meeting side length constraint

            squares.append((x, y, w, h))
            side_lengths.append(side_length)

            # Draw green square on white background
            cv2.rectangle(img_with_squares, (x, y), (x + w, y + h), (0, 255, 0), -1)

    if side_lengths:
        print(f"Detected {len(side_lengths)} squares")
        for i, side_length in enumerate(side_lengths, 1):
            print(f"Square {i}: side length = {side_length:.2f} pixels")

        # Calculate and output average square side length
        square_side_length_pixels = np.mean(side_lengths)
        print(f"Average square side length: {square_side_length_pixels:.2f} pixels")
        print(f"Known actual square side length: {square_side_length_mm}mm")
        print(f"Pixel to real size ratio: {square_side_length_mm / square_side_length_pixels:.4f} mm/pixel")

        # Display binary image and detected squares
        cv2.imshow('Binary Image', binary)
        cv2.imshow('Detected Squares (Original Method)', img_with_squares)
    else:
        print("No squares detected")
        square_side_length_pixels = None


# Stone area calculation function (unchanged)
def calculate_stone_area(image):
    global square_side_length_pixels, square_side_length_mm

    # Convert to HSV color space to extract blue regions
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_no_blue = cv2.bitwise_not(mask_blue)
    stone_region = cv2.bitwise_and(image, image, mask=mask_no_blue)

    # Convert to grayscale for contour detection
    gray_image = cv2.cvtColor(stone_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Find contours of stone region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        stone_area_pixels = cv2.contourArea(largest_contour)

        if square_side_length_pixels is not None:
            pixel_to_mm_ratio = square_side_length_mm / square_side_length_pixels
            stone_area_mm2 = stone_area_pixels * (pixel_to_mm_ratio ** 2)
            print(f"Stone area: {stone_area_pixels:.2f} pixels ({stone_area_mm2:.2f} square mm)")
        else:
            print(f"Stone area: {stone_area_pixels:.2f} pixels (No square detected, cannot convert to actual size)")

        # Visualize results
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        result_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_img_with_contours = result_img.copy()
        cv2.drawContours(result_img_with_contours, [largest_contour], -1, (255, 0, 0), 5)

        plt.figure(figsize=(15, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(thresh, cmap='gray')
        plt.title("Thresholded Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(result_img_with_contours)
        title = f"Stone Area with Red Contours\n(Area: {stone_area_pixels:.2f} pixels"
        if square_side_length_pixels is not None:
            title += f", {stone_area_mm2:.2f} mm²"
        plt.title(title)
        plt.axis('off')
        plt.show()


# Main program
def main():
    global img_bgr, img_bgr_resized, scale

    # Read image
    image_path = "photo path"  # Make sure the path is correct
    img_pil = Image.open(image_path)
    img = np.array(img_pil)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize image to fit screen
    img_bgr_resized, scale = resize_image(img_bgr, screen_width, screen_height)

    # Create window and set mouse callback
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    # Display resized image
    cv2.imshow('image', img_bgr_resized)
    print("Please use mouse to drag and select 5mm×5mm square area")

    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()