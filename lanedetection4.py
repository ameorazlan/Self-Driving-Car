import cv2
import numpy as np
import os

def detect_lane_in_image(image_path):
    # Load and resize the image to the expected dimensions
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    resized_image = cv2.resize(image, (320, 160))

    # Convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve contrast in the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blur, 50, 150)

    # Masking the region of interest
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (30, 130),
        (290, 130),
        (320, 80),
        (0, 80),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Line Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=15, minLineLength=10, maxLineGap=20)
    
    # Drawing lines on the original image
    line_image = np.copy(resized_image) * 0  # Creating a blank to draw lines on
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # Creating a "weighted" image to overlay lines on original
    combo_image = cv2.addWeighted(resized_image, 0.8, line_image, 1, 0)
    
    # Check if output directories exist, if not, create them
    output_dir_success = "Algorithm 4 Lane Images"
    output_dir_unsuccessful = "Algorithm 4 Unsuccessful"
    if not os.path.isdir(output_dir_success):
        os.makedirs(output_dir_success)
    if not os.path.isdir(output_dir_unsuccessful):
        os.makedirs(output_dir_unsuccessful)

    # Save the final image
    base_fname = os.path.splitext(os.path.basename(image_path))[0]
    if lines is not None:
        save_path = os.path.join(output_dir_success, base_fname + ".jpg")
    else:
        save_path = os.path.join(output_dir_unsuccessful, base_fname + ".jpg")
    
    cv2.imwrite(save_path, combo_image)

    print("Lane detection completed. Output saved to:", save_path)

image_dir = r"new_jungle_mouse_data\IMG"
# List all files in the directory
images = os.listdir(image_dir)

center_images = [file for file in images if file.startswith('center')]

# Loop over the image files and apply the lane_detection function
for i, image_file in enumerate(center_images):
    # Construct the full image path
    image_path = os.path.join(image_dir, image_file)
    print(image_path)
    
    # Run the lane_detection function
    detect_lane_in_image(image_path)
    
    print("saved image {}".format(i))
