import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

HEIGHT = 160
WIDTH = 320

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)

    # Create a match color with the same color channel counts.
    match_mask_color = 255
      
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):

    # If there are no lines to draw, exit.
    if lines is None:
        return
        
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img

def lane_detection(fname):

    # Original Image
    image = mpimg.imread(fname)
    #plt.imshow(image)
    #plt.show()

    # Convert to grayscale and show edges
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    #plt.figure()
    #plt.imshow(cannyed_image)
    #plt.show()

    # Crop Image
    region_of_interest_vertices = [
        (30, (HEIGHT-25)),
        (WIDTH / 2, HEIGHT / 2),
        (290, (HEIGHT-25)),
    ]

    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32),
    )

    #plt.figure()
    #plt.imshow(cropped_image)
    #plt.show()

    # Hough transform
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=50,
        lines=np.array([]),
        minLineLength=30,
        maxLineGap=25
    )
    print(lines)

    # Combine lines into 1
    line_x = []
    line_y = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                line_x.extend([x1,x2])
                line_y.extend([y1, y2])

        min_y = HEIGHT * 7/10
        max_y = HEIGHT - 25

        poly_left = np.poly1d(np.polyfit(
            line_y,
            line_x,
            deg=1
        ))
        x_start = int(poly_left(max_y))
        x_end = int(poly_left(min_y))

        # Ensure all coordinate values are integers
        max_y = int(max_y)
        min_y = int(min_y)

        # Draw lines
        line_image = draw_lines(image, [[
            [x_start, max_y, x_end, min_y]
            ]], 
            thickness=5) 
        #plt.figure()
        #plt.imshow(line_image)
        #plt.show()

        output_dir = "Lane Images"
        # Save the final image
        base_fname = os.path.splitext(os.path.basename(fname))[0]
        # Construct the full path with the new directory
        save_path = os.path.join(output_dir, base_fname + ".jpg")
        # Convert the image from RGB to BGR color space before saving with cv2.imwrite
        cv2.imwrite(save_path, cv2.cvtColor(line_image, cv2.COLOR_RGB2BGR))
    else:
        output_dir = "Unsuccessful Lane Images"
        # Save the final image
        base_fname = os.path.splitext(os.path.basename(fname))[0]
        # Construct the full path with the new directory
        save_path = os.path.join(output_dir, base_fname + ".jpg")
        # Convert the image from RGB to BGR color space before saving with cv2.imwrite
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def lane_detection_print(fname):

    # Original Image
    image = mpimg.imread(fname)
    #plt.imshow(image)
    #plt.show()

    # Convert to grayscale and show edges
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    plt.figure()
    plt.imshow(cannyed_image)
    plt.show()

    # Crop Image
    region_of_interest_vertices = [
        (30, (HEIGHT-25)),
        (30, HEIGHT / 2),
        (270, HEIGHT / 2),
        (270, (HEIGHT-25)),
    ]

    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32),
    )

    plt.figure()
    plt.imshow(cropped_image)
    plt.show()

    # Hough transform
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=50,
        lines=np.array([]),
        minLineLength=30,
        maxLineGap=25
    )
    print(lines)

    # Combine lines into 1
    line_x = []
    line_y = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                line_x.extend([x1,x2])
                line_y.extend([y1, y2])

        min_y = HEIGHT * 7/10
        max_y = HEIGHT - 25

        poly_left = np.poly1d(np.polyfit(
            line_y,
            line_x,
            deg=1
        ))
        x_start = int(poly_left(max_y))
        x_end = int(poly_left(min_y))

        # Ensure all coordinate values are integers
        max_y = int(max_y)
        min_y = int(min_y)

        # Draw lines
        line_image = draw_lines(image, [[
            [x_start, max_y, x_end, min_y]
            ]], 
            thickness=5) 
        plt.figure()
        plt.imshow(line_image)
        plt.show()

        output_dir = "Lane Images"
        # Save the final image
        base_fname = os.path.splitext(os.path.basename(fname))[0]
        # Construct the full path with the new directory
        save_path = os.path.join(output_dir, base_fname + ".jpg")
        # Convert the image from RGB to BGR color space before saving with cv2.imwrite
        cv2.imwrite(save_path, cv2.cvtColor(line_image, cv2.COLOR_RGB2BGR))
    else:
        output_dir = "Unsuccessful Lane Images"
        # Save the final image
        base_fname = os.path.splitext(os.path.basename(fname))[0]
        # Construct the full path with the new directory
        save_path = os.path.join(output_dir, base_fname + ".jpg")
        # Convert the image from RGB to BGR color space before saving with cv2.imwrite
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

image_dir = r"new_jungle_mouse_data\IMG"
# List all files in the directory
images = os.listdir(image_dir)

center_images = [file for file in images if file.startswith('center')]

# Loop over the image files and apply the lane_detection function
#for i, image_file in enumerate(center_images):
    # Construct the full image path
#    image_path = os.path.join(image_dir, image_file)
#    print(image_path)
    
    # Run the lane_detection function
#    lane_detection(image_path)
    
#    print("saved image {}".format(i))

lane_detection_print(r"center_2024_01_19_03_16_59_940.jpg")
