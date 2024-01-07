import cv2, os
import numpy as np
import matplotlib.image as mpimg


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # Assuming image.shape is in the form of (height, width, channels)
    IMAGE_HEIGHT, IMAGE_WIDTH, _ = image.shape

    # (x1, y1) and (x2, y2) forms a line
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT

    # Create an empty mask with the same width and height as the image
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.bool_)

    # Generate the mask using the line equation derived from (x1, y1) and (x2, y2)
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = True

    # Choose which side of the line is shadowed and adjust saturation
    cond = np.random.randint(2)
    if cond == 0:
        mask = ~mask  # Invert mask if cond is 0

    # Convert image to HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # Adjust saturation in HLS(Hue, Light, Saturation)
    s_ratio = np.random.uniform(low=0.2, high=0.5)
    hls[..., 1][mask] = hls[..., 1][mask] * s_ratio

    # Convert back to RGB and return
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):

    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    processed_steering_angles = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            #if is_training and np.random.rand() < 0.6:
            #    image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            #else:
            image = load_image(data_dir, center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            #print("done preprocess")
            #images[i] = image
            processed_steering_angles[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, processed_steering_angles

def preprocess_pytorch(data_dir, image_paths, steering_angles, image_count, is_training):
    images = np.empty([image_count, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    processed_steering_angles = np.empty(image_count)
    for index in range(image_count):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center) 
                # add the image and steering angle to the batch
            images[index] = preprocess(image)
            #print("done preprocess")
            #images[i] = image
            processed_steering_angles[index] = steering_angle
    print("Finished preprocessing data")
    return images, processed_steering_angles