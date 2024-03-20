import cv2, os
import numpy as np
import matplotlib.image as mpimg
import torch
from torch.utils.data import Dataset

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
    Generates and adds random shadow to an image.
    :param image: Input image in RGB format.
    :return: Image with random shadow.
    """
    # Extracting the dimensions of the input image
    IMAGE_HEIGHT, IMAGE_WIDTH = image.shape[:2]

    # (x1, y1) and (x2, y2) form a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # Creating a mask where the shadow will be applied
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # Choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # Convert to HLS to adjust saturation
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
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


def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augmented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    num_samples = image_paths.shape[0]
    while True:  # Loop forever so the generator never terminates
        # Shuffle indices once per epoch
        shuffled_indices = np.random.permutation(np.arange(num_samples))
        for offset in range(0, num_samples, batch_size):
            # Select a batch of shuffled indices
            batch_indices = shuffled_indices[offset:offset+batch_size]
            
            # Initialize arrays for storing batch data
            images = np.empty([len(batch_indices), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
            steers = np.empty(len(batch_indices))
            
            # Iterate over the batch and populate the arrays
            for i, batch_index in enumerate(batch_indices):
                center, left, right = image_paths[batch_index]
                steering_angle = steering_angles[batch_index]
                
                # Argumentation
                if is_training and np.random.rand() < 0.6:
                    image, steering_angle = augment(data_dir, center, left, right, steering_angle)
                else:
                    image = load_image(data_dir, center)
                
                # Preprocess the image and add to the batch
                images[i] = preprocess(image)
                steers[i] = steering_angle
            
            yield images, steers

def batch_generator_with_speed_input(data_dir, image_paths_and_speeds, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths, speed, and associated steering angles
    """
    num_samples = len(image_paths_and_speeds)
    while True:  # Loop forever so the generator never terminates
        shuffled_indices = np.random.permutation(np.arange(num_samples))
        for offset in range(0, num_samples, batch_size):
            batch_indices = shuffled_indices[offset:offset + batch_size]
            
            images = np.empty([len(batch_indices), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
            speeds = np.empty(len(batch_indices))
            steers = np.empty(len(batch_indices))
            
            for i, batch_index in enumerate(batch_indices):
                center, left, right, speed = image_paths_and_speeds[batch_index]  # Adjusted to include speed
                steering_angle = steering_angles[batch_index]
                
                if is_training and np.random.rand() < 0.6:
                    image, steering_angle = augment(data_dir, center, left, right, steering_angle)
                else:
                    image = load_image(data_dir, center)
                
                images[i] = preprocess(image)
                speeds[i] = speed  # Capture the speed value
                steers[i] = steering_angle
            
            # Modify the yield to include speed as an input alongside images
            yield [images, speeds], steers

def batch_generator_with_speed_throttle(data_dir, image_paths_and_speeds, controls, batch_size, is_training):
    """
    Generate training image give image paths, speed, and associated steering angles
    """
    num_samples = len(image_paths_and_speeds)
    while True:  # Loop forever so the generator never terminates
        shuffled_indices = np.random.permutation(np.arange(num_samples))
        for offset in range(0, num_samples, batch_size):
            batch_indices = shuffled_indices[offset:offset + batch_size]
            
            images = np.empty([len(batch_indices), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
            speeds = np.empty(len(batch_indices))
            steers = np.empty(len(batch_indices))
            throttles = np.empty(len(batch_indices))
            
            for i, batch_index in enumerate(batch_indices):
                center, left, right, speed = image_paths_and_speeds[batch_index]  # Adjusted to include speed
                steering_angle, throttle = controls[batch_index]
                
                if is_training and np.random.rand() < 0.6:
                    image, steering_angle = augment(data_dir, center, left, right, steering_angle)
                else:
                    image = load_image(data_dir, center)
                
                images[i] = preprocess(image)
                speeds[i] = speed  # Capture the speed value
                steers[i] = steering_angle
                throttles[i] = throttle
            
            # Modify the yield to include speed as an input alongside images
            yield [images, speeds], [steers, throttles]

def preprocess_pytorch_tensor(X, Y, data_dir, is_training):

    num_images = len(X)
    image_shape = (3, 66, 200)  
    augmented_inputs = torch.empty((num_images, *image_shape), dtype=torch.float32)
    augmented_outputs = torch.empty((num_images, 1), dtype=torch.float32)  
    for i in range(num_images):
        center, left, right = X[i]
        steering_angle = Y[i]

        if is_training and torch.rand(1).item() < 0.6:
            # Apply augmentation
            image, new_steering_angle = augment(data_dir, center, left, right, steering_angle)
        else:
            # Load the center image
            image = load_image(data_dir, center)
            new_steering_angle = steering_angle

        image = preprocess(image)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # Rearrange [H, W, C] to [C, H, W]

        augmented_inputs[i] = image
        augmented_outputs[i] = torch.tensor(new_steering_angle, dtype=torch.float32)

    return augmented_inputs, augmented_outputs

def preprocess_pytorch_speed(X, Y, data_dir, is_training):
    num_images = len(X)
    
    augmented_inputs = []
    augmented_outputs = torch.empty((num_images, 1), dtype=torch.float32)
    
    for i in range(num_images):
        center, left, right, current_speed = X[i]
        steering_angle = Y[i]

        if is_training and torch.rand(1).item() < 0.6:
            # Apply augmentation
            image, new_steering_angle = augment(data_dir, center, left, right, steering_angle)
        else:
            # Load the center image
            image = load_image(data_dir, center)
            new_steering_angle = steering_angle

        image = preprocess(image)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # Rearrange [H, W, C] to [C, H, W]

        # Store the image and corresponding speed as a tuple
        augmented_inputs.append((image, torch.tensor([current_speed], dtype=torch.float32)))

        augmented_outputs[i] = torch.tensor(new_steering_angle, dtype=torch.float32)

    images = torch.stack([x[0] for x in augmented_inputs])
    speeds = torch.stack([x[1] for x in augmented_inputs]).squeeze(1)  # Remove extra dimension from speed values

    return (images, speeds), augmented_outputs

class CustomDataset(Dataset):


    def __init__(self, inputs, targets):
        """
        Args:
            inputs (tuple): A tuple containing two tensors - images and speeds.
            targets (Tensor): A tensor containing the target values (e.g., steering angles).
        """
        self.images = inputs[0]
        self.speeds = inputs[1]
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.images[idx]
        speed = self.speeds[idx]
        target = self.targets[idx]
        return (image, speed), target
    
def preprocess_pytorch_speed_throttle(X, Y, data_dir, is_training):
    num_images = len(X)
    
    augmented_inputs = []
    augmented_outputs = torch.empty((num_images, 2), dtype=torch.float32)  # Adjust for 2 outputs
    
    for i in range(num_images):
        center, left, right, current_speed = X[i]
        steering_angle, throttle = Y[i]  # Unpack both steering and throttle values

        if is_training and torch.rand(1).item() < 0.6:
            image, new_steering_angle = augment(data_dir, center, left, right, steering_angle)
            # Assume augment function does not change throttle value, so we use the original throttle value
        else:
            image = load_image(data_dir, center)
            new_steering_angle = steering_angle

        image = preprocess(image)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)

        augmented_inputs.append((image, torch.tensor([current_speed], dtype=torch.float32)))
        augmented_outputs[i] = torch.tensor([new_steering_angle, throttle], dtype=torch.float32)

    images = torch.stack([x[0] for x in augmented_inputs])
    speeds = torch.stack([x[1] for x in augmented_inputs]).squeeze(1)

    return (images, speeds), augmented_outputs