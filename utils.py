import torch
import os
import numpy as np
from PIL import Image
import cv2
import random
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def set_log(log_dir):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
        # datefmt='%a, %d %b %Y %H:%M:%S',
        filename=f"{log_dir}/train.log",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe


def set_training_dir(dir_name=None, project_dir=None):
    """
    This functions counts the number of training directories already present
    and creates a new one in `outputs/training/`.
    And returns the directory path.
    """
    if project_dir is not None:
        os.makedirs(project_dir, exist_ok=True)
        return project_dir
    if not os.path.exists('outputs/training'):
        os.makedirs('outputs/training')
    if dir_name:
        new_dir_name = f"outputs/training/{dir_name}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name
    else:
        num_train_dirs_present = len(os.listdir('outputs/training/'))
        next_dir_num = num_train_dirs_present + 1
        new_dir_name = f"outputs/training/res_{next_dir_num}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name


def calculate_mean_std(image_dir):
    # List to store all pixel values
    pixel_values = []

    # Loop over all files in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Open image and convert to RGB
            img = Image.open(os.path.join(image_dir, filename))
            # Convert image to numpy array
            img_array = np.array(img) / 255.0
            pixel_values.append(img_array)
    # Concatenate all pixel values
    # all_pixels = np.concatenate(pixel_values, axis=0)
    all_pixels = np.concatenate([img.flatten() for img in pixel_values])

    # Calculate mean and standard deviation
    mean = np.mean(all_pixels, axis=0)
    std = np.std(all_pixels, axis=0)
    print(f"The calculated mean is equal to {mean} and The calculated std is equal to {std}")
    return mean, std


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


if __name__ == "__main__":
    # Directory containing your images
    image_directory = '/media/mostafahaggag/Shared_Drive/selfdevelopment/datasets/mvist/screw/train/good'

    mean, std = calculate_mean_std(image_directory)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")