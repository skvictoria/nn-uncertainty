import os
import numpy as np
from PIL import Image

def add_gaussian_noise(image, mean=10, sigma=25):
    """
    Add Gaussian noise to an image.

    Parameters:
    - image: PIL Image object.
    - mean: Mean of the Gaussian noise.
    - sigma: Standard deviation of the Gaussian noise.

    Returns:
    - Noisy image as a PIL Image object.
    """
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Generate Gaussian noise
    gauss = np.random.normal(mean, sigma, img_array.shape)
    
    # Add the Gaussian noise to the image
    noisy_array = img_array + gauss
    
    # Clip the values to be between 0 and 255 and convert to uint8
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
    
    # Convert the noisy image back to a PIL Image object
    return Image.fromarray(noisy_array)

def process_images(source_dir, target_dir, sigma):
    """
    Process images by adding Gaussian noise and saving them to the target directory.

    Parameters:
    - source_dir: Path to the source directory.
    - target_dir: Path to the target directory.
    - sigma: Standard deviation of the Gaussian noise.
    """
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".JPEG"):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, os.path.relpath(root, source_dir), file)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # Open the image
                image = Image.open(source_path)
                
                # Add Gaussian noise to the image
                noisy_image = add_gaussian_noise(image, sigma=sigma)
                
                # Save the noisy image
                noisy_image.save(target_path)

# Example usage
source_directory = "/home/seulgi/work/data/random-sampled-imagenet"
target_directory = "/home/seulgi/work/data/noisy-imagenet"
sigma = 50  # Adjust the level of Gaussian noise

process_images(source_directory, target_directory, sigma)
