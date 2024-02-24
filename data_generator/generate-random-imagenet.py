import os
import shutil
import random

def create_random_sampled_imagenet_dataset(source_dir, target_dir, total_samples):
    """
    Randomly samples a specified total number of images across all classes in the ImageNet dataset,
    from 'class name/images/image.jpeg', and saves them as 'class name/image.jpeg' in the target directory,
    ensuring the total number of images in the target matches total_samples.
    
    Parameters:
    - source_dir: Path to the original ImageNet dataset.
    - target_dir: Path to the target directory for the sampled dataset.
    - total_samples: Total number of images to sample across the entire dataset.
    """
    # Gather all image paths across all classes
    all_images = []
    for class_name in os.listdir(source_dir):
        images_dir = os.path.join(source_dir, class_name, "images")
        if os.path.isdir(images_dir):
            images = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.lower().endswith('.jpeg')]
            all_images.extend(images)

    # Ensure we do not exceed the number of available images
    total_samples = min(total_samples, len(all_images))

    # Randomly sample from the list of all images
    sampled_images = random.sample(all_images, total_samples)
    
    # Copy the sampled images to the target directory
    for img_path in sampled_images:
        # Extract class name from the path
        class_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))  # Adjust for 'images' subdirectory
        # Prepare target directory path for the class
        target_class_dir = os.path.join(target_dir, class_name)
        # Ensure the target class directory exists
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)
        
        # Define the target path for the image
        target_img_path = os.path.join(target_class_dir, os.path.basename(img_path))
        # Copy the image to the target path
        shutil.copy2(img_path, target_img_path)
        print(f"Copied {img_path} to {target_img_path}")

# Example usage
source_imagenet_dir = '/home/seulgi/Downloads/tiny-imagenet-200/train'  # Update this path to your ImageNet source directory
target_imagenet_dir = '/home/seulgi/work/data/random-sampled-imagenet'  # Update this path to where you want the sampled dataset
total_samples = 500  # Total number of images to sample across all classes

create_random_sampled_imagenet_dataset(source_imagenet_dir, target_imagenet_dir, total_samples)
