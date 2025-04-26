import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Define the paths to the train/test images and masks
train_images_dir = "BBBC005_v1/train_images"
test_images_dir = "BBBC005_v1/test_images"
train_masks_dir = "BBBC005_v1/train_masks"
test_masks_dir = "BBBC005_v1/test_masks"

# Define the path to save sample images
sample_images_dir = "BBBC005_v1/sample_img"
os.makedirs(sample_images_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Get the list of image files
train_image_files = os.listdir(train_images_dir)
test_image_files = os.listdir(test_images_dir)
train_mask_files = os.listdir(train_masks_dir)
test_mask_files = os.listdir(test_masks_dir)

# 1. Dataset Dimensions and Size
print(f"Number of training images: {len(train_image_files)}")
print(f"Number of testing images: {len(test_image_files)}")

# Let's load a sample image to get its dimensions
sample_image_path = os.path.join(train_images_dir, train_image_files[0])
sample_image = Image.open(sample_image_path)
image_size = np.array(sample_image).shape
print(f"Sample image dimensions (Height, Width, Channels): {image_size}")

# 2. Number of Classes (Foreground and Background)
# Since this is a segmentation task, we can assume two classes: foreground and background
num_classes = 2
print(f"Number of classes: {num_classes} (Foreground and Background)")

# 3. Visualize Sample Images and Masks
def visualize_samples(image_files, mask_files, image_dir, mask_dir, title_prefix=""):
    # Plot a few sample images and masks
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    for i in range(3):  # Display 3 samples
        # Load an image and its corresponding mask
        image_path = os.path.join(image_dir, image_files[i])
        mask_path = os.path.join(mask_dir, mask_files[i])
        
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Display the image and mask side by side
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"{title_prefix} Sample Image {i+1}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f"{title_prefix} Sample Mask {i+1}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    output_path = os.path.join(sample_images_dir, f"{title_prefix}_samples.png")
    plt.savefig(output_path)  # Save the plot as an image file in the sample_img folder
    plt.close()
    print(f"Saved {title_prefix} samples to {output_path}")

# Visualize 3 sample images and masks from the training set
visualize_samples(train_image_files, train_mask_files, train_images_dir, train_masks_dir, title_prefix="Training")

# Visualize 3 sample images and masks from the testing set
visualize_samples(test_image_files, test_mask_files, test_images_dir, test_masks_dir, title_prefix="Testing")
