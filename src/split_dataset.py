import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

# Define directories for images and masks
images_dir = Path("BBBC005_v1/BBBC005_v1_ground_truth")
masks_dir = Path("BBBC005_v1/synthetic_2_ground_truth")

# Define directories for train and test splits
train_images_dir = Path("BBBC005_v1/train_images")
test_images_dir = Path("BBBC005_v1/test_images")
train_masks_dir = Path("BBBC005_v1/train_masks")
test_masks_dir = Path("BBBC005_v1/test_masks")

# Create new directories for train and test splits
train_images_dir.mkdir(parents=True, exist_ok=True)
test_images_dir.mkdir(parents=True, exist_ok=True)
train_masks_dir.mkdir(parents=True, exist_ok=True)
test_masks_dir.mkdir(parents=True, exist_ok=True)

# List all image and mask files
image_files = os.listdir(images_dir)
mask_files = os.listdir(masks_dir)

# Print the files in both directories to confirm
print(f"Found {len(image_files)} image files in {images_dir}")
print(f"Found {len(mask_files)} mask files in {masks_dir}")
print(f"Sample image files: {image_files[:5]}")  # Print first 5 image filenames
print(f"Sample mask files: {mask_files[:5]}")  # Print first 5 mask filenames

# Split the data into training and testing (80% train, 20% test)
train_images, test_images = train_test_split(image_files, test_size=0.2, random_state=42)
train_masks, test_masks = train_test_split(mask_files, test_size=0.2, random_state=42)

# Move the images and masks into their respective train and test directories
for img in train_images:
    shutil.move(images_dir / img, train_images_dir / img)

for img in test_images:
    shutil.move(images_dir / img, test_images_dir / img)

for mask in train_masks:
    shutil.move(masks_dir / mask, train_masks_dir / mask)

for mask in test_masks:
    shutil.move(masks_dir / mask, test_masks_dir / mask)

print("✅ Data has been split into training and testing sets.")
