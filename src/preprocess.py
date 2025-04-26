import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

# Define paths (adjust the paths if necessary)
train_images_dir = "BBBC005_v1/train_images"
test_images_dir = "BBBC005_v1/test_images"
train_masks_dir = "BBBC005_v1/train_masks"
test_masks_dir = "BBBC005_v1/test_masks"

# Load image files (We are assuming the images are already moved into train and test directories)
train_image_files = os.listdir(train_images_dir)
test_image_files = os.listdir(test_images_dir)

# Load the images and masks into numpy arrays (images in the range [0, 1] and masks as binary)
def load_images_and_masks(image_files, mask_files, image_dir, mask_dir):
    images = []
    masks = []
    for img_file, mask_file in zip(image_files, mask_files):
        try:
            # Load image and mask
            image_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            image = Image.open(image_path).convert('RGB')  # Convert to RGB for consistent size (3 channels)
            image = np.array(image) / 255.0  # Scale the image to [0, 1]

            mask = Image.open(mask_path).convert('L')  # Convert to grayscale for mask
            mask = np.array(mask) / 255.0  # Scale the mask to [0, 1] (binary mask)

            images.append(image)
            masks.append(mask)
        except Exception as e:
            print(f"Error loading image/mask pair {img_file}, {mask_file}: {e}")
    
    return np.array(images), np.array(masks)

# Debug: Print the image dimensions of the first image to ensure they are correct
image_path = os.path.join(train_images_dir, train_image_files[0])
sample_image = Image.open(image_path)
print(f"Sample image size: {sample_image.size}")

# Load a subset of the data for debugging
train_image_files = train_image_files[:50]  # Use only the first 50 images for testing
test_image_files = test_image_files[:10]    # Use only the first 10 test images for testing

# Load training and testing data (subset for debugging)
train_images, train_masks = load_images_and_masks(train_image_files, train_image_files, train_images_dir, train_masks_dir)
test_images, test_masks = load_images_and_masks(test_image_files, test_image_files, test_images_dir, test_masks_dir)

# Display a sample image to ensure they are loaded correctly
plt.imshow(train_images[0])
plt.show()

# 1. Split Data into Training (80%) and Testing (20%) Sets
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks, test_size=0.2, random_state=42)

# 2. Create a StandardScaler Object or MinMaxScaler for images (MinMaxScaler is more common for images)
scaler = MinMaxScaler()  # You can switch to StandardScaler if desired

# Flatten the images to 2D (N, H*W*C) for scaling
X_train_flattened = X_train.reshape(X_train.shape[0], -1)  # Flatten the images
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Fit the scaler on the training data and transform both the training and testing data
X_train_scaled = scaler.fit_transform(X_train_flattened)
X_test_scaled = scaler.transform(X_test_flattened)

# Reshape the scaled data back to the original image shape
X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_test_scaled = X_test_scaled.reshape(X_test.shape)

# 3. Print the shapes of the resulting training and test sets
print(f"Training data shape: {X_train_scaled.shape}")
print(f"Testing data shape: {X_test_scaled.shape}")

# 4. Print dataset characteristics:
print(f"Dataset dimensions and size:")
print(f"Number of training images: {X_train_scaled.shape[0]}")
print(f"Number of testing images: {X_test_scaled.shape[0]}")
print(f"Sample image dimensions (Height, Width, Channels): {X_train_scaled.shape[1:]}")

# For binary segmentation, number of classes is 2: foreground and background
print(f"Number of classes: 2 (Foreground and Background)")

# Save preprocessed data
preprocessed_dir = "BBBC005_v1/preprocessed"
os.makedirs(preprocessed_dir, exist_ok=True)  # Create directory if it doesn't exist

np.save(os.path.join(preprocessed_dir, "X_train_scaled.npy"), X_train_scaled)
np.save(os.path.join(preprocessed_dir, "X_test_scaled.npy"), X_test_scaled)
np.save(os.path.join(preprocessed_dir, "y_train.npy"), y_train)
np.save(os.path.join(preprocessed_dir, "y_test.npy"), y_test)

print("Preprocessed data saved successfully in 'BBBC005_v1/preprocessed/'")
