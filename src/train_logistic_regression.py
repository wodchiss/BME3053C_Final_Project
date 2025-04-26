import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the path to the preprocessed directory
preprocessed_dir = "BBBC005_v1/preprocessed"

# Debugging: Check if preprocessed files exist
required_files = [
    "X_train_scaled.npy",
    "X_test_scaled.npy",
    "y_train.npy",
    "y_test.npy"
]

for file in required_files:
    file_path = os.path.join(preprocessed_dir, file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file '{file_path}' not found. Please run 'preprocess.py' first.")

# Load preprocessed data
X_train_scaled = np.load(os.path.join(preprocessed_dir, "X_train_scaled.npy"))
X_test_scaled = np.load(os.path.join(preprocessed_dir, "X_test_scaled.npy"))
y_train = np.load(os.path.join(preprocessed_dir, "y_train.npy"))
y_test = np.load(os.path.join(preprocessed_dir, "y_test.npy"))

# Debugging: Print shapes of the loaded arrays
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# 1. Reshape data for pixel-level classification
# Flatten images and masks so each pixel is treated as a sample
X_train_flattened = X_train_scaled.reshape(-1, X_train_scaled.shape[-1])  # Shape: (40*520*696, 3)
X_test_flattened = X_test_scaled.reshape(-1, X_test_scaled.shape[-1])    # Shape: (10*520*696, 3)
y_train_flattened = y_train.flatten()  # Shape: (40*520*696,)
y_test_flattened = y_test.flatten()    # Shape: (10*520*696,)

# Debugging: Print shapes after flattening
print(f"X_train_flattened shape: {X_train_flattened.shape}")
print(f"X_test_flattened shape: {X_test_flattened.shape}")
print(f"y_train_flattened shape: {y_train_flattened.shape}")
print(f"y_test_flattened shape: {y_test_flattened.shape}")

# 2. Train Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000, random_state=42)  # max_iter=1000 to allow convergence
log_reg.fit(X_train_flattened, y_train_flattened)  # Train on pixel-level data

# 3. Predict using the trained model
y_pred_flattened = log_reg.predict(X_test_flattened)

# 4. Evaluate the model
accuracy = accuracy_score(y_test_flattened, y_pred_flattened)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display Classification Report and Confusion Matrix
print("Classification Report:")
print(classification_report(y_test_flattened, y_pred_flattened))
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test_flattened, y_pred_flattened)
print(conf_matrix)

# 5. Visualize Confusion Matrix and save as .png
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Background", "Foreground"], yticklabels=["Background", "Foreground"])
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Save the confusion matrix as a .png file
output_dir = "BBBC005_v1/train_images"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
conf_matrix_path = os.path.join(output_dir, "logistic_regression_confusion_matrix.png")
plt.savefig(conf_matrix_path)
print(f"Confusion matrix saved to {conf_matrix_path}")

plt.show()

# 6. Reshape predictions back to original image dimensions for visualization
y_pred = y_pred_flattened.reshape(y_test.shape)  # Reshape to (10, 520, 696)

# 7. Visualize some test predictions (visualizing first 5 test images and their predictions)
for i in range(5):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(X_test_scaled[i])  # Visualize the original image
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(y_test[i], cmap='gray')  # Visualize the ground truth mask
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(y_pred[i], cmap='gray')  # Visualize the predicted mask
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.show()


