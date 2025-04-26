# BME3053C_Final_Project: Nucleus Segmentation

This project performs nucleus segmentation using logistic regression on microscopy images. The pipeline includes preprocessing the dataset, training a logistic regression model, and evaluating its performance.

---

## Setup Instructions

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/BME3053C_Final_Project.git
cd BME3053C_Final_Project
```

### 2. Install Required Python Packages

Install the required Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Workflow

### 3. Preprocess the Dataset

The `preprocess.py` script prepares the dataset for training and testing. It performs the following steps:

1. **Load Images and Masks**:
   - Loads images and their corresponding masks from the `BBBC005_v1/train_images` and `BBBC005_v1/train_masks` directories.
   - Converts images to RGB format and scales pixel values to the range `[0, 1]`.
   - Converts masks to grayscale and scales them to binary values in the range `[0, 1]`.

2. **Split Data**:
   - Splits the dataset into training (80%) and testing (20%) sets using `train_test_split`.

3. **Save Preprocessed Data**:
   - Saves the preprocessed data as `.npy` files in the `BBBC005_v1/preprocessed/` directory:
     - `X_train_scaled.npy`: Scaled training images.
     - `X_test_scaled.npy`: Scaled testing images.
     - `y_train.npy`: Training masks.
     - `y_test.npy`: Testing masks.

#### How to Run

Run the preprocessing script:

```bash
python src/preprocess.py
```

#### Output

The preprocessed data will be saved in the `BBBC005_v1/preprocessed/` directory. You should see the following files:
- `X_train_scaled.npy`
- `X_test_scaled.npy`
- `y_train.npy`
- `y_test.npy`

---

### 4. Train the Logistic Regression Model

The `train_logistic_regression.py` script trains a logistic regression model on the preprocessed data and evaluates its performance. It performs the following steps:

1. **Load Preprocessed Data**:
   - Loads the preprocessed `.npy` files from the `BBBC005_v1/preprocessed/` directory.

2. **Reshape Data for Pixel-Level Classification**:
   - Flattens the images and masks so that each pixel is treated as a sample with 3 features (RGB channels).

3. **Train Logistic Regression**:
   - Trains a logistic regression model using the flattened training data.

4. **Evaluate the Model**:
   - Predicts the labels for the test data.
   - Computes accuracy, precision, recall, F1-score, and confusion matrix.

5. **Visualize Predictions**:
   - Reshapes the predictions back to the original image dimensions.
   - Visualizes the original image, ground truth mask, and predicted mask for the first 5 test samples.

#### How to Run

Run the training script:

```bash
python src/train_logistic_regression.py
```

#### Output

1. **Model Performance**:
   - Accuracy, classification report, and confusion matrix will be printed in the terminal.

2. **Visualizations**:
   - The script will display visualizations of the original image, ground truth mask, and predicted mask for the first 5 test samples.

---

## Repository Structure

```
BME3053C_Final_Project/
├── BBBC005_v1/
│   ├── preprocessed/               # Directory where preprocessed files are saved
│       ├── X_train_scaled.npy
│       ├── X_test_scaled.npy
│       ├── y_train.npy
│       ├── y_test.npy
│   ├── train_images/               # Training images
│   ├── train_masks/                # Training masks
│   ├── test_images/                # Testing images
│   ├── test_masks/                 # Testing masks
├── src/
│   ├── preprocess.py               # Preprocessing script
│   ├── train_logistic_regression.py # Training script
├── README.md                       # Project overview and instructions
├── requirements.txt                # Python package dependencies
```

---

## Notes

- Ensure that the dataset is properly organized in the `BBBC005_v1/` directory before running the scripts.
- If you encounter any issues, verify that the preprocessed files are saved correctly in the `BBBC005_v1/preprocessed/` directory.

---
