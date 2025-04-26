# Logistic Regression for Image Classification

This project implements a logistic regression model for image classification tasks. It includes data preprocessing, model training, evaluation, and visualization of results.

## Features
- Preprocessing of image datasets (`BBBC005_v1_ground_truth` and `synthetic_2_ground_truth`).
- Training a logistic regression model on the processed data.
- Evaluation of the model using a confusion matrix and classification metrics.
- Visualization of test predictions and misclassified examples.
- Saving the confusion matrix as a `.png` file in the `plots` folder.

---

## Setup Instructions

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/wodchiss/BME3053C_Final_Project.git
cd BME3053C_Final_Project
```

### 2. Install Required Python Packages

Install the required Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Workflow

### 1. Preprocess the Data

Run the `split.py` script to load and explore the datasets:

```bash
python check/split.py
```

### 2. Train the Logistic Regression Model

Run the `train_logistic_regression.py` script to train the logistic regression model:

```bash
python src/train_logistic_regression.py
```

### 3. View Results

- The confusion matrix will be saved as a `.png` file in the `plots` folder:
  ```
  plots/logistic_regression_confusion_matrix.png
  ```
- Test predictions and misclassified examples will also be visualized during execution:
  - **Test Predictions**: The script visualizes the original image, ground truth mask, and predicted mask for the first 5 test images.
  - **Misclassified Examples**: The script identifies and visualizes up to 5 misclassified pixels, showing the original image and the corresponding ground truth mask.

---

## Folder Structure

```
BME3053C_Final_Project/
├── check/
│   ├── BBBC005_v1_ground_truth/
│   ├── synthetic_2_ground_truth/
│   ├── split.py
│   ├── preprocess.py
│   ├── model.py
│   └── plot.py
├── src/
│   ├── train_logistic_regression.py
│   └── ...
├── plots/
│   └── logistic_regression_confusion_matrix.png
└── README.md
```

---

## Requirements

- Python 3.x
- Required libraries:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `Pillow`
  - `seaborn`

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Notes

- Ensure the datasets (`BBBC005_v1_ground_truth` and `synthetic_2_ground_truth`) are placed in the `check` folder before running the scripts.
- The `plots` folder will be created automatically if it does not exist.

---

## License

This project is licensed under the MIT License.
