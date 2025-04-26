# BME3053C_Final_Project

# **Cell Detection Using Logistic Regression**

## **Team Name:** *Your Team Name*

### **Dataset:**
The dataset used for this project is the **BBBC005_v1** dataset, which consists of synthetic microscopy images with corresponding foreground/background segmentation ground truth. The images are named with the cell count encoded in the filename, and the segmentation masks are provided in 8-bit TIF files. This dataset allows for the task of **cell detection** and **foreground/background segmentation**.

You can access the full dataset here: [BBBC005 Dataset](https://bbbc.broadinstitute.org/BBBC005).

### **Files Included:**
- `BBBC005_v1_ground_truth.zip`: Contains 1200 image files and their corresponding segmentation masks (foreground and background).
- `BBBC005_results_bray.csv` (Optional): Contains cell count results used for evaluating the dataset.
- `images/`: Contains the synthetic microscopy images.
- `masks/`: Contains the corresponding segmentation masks.
  
### **How to Use This Repository:**

1. **Download and Setup:**
   - Download the repository to your local machine:
     ```bash
     git clone https://github.com/yourusername/yourrepository.git
     ```
   - Unzip `BBBC005_v1_ground_truth.zip` to access the images and masks.
   
2. **Running the Model:**
   - Install required dependencies (e.g., `scikit-learn`, `opencv-python`, `matplotlib`):
     ```bash
     pip install -r requirements.txt
     ```
   - After unzipping the dataset, run the code to preprocess the images and train the Logistic Regression model:
     ```bash
     python main.py
     ```
   - The code will extract image patches, compute features (mean, standard deviation, etc.), label them as cell or background based on the segmentation mask, and train a logistic regression model.

3. **Evaluating the Model:**
   - The evaluation is done using accuracy, precision, and recall metrics, which can be found in the code after running the training script.

4. **Results:**
   - The results of the model training and evaluation are stored in a file or displayed on the console. For further results, the `BBBC005_results_bray.csv` file can be used as a benchmark for comparison.

### **File Structure:**
```
YourTeamName_Data/
├── BBBC005_v1_ground_truth.zip
├── BBBC005_results_bray.csv (optional)
├── images/ (if separated)
├── masks/ (if separated)
└── main.py (or notebook file)
```

### **Citation:**
If you use this dataset, please cite the original paper:
- Bray et al., *J. Biomol Screen*, 2011.  
  Dataset: [BBBC005 dataset citation](https://bbbc.broadinstitute.org/BBBC005).
