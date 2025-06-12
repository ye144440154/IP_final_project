# IP_final_project

This project is for **Mango Grading and Classification** based on this dataset from Kaggle: 
[Mango Varieties Classification Dataset] (https://www.kaggle.com/datasets/saurabhshahane/mango-varieties-classification/data)

We first apply the images segmentation techniques on mango images, extract features, and then finally classify the mangos images using several methods such as KNN, MLP, and Random Forest algorithm.

### 🔹 Classifier/
Including feature files, training/testing program, and result comparison scripts. 
This folder includes classification for both grading and variety classificaion, using MLP and Random Forest.

1. **Feature files for training**: `Classification_features.csv`, `Classification_features_1.csv`, `Classification_features_modified.csv`, `combined_output.csv`
2. **Main classificationto programs**: 
`classifier_mlp.py` (MLP), `classifier_rf.py`(Random Forest)
3. **Result files**: 
`test_predictions.csv`, `test_predictions_rf.csv`

### 🔹 Compare/
Stores and compares segmented mango images.

### 🔹 Dataset/
Stores original mango images.

### 🔹 Mask/
Stores segmentation masks files for each mango image.

### 🔹 Output/
Stores the segmented mango images.

## Other files

| File Names | Explanation |
|------|------|
| `classifier_demo.ipynb` | Classification using **KNN** |
| `cropper.ipynb`, `morphology.py` | Image segmentation pipeline |
| `feature_extraction.py`, `feature_extraction_1.py`, `feature_extraction_modified.py` | Feature extaction scripts |

## How to execute our programs
1. **Segmentation**
    ```bash
    # Run Jupyter Notebook
    cropper.ipynb
    ```
2. **Feature Extraction**
    ```bash
    python feature_extraction.py
    python feature_extraction_1.py
    python feature_extraction_modified.py
    ```
3. **Classification**
    - KNN:
        ```bash
        # Run Jupyter Notebook
        classifier_demo.ipynb
        ```
    - MLP:
        ```bash
        python classifier_mlp.py
        ```
    - Random Forest:
        ```bash
        python classifier_rf.py
        ```