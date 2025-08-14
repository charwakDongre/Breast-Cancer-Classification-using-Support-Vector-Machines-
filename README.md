# Breast Cancer Classification using Support Vector Machines (SVM)

This project demonstrates the use of Support Vector Machines (SVM) for binary classification. The goal is to build a model that can accurately predict whether a breast tumor is malignant or benign based on a set of features. The analysis covers the use of different SVM kernels, hyperparameter tuning, and robust evaluation with cross-validation.

---

## The Dataset

The project uses the **Wisconsin Breast Cancer dataset** (`breast-cancer.csv`), which contains 30 features computed from digitized images of breast mass samples. The target variable, 'diagnosis', is binary (Malignant/Benign).

---

## Project Workflow

### 1. Data Preprocessing
- **Data Loading & Cleaning**: The dataset was loaded and the non-essential 'id' column was removed.
- **Encoding**: The categorical target 'diagnosis' was converted into a numerical format (Malignant=1, Benign=0).
- **Feature Scaling**: All 30 numerical features were standardized using `StandardScaler`. This is a critical step for SVMs, as they are sensitive to the scale of the input data.

### 2. SVM with Different Kernels
Two baseline SVM models were trained to compare their performance:
- **Linear Kernel**: This kernel creates a simple, straight-line decision boundary. It is effective for datasets that are linearly separable.
- **RBF (Radial Basis Function) Kernel**: This kernel can create a complex, non-linear boundary, allowing it to capture more intricate patterns in the data.

The performance of both kernels was strong, but this step highlighted the need for careful tuning, especially for the RBF kernel. The decision boundaries for both were visualized using a 2D plot for better intuition.

### 3. Hyperparameter Tuning and Cross-Validation
To find the optimal settings for the RBF SVM, `GridSearchCV` was used.
- **Hyperparameters**: We tuned the `C` (regularization) and `gamma` parameters to find the combination that yielded the best performance.
- **Cross-Validation**: `GridSearchCV` uses 5-fold cross-validation, providing a robust evaluation of the model by training and testing it on different subsets of the data.
- **Best Model**: The grid search identified the optimal parameters, which resulted in a model with a mean cross-validation accuracy of **~97%**.

### 4. Final Evaluation
The best model found during the tuning process was evaluated on the final, unseen test set.
- **Test Accuracy**: The fine-tuned model achieved an excellent accuracy of **95.3%** on the test data.
- **Classification Report**: A detailed report showed that the model was particularly good at identifying benign tumors, demonstrating high precision and recall.

---

## How to Run the Code
1.  Ensure you have Python and Jupyter Notebook installed.
2.  Clone this repository to your local machine.
3.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
4.  Open and run the Jupyter Notebook to see the full analysis.

---

## Libraries Used
- **pandas**
- **numpy**
- **scikit-learn**
- **matplotlib**
- **seaborn**
