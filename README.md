# Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Objective

The objective of this project is to develop a K-Nearest Neighbors (KNN) classification model to classify breast cancer tumors as malignant or benign using diagnostic measurements. The model is evaluated using standard classification metrics to measure its performance.

## Dataset Link

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Libraries Used

- pandas
- matplotlib
- scikit-learn

## Methodology

1. Loaded the Breast Cancer Wisconsin dataset using Pandas.
2. Displayed the first five records.
3. Identified the numerical features and target variable.
4. Displayed dataset information and summary statistics.
5. Checked for missing values.
6. Removed unnecessary columns (`id` and `Unnamed: 32`).
7. Encoded the target variable (`diagnosis`) using LabelEncoder.
8. Standardized the feature values using StandardScaler.
9. Split the dataset into 80% training and 20% testing sets.
10. Trained the K-Nearest Neighbors (KNN) classifier with K = 5.
11. Predicted the class labels for the test dataset.
12. Evaluated the model using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.

## Results

- **Accuracy:** 94.74%
- **Precision:** 93.02%
- **Recall:** 93.02%
- **F1 Score:** 93.02%

The KNN classifier achieved excellent classification performance, correctly classifying the majority of breast cancer cases with high accuracy and balanced precision and recall.

## Conclusion

This project successfully implemented a K-Nearest Neighbors (KNN) classifier to classify breast cancer tumors as malignant or benign. The dataset was preprocessed by removing unnecessary columns, encoding the target variable, and standardizing the feature values before model training. The classifier achieved an accuracy of **94.74%**, along with high precision, recall, and F1 score, demonstrating strong predictive performance. Unlike regression algorithms, KNN is a supervised classification algorithm that classifies new samples based on the nearest neighboring data points. One major advantage of KNN is its simplicity and effectiveness for well-scaled datasets. However, its prediction time increases with larger datasets because it computes distances to all training samples.
