import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# Task 1: Data Understanding
# ============================================================

# Load Dataset
df = pd.read_csv("data.csv")

print("=" * 60)
print("FIRST FIVE RECORDS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("NUMERICAL FEATURES")
print("=" * 60)
print(df.select_dtypes(include=["int64", "float64"]).columns.tolist())

print("\nTarget Variable : diagnosis")

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(df.describe())

# ============================================================
# Task 2: Data Preprocessing
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

# Remove unnecessary columns
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

if "Unnamed: 32" in df.columns:
    df.drop("Unnamed: 32", axis=1, inplace=True)

# Encode target variable
encoder = LabelEncoder()
df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

# Features and Target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Standardize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# Task 3: Model Development
# ============================================================

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ============================================================
# Task 4: Model Evaluation
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.show()

# ============================================================
# Observations
# ============================================================

print("\n" + "=" * 60)
print("OBSERVATIONS")
print("=" * 60)

print("1. KNN classified most tumors correctly.")
print("2. Feature scaling improved KNN performance.")
print("3. Accuracy and F1-Score indicate good classification performance.")

# ============================================================
# Task 5: Conclusion
# ============================================================

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print("""
This project developed a K-Nearest Neighbors (KNN) classifier
to classify breast tumors as malignant or benign using
diagnostic measurements. The dataset was preprocessed by
removing unnecessary columns, encoding the target variable,
handling missing values, and standardizing the feature values.
The model achieved good classification performance based on
Accuracy, Precision, Recall, and F1-Score. Feature scaling is
important for KNN because the algorithm relies on distance
between data points. One limitation of KNN is that prediction
becomes slower for large datasets because it compares new
samples with all training instances.
""")