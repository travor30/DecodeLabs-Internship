# Data Classification Using AI
# Project 2 - DecodeLabs Industrial Training | Batch 2026
# Topic: Supervised Learning - K-Nearest Neighbors on the Iris Dataset

# ------------------------------------------------------
# STEP 0: IMPORT LIBRARIES
# Before coding, I need to bring in the tools I'll use.
#
# pandas     → loading and exploring the dataset
# numpy      → numerical operations on arrays
# sklearn    → the machine learning library (scikit-learn)
#   datasets         → gives us the built-in Iris dataset
#   train_test_split → splits data into training and testing
#   StandardScaler   → scales (normalizes) features evenly
#   KNeighborsClassifier → the KNN algorithm itself
#   metrics          → tools to measure model performance
# -------------------------------------------------------
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# -------------------------------------------------------
# STEP 1: LOAD AND UNDERSTAND THE DATASET
# The Iris dataset is a classic benchmark dataset in ML.
# It has 150 flower samples, 3 species, 4 measurements.
#
# Features (what we measure):
#   - Sepal Length (cm)
#   - Sepal Width  (cm)
#   - Petal Length (cm)
#   - Petal Width  (cm)
#
# Target (what we want to predict):
#   - 0 = Iris Setosa
#   - 1 = Iris Versicolor
#   - 2 = Iris Virginica
# -------------------------------------------------------
def load_dataset():
    print("=" * 55)
    print("     PROJECT 2: Data Classification Using AI")
    print("     K-Nearest Neighbors | Iris Dataset")
    print("=" * 55)
    print()

    # Load the dataset from sklearn's built-in datasets
    iris = load_iris()

    # X holds the features (input data) — shape: (150, 4)
    # y holds the labels (correct answers) — shape: (150,)
    X = iris.data
    y = iris.target

    # Convert to a DataFrame so we can see it clearly
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["species"] = [iris.target_names[label] for label in y]

    print("[STEP 1] Dataset Loaded Successfully")
    print(f"  Total samples   : {X.shape[0]}")
    print(f"  Total features  : {X.shape[1]}")
    print(f"  Classes         : {list(iris.target_names)}")
    print()
    print("  First 5 rows of the dataset:")
    print(df.head().to_string(index=False))
    print()

    # Show basic statistics — mean, min, max, etc.
    print("  Dataset Statistics:")
    print(df.describe().round(2).to_string())
    print()

    return X, y, iris


# -------------------------------------------------------
# STEP 2: FEATURE SCALING (STANDARDIZATION)
# KNN works by measuring DISTANCE between data points.
# If one feature is on a scale of 0-1000 and another
# is on a scale of 0-5, the first feature will DOMINATE
# and give unfair results.
#
# StandardScaler fixes this by making every feature
# have a Mean = 0 and Standard Deviation = 1.
# This is called the "Gatekeeper Rule" — no feature
# is allowed to bully the others.
# -------------------------------------------------------
def scale_features(X_train, X_test):
    # Create the scaler object
    scaler = StandardScaler()

    # fit_transform on TRAINING data:
    # - fit()      → calculates the mean and std of training data
    # - transform() → subtracts mean, divides by std
    X_train_scaled = scaler.fit_transform(X_train)

    # transform() ONLY on TEST data (no re-fitting!)
    # We use the SAME mean/std from training — this prevents data leakage
    X_test_scaled = scaler.transform(X_test)

    print("[STEP 3] Feature Scaling Applied (StandardScaler)")
    print(f"  Training set mean after scaling  : {X_train_scaled.mean():.4f}  (should be ~0)")
    print(f"  Training set std after scaling   : {X_train_scaled.std():.4f}   (should be ~1)")
    print()

    return X_train_scaled, X_test_scaled, scaler


# -------------------------------------------------------
# STEP 3: TRAIN-TEST SPLIT
# We cannot test a model on data it was trained on —
# that's like giving a student the exam answers to study.
#
# Standard split: 80% training, 20% testing
# random_state=42 → fixes the shuffle so results are
#                   reproducible every time we run this
# stratify=y      → ensures each class is proportionally
#                   represented in both sets (balanced split)
# -------------------------------------------------------
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,       # 20% for testing = 30 samples
        random_state=42,     # seed for reproducibility
        stratify=y           # keep class balance in both sets
    )

    print("[STEP 2] Train-Test Split Complete")
    print(f"  Training samples  : {X_train.shape[0]}  (80%)")
    print(f"  Testing samples   : {X_test.shape[0]}   (20%)")
    print()

    return X_train, X_test, y_train, y_test


# -------------------------------------------------------
# STEP 4: BUILD AND TRAIN THE KNN MODEL
# K-Nearest Neighbors (KNN) is built on the
# "Proximity Principle": similar things exist in
# close proximity.
#
# When classifying a new flower:
#   1. Calculate the distance to ALL training examples
#   2. Find the K closest ones (K=5 here)
#   3. Take a majority vote among those K neighbors
#   4. Assign the majority class as the prediction
#
# Why K=5?
# - K=1 is too sensitive to noise (overfitting)
# - K=100 is too generic (underfitting)
# - K=5 is a commonly used starting point (the "elbow")
# -------------------------------------------------------
def train_model(X_train_scaled, y_train):
    # Instantiate the classifier with K=5
    model = KNeighborsClassifier(n_neighbors=5)

    # .fit() is where training happens
    # For KNN, "training" means memorizing the labeled data
    # (KNN is a lazy learner — it does all the work at prediction time)
    model.fit(X_train_scaled, y_train)

    print("[STEP 4] KNN Model Trained (K=5)")
    print("  The model has memorized the training data.")
    print("  It is now ready to classify new flower measurements.")
    print()

    return model


# -------------------------------------------------------
# STEP 5: MAKE PREDICTIONS
# Pass the SCALED test features through the model.
# The model looks at each test point, finds the 5
# nearest neighbors in the training set, and votes.
# -------------------------------------------------------
def make_predictions(model, X_test_scaled):
    predictions = model.predict(X_test_scaled)
    return predictions


# -------------------------------------------------------
# STEP 6: EVALUATE THE MODEL
# We have several metrics to judge performance:
#
# ACCURACY  → % of total correct predictions
#             (good for balanced datasets like Iris)
#
# F1 SCORE  → harmonic mean of Precision and Recall
#             Better metric when classes could be imbalanced
#             Formula: 2 * (Precision * Recall) / (Precision + Recall)
#
# CONFUSION MATRIX → shows WHERE the model made mistakes
#   - Rows = Actual class
#   - Columns = Predicted class
#   - Diagonal = correct predictions
#   - Off-diagonal = mistakes
#
# CLASSIFICATION REPORT → Precision, Recall, F1 per class
# -------------------------------------------------------
def evaluate_model(y_test, predictions, iris):
    accuracy = accuracy_score(y_test, predictions)
    f1       = f1_score(y_test, predictions, average="weighted")
    cm       = confusion_matrix(y_test, predictions)

    print("[STEP 5] Model Evaluation")
    print("-" * 40)
    print(f"  Accuracy Score  : {accuracy * 100:.2f}%")
    print(f"  F1 Score        : {f1 * 100:.2f}%")
    print()

    # Print the confusion matrix in a readable format
    print("  Confusion Matrix:")
    print("  (Rows = Actual, Columns = Predicted)")
    print()
    class_names = iris.target_names
    header = "  " + " " * 12 + "  ".join([f"{n:>10}" for n in class_names])
    print(header)
    for i, row in enumerate(cm):
        row_label = f"  {class_names[i]:>12}"
        row_vals  = "  ".join([f"{val:>10}" for val in row])
        print(row_label + "  " + row_vals)
    print()

    # Full breakdown per class
    print("  Classification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    ))

    return accuracy, f1, cm


# -------------------------------------------------------
# STEP 7: PREDICT ON CUSTOM INPUT
# This lets us test the model with a brand new flower
# measurement that wasn't in the original dataset.
# We scale it using the SAME scaler from training first.
# -------------------------------------------------------
def predict_custom(model, scaler, iris):
    print("-" * 40)
    print("[STEP 6] Custom Prediction Demo")
    print("  Testing the model with a brand new flower...")
    print()

    # A sample flower with these measurements (in cm):
    # sepal length=5.1, sepal width=3.5, petal length=1.4, petal width=0.2
    # (This should be Setosa based on the measurements)
    sample_1 = np.array([[5.1, 3.5, 1.4, 0.2]])
    sample_2 = np.array([[6.3, 3.3, 4.7, 1.6]])   # Should be Versicolor
    sample_3 = np.array([[7.2, 3.0, 5.8, 1.6]])   # Should be Virginica

    samples = [sample_1, sample_2, sample_3]
    labels  = ["Sample 1", "Sample 2", "Sample 3"]

    for label, sample in zip(labels, samples):
        # IMPORTANT: scale the new input using the fitted scaler
        sample_scaled = scaler.transform(sample)
        prediction    = model.predict(sample_scaled)[0]
        species       = iris.target_names[prediction]

        print(f"  {label}: sepal=[{sample[0][0]}, {sample[0][1]}], petal=[{sample[0][2]}, {sample[0][3]}]")
        print(f"  → Predicted Species: {species.upper()}")
        print()


# -------------------------------------------------------
# STEP 8: K VALUE COMPARISON
# Shows how different K values affect accuracy.
# This demonstrates the bias-variance tradeoff and
# helps explain why K=5 was chosen.
# -------------------------------------------------------
def compare_k_values(X_train_scaled, X_test_scaled, y_train, y_test):
    print("-" * 40)
    print("[BONUS] Comparing Different K Values")
    print()
    print(f"  {'K Value':>8}  {'Accuracy':>10}  {'F1 Score':>10}")
    print("  " + "-" * 34)

    for k in [1, 3, 5, 7, 9, 11, 15]:
        temp_model = KNeighborsClassifier(n_neighbors=k)
        temp_model.fit(X_train_scaled, y_train)
        temp_preds  = temp_model.predict(X_test_scaled)
        temp_acc    = accuracy_score(y_test, temp_preds) * 100
        temp_f1     = f1_score(y_test, temp_preds, average="weighted") * 100
        marker = " ← chosen" if k == 5 else ""
        print(f"  {k:>8}  {temp_acc:>9.2f}%  {temp_f1:>9.2f}%{marker}")

    print()


# -------------------------------------------------------
# MAIN FUNCTION — runs the full pipeline in order
# -------------------------------------------------------
def main():
    # --- PHASE 1: INPUT ---
    X, y, iris = load_dataset()

    # --- PHASE 2: PROCESS ---
    # Split first, then scale (IMPORTANT ORDER)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Train the model
    model = train_model(X_train_scaled, y_train)

    # Make predictions
    predictions = make_predictions(model, X_test_scaled)

    # --- PHASE 3: OUTPUT ---
    accuracy, f1, cm = evaluate_model(y_test, predictions, iris)

    # Custom prediction demo
    predict_custom(model, scaler, iris)

    # K value comparison
    compare_k_values(X_train_scaled, X_test_scaled, y_train, y_test)

    print("=" * 55)
    print("  Pipeline Complete!")
    print(f"  Final Accuracy : {accuracy * 100:.2f}%")
    print(f"  Final F1 Score : {f1 * 100:.2f}%")
    print("=" * 55)


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------
if __name__ == "__main__":
    main()
