# Task 2 — Data Classification Using AI

**Decode Labs Industrial Training | Batch 2026 | Artificial Intelligence**



## Task 2 Overview

A supervised machine learning pipeline that classifies iris flowers into three species using the **K-Nearest Neighbors (KNN)** algorithm. The project covers the complete ML workflow: loading a dataset, exploratory data analysis, train-test splitting, feature scaling, model training, prediction, and multi-metric evaluation.

> *"We do not write the rules. We provide history, and the machine derives the logic."*
> — Decode Labs Project Brief



## Objectives

 Master the **full supervised learning pipeline** from raw data to evaluated model
 Understand why **feature scaling** is critical for distance-based algorithms
 Learn to evaluate models beyond accuracy using **F1 Score and Confusion Matrix**
 Apply the **train-test split** correctly to prevent data leakage



##  Results

 Metric                             Score 

 **Accuracy**                **93.33%** (28/30 correct) 
 **F1 Score (weighted)**     **93.27%** 
 **Setosa accuracy**         100% — perfectly classified 
 **Versicolor accuracy**     100% — perfectly classified 
 **Virginica accuracy**      80% — 2 misclassified as Versicolor 

### Confusion Matrix

                Setosa   Versicolor   Virginica
Setosa            10          0           0
Versicolor         0         10           0
Virginica          0          2           8

The 2 errors are biologically expected — Virginica and Versicolor share overlapping measurements.



## Pipeline Architecture


INPUT                  PROCESS                    OUTPUT

Iris Dataset          Train-Test Split          Accuracy: 93.33%
(150 samples)          (80% / 20%)                F1 Score: 93.27%
4 features                                       Confusion Matrix
3 classes              StandardScaler             Classification Report
                       (Mean=0, Std=1)
                       
                       KNN (K=5)
                       model.fit()
                       model.predict()




## Dataset — The Iris Benchmark

 Property               Value 

 Total Samples       150 (balanced — 50 per class) 
 Features            4 (sepal length, sepal width, petal length, petal width) 
 Classes             3 (Setosa, Versicolor, Virginica) 
 Source               Built-in `sklearn.datasets.load_iris()` 



##  Key Concepts Demonstrated

### Why Feature Scaling Matters for KNN
KNN classifies by measuring **Euclidean distance**. Without scaling, features on larger scales dominate unfairly. `StandardScaler` brings all features to Mean=0, Std=1 — equal contribution guaranteed.

### The Data Leakage Rule

CORRECT order:  Split data  →  Fit scaler on TRAIN  →  Transform TRAIN  →  Transform TEST
WRONG order:    Fit scaler on ALL data  →  Split  (test data contaminates training statistics)


### K Value Selection
 K         Accuracy                                 Note 

 1         96.67%                        Overfits — too sensitive to noise 
 **5**    **93.33%**                    **Chosen — stable, standard starting point** 
 15        96.67%                        Underfitting risk at larger K 
 
### Accuracy vs F1 Score
Accuracy alone can be misleading on imbalanced datasets (the "Accuracy Mirage"). F1 Score — the harmonic mean of Precision and Recall — is the professional standard for classification evaluation.



##  File Structure


Project-2-Data-Classification/

 classifier.py        Full ML pipeline (run this)
 README.md            This file




##  How to Run

**Requirements:** Python 3.6+

bash
# Install dependencies
pip install scikit-learn pandas NumPy

# Run the classifier
python classifier.py


**Output Preview:**

[STEP 1] Dataset Loaded — 150 samples, 4 features, 3 classes
[STEP 2] Train-Test Split — 120 training / 30 testing
[STEP 3] Feature Scaling Applied (Mean ≈ 0, Std ≈ 1)
[STEP 4] KNN Model Trained (K=5)
[STEP 5] Accuracy: 93.33% | F1: 93.27%
[STEP 6] Custom Prediction → Sample [5.1, 3.5, 1.4, 0.2] = SETOSA ✓
[BONUS]  K=1→96.67%, K=5→93.33%, K=15→96.67%







