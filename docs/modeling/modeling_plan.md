# Industrial-Level Modeling Strategy & Testing Plan

This document outlines the detailed architecture and testing strategy used to achieve industrial-level accuracy for the OctWave 3.0 Fraud Detection system, focusing heavily on robust hyperparameter tuning and model ensembling.

## 1. Objective
To build a highly accurate, robust machine learning pipeline capable of identifying fraudulent transactions in a severely imbalanced dataset (65:1 ratio). The goal is to maximize the model's ability to catch fraud (Recall) while maintaining a high Precision, ultimately optimizing the **PR-AUC (Precision-Recall Area Under Curve)** and **F1-Score**.

## 2. Exhaustive Model Evaluation
Given the tabular nature of the dataset and the extreme class imbalance, we expanded our evaluation to an exhaustive list of 12 state-of-the-art models:

1.  **XGBoost (Extreme Gradient Boosting)**
2.  **LightGBM (Light Gradient Boosting Machine)**
3.  **CatBoost (Categorical Boosting)**
4.  **Random Forest**
5.  **ExtraTrees Classifier**
6.  **Gradient Boosting Classifier**
7.  **AdaBoost**
8.  **Logistic Regression**
9.  **Support Vector Classifier (SVC)**
10. **K-Nearest Neighbors (KNN)**
11. **Gaussian Naive Bayes**
12. **Multi-Layer Perceptron (Neural Network)**

## 3. Addressing the 65:1 Class Imbalance
The primary challenge in this dataset is the extreme scarcity of fraudulent cases. We successfully tackled this using **Algorithm-Level Interventions**:
*   Utilized class weighting parameters native to each model:
    *   XGBoost/LightGBM: `scale_pos_weight = count(negative) / count(positive)`
    *   CatBoost: `auto_class_weights='Balanced'`
    *   Scikit-Learn Models: `class_weight='balanced'`
*   *Note: Because the native algorithmic weights pushed the F1-Scores to near-perfect levels (>0.99), data-level interventions like SMOTE were deemed unnecessary for the baseline models.*

## 4. Hyperparameter Tuning with Optuna
We leveraged **Optuna** for automated Bayesian optimization to efficiently traverse the hyperparameter space.

### Optuna Strategy
*   **Trials**: 20 trials per base model (to balance the execution time across 12 models).
*   **Optimization Metric**: **F1-Score** (as mandated by competition rules).
*   **Cross-Validation**: 5-Fold Stratified Cross-Validation for each trial.

## 5. Artifact Storage Architecture
To maintain a production-grade workspace, all models and predictions were isolated into their own dedicated subdirectories:
*   **Models**: `models/<model_name>/best_model.pkl`
*   **Predictions**: `outputs/<model_name>/predictions.csv`
*   **Logging**: A central master table at `outputs/model_results_log.csv` tracking all metrics and hyperparameters.

## 6. Model Ensembling (Next Phase)
To achieve the highest possible accuracy and generalization, we will combine the optimally tuned top base models (AdaBoost, XGBoost, CatBoost).
*   **Soft Voting Classifier**: Average the predicted probabilities of the top models.
*   **Meta-Optimization**: We will use Optuna again to discover the mathematical optimum weighting for the ensemble members (e.g., `0.5 * Ada + 0.3 * XGB + 0.2 * Cat`).
