# Industrial-Level Modeling Strategy & Testing Plan

This document outlines the detailed architecture and testing strategy to achieve industrial-level accuracy for the OctWave 3.0 Fraud Detection system, focusing heavily on robust hyperparameter tuning and model ensembling.

## 1. Objective
To build a highly accurate, robust machine learning pipeline capable of identifying fraudulent transactions in a severely imbalanced dataset (65:1 ratio). The goal is to maximize the model's ability to catch fraud (Recall) while maintaining a high Precision, ultimately optimizing the **PR-AUC (Precision-Recall Area Under Curve)** and **F1-Score**.

## 2. Base Models for Evaluation
Given the tabular nature of the dataset and the extreme class imbalance, we will evaluate the following state-of-the-art tree-based models:

1.  **XGBoost (Extreme Gradient Boosting)**
    *   **Why**: Industry standard for tabular data, handles missing values naturally, and supports scale_pos_weight for imbalance.
2.  **LightGBM (Light Gradient Boosting Machine)**
    *   **Why**: Extremely fast, efficient memory usage, handles large feature spaces well, and has strong native support for imbalanced datasets via `is_unbalance=True` or `scale_pos_weight`.
3.  **CatBoost (Categorical Boosting)**
    *   **Why**: Exceptional out-of-the-box performance, highly resistant to overfitting, and natively handles categorical features well. Includes `auto_class_weights='Balanced'`.
4.  **Random Forest**
    *   **Why**: Excellent baseline model. While boosting models often perform better on imbalanced data, RF is highly parallelizable and less prone to overfitting the minority class.

## 3. Addressing the 65:1 Class Imbalance
The primary challenge in this dataset is the extreme scarcity of fraudulent cases. We will tackle this at two levels:

### Algorithm-Level Interventions
*   Utilize class weighting parameters native to each model:
    *   XGBoost/LightGBM: `scale_pos_weight = count(negative) / count(positive)`
    *   CatBoost: `auto_class_weights='Balanced'`

### Data-Level Interventions (Resampling)
*   **SMOTE (Synthetic Minority Over-sampling Technique)**: We will test generating synthetic minority class samples.
*   **ADASYN**: Similar to SMOTE but focuses on generating samples next to the original samples which are wrongly classified using a k-Nearest Neighbors classifier.
*   *Note: Oversampling will ONLY be applied to the training folds during Cross-Validation to prevent data leakage.*

## 4. Hyperparameter Tuning with Optuna
We will leverage **Optuna** for automated Bayesian optimization to efficiently traverse the hyperparameter space.

### Optuna Strategy
*   **Trials**: 50-100 trials per base model.
*   **Optimization Metric**: **PR-AUC** or **F1-Score** (not ROC-AUC, as it can be overly optimistic on highly imbalanced data).
*   **Cross-Validation**: 5-Fold Stratified Cross-Validation for each trial.

### Search Spaces
*   **XGBoost/LightGBM**:
    *   `learning_rate`: [0.01, 0.3] (log uniform)
    *   `max_depth`: [3, 10]
    *   `subsample` / `bagging_fraction`: [0.5, 1.0]
    *   `colsample_bytree` / `feature_fraction`: [0.5, 1.0]
    *   `min_child_weight`: [1, 10]
    *   Regularization (`reg_alpha`, `reg_lambda`): [1e-8, 10.0] (log uniform)
*   **CatBoost**:
    *   `learning_rate`: [0.01, 0.3] (log uniform)
    *   `depth`: [4, 10]
    *   `l2_leaf_reg`: [1, 10]

## 5. Model Ensembling
To achieve the highest possible accuracy and generalization, we will combine the optimally tuned base models.

### Approach 1: Soft Voting Classifier
*   Average the predicted probabilities of the top-performing XGBoost, LightGBM, and CatBoost models.
*   Weighting can be applied based on each model's individual cross-validation score.

### Approach 2: Stacking (Meta-Learning)
*   Use a Level-0 layer consisting of our tuned models (XGBoost, LightGBM, CatBoost).
*   Train a Level-1 Meta-Learner (typically a Logistic Regression or a shallow, heavily regularized tree model) on the out-of-fold predictions from Level-0.
*   This approach often squeezes out an extra 1-3% in PR-AUC by learning the specific biases of the base models.

## 6. Evaluation Protocol
*   **Cross-Validation Strategy**: We strictly enforce **Stratified K-Fold (k=5 or 10)** splitting. This ensures that the 1.51% fraud ratio is maintained consistently across all training and validation folds.
*   **Primary Metrics**:
    *   **PR-AUC (Precision-Recall AUC)**: The absolute best metric for extreme class imbalance.
    *   **F1-Score**: The harmonic mean of Precision and Recall.
*   **Secondary Metrics**:
    *   **Recall (Sensitivity)**: How many actual frauds did we catch? (Crucial for minimizing financial loss).
    *   **Precision**: When we flag a transaction as fraud, how often are we right? (Crucial for minimizing customer friction).

## 7. Implementation Roadmap
1.  **Script `src/modeling/train.py`**:
    *   Setup the Optuna objective functions for XGBoost, LightGBM, and CatBoost.
    *   Integrate Stratified K-Fold CV.
    *   Implement saving logic for the best tuned models.
2.  **Script `src/modeling/ensemble.py` (Optional/Later)**:
    *   Load the tuned base models.
    *   Implement Voting and Stacking classifiers.
3.  **Evaluate & Validate**:
    *   Generate a detailed report of the final ensemble's performance on a holdout test set.
