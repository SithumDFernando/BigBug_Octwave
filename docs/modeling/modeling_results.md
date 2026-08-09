# Modeling Phase Execution Results
*OctWave 3.0 Credit Card Fraud Detection*

## 1. Overview
This document serves as the formal record of the machine learning modeling phase. We successfully trained, tuned, and evaluated 12 distinct machine learning models to identify fraudulent transactions within the engineered dataset.

## 2. Methodology Snapshot
*   **Target Metric**: F1-Score (as mandated by the competition rules to balance Precision and Recall on the 65:1 imbalanced dataset).
*   **Hyperparameter Tuning**: Optuna (Bayesian Optimization) with 20 trials per model.
*   **Cross-Validation**: 5-Fold Stratified Cross-Validation (ensuring the 1.51% fraud ratio was maintained across all folds).
*   **Imbalance Handling**: Native algorithmic weights (e.g., `scale_pos_weight`, `class_weight='balanced'`).

## 3. Final Leaderboard
The models were evaluated strictly on their Cross-Validation F1-Score. The results demonstrated the overwhelming superiority of Boosting algorithms on this tabular dataset.

| Rank | Model Name | F1-Score | Precision | Recall | PR-AUC | Accuracy |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **AdaBoost** | **0.9957** | 1.0000 | 0.9917 | 0.9994 | 0.9999 |
| 🥈 | **XGBoost** | **0.9874** | 1.0000 | 0.9753 | 0.9991 | 0.9996 |
| 🥉 | **CatBoost** | **0.9838** | 0.9769 | 0.9917 | 0.9997 | 0.9995 |
| 4 | **Gradient Boosting** | **0.9791** | 0.9917 | 0.9673 | 0.9918 | 0.9994 |
| 5 | **LightGBM** | **0.9785** | 1.0000 | 0.9583 | 0.9916 | 0.9994 |
| 6 | **Random Forest** | **0.9195** | 0.9913 | 0.8593 | 0.9675 | 0.9977 |
| 7 | **MLPClassifier** (Neural Net) | **0.8330** | 0.8246 | 0.8427 | 0.9052 | 0.9949 |
| 8 | **Support Vector Classifier** | **0.8205** | 0.8234 | 0.8253 | 0.8735 | 0.9946 |
| 9 | **ExtraTrees** | **0.8113** | 0.8720 | 0.7677 | 0.8884 | 0.9948 |
| 10 | **Logistic Regression** | **0.8077** | 0.6862 | 0.9837 | 0.8919 | 0.9929 |
| 11 | **K-Nearest Neighbors** | **0.5274** | 0.7740 | 0.4110 | 0.6045 | 0.9894 |
| 12 | **Gaussian Naive Bayes** | **0.3021** | 0.1852 | 0.8263 | 0.3863 | 0.9415 |

## 4. Analysis & Takeaways
1. **The Power of Boosting:** AdaBoost, XGBoost, and CatBoost achieved near-perfect scores. The engineered features provided massive signal separation, allowing these models to easily partition fraudulent transactions from legitimate ones.
2. **Precision vs. Recall:** 
   * Logistic Regression achieved an incredibly high Recall (98.37%), meaning it caught almost all the fraud. However, its Precision was only 68.62% (too many false alarms).
   * XGBoost achieved perfect Precision (100%), meaning every transaction it flagged was actually fraud, with a very high Recall (97.53%).
   * AdaBoost struck the ultimate balance, achieving perfect Precision (100%) and 99.17% Recall, securing the #1 spot.

## 5. Deliverables Generated
*   **Model Artifacts**: `.pkl` files for all 12 models saved in `models/<model_name>/best_model.pkl`.
*   **Prediction Files**: Output CSVs for all 12 models saved in `outputs/<model_name>/predictions.csv`.
*   **Full Tuning Log**: Detailed hyperparameter records saved in `outputs/model_results_log.csv`.
