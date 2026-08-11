# OctWave 3.0 Challenge - Submission Report

**Team Details:** Team BigBug

## 1. Approach

Our approach centers on domain-driven feature engineering and optimized boosting algorithms, avoiding synthetic resampling techniques (like SMOTE). We handled the extreme class imbalance (1.51% fraud) by applying algorithmic class weights to penalize minority class misclassification. After engineering 16 new features, we systematically evaluated 12 classification algorithms. We utilized Bayesian Optimization (Optuna) for hyperparameter tuning (20 trials per model), validating results via 5-Fold Stratified Cross-Validation.

## 2. Final Model

Our final model is an **Optuna-Weighted Soft Voting Ensemble** that blends the predicted probabilities of our top 3 performing gradient boosting algorithms:

- AdaBoost (weight: 0.284)
- XGBoost (weight: 0.423)
- CatBoost (weight: 0.292)

By blending diverse boosting models, this ensemble minimizes variance and prevents model-specific overfitting. It generates final predictions using a standard 0.5 decision threshold, prioritizing zero false alarms.

## 3. Validation Method

We used **5-Fold Stratified Cross-Validation**. All metrics reported are Out-of-Fold (OOF) predictions, providing an unbiased estimate of our true performance on unseen data.

## 4. Final Performance (OOF)

- **F1-Score:** 0.9959
- **Precision:** 1.0000
- **Recall:** 0.9917
- **PR-AUC:** 0.9991

## 5. Key Findings & Challenges

- **Feature Engineering was Decisive:** We engineered 16 custom features from the raw data. Features like `location_anomaly_score`, `amount_to_trust_ratio`, and `is_night_transaction` provided massive signal uplift.
- **Algorithmic Class Weighting Outperforms Resampling:** Native class weight scaling enabled us to reach F1 scores above 0.99 without the data distortion risks associated with oversampling (SMOTE).
- **Boosting Algorithms Dominate:** All five of our best individual models were gradient boosting variants, heavily outperforming linear models and traditional random forests.
- **Highly Predictable Synthetic Dataset:** The near-perfect metrics across multiple models confirm that the simulated dataset follows learnable, deterministic rules, allowing for extreme precision.
- **Ensemble Diversity as Insurance:** Blending 3 distinct boosting algorithms offers the best protection against overfitting to the public leaderboard, maximizing stability for the 70% unseen private test data.
