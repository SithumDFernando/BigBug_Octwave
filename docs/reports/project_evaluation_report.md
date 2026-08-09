# Project Evaluation & 1.00 F1-Score Strategy Report

## 1. Current State of the Project
We have built a highly robust, industrial-grade pipeline:
- **Feature Engineering**: 24 features including time-bins, trust ratios, and risk indicators.
- **Modeling**: Exhaustive evaluation of 12 algorithms.
- **Hyperparameter Tuning**: Optuna-optimized models with Stratified K-Fold CV.
- **Ensembling**: Optuna-weighted soft voting of AdaBoost, XGBoost, and CatBoost.
- **Current CV Score**: **0.9959 F1-Score**.

Reaching a 0.9959 F1-Score on an imbalanced dataset is an exceptional feat. However, if the Kaggle Public Leaderboard shows that a perfect **1.00 F1-Score** is possible, it tells us something very specific about the dataset.

---

## 2. Why is 1.00 F1 Possible? (The Analysis)

In real-world credit card fraud, a 1.00 F1-Score is statistically impossible due to human unpredictability. If a perfect score is achievable on the Kaggle leaderboard, it implies one of two things:

1. **A Data Leakage exists:** Information from the target variable (`is_fraud`) is unintentionally encoded in the features (e.g., a specific combination of `merchant_category` and `transaction_hour` perfectly predicts fraud).
2. **Deterministic Generation:** The dataset is synthetic, and the creator used hard-coded rules to generate the fraud labels (e.g., `IF amount > 500 AND device_trust_score < 20 THEN fraud`).

If the top score is 1.00, we do not need "better machine learning algorithms" (we already maxed those out). We need to find the **Deterministic Rules** or optimize our **Probability Thresholds**.

---

## 3. How to Update the Project to Hit 1.00 F1

To bridge the gap from 0.9959 to 1.0000, we need to implement the following advanced strategies:

### Strategy A: Probability Threshold Optimization
Currently, our ensemble uses the default threshold of `0.5` (if probability >= 0.5, predict Fraud). 
- **The Fix:** Because F1-Score is extremely sensitive to the balance of Precision and Recall, the optimal threshold is rarely exactly 0.5. We must write a script that iterates through thresholds from `0.01` to `0.99` (e.g., `0.54` or `0.61`) to find the exact mathematical cut-off that perfectly maximizes the F1-Score on our CV folds.

### Strategy B: Decision Tree Rule Extraction (Finding the "Leak")
Since a 1.00 score exists, there is likely a hard-coded rule in the data.
- **The Fix:** We train a single `DecisionTreeClassifier` with no depth limit on the data. We then traverse the tree to extract the exact `IF/THEN` rules that result in 100% pure fraud leaves. 
- **Rule-Based Overrides:** Once we find these rules, we can apply them *on top* of our Ensemble. For example: `If Ensemble predicts 0, but Transaction_Amount > 800 and Location_Mismatch == 1, OVERRIDE to 1`.

### Strategy C: Level-1 Stacking (Meta-Learner)
Soft Voting (our current ensemble) just averages probabilities. 
- **The Fix:** We upgrade to a **Stacking Classifier**. We use the predictions of XGBoost, AdaBoost, and CatBoost as *inputs* to a new Logistic Regression model (the Meta-Learner). The Logistic Regression model learns exactly when XGBoost is right and when CatBoost is wrong, allowing it to correct the final few errors preventing a perfect score.

---

## 4. Proposed Next Steps
If you want to push for the perfect 1.00 F1-Score, I recommend we update the project by creating a `src/modeling/advanced_ensemble.py` script that does two things:
1. **Implements Stacking (Strategy C)** instead of Soft Voting.
2. **Implements Threshold Optimization (Strategy A)** to find the exact probability cut-off.

Would you like me to draft an Implementation Plan to execute these advanced 1.00 F1-Score techniques?
