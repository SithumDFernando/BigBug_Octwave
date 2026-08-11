
================================================================================
NOTEBOOK: 01_EDA
Total cells: 50
================================================================================

--- [MARKDOWN Cell 0] ---
# 📊 Exploratory Data Analysis — OctWave 3.0 Fraud Detection

**Objective:** Explore and understand the credit card transaction dataset before building a fraud detection model.  
**Dataset:** Simulated credit card transactions with 8,000 training samples and 2,000 test samples.  
**Target:** `is_fraud` — binary classification (0 = Legitimate, 1 = Fraudulent)

---

--- [MARKDOWN Cell 1] ---
## 1. Setup & Imports

--- [MARKDOWN Cell 3] ---
## 2. Data Loading

We load both the training and test datasets to explore their structure and ensure consistency.

--- [MARKDOWN Cell 5] ---
## 3. First Look at the Data

### 3.1 Training Set — First 10 Rows

--- [MARKDOWN Cell 7] ---
### 3.2 Data Types & Memory Usage

--- [MARKDOWN Cell 9] ---
### 3.3 Statistical Summary (Numerical Features)

--- [MARKDOWN Cell 11] ---
### 3.4 Categorical Feature — `merchant_category`

--- [MARKDOWN Cell 13] ---
## 4. Data Quality Assessment

### 4.1 Missing Values

--- [MARKDOWN Cell 15] ---
### 4.2 Duplicate Rows

--- [MARKDOWN Cell 17] ---
## 5. Target Variable — Class Distribution

The most critical characteristic of this dataset is the **severe class imbalance**.

--- [MARKDOWN Cell 19] ---
## 6. Feature Distributions

### 6.1 Numerical Feature Histograms

--- [MARKDOWN Cell 21] ---
### 6.2 Boxplots — Fraud vs Legitimate Transactions

--- [MARKDOWN Cell 23] ---
### 6.3 Binary Feature Distributions

--- [MARKDOWN Cell 25] ---
### 6.4 Merchant Category Distribution

--- [MARKDOWN Cell 27] ---
## 7. Correlation Analysis

### 7.1 Correlation Heatmap

--- [MARKDOWN Cell 29] ---
### 7.2 Correlation with Target Variable (`is_fraud`)

--- [MARKDOWN Cell 31] ---
## 8. Fraud Signal Deep Dive

### 8.1 Transaction Hour — Fraud Distribution

--- [MARKDOWN Cell 33] ---
### 8.2 Device Trust Score — Fraud Distribution

--- [MARKDOWN Cell 35] ---
### 8.3 Amount — Fraud Distribution

--- [MARKDOWN Cell 37] ---
### 8.4 Velocity (Last 24h) — Fraud Distribution

--- [MARKDOWN Cell 39] ---
## 9. Feature Interactions

### 9.1 Foreign Transaction × Location Mismatch (Key Fraud Signals)

--- [MARKDOWN Cell 41] ---
### 9.2 Pairplot — Key Features Colored by Fraud

--- [MARKDOWN Cell 43] ---
## 10. Outlier Analysis

Examining outliers using the IQR (Interquartile Range) method.

--- [MARKDOWN Cell 45] ---
## 11. Train vs Test Distribution Comparison

Ensuring no significant distribution drift between training and test sets.

--- [MARKDOWN Cell 47] ---
## 12. Zero & Edge Value Analysis

--- [MARKDOWN Cell 49] ---
## 13. EDA Summary & Key Takeaways

### Data Quality
| Check | Result |
|---|---|
| Missing values | ✅ None |
| Duplicates | ✅ None |
| Data type issues | ✅ None |
| Train/Test drift | ✅ Distributions aligned |
| Negative values | ✅ None |

### Critical Finding: Severe Class Imbalance
- **98.49% Legitimate vs 1.51% Fraud** (65:1 ratio)
- Must be addressed during modeling with SMOTE, class weights, or ensemble methods

### Fraud Signal Strength (Ranked)

| Rank | Feature | Signal Strength | Details |
|---|---|---|---|
| 1 | `location_mismatch` | 🔴 **Strong** (5.5×) | 8.24% fraud rate when 1, vs 1.51% baseline |
| 2 | `foreign_transaction` | 🔴 **Strong** (5.4×) | 8.12% fraud rate when 1, vs 1.51% baseline |
| 3 | `device_trust_score` | 🔴 **Strong** (4×) | 6.04% fraud rate when score 20-40 |
| 4 | `transaction_hour` | 🟠 **Moderate** (3.3×) | 4.91% fraud rate during night hours (0-5) |
| 5 | `velocity_last_24h` | 🟡 **Mild** | Positive correlation (+0.110) |
| 6 | `amount` | 🟡 **Mild** | 3.04% fraud rate for $500+ (2× baseline) |
| 7 | `cardholder_age` | ⚪ **None** | Zero correlation with fraud |

### Feature Interactions
- **`foreign_transaction` × `location_mismatch`** is the strongest combined signal
- This interaction should be engineered as a feature during preprocessing

### Recommendations for Next Steps
1. **Drop** `transaction_id` (no predictive value)
2. **Encode** `merchant_category` (one-hot encoding — only 5 categories)
3. **Engineer** interaction features (`foreign_transaction × location_mismatch`, `is_night`)
4. **Address imbalance** using SMOTE or class weights during modeling
5. **Consider dropping** `cardholder_age` (zero signal) or keep for completeness
6. **Use tree-based models** (XGBoost, LightGBM) — they handle this type of data well

---
*Notebook completed. Proceed to `02_Data_Preprocessing.ipynb` for data cleaning and feature engineering.*
================================================================================
NOTEBOOK: 02_Data_Preprocessing
Total cells: 42
================================================================================

--- [MARKDOWN Cell 0] ---
# 🔧 Data Cleaning & Feature Engineering — OctWave 3.0 Fraud Detection

**Objective:** Transform raw credit card transaction data into a clean, feature-rich dataset ready for model training.  
**Pipeline:** Load Raw Data → Validate & Clean → Feature Engineering → Encoding → Save Processed Data  
**Key Principle:** Every transformation is deliberate and justified by insights from our EDA (`01_EDA.ipynb`).

---

--- [MARKDOWN Cell 1] ---
## 1. Setup & Imports

--- [MARKDOWN Cell 3] ---
## 2. Load Raw Data

We load both the training and test datasets from the raw data directory. The test set's `transaction_id` values are preserved separately for final submission formatting.

--- [MARKDOWN Cell 6] ---
## 3. Data Quality Validation

Before any transformation, we systematically verify data quality. This confirms our EDA findings and ensures no issues have been introduced during data collection.

### 3.1 Missing Values Check

--- [MARKDOWN Cell 8] ---
### 3.2 Duplicate Rows Check

--- [MARKDOWN Cell 10] ---
### 3.3 Data Type Verification

--- [MARKDOWN Cell 12] ---
### 3.4 Binary Column Verification

Confirm that `foreign_transaction` and `location_mismatch` are truly binary (0/1) with no unexpected values.

--- [MARKDOWN Cell 14] ---
### 3.5 Edge Values & Outlier Decision

From our EDA, we identified some edge values. Here we document our decision **not to remove outliers**.

--- [MARKDOWN Cell 16] ---
---

## 4. Data Cleaning Pipeline

### 4.1 Drop `transaction_id`

`transaction_id` is an arbitrary unique identifier with **no predictive value**. It must be removed from the feature set before training. However, we preserve the test set IDs for final submission formatting.

--- [MARKDOWN Cell 18] ---
### 4.2 Separate Target Variable

We separate `is_fraud` from the training features. The test set does not have this column.

--- [MARKDOWN Cell 20] ---
---

## 5. Feature Engineering

We now create new features based on the **fraud signals** identified in our EDA. Each feature is motivated by a specific data insight.

To ensure consistency, we combine train and test sets for feature engineering, then split them back afterward.

### Feature Engineering Strategy

| # | Feature | Type | Motivation |
|---|---------|------|------------|
| 1 | `is_night_transaction` | Binary | Night hours (0–5) have **3.3× higher** fraud rate (4.91% vs 1.51% baseline) |
| 2 | `time_of_day_category` | Categorical | Captures broader temporal patterns beyond just night hours |
| 3 | `amount_to_trust_ratio` | Continuous | High amounts on low-trust devices are suspicious |
| 4 | `amount_velocity_ratio` | Continuous | Average spend per recent transaction — detects account takeover |
| 5 | `is_high_risk_location` | Binary | Either `foreign_transaction=1` OR `location_mismatch=1` (~8% fraud rate) |
| 6 | `location_anomaly_score` | Integer | Combined location risk severity (0–2) |
| 7 | `is_high_amount` | Binary | Transactions >$500 have **2× baseline** fraud rate |
| 8 | `is_low_trust` | Binary | Device trust score <40 has **4× baseline** fraud rate |

--- [MARKDOWN Cell 22] ---
### 5.1 Time-Based Risk Features

From EDA, we know that **night-time transactions (hours 0–5) have a fraud rate of 4.91%** — over 3× the baseline of 1.51%. We create two features to capture this temporal pattern.

--- [MARKDOWN Cell 24] ---
### 5.2 Trust & Anomaly Ratio Features

These continuous features capture **interactions** between raw features that are individually predictive of fraud:
- **`amount_to_trust_ratio`**: Fraudsters often attempt high-value transactions on low-trust devices.
- **`amount_velocity_ratio`**: High spend-per-recent-transaction indicates potential account takeover.

--- [MARKDOWN Cell 26] ---
### 5.3 High-Risk Location Features

From EDA, `foreign_transaction` (5.4× uplift) and `location_mismatch` (5.5× uplift) are the **two strongest individual fraud signals**. We create composite features to capture their interaction:
- **`is_high_risk_location`**: Either flag is set → high risk.
- **`location_anomaly_score`**: Sum of both flags (0, 1, or 2) → severity of location anomaly.

--- [MARKDOWN Cell 28] ---
### 5.4 Strategic Binning Features

We create binary threshold features for key continuous variables based on EDA-identified breakpoints:
- **`is_high_amount`**: Transactions > $500 have a 3.04% fraud rate (2× baseline).
- **`is_low_trust`**: Trust score < 40 has a ~6% fraud rate (4× baseline).

--- [MARKDOWN Cell 30] ---
---

## 6. Categorical Encoding

### 6.1 One-Hot Encoding

We apply **One-Hot Encoding** to the categorical columns:
- `merchant_category` — 5 unique values: Food, Clothing, Travel, Electronics, Grocery
- `time_of_day_category` — 4 unique values: Night, Morning, Afternoon, Evening

**Why One-Hot Encoding?**
- Only 5 + 4 = 9 new columns — minimal dimensionality increase.
- No ordinal relationship exists between categories.
- Works well with both tree-based and linear models.
- We use `drop_first=False` to preserve all information for tree-based models.

--- [MARKDOWN Cell 32] ---
---

## 7. Feature Scaling Discussion

Feature scaling is **model-dependent**:

| Model Type | Scaling Needed? | Reason |
|---|---|---|
| Random Forest, XGBoost, LightGBM, CatBoost | ❌ No | Tree-based models split on feature values — scale invariant |
| Logistic Regression, SVM, Neural Networks | ✅ Yes | Distance/gradient-based models are scale-sensitive |

**Decision:** We **do not apply scaling** in this preprocessing step because:
1. Our primary models will be tree-based (XGBoost, LightGBM, Random Forest).
2. If we later add a Logistic Regression baseline, we'll apply `StandardScaler` in the model pipeline (not in the data itself).
3. Keeping raw feature values preserves interpretability and allows tree-based models to access natural distributions.

**Features that would need scaling if used with linear models:**
- `amount` (range: 0 – 1,471)
- `device_trust_score` (range: 25 – 99)
- `velocity_last_24h` (range: 0 – 9)
- `cardholder_age` (range: 18 – 69)
- `transaction_hour` (range: 0 – 23)
- `amount_to_trust_ratio` (continuous, wide range)
- `amount_velocity_ratio` (continuous, wide range)

--- [MARKDOWN Cell 34] ---
---

## 8. Split & Save Processed Data

### 8.1 Split Combined Dataset Back to Train/Test

--- [MARKDOWN Cell 36] ---
### 8.2 Save to Disk

--- [MARKDOWN Cell 38] ---
---

## 9. Final Feature Matrix Overview

A comprehensive view of the final engineered feature matrix.

--- [MARKDOWN Cell 41] ---
---

## 10. Summary

### Data Cleaning
| Step | Action | Result |
|---|---|---|
| Missing values | Verified none exist | ✅ Clean |
| Duplicates | Verified none exist | ✅ Clean |
| Data types | Verified consistency | ✅ Consistent |
| Binary columns | Verified {0,1} values | ✅ Valid |
| Outliers | Kept (signal, not noise) | ✅ Preserved |
| `transaction_id` | Dropped from features | ✅ Saved separately |

### Feature Engineering
| Feature | Type | Motivation |
|---|---|---|
| `is_night_transaction` | Binary | Night hours have 3.3× fraud uplift |
| `time_of_day_category` | One-Hot (4 cols) | Broader temporal patterns |
| `amount_to_trust_ratio` | Continuous | High amount + low trust = suspicious |
| `amount_velocity_ratio` | Continuous | Spend per recent transaction |
| `is_high_risk_location` | Binary | Location-based risk flag |
| `location_anomaly_score` | Integer (0–2) | Combined location risk severity |
| `is_high_amount` | Binary | Amount >$500 → 2× fraud rate |
| `is_low_trust` | Binary | Trust <40 → 4× fraud rate |
| `merchant_category` | One-Hot (5 cols) | Categorical encoding |

### Output Files
- `data/processed/train_engineered.csv` — 24 columns (23 features + target)
- `data/processed/test_engineered.csv` — 23 columns (features only)
- `data/processed/test_transaction_ids.csv` — Preserved for submission

### Key Decisions
1. **No outlier removal** — Outliers are fraud signals in this domain.
2. **No feature scaling** — Primary models are tree-based (scale invariant). Scaling will be applied in model pipelines if needed.
3. **Raw features preserved** — Originals kept alongside engineered features for tree-based models to access natural distributions.
4. **One-Hot Encoding without drop_first** — Preserves all information; works best with tree models.

**→ Next: `03_Model_Training.ipynb` — Model development, comparison, and selection.**
================================================================================
NOTEBOOK: 03_Model_Training
Total cells: 11
================================================================================

--- [MARKDOWN Cell 0] ---
# Exhaustive Model Training & Ensembling Pipeline

This notebook implements the exhaustive 12-model pipeline with centralized logging, Optuna hyperparameter optimization, and a custom Optuna-weighted Soft Voting Ensemble.

**Objectives:**
1. Evaluate 12 classification models (Tree-based, Linear, Distance, Neural Network).
2. Use **Optuna** to run 20 trials per model, optimizing for **F1-Score**.
3. Handle extreme class imbalance (65:1) using algorithmic weights.
4. Save models and predictions in an organized nested folder structure (`models/<name>/`, `outputs/<name>/`).
5. Track all metrics in a centralized log (`outputs/model_results_log.csv`).
6. Ensembling Phase: Pick top 5 models and find optimal soft voting weights via Optuna.

--- [MARKDOWN Cell 4] ---
## Dynamic Optuna Pipeline

We define a general-purpose Optuna objective. The pipeline trains models via 5-Fold Stratified CV, logging the results of the best trial to our CSV tracker, and storing artifacts securely.

--- [MARKDOWN Cell 7] ---
## Train All Models

*Note: Running 20 trials for all 12 models can take significant time. Uncomment the loop to execute the exhaustive search.*

--- [MARKDOWN Cell 9] ---
## Optuna-Weighted Ensembling

Here we select the top 3 to 5 performing models from our log, load their Out-Of-Fold predictions, and use Optuna to find the optimal blending weights to maximize the ensemble's F1-Score.
================================================================================
NOTEBOOK: 04_Evaluation_and_Prediction
Total cells: 29
================================================================================

--- [MARKDOWN Cell 0] ---
# 🏆 Final Evaluation & Prediction Pipeline

This capstone notebook performs the **final evaluation** of our trained models and generates the **Kaggle submission file**.

**Objectives:**
1. Load the best models from Notebook 03 (9 base models + Optuna-weighted Ensemble).
2. Perform comprehensive evaluation: Confusion Matrices, ROC Curves, Precision-Recall Curves.
3. Threshold Analysis — sweep thresholds (0.0 → 1.0) to find the optimal decision boundary.
4. Feature Importance — cross-model comparison of the most predictive features.
5. Generate final predictions on the unseen test set and create `submission.csv`.
6. Validate the submission format against `sample_submission.csv`.

---

--- [MARKDOWN Cell 1] ---
## Section 1: Setup & Data Loading

--- [MARKDOWN Cell 5] ---
---
## Section 2: Load Best Models

We load the top 5 individual models and the Optuna-weighted Soft Voting Ensemble metadata.

--- [MARKDOWN Cell 8] ---
---
## Section 3: Final Cross-Validation Evaluation

We re-evaluate the top models using **5-Fold Stratified CV** to generate out-of-fold predictions. These OOF predictions allow us to compute unbiased confusion matrices, ROC curves, and Precision-Recall curves.

--- [MARKDOWN Cell 14] ---
---
## Section 4: Threshold Analysis

The default classification threshold is **0.5**. However, with imbalanced data, a different threshold might yield better F1-Score. We sweep thresholds from 0.0 to 1.0 for both the **Ensemble** and **AdaBoost** models.

We evaluate two strategies:
- **Standard (0.5)**: The default decision boundary.
- **Aggressive (~0.3)**: A lower threshold that maximizes recall at the cost of precision.

--- [MARKDOWN Cell 18] ---
### Threshold Analysis Takeaways

- The **standard threshold (0.5)** already achieves near-perfect F1 for both the Ensemble and AdaBoost, which is expected given their already-high confidence outputs.
- The **aggressive threshold (0.3)** may catch additional borderline fraud cases (higher Recall) but at the cost of more false positives (lower Precision).
- The **optimal threshold** (as determined by the F1 sweep) confirms whether 0.5 is indeed the best or if a slight adjustment improves performance.

For our final submission, we will use the **standard 0.5 threshold** unless the optimal threshold shows a clear improvement.

--- [MARKDOWN Cell 19] ---
---
## Section 5: Feature Importance Analysis

We extract feature importances from the top tree-based models to understand which features drive fraud detection.

--- [MARKDOWN Cell 23] ---
---
## Section 6: Generate Test Predictions & Submission

We generate predictions on the unseen test data using the **Optuna-weighted Soft Voting Ensemble**, validate the submission format, and save the final output.

--- [MARKDOWN Cell 28] ---
---
## Section 7: Summary & Conclusion

### End-to-End Methodology

| Phase | Notebook | Description |
|---|---|---|
| **EDA** | `01_EDA.ipynb` | Explored distributions, correlations, and fraud patterns across 9 features. Identified `foreign_transaction`, `location_mismatch`, and `device_trust_score` as dominant fraud signals. |
| **Preprocessing** | `02_Data_Preprocessing.ipynb` | Cleaned data, one-hot encoded `merchant_category`, and engineered 16 new features including `is_night_transaction`, `amount_to_trust_ratio`, `is_high_risk_location`, and time-of-day categories. |
| **Model Training** | `03_Model_Training.ipynb` | Trained 9 models with Optuna-tuned hyperparameters (20 trials each, 5-Fold Stratified CV). Built an Optuna-weighted Soft Voting Ensemble from the top 5 models. |
| **Evaluation** | `04_Evaluation_and_Prediction.ipynb` | Performed final OOF evaluation with confusion matrices, ROC curves, PR curves, threshold analysis, and feature importance. Generated validated submission files. |

### Final Model Selection Rationale

We selected the **Optuna-weighted Soft Voting Ensemble** as our primary model because:

1. **Diversity**: It combines 5 different algorithms (AdaBoost, XGBoost, LightGBM, CatBoost, GradientBoosting), reducing the risk of model-specific overfitting.
2. **Optimized Weights**: Optuna discovered the mathematically optimal weighting for each member, outperforming equal-weight averaging.
3. **Near-Perfect Metrics**: CV F1 = 0.9959, Precision = 1.0, Recall = 0.9917.
4. **Generalization**: The ensemble's diverse decision boundaries provide the most robust predictions for the unseen private leaderboard data.

### Key Findings

1. **Boosting dominates**: All top-5 models are boosting algorithms, confirming their superiority on structured/tabular fraud detection datasets.
2. **Feature engineering matters**: The 16 engineered features (especially `is_high_risk_location`, `amount_to_trust_ratio`, `is_night_transaction`) provided massive signal that simpler models couldn't leverage.
3. **Class imbalance is manageable algorithmically**: Native `class_weight='balanced'` and `scale_pos_weight` parameters eliminated the need for SMOTE or other resampling techniques.
4. **The dataset is highly predictable**: Near-perfect F1 across multiple model families suggests the synthetic data follows learnable deterministic rules.

---