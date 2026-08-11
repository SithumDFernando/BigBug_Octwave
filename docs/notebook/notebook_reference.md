# 📓 Comprehensive Notebook Reference — Team BigBug, OctWave 3.0

> This document is a complete, chronological reference of **everything** we did, **every decision** we made, and **every finding** we discovered — from raw data loading to final submission generation. It is intended as the single source of truth for building the final consolidated Jupyter Notebook.

---

## Table of Contents

1. [Competition Context](#1-competition-context)
2. [Phase 1: Data Loading & Initial Inspection](#2-phase-1-data-loading--initial-inspection)
3. [Phase 2: Exploratory Data Analysis (EDA)](#3-phase-2-exploratory-data-analysis-eda)
4. [Phase 3: Data Cleaning & Validation](#4-phase-3-data-cleaning--validation)
5. [Phase 4: Feature Engineering](#5-phase-4-feature-engineering)
6. [Phase 5: Model Training & Hyperparameter Tuning](#6-phase-5-model-training--hyperparameter-tuning)
7. [Phase 6: Ensemble Construction](#7-phase-6-ensemble-construction)
8. [Phase 7: Advanced Ensemble (Stacking)](#8-phase-7-advanced-ensemble-stacking)
9. [Phase 8: Final Model Selection & Submission](#9-phase-8-final-model-selection--submission)
10. [Final Evaluation, Visualization & Submission Strategy](#10-final-evaluation-visualization--submission-strategy)
11. [Final Results Summary](#11-final-results-summary)

---

## 1. Competition Context

### Problem Statement
Build a binary classifier to detect **fraudulent credit card transactions** (`is_fraud = 0 or 1`) from a simulated dataset with extreme class imbalance.

### Key Constraints
- **Evaluation Metric:** F1-Score (balances Precision and Recall)
- **Timeline:** 9th August 2026 (12:00 NOON) → 11th August 2026 (11:59 PM)
- **Submission Limit:** Maximum 10 submissions per day
- **Final Evaluation:** Up to 2 submissions may be selected for the Private Leaderboard
- **External Data:** Strictly prohibited

### Dataset Statistics
| Property | Train (`train.csv`) | Test (`test.csv`) |
|---|---|---|
| Rows | 8,000 | 2,000 |
| Features | 9 + target (`is_fraud`) | 9 (no target) |
| File size | ~297 KB | ~70 KB |

### Raw Features (9 columns + target)
| Feature | Type | Range / Values | Description |
|---|---|---|---|
| `transaction_id` | Integer | 1 – 10,000 | Unique identifier (NOT a predictive feature) |
| `amount` | Float | 0.00 – 1,471.04 | Monetary value of the transaction |
| `transaction_hour` | Integer | 0 – 23 | Hour of the day |
| `merchant_category` | Categorical | Food, Clothing, Travel, Electronics, Grocery | Merchant type |
| `foreign_transaction` | Binary | 0, 1 | Was the transaction foreign? |
| `location_mismatch` | Binary | 0, 1 | Location mismatch flag |
| `device_trust_score` | Integer | 25 – 99 | Device trust rating |
| `velocity_last_24h` | Integer | 0 – 9 | Transactions in last 24 hours |
| `cardholder_age` | Integer | 18 – 69 | Age of the cardholder |
| `is_fraud` | Binary | 0, 1 | **Target variable** (train only) |

### Class Distribution (Target)
| Class | Count | Percentage |
|---|---|---|
| 0 (Legitimate) | 7,879 | 98.49% |
| 1 (Fraud) | 121 | 1.51% |

**Imbalance Ratio: 65:1** — This is extreme imbalance and the single most critical challenge.

---

## 2. Phase 1: Data Loading & Initial Inspection

**Source:** `src/data_processing/analyze_data.py`

### What We Did
```python
import pandas as pd
train_df = pd.read_csv('data/raw/train.csv')
test_df  = pd.read_csv('data/raw/test.csv')
```

### Checks Performed
1. **Shape verification:**
   - `train_df.shape` → (8000, 10) — 9 features + 1 target ✅
   - `test_df.shape` → (2000, 9) — 9 features, no target ✅

2. **Missing values check:**
   ```python
   train_df.isnull().sum()  # → 0 for every column ✅
   test_df.isnull().sum()   # → 0 for every column ✅
   ```
   **Finding:** Zero missing values in both datasets. No imputation required.

3. **Data types inspection:**
   ```python
   train_df.dtypes
   ```
   **Finding:** All columns have consistent data types between train and test. `merchant_category` is the only string/object column — all others are numeric (int64 or float64). ✅

4. **Descriptive statistics:**
   ```python
   train_df.describe()
   ```
   This gave us the ranges, means, and standard deviations for all numerical columns.

5. **Class balance check:**
   ```python
   train_df['is_fraud'].value_counts(normalize=True)
   ```
   **Finding:** Only 1.51% of transactions are fraudulent (121 out of 8,000). This is a **65:1 imbalance ratio**.

6. **Merchant category distribution:**
   ```python
   train_df['merchant_category'].value_counts()
   ```
   **Finding:** 5 unique categories — Food, Clothing, Travel, Electronics, Grocery. Roughly balanced distribution across categories.

---

## 3. Phase 2: Exploratory Data Analysis (EDA)

**Source:** `notebooks/01_EDA.ipynb`, `docs/data_cleaning/data_cleaning_plan.md`

### 3.1 Duplicate Analysis
- ✅ **No duplicate rows** in train or test
- ✅ **No duplicate `transaction_id`** values in either set
- ✅ **No overlapping `transaction_id`** between train and test
- **Action:** No deduplication required

### 3.2 Outlier Analysis (IQR Method)
| Feature | Outliers | % of Train |
|---|---|---|
| `amount` | 401 | 5.01% |
| `velocity_last_24h` | 41 | 0.51% |
| `transaction_hour` | 0 | 0.00% |
| `device_trust_score` | 0 | 0.00% |
| `cardholder_age` | 0 | 0.00% |

**Critical Decision:** We decided **NOT to remove outliers**. In fraud detection, outliers are **signal, not noise**. High-amount transactions and high-velocity transactions are correlated with fraud. Removing them would destroy predictive power.

### 3.3 Zero/Edge Value Analysis
| Feature | Zeros | Notes |
|---|---|---|
| `amount` | 1 | A $0.00 transaction — could be a test/probe transaction (common fraud pattern) |
| `transaction_hour` | 341 | Valid — hour `0` means midnight |
| `velocity_last_24h` | 1,120 | Valid — 0 prior transactions in 24h |

**Decision:** All zeros are valid and meaningful. The single `amount = 0` record was kept (zero-dollar test charges are a known fraud pattern). No negative values found anywhere.

### 3.4 Fraud Signal Discovery
This was the most critical part of EDA — understanding which features carry the strongest fraud signal.

| Signal | Fraud Rate | Baseline (1.51%) | Uplift |
|---|---|---|---|
| `location_mismatch = 1` | 8.24% | 1.51% | 🔴 **5.5×** |
| `foreign_transaction = 1` | 8.12% | 1.51% | 🔴 **5.4×** |
| `device_trust_score` 20-40 | 6.04% | 1.51% | 🔴 **4.0×** |
| Night transactions (0-5h) | 4.91% | 1.51% | 🟠 **3.3×** |
| `amount` > 500 | 3.04% | 1.51% | 🟡 **2.0×** |
| High `velocity_last_24h` | corr: +0.110 | — | 🟡 Mild |
| `cardholder_age` | corr: +0.000 | — | ⚪ None |

### 3.5 Correlation Analysis with `is_fraud`
| Feature | Correlation |
|---|---|
| `foreign_transaction` | **+0.179** |
| `location_mismatch` | **+0.168** |
| `device_trust_score` | **−0.138** |
| `transaction_hour` | **−0.135** |
| `velocity_last_24h` | +0.110 |
| `amount` | +0.034 |
| `cardholder_age` | +0.000 |

**Key Insight:** `cardholder_age` has zero correlation with fraud — it provides no predictive signal at all. However, we kept it in the feature set because tree-based models can safely ignore irrelevant features, and removing it could hurt models that find subtle interactions.

### 3.6 Train vs Test Distribution Consistency
| Feature | Train Mean | Test Mean | Drift? |
|---|---|---|---|
| `amount` | ~175 | ~178 | ✅ No drift |
| `device_trust_score` | ~62 | ~61 | ✅ No drift |

**Finding:** All feature distributions are consistent between train and test. No distribution drift correction needed.

### 3.7 Critical Visualizations Performed
The following visualizations were instrumental in discovering the patterns above and must be reproduced in the final notebook:
- **Numerical feature histograms**
- **Side-by-side boxplots** (Fraud vs Legitimate) for continuous features
- **Merchant category distribution**
- **Correlation heatmap** (full feature matrix)
- **Deep dive plots** for `transaction_hour`, `device_trust_score`, `amount`, and `velocity_last_24h` split by fraud class
- **Feature interaction plot** (`foreign_transaction` × `location_mismatch`)
- **Pairplot** colored by fraud class
- **Train vs Test distribution overlay** plots

---

## 4. Phase 3: Data Cleaning & Validation

**Source:** `src/data_processing/preprocess.py` (lines 43-105)

### What We Did (Step by Step)

#### Step 1: Load Raw Data
```python
train = pd.read_csv('data/raw/train.csv')
test  = pd.read_csv('data/raw/test.csv')
```

#### Step 2: Preserve Test IDs
```python
test_ids = test['transaction_id']
test_ids.to_csv('data/processed/test_transaction_ids.csv', index=False)
```
**Rationale:** `transaction_id` is needed for the final `submission.csv` but has zero predictive value. We save it separately before dropping it from the feature matrix.

#### Step 3: Drop Identifier Column
```python
train.drop(columns=['transaction_id'], inplace=True)
test.drop(columns=['transaction_id'], inplace=True)
```

#### Step 4: Separate Target Variable
```python
y_train = train['is_fraud']
train.drop(columns=['is_fraud'], inplace=True)
```

#### Step 5: Combine Train & Test for Consistent Engineering
```python
combined = pd.concat([train, test], axis=0)
```
**Rationale:** By combining them before feature engineering, we guarantee that one-hot encoding produces the exact same columns for both sets (no missing category columns in test).

#### Step 6: Sanity Checks (Post-Processing)
```python
assert train_engineered.isnull().sum().sum() == 0, "Null values found in train!"
assert test_engineered.isnull().sum().sum() == 0, "Null values found in test!"
assert 'is_fraud' in train_engineered.columns, "is_fraud missing from train!"
assert 'is_fraud' not in test_engineered.columns, "is_fraud found in test!"
```
**These 4 assertions guarantee:**
1. No null values were introduced during feature engineering
2. No null values exist in test
3. The target column is present in train
4. The target column has NOT leaked into test (critical data leakage check)

---

## 5. Phase 4: Feature Engineering

**Source:** `src/data_processing/preprocess.py` → `engineer_features()` function (lines 11-41)

Starting from **9 raw features** (after dropping `transaction_id`), we engineered **16 additional features** for a total of **23 model-ready features** (excluding the target).

### 5.1 Time-Based Risk Features

#### `is_night_transaction` (Binary)
```python
df['is_night_transaction'] = df['transaction_hour'].apply(lambda x: 1 if 0 <= x <= 5 else 0)
```
**Rationale:** EDA showed nighttime transactions (midnight–5am) have a **3.3× higher fraud rate** (4.91% vs 1.51% baseline).

#### `time_of_day_category` (Categorical → One-Hot Encoded)
```python
def get_time_category(hour):
    if 0 <= hour <= 5: return 'Night'
    elif 6 <= hour <= 11: return 'Morning'
    elif 12 <= hour <= 17: return 'Afternoon'
    else: return 'Evening'

df['time_of_day_category'] = df['transaction_hour'].apply(get_time_category)
```
**Rationale:** Provides coarser time granularity that tree-based models can use as clean decision boundaries instead of arbitrary hour integers.
**Result:** This later gets one-hot encoded into 4 binary columns: `time_of_day_category_Night`, `time_of_day_category_Morning`, `time_of_day_category_Afternoon`, `time_of_day_category_Evening`.

### 5.2 Trust and Anomaly Ratios

#### `amount_to_trust_ratio` (Continuous)
```python
df['amount_to_trust_ratio'] = df['amount'] / (df['device_trust_score'] + 1e-6)
```
**Rationale:** Fraudsters attempt high-value transactions on low-trust devices. This ratio captures that interaction. The `1e-6` epsilon prevents division by zero (though the minimum trust score is 25, so it's a safety measure).

#### `amount_velocity_ratio` (Continuous)
```python
df['amount_velocity_ratio'] = df['amount'] / (df['velocity_last_24h'] + 1)
```
**Rationale:** Represents the average spend per recent transaction. High velocity coupled with large amounts is a classic indicator of account takeover. The `+1` prevents division by zero (velocity can be 0).

### 5.3 High-Risk Location Flags

#### `is_high_risk_location` (Binary)
```python
df['is_high_risk_location'] = ((df['foreign_transaction'] == 1) | (df['location_mismatch'] == 1)).astype(int)
```
**Rationale:** Both `foreign_transaction` and `location_mismatch` individually correspond to an ~8% fraud rate (5.4× and 5.5× uplift). This feature flags ANY location anomaly.

#### `location_anomaly_score` (Integer, 0-2)
```python
df['location_anomaly_score'] = df['foreign_transaction'] + df['location_mismatch']
```
**Rationale:** Captures the *severity* of location anomalies. A score of 2 (both foreign AND location mismatch) is a much stronger fraud signal than either alone.

### 5.4 Strategic Binning

#### `is_high_amount` (Binary)
```python
df['is_high_amount'] = (df['amount'] > 500).astype(int)
```
**Rationale:** EDA showed the fraud rate doubles for transactions above $500 (3.04% vs 1.51% baseline).

#### `is_low_trust` (Binary)
```python
df['is_low_trust'] = (df['device_trust_score'] < 40).astype(int)
```
**Rationale:** The fraud rate is approximately 4× higher (6.04%) when the device trust score is below 40.

### 5.5 Categorical Encoding

#### One-Hot Encoding
```python
cat_cols = ['merchant_category', 'time_of_day_category']
combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=False)
```
**Decision: `drop_first=False`** — We chose NOT to drop the first category. While dropping one is standard for linear models (to avoid multicollinearity), our primary models are tree-based (XGBoost, CatBoost, AdaBoost), which are immune to multicollinearity. Keeping all categories preserves full information.

**Result:** `merchant_category` → 5 binary columns; `time_of_day_category` → 4 binary columns.

#### Boolean to Integer Conversion
```python
for col in combined_encoded.columns:
    if combined_encoded[col].dtype == 'bool':
        combined_encoded[col] = combined_encoded[col].astype(int)
```
**Rationale:** `pd.get_dummies()` creates boolean columns. Some models (especially CatBoost) can be sensitive to data types, so we convert bools to integers for universal compatibility.

### 5.6 Split Back & Save
```python
train_engineered = combined_encoded.iloc[:len(train)].copy()
test_engineered  = combined_encoded.iloc[len(train):].copy()
train_engineered['is_fraud'] = y_train.values

train_engineered.to_csv('data/processed/train_engineered.csv', index=False)
test_engineered.to_csv('data/processed/test_engineered.csv', index=False)
```

### 5.7 Preserved Raw Features
**Critical Decision:** We kept the original numerical columns (`amount`, `transaction_hour`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`) intact alongside their engineered counterparts. Tree-based models benefit from having access to both raw distributions AND engineered signals.

### 5.8 Final Feature Count Summary
| Category | Features | Count |
|---|---|---|
| Raw numerical | `amount`, `transaction_hour`, `foreign_transaction`, `location_mismatch`, `device_trust_score`, `velocity_last_24h`, `cardholder_age` | 7 |
| Engineered numerical | `is_night_transaction`, `amount_to_trust_ratio`, `amount_velocity_ratio`, `is_high_risk_location`, `location_anomaly_score`, `is_high_amount`, `is_low_trust` | 7 |
| One-Hot: merchant_category | `Food`, `Clothing`, `Travel`, `Electronics`, `Grocery` | 5 |
| One-Hot: time_of_day_category | `Night`, `Morning`, `Afternoon`, `Evening` | 4 |
| **Total features** | | **23** |

**Output Files:**
- `data/processed/train_engineered.csv` → 23 features + `is_fraud` target (24 columns)
- `data/processed/test_engineered.csv` → 23 features (no target)
- `data/processed/test_transaction_ids.csv` → Transaction IDs for submission

### 5.9 Feature Scaling Rationale
**Source:** `notebooks/02_Data_Preprocessing.ipynb`

Scaling was intentionally **deferred** to the modeling pipeline rather than being applied during data preprocessing.

| Model Type | Scaling Needed? | Reason |
|---|---|---|
| Random Forest, XGBoost, LightGBM, CatBoost | ❌ No | Tree-based models split on feature values and are scale-invariant. |
| Logistic Regression, SVM, Neural Networks | ✅ Yes | Distance/gradient-based models are highly sensitive to feature scales. |

**Decision:** Because our primary models are tree-based, keeping raw feature values preserves interpretability and allows tree-based models to access natural distributions. Models that require scaling will apply `StandardScaler` inside their own Scikit-Learn pipelines.

---

## 6. Phase 5: Model Training & Hyperparameter Tuning

**Source:** `src/modeling/train.py`

### 6.1 Strategy Overview
- **Objective:** Maximize F1-Score
- **Tuning Engine:** Optuna (Bayesian Optimization), 20 trials per model, 10-minute timeout per model
- **Validation:** 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`)
- **Imbalance Handling:** Algorithmic class weights (NO SMOTE / NO resampling)

### 6.2 Class Imbalance Strategy
**Decision: Use native algorithmic class weights instead of SMOTE.**

| Algorithm Family | Weight Parameter | How It Works |
|---|---|---|
| XGBoost / LightGBM | `scale_pos_weight = 65.12` | Calculated as `count(negative) / count(positive)` = 7879 / 121 |
| CatBoost | `auto_class_weights = 'Balanced'` | CatBoost internally calculates balanced weights |
| Scikit-Learn (RF, ExtraTrees, LR, SVC) | `class_weight = 'balanced'` | Scikit-learn adjusts weights inversely proportional to class frequencies |
| AdaBoost, GradientBoosting | No native weight param | These models rely on their sequential boosting nature to focus on misclassified (minority) samples |
| KNN, GaussianNB, MLP | No class weight | These models don't support class weights natively |

**Why no SMOTE?** The native algorithmic weights pushed F1 scores above 0.99 without any data resampling. SMOTE would introduce synthetic samples that could distort decision boundaries, especially risky for a competition where the test distribution must match the training distribution exactly.

### 6.3 Feature Scaling Strategy
**Decision: Scale only for models that require it.**

```python
needs_scaling = model_name in ['LogisticRegression', 'SVC', 'KNN', 'GaussianNB', 'MLPClassifier']

if needs_scaling:
    model = Pipeline([('scaler', StandardScaler()), ('clf', model)])
```

- **Tree-based models (XGBoost, CatBoost, RF, etc.):** No scaling needed — trees split on feature values, so scale doesn't matter.
- **Distance/gradient-based models (LR, SVC, KNN, GNB, MLP):** `StandardScaler` applied via a `Pipeline` to prevent data leakage during cross-validation (the scaler is fit only on the training folds, not the validation fold).

### 6.4 The 12 Models Evaluated

Every model was tuned with Optuna (20 trials) and evaluated with 5-Fold Stratified CV:

#### Model 1: Logistic Regression
```python
LogisticRegression(
    C=trial.suggest_float('C', 1e-4, 10.0, log=True),
    class_weight='balanced', random_state=42, max_iter=1000
)
```
- Scaled with `StandardScaler`
- **Result:** F1 = 0.8077, Precision = 0.6862, Recall = 0.9837
- **Insight:** Very high recall (caught 98.37% of fraud) but too many false positives

#### Model 2: Random Forest
```python
RandomForestClassifier(
    n_estimators=trial.suggest_int('n_estimators', 50, 200),
    max_depth=trial.suggest_int('max_depth', 3, 10),
    class_weight='balanced', random_state=42
)
```
- **Result:** F1 = 0.9195, Precision = 0.9913, Recall = 0.8593

#### Model 3: ExtraTrees
```python
ExtraTreesClassifier(
    n_estimators=trial.suggest_int('n_estimators', 50, 200),
    max_depth=trial.suggest_int('max_depth', 3, 15),
    class_weight='balanced', random_state=42
)
```
- **Result:** F1 = 0.8113, Precision = 0.8720, Recall = 0.7677

#### Model 4: XGBoost
```python
XGBClassifier(
    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    max_depth=trial.suggest_int('max_depth', 3, 9),
    scale_pos_weight=65.12, eval_metric='logloss', random_state=42
)
```
- **Result:** F1 = 0.9874, Precision = 1.0000, Recall = 0.9753
- **Best params:** `learning_rate=0.151, max_depth=5`

#### Model 5: LightGBM
```python
LGBMClassifier(
    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    max_depth=trial.suggest_int('max_depth', 3, 9),
    scale_pos_weight=65.12, random_state=42, verbose=-1
)
```
- **Result:** F1 = 0.9785, Precision = 1.0000, Recall = 0.9583
- **Best params:** `learning_rate=0.110, max_depth=8`

#### Model 6: CatBoost
```python
CatBoostClassifier(
    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    depth=trial.suggest_int('depth', 4, 8),
    auto_class_weights='Balanced', random_state=42, verbose=0
)
```
- **Result:** F1 = 0.9838, Precision = 0.9769, Recall = 0.9917
- **Best params:** `learning_rate=0.081, depth=6`

#### Model 7: AdaBoost
```python
AdaBoostClassifier(
    n_estimators=trial.suggest_int('n_estimators', 50, 200),
    learning_rate=trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
    random_state=42
)
```
- **Result:** F1 = 0.9957, Precision = 1.0000, Recall = 0.9917 🏆
- **Best params:** `n_estimators=124, learning_rate=0.669`
- **Top single model!**

#### Model 8: Gradient Boosting
```python
GradientBoostingClassifier(
    n_estimators=trial.suggest_int('n_estimators', 50, 200),
    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    max_depth=trial.suggest_int('max_depth', 3, 8),
    random_state=42
)
```
- **Result:** F1 = 0.9791, Precision = 0.9917, Recall = 0.9673

#### Model 9: SVC (Support Vector Classifier)
```python
SVC(
    C=trial.suggest_float('C', 0.1, 10.0, log=True),
    class_weight='balanced', probability=True, random_state=42
)
```
- Scaled with `StandardScaler`
- **Result:** F1 = 0.8205, Precision = 0.8234, Recall = 0.8253

#### Model 10: KNN (K-Nearest Neighbors)
```python
KNeighborsClassifier(
    n_neighbors=trial.suggest_int('n_neighbors', 3, 15),
    weights=trial.suggest_categorical('weights', ['uniform', 'distance'])
)
```
- Scaled with `StandardScaler`
- **Result:** F1 = 0.5274, Precision = 0.7740, Recall = 0.4110
- **Worst performer after GNB** — distance-based models struggle with high-dimensional sparse features

#### Model 11: Gaussian Naive Bayes
```python
GaussianNB(
    var_smoothing=trial.suggest_float('var_smoothing', 1e-9, 1e-2, log=True)
)
```
- Scaled with `StandardScaler`
- **Result:** F1 = 0.3021, Precision = 0.1852, Recall = 0.8263
- **Worst performer** — the Gaussian assumption is violated by the binary/engineered features

#### Model 12: MLP Classifier (Neural Network)
```python
MLPClassifier(
    hidden_layer_sizes=trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50)]),
    alpha=trial.suggest_float('alpha', 1e-4, 1e-1, log=True),
    random_state=42, max_iter=500
)
```
- Scaled with `StandardScaler`
- **Result:** F1 = 0.8330, Precision = 0.8246, Recall = 0.8427

### 6.5 Final Model Leaderboard

| Rank | Model | F1-Score | Precision | Recall | PR-AUC |
|:---:|---|:---:|:---:|:---:|:---:|
| 🥇 | **AdaBoost** | **0.9957** | 1.0000 | 0.9917 | 0.9994 |
| 🥈 | **XGBoost** | **0.9874** | 1.0000 | 0.9753 | 0.9991 |
| 🥉 | **CatBoost** | **0.9838** | 0.9769 | 0.9917 | 0.9997 |
| 4 | Gradient Boosting | 0.9791 | 0.9917 | 0.9673 | 0.9918 |
| 5 | LightGBM | 0.9785 | 1.0000 | 0.9583 | 0.9916 |
| 6 | Random Forest | 0.9195 | 0.9913 | 0.8593 | 0.9675 |
| 7 | MLPClassifier | 0.8330 | 0.8246 | 0.8427 | 0.9052 |
| 8 | SVC | 0.8205 | 0.8234 | 0.8253 | 0.8735 |
| 9 | ExtraTrees | 0.8113 | 0.8720 | 0.7677 | 0.8884 |
| 10 | Logistic Regression | 0.8077 | 0.6862 | 0.9837 | 0.8919 |
| 11 | KNN | 0.5274 | 0.7740 | 0.4110 | 0.6045 |
| 12 | Gaussian Naive Bayes | 0.3021 | 0.1852 | 0.8263 | 0.3863 |

### 6.6 Key Takeaways from Model Training
1. **Boosting algorithms dominate** — all 5 top models are gradient boosting variants
2. **AdaBoost was the surprise champion** — despite being the oldest boosting algorithm, it achieved the best single-model F1 (0.9957) with perfect precision
3. **XGBoost achieved perfect Precision** (100%) — every transaction it flagged was actually fraud
4. **CatBoost achieved the highest Recall** (99.17%) — it caught nearly every fraudulent transaction
5. **Non-boosting models severely underperformed** — the gap between boosting (>0.97) and non-boosting (<0.92) is massive

### 6.7 Artifact Storage
For every model, we saved:
- **Model file:** `models/<model_name>/best_model.pkl` (pickle serialized)
- **Predictions:** `outputs/<model_name>/predictions.csv`
- **Central log:** `outputs/model_results_log.csv` (all metrics + hyperparameters)

---

## 7. Phase 6: Ensemble Construction (3-Model Soft Voting)

**Source:** `src/modeling/ensemble.py`

### 7.1 Strategy
Combine the top 3 models (AdaBoost, XGBoost, CatBoost) into an Optuna-weighted Soft Voting Ensemble.

**Why only 3 models?** These 3 represent the clear top tier (F1 > 0.98). Adding weaker models (LightGBM at 0.9785, GradientBoosting at 0.9791) would dilute the ensemble quality without adding meaningful diversity.

### 7.2 Step-by-Step Process

#### Step 1: Load Pre-Trained Models
```python
models = {}
for name in ['adaboost', 'xgboost', 'catboost']:
    path = os.path.join(base_dir, "models", name, "best_model.pkl")
    with open(path, 'rb') as f:
        models[name] = pickle.load(f)
```

#### Step 2: Generate Out-Of-Fold (OOF) Probabilities
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    probs = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    oof_probs[name] = probs
```
**Why OOF?** If we used regular training predictions to optimize weights, the weights would overfit to training data. OOF predictions are generated by models that never saw the data they're predicting on — this gives us an unbiased estimate.

#### Step 3: Optimize Weights with Optuna (100 Trials)
```python
def objective(trial):
    w_ada = trial.suggest_float('w_ada', 0.0, 1.0)
    w_xgb = trial.suggest_float('w_xgb', 0.0, 1.0)
    w_cat = trial.suggest_float('w_cat', 0.0, 1.0)

    total = w_ada + w_xgb + w_cat
    w_ada /= total  # Normalize to sum to 1
    w_xgb /= total
    w_cat /= total

    blended_probs = (w_ada * oof_probs['adaboost'] +
                     w_xgb * oof_probs['xgboost'] +
                     w_cat * oof_probs['catboost'])

    preds = (blended_probs >= 0.5).astype(int)
    return f1_score(y, preds)
```
**Note:** Optuna ran 100 trials (not 20 like base models) because weight optimization is computationally cheap — it's just weighted averaging of pre-computed probabilities.

#### Step 4: Optimal Weights Found
| Model | Weight | Trust % |
|---|---|---|
| **XGBoost** | 0.423 | 42.3% |
| **CatBoost** | 0.292 | 29.2% |
| **AdaBoost** | 0.284 | 28.4% |

**Interesting finding:** Despite AdaBoost being the best single model (F1=0.9957), Optuna assigned it the lowest weight. This is because XGBoost provides the most *complementary* signal — it catches different fraud patterns than the other two.

#### Step 5: Build VotingClassifier & Train on Full Data
```python
estimators = [('adaboost', models['adaboost']),
              ('xgboost', models['xgboost']),
              ('catboost', models['catboost'])]

ensemble_model = VotingClassifier(estimators=estimators, voting='soft', weights=best_weights)
ensemble_model.fit(X, y)
```

#### Step 6: Decision Threshold
```python
preds = (blended_probs >= 0.5).astype(int)
```
**Decision:** We used the standard **0.5 threshold**. This was validated to give perfect precision (zero false alarms) with only 1 missed fraud case.

### 7.3 Ensemble Results
| Metric | Score |
|---|---|
| **F1-Score** | **0.9959** |
| **Precision** | **1.0000** |
| **Recall** | **0.9917** |
| **PR-AUC** | **0.9991** |
| **Accuracy** | **0.9999** |

**The ensemble marginally outperformed AdaBoost alone** (0.9959 vs 0.9957). The improvement is tiny but the ensemble provides better generalization insurance for the private leaderboard.

---

## 8. Phase 7: Advanced Ensemble (Stacking Meta-Learner)

**Source:** `src/modeling/advanced_ensemble.py`

### 8.1 Strategy
We attempted a more advanced 2-level stacking approach using all 5 boosting models:

- **Level-0 (Base Models):** AdaBoost, XGBoost, CatBoost, LightGBM, GradientBoosting
- **Level-1 (Meta-Learner):** Ridge Logistic Regression (`LogisticRegression(penalty='l2', C=1.0)`)

### 8.2 Process

#### Step 1: Generate OOF Meta-Features
```python
meta_features = np.zeros((X.shape[0], len(models)))  # Shape: (8000, 5)
for i, (name, model) in enumerate(models.items()):
    probs = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    meta_features[:, i] = probs
```
Each row becomes a 5-dimensional vector of fraud probabilities from each base model.

#### Step 2: Train Meta-Learner on OOF Features
```python
meta_model = LogisticRegression(penalty='l2', C=1.0, random_state=42)
meta_oof_probs = cross_val_predict(meta_model, X_meta, y, cv=cv, method='predict_proba')[:, 1]
```
**Why Logistic Regression?** As documented in our overfitting analysis report, a simple, regularized linear model as the meta-learner prevents overfitting. Using a complex model (like another XGBoost) as the meta-learner would memorize noise.

#### Step 3: Dynamic Threshold Optimization
```python
thresholds = np.arange(0.01, 1.0, 0.01)
for thresh in thresholds:
    preds = (y_probs >= thresh).astype(int)
    score = f1_score(y_true, preds)
    if score > best_f1:
        best_f1 = score
        best_thresh = thresh
```
**Result:** Optimal threshold found at **0.02** (very aggressive — catches maximum fraud).

### 8.3 Advanced Ensemble Results
| Metric | Score |
|---|---|
| **F1-Score** | **0.9959** |
| **Precision** | **1.0000** |
| **Recall** | **0.9917** |
| **PR-AUC** | **0.9990** |

**The advanced stacking ensemble tied the basic soft-voting ensemble** — both achieved F1 = 0.9959. The additional complexity of stacking 5 models through a meta-learner provided zero improvement.

---

## 9. Phase 8: Final Model Selection & Submission

**Source:** `src/modeling/ensemble.py` (lines 151-167) and `src/modeling/advanced_ensemble.py` (lines 147-160)

### 9.1 Selection Logic

Both scripts contain explicit model selection logic:

**In `ensemble.py`:**
```python
if ensemble_metrics['F1_Score'] > best_single_f1:  # best_single_f1 = 0.9957 (AdaBoost)
    ens_sub.to_csv(final_sub_path, index=False)  # Use ensemble
else:
    best_preds = pd.read_csv(best_single_preds_path)  # Use AdaBoost
    best_preds.to_csv(final_sub_path, index=False)
```
**Result:** Ensemble (0.9959) > AdaBoost (0.9957) → **Ensemble predictions saved to `FINAL_SUBMISSION.csv`**

**In `advanced_ensemble.py`:**
```python
if metrics['F1_Score'] > best_single_f1:  # best_single_f1 = 0.9959 (basic ensemble)
    ens_sub.to_csv(final_sub_path, index=False)  # Overwrite with stacking
else:
    print("Keeping the previous FINAL_SUBMISSION.csv intact.")
```
**Result:** Advanced (0.9959) is NOT > basic ensemble (0.9959) → **Previous `FINAL_SUBMISSION.csv` kept intact**

### 9.2 Final Decision
The **3-model Soft Voting Ensemble** (AdaBoost + XGBoost + CatBoost) was selected as the final submission model because:
1. It achieved the highest F1-Score (0.9959)
2. The more complex 5-model stacking approach failed to beat it
3. Simpler models generalize better on unseen private leaderboard data (Occam's Razor)

### 9.3 Pipeline Execution Order
**Source:** `src/run_pipeline.py`

The entire pipeline was orchestrated by a single runner script:
```python
pipeline_scripts = [
    "src/data_processing/preprocess.py",      # Step 1: Feature Engineering
    "src/modeling/train.py",                   # Step 2: Train 12 Models
    "src/modeling/ensemble.py",                # Step 3: 3-Model Soft Voting
    "src/modeling/advanced_ensemble.py",        # Step 4: 5-Model Stacking
    "src/inference/predict.py"                 # Step 5: (Empty — predictions already generated)
]
```
**Note:** `src/inference/predict.py` is an empty file — predictions were already generated within the ensemble scripts themselves.

---

## 10. Final Evaluation, Visualization & Submission Strategy

**Source:** `notebooks/04_Evaluation_and_Prediction.ipynb`

### 10.1 Evaluation Visualizations
To thoroughly validate our final models, the following diagnostic plots were generated on the Out-Of-Fold predictions:
- **Confusion Matrices** for the top models (AdaBoost and the Ensemble)
- **ROC Curves** (Receiver Operating Characteristic)
- **Precision-Recall (PR) Curves** (Crucial for severe class imbalance)
- **Threshold Sweep Plot** (F1, Precision, Recall plotted against threshold values from 0.01 to 0.99)
- **Cross-Model Feature Importance** (Bar charts showing normalized importance ranking averaged across models)

### 10.2 Dual Submission Strategy
Based on the threshold sweep, two distinct submission strategies were explored:
- **Standard Submission (Threshold = 0.5):** The default decision boundary, achieving perfect precision.
- **Aggressive Submission (Threshold = 0.3):** A lower threshold that maximizes recall at the cost of precision, designed to catch borderline fraud cases that the 0.5 threshold might miss.

### 10.3 Submission Validation Logic
Before saving `submission.csv`, a validation function was applied to ensure strict compliance with Kaggle's expected format. It checked:
1. **Shape match:** Exactly matches `sample_submission.csv` (2000 rows).
2. **Column names:** Exactly `transaction_id` and `is_fraud`.
3. **Data types:** Both columns must be integers.
4. **Value range:** `is_fraud` must contain only 0s and 1s.
5. **Transaction IDs:** Must perfectly match the test set IDs.

---

## 11. Final Results Summary

### The Winning Model
**Optuna-Weighted 3-Model Soft Voting Ensemble**
- AdaBoost (weight: 0.284)
- XGBoost (weight: 0.423)
- CatBoost (weight: 0.292)
- Decision threshold: 0.5

### Final OOF Metrics
| Metric | Value |
|---|---|
| **F1-Score** | **0.9959** |
| **Precision** | **1.0000** |
| **Recall** | **0.9917** |
| **PR-AUC** | **0.9991** |
| **Accuracy** | **0.9999** |

### What This Means in Practice
- **Precision = 1.0000:** Every single transaction our model flagged as fraud WAS actually fraud. Zero false alarms.
- **Recall = 0.9917:** Out of 121 fraudulent transactions in training, we correctly identified 120. We missed exactly **1** fraud case.
- **F1 = 0.9959:** The harmonic mean of perfect precision and near-perfect recall.

### Output Files
| File | Location | Description |
|---|---|---|
| Final submission | `outputs/FINAL_SUBMISSION.csv` | The CSV uploaded to Kaggle |
| Ensemble model | `models/ensemble/best_model.pkl` | Serialized VotingClassifier |
| All model results | `outputs/model_results_log.csv` | Central log of all 12 models + 2 ensembles |

---

*Team BigBug — OctWave 3.0 Credit Card Fraud Detection Challenge, August 2026*
