# Data Cleaning Plan — OctWave 3.0 Fraud Detection

## 1. Data Summary

| Property | Train | Test |
|---|---|---|
| Rows | 8,001 (header + 8,000 records) | 2,001 (header + 2,000 records) |
| Features | 9 + target (`is_fraud`) | 9 (no target) |
| File size | ~297 KB | ~70 KB |

**Features**: `transaction_id`, `amount`, `transaction_hour`, `merchant_category`, `foreign_transaction`, `location_mismatch`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`

---

## 2. Key Findings from Analysis

### 2.1 Missing Values
- ✅ **No missing values** in either train or test set.
- **Action**: No imputation required.

### 2.2 Duplicates
- ✅ **No duplicate rows** in train or test.
- ✅ **No duplicate `transaction_id`** values in either set.
- ✅ **No overlapping `transaction_id`** between train and test.
- **Action**: No deduplication required.

### 2.3 Data Type Consistency
- ✅ All columns have **consistent data types** between train and test.
- `merchant_category` is the only string/categorical column — all others are numeric.
- **Action**: No type casting fixes needed.

### 2.4 Target Variable — Severe Class Imbalance

| Class | Count | Percentage |
|---|---|---|
| 0 (Legitimate) | 7,879 | 98.49% |
| 1 (Fraud) | 121 | 1.51% |

**Imbalance ratio: 65:1** — This is extreme imbalance.

> ⚠️ **Critical**: This is the single most important data issue. Standard models trained without addressing this will be biased towards predicting "not fraud" and will achieve a very poor F1-score.

**Action (during modeling — NOT data cleaning):**
- Apply **SMOTE** (Synthetic Minority Oversampling) or **ADASYN** on the training set.
- Alternatively, use `class_weight='balanced'` in models that support it.
- Consider **undersampling** the majority class.
- Ensemble approaches like **BalancedRandomForest** or **EasyEnsemble**.

### 2.5 Outliers (IQR Method)

| Feature | Outliers | % of Train |
|---|---|---|
| `amount` | 401 | 5.01% |
| `velocity_last_24h` | 41 | 0.51% |
| `transaction_hour` | 0 | 0.00% |
| `device_trust_score` | 0 | 0.00% |
| `cardholder_age` | 0 | 0.00% |

**Action**:
- **DO NOT remove outliers.** Fraud detection is inherently about identifying anomalous patterns. High-amount transactions and high-velocity transactions are likely correlated with fraud (confirmed: amount `500+` has a 3.04% fraud rate vs ~1.3% baseline, and `velocity_last_24h` has a positive correlation of 0.110 with `is_fraud`).
- Outliers are **signal, not noise** in this problem domain.

### 2.6 Zero/Edge Values

| Feature | Zeros | Notes |
|---|---|---|
| `amount` | 1 | A $0.00 transaction — could be a test/probe transaction (common in fraud) |
| `transaction_hour` | 341 | Valid — hour `0` means midnight |
| `velocity_last_24h` | 1,120 | Valid — 0 prior transactions in 24h |

**Action**:
- The single `amount = 0` record is valid and should be kept (zero-dollar test charges are a known fraud pattern).
- No negative values found anywhere. ✅

### 2.7 Feature Distributions — Train vs Test Consistency
Distributions are very consistent between train and test:
- `amount`: mean ~175 (train) vs ~178 (test) ✅
- `device_trust_score`: mean ~62 (train) vs ~61 (test) ✅
- All other features show similar distribution characteristics.

**Action**: No distribution drift correction needed.

---

## 3. Data Cleaning Steps (Ordered)

### Step 1: Validate & Load Data
- [x] Load `train.csv` and `test.csv`
- [x] Confirm no missing values
- [x] Confirm no duplicates
- [x] Confirm data types are consistent

### Step 2: Drop Identifier Column
- Drop `transaction_id` from the feature set before training (store separately for submission).
- `transaction_id` is an arbitrary unique identifier with no predictive value.

### Step 3: Encode Categorical Variable
- `merchant_category` has 5 unique values: `Food`, `Clothing`, `Travel`, `Electronics`, `Grocery`.
- **Recommended approach**: **One-Hot Encoding** (since only 5 categories, dimensionality increase is minimal).
- Alternative: **Label Encoding** if using tree-based models (XGBoost, LightGBM handle ordinal encoding natively).

### Step 4: Verify Binary Columns
- `foreign_transaction` and `location_mismatch` are already binary (0/1). ✅
- No cleaning needed — just confirm they stay as integers.

### Step 5: Feature Scaling (Model-Dependent)
- For **tree-based models** (Random Forest, XGBoost, LightGBM): **No scaling needed**.
- For **Logistic Regression / Neural Networks**: Apply **StandardScaler** or **MinMaxScaler** to:
  - `amount` (range: 0 – 1,471)
  - `device_trust_score` (range: 25 – 99)
  - `velocity_last_24h` (range: 0 – 9)
  - `cardholder_age` (range: 18 – 69)
  - `transaction_hour` (range: 0 – 23)

### Step 6: Save Cleaned Data
- Save cleaned train and test sets to `data/processed/` directory.
- Maintain a separate file for `transaction_id` mapping for test submission.

---

## 4. Fraud Signal Summary (For Feature Engineering Phase)

These insights from the analysis will inform feature engineering (next phase, not part of cleaning):

| Signal | Fraud Rate | Baseline | Strength |
|---|---|---|---|
| `foreign_transaction = 1` | 8.12% | 1.51% | 🔴 **Strong** (5.4x) |
| `location_mismatch = 1` | 8.24% | 1.51% | 🔴 **Strong** (5.5x) |
| `device_trust_score` 20-40 | 6.04% | 1.51% | 🔴 **Strong** (4x) |
| Night transactions (0-5h) | 4.91% | 1.51% | 🟠 **Moderate** (3.3x) |
| `amount` 500+ | 3.04% | 1.51% | 🟡 **Mild** (2x) |
| `velocity_last_24h` (high) | corr: 0.110 | — | 🟡 **Mild** |
| `cardholder_age` | corr: 0.000 | — | ⚪ **None** |

### Correlation with `is_fraud`
| Feature | Correlation |
|---|---|
| `foreign_transaction` | +0.179 |
| `location_mismatch` | +0.168 |
| `device_trust_score` | −0.138 |
| `transaction_hour` | −0.135 |
| `velocity_last_24h` | +0.110 |
| `amount` | +0.034 |
| `cardholder_age` | +0.000 |

---

## 5. What Does NOT Need Cleaning

| Item | Reason |
|---|---|
| Missing values | None exist |
| Duplicates | None exist |
| Negative values | None exist |
| Data type mismatches | All consistent |
| Train/test distribution drift | Distributions align well |
| Outliers | Meaningful signal for fraud detection |

---

## 6. Summary

> **This is a clean dataset.** The data requires minimal preprocessing — primarily encoding `merchant_category`, dropping `transaction_id` from features, and optional scaling. The critical challenge is the **65:1 class imbalance**, which must be addressed during the modeling phase (not during data cleaning) using resampling techniques or class-weighted algorithms.

### Cleaning Checklist

| # | Task | Status |
|---|---|---|
| 1 | Handle missing values | ✅ None found |
| 2 | Remove duplicates | ✅ None found |
| 3 | Fix data types | ✅ All consistent |
| 4 | Drop `transaction_id` from features | 🔲 Pending |
| 5 | Encode `merchant_category` | 🔲 Pending |
| 6 | Verify binary columns | ✅ Already correct |
| 7 | Feature scaling (if needed) | 🔲 Model-dependent |
| 8 | Save cleaned data | 🔲 Pending |
